import os
import time
import numpy as np
import torch
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from tqdm import tqdm
from collections import deque

from log.wandb_logger import WandbLogger
from policy.base import Base

from utils.sampler import OnlineSampler

COLORS = {
    "0": "magenta",
    "1": "red",
    "2": "blue",
    "3": "green",
    "4": "yellow",
    "5": "orange",
    "6": "purple",
    "7": "pink",
    "8": "brown",
    "9": "grey",
}


# model-free policy trainer
class Trainer:
    def __init__(
        self,
        env: gym.Env,
        policy: Base,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        timesteps: int = 1e6,
        lr_scheduler: torch.optim.lr_scheduler = None,
        log_interval: int = 2,
        eval_episodes: int = 10,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.policy = policy
        self.sampler = sampler
        self.eval_episodes = eval_episodes

        self.logger = logger
        self.writer = writer

        # training parameters
        self.timesteps = timesteps
        self.eval_num = 0
        self.eval_interval = int(self.timesteps / log_interval)
        self.lr_scheduler = lr_scheduler

        # initialize the essential training components
        self.last_max_reward = -1e10
        self.std_limit = 0.5

        self.log_interval = log_interval
        self.seed = seed

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_reward_mean = deque(maxlen=3)
        self.last_reward_std = deque(maxlen=3)

        # Train loop
        with tqdm(total=self.timesteps, desc=f"PPO Training (Timesteps)") as pbar:
            while pbar.n < self.timesteps:
                self.policy.train()
                batch, sample_time = self.sampler.collect_samples(
                    env=self.env, policy=self.policy, seed=self.seed
                )
                loss_dict, ppo_timesteps, update_time = self.policy.learn(batch)

                # Calculate expected remaining time
                pbar.update(ppo_timesteps)

                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / pbar.n
                remaining_time = avg_time_per_iter * (self.timesteps - pbar.n)

                # Update environment steps and calculate time metrics
                loss_dict[f"{self.policy.name}/analytics/timesteps"] = pbar.n
                loss_dict[f"{self.policy.name}/analytics/sample_time"] = sample_time
                loss_dict[f"{self.policy.name}/analytics/update_time"] = update_time
                loss_dict[f"{self.policy.name}/analytics/remaining_time (hr)"] = (
                    remaining_time / 3600
                )  # Convert to hours

                self.write_log(loss_dict, step=pbar.n)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                if pbar.n >= self.eval_interval * (self.eval_num + 1):
                    ### Eval Loop
                    self.policy.eval()
                    self.eval_num += 1

                    eval_dict, traj_plot = self.evaluate()

                    # Manual logging
                    self.write_log(eval_dict, step=pbar.n, eval_log=True)
                    self.write_image(
                        traj_plot, step=pbar.n, logdir=f"{self.policy.name}"
                    )

                    self.last_reward_mean.append(
                        eval_dict[f"{self.policy.name}/eval/rew_mean"]
                    )
                    self.last_reward_std.append(
                        eval_dict[f"{self.policy.name}/eval/rew_std"]
                    )

                    self.save_model(pbar.n)

            torch.cuda.empty_cache()

        self.logger.print(
            "total PPO training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

    def evaluate(self, dimension: int = 2):
        assert dimension in [2, 3], "Dimension must be 2 or 3"

        # Define the figure and draw the reference trajectory
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            self.env.xref[:, 0],
            self.env.xref[:, 1],
            linestyle="--",
            c="black",
            label="Reference",
        )

        ep_buffer = []
        for num_episodes in range(self.eval_episodes):
            ep_reward, ep_tracking_error, ep_control_effort = 0, 0, 0

            # Env initialization
            obs, _ = self.env.reset(seed=self.seed)
            trajectory = [obs[:dimension]]

            done = False
            while not done:
                with torch.no_grad():
                    a, _ = self.policy(obs, deterministic=False)
                    a = a.cpu().numpy().squeeze()
                    if a.shape == ():  # Ensure it's an array
                        a = np.array([a.item()])

                next_obs, rew, term, trunc, infos = self.env.step(a)
                trajectory.append(next_obs[:dimension])  # Store trajectory point
                done = term or trunc

                obs = next_obs
                ep_reward += rew
                ep_tracking_error += infos["tracking_error"]
                ep_control_effort += infos["control_effort"]

                if done:
                    ep_buffer.append(
                        {
                            "reward": ep_reward,
                            "tracking_error": ep_tracking_error,
                            "control_effort": ep_control_effort,
                        }
                    )

                    trajectory = np.array(trajectory)
                    ax.plot(
                        trajectory[:, 0],
                        trajectory[:, 1],
                        linestyle="-",
                        alpha=0.7,
                        c=COLORS[str(num_episodes)],
                        label=str(num_episodes),
                    )

        # Convert figure to a NumPy array
        # Render the figure to update the canvas
        fig.canvas.draw()

        # Extract image from the figure as a NumPy array (RGBA format)
        image_array = np.array(fig.canvas.renderer.buffer_rgba())

        # Close the figure to free memory
        plt.close(fig)

        rew_list = [ep_info["reward"] for ep_info in ep_buffer]
        trk_list = [ep_info["tracking_error"] for ep_info in ep_buffer]
        ctr_list = [ep_info["control_effort"] for ep_info in ep_buffer]

        rew_mean, rew_std = np.mean(rew_list), np.std(rew_list)
        trk_mean, trk_std = np.mean(trk_list), np.std(trk_list)
        ctr_mean, ctr_std = np.mean(ctr_list), np.std(ctr_list)

        eval_dict = {
            f"{self.policy.name}/eval/rew_mean": rew_mean,
            f"{self.policy.name}/eval/rew_std": rew_std,
            f"{self.policy.name}/eval/trk_mean": trk_mean,
            f"{self.policy.name}/eval/trk_std": trk_std,
            f"{self.policy.name}/eval/ctr_mean": ctr_mean,
            f"{self.policy.name}/eval/ctr_std": ctr_std,
        }

        return eval_dict, image_array

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, image: np.ndarray, step: int, logdir: str):
        image_list = [image]
        path_image_path = os.path.join(logdir, "traj_plot")
        self.logger.write_images(step=step, images=image_list, logdir=path_image_path)

    def save_model(self, e):
        # save checkpoint
        if e % self.log_interval == 0:
            self.policy.save_model(self.logger.checkpoint_dirs[4], e)

        # save the best model
        if (
            np.mean(self.last_reward_mean) > self.last_max_reward
            and np.mean(self.last_reward_std) <= self.std_limit
        ):
            self.policy.save_model(self.logger.log_dirs[4], e, is_best=True)
            self.last_max_reward = np.mean(self.last_reward_mean)

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values for each key
        sum_dict = {key: 0 for key in dict_list[0].keys()}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                sum_dict[key] += value

        # Calculate the average for each key
        avg_dict = {key: sum_val / len(dict_list) for key, sum_val in sum_dict.items()}

        return avg_dict
