import os
import time
import pickle
import numpy as np
import torch
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt

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
        log_interval: int = 2,
        eval_num: int = 10,
        eval_episodes: int = 10,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.policy = policy
        self.sampler = sampler
        self.eval_num = eval_num
        self.eval_episodes = eval_episodes

        self.logger = logger
        self.writer = writer

        # training parameters
        self.timesteps = timesteps
        self.nupdates = self.timesteps // self.policy.minibatch_size

        self.log_interval = log_interval
        self.eval_interval = int(self.timesteps / self.log_interval)

        # initialize the essential training components
        self.last_max_reward = -1e10
        self.std_limit = 0.5

        self.seed = seed

    def train(self, scheduler: str | None = None) -> dict[str, float]:
        start_time = time.time()

        self.last_reward_mean = deque(maxlen=3)
        self.last_reward_std = deque(maxlen=3)

        ### DEFINE LR SCHEDULER ####
        if scheduler == "lambda":

            def lr_lambda(step):
                # linearly decay from 1.0 at step=0 to 0.0 at step=max_steps
                return 1.0 - float(step) / float(self.nupdates)

            lr_scheduler = LambdaLR(self.policy.optimizer, lr_lambda=lr_lambda)
        else:
            lr_scheduler = None

        # Train loop
        eval_idx = 0
        with tqdm(total=self.timesteps, desc=f"PPO Training (Timesteps)") as pbar:
            while pbar.n < self.timesteps:
                step = pbar.n + 1  # + 1 to avoid zero division

                self.policy.train()
                batch, sample_time = self.sampler.collect_samples(
                    env=self.env, policy=self.policy, seed=self.seed
                )
                loss_dict, ppo_timesteps, update_time = self.policy.learn(batch)

                # Calculate expected remaining time
                pbar.update(ppo_timesteps)

                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / step
                remaining_time = avg_time_per_iter * (self.timesteps - step)

                # Update environment steps and calculate time metrics
                loss_dict[f"{self.policy.name}/analytics/timesteps"] = step
                loss_dict[f"{self.policy.name}/analytics/sample_time"] = sample_time
                loss_dict[f"{self.policy.name}/analytics/update_time"] = update_time
                loss_dict[f"{self.policy.name}/analytics/learning_rate"] = (
                    self.policy.optimizer.param_groups[0]["lr"]
                )
                loss_dict[f"{self.policy.name}/analytics/remaining_time (hr)"] = (
                    remaining_time / 3600
                )  # Convert to hours

                self.write_log(loss_dict, step=step)

                #### LR-SCHEDULING ####
                if lr_scheduler is not None:
                    lr_scheduler.step()

                #### EVALUATIONS ####
                if step >= self.eval_interval * (eval_idx + 1):
                    ### Eval Loop
                    self.policy.eval()
                    eval_idx += 1

                    eval_dict_list = []
                    for i in range(self.eval_num):
                        eval_dict, traj_plot = self.evaluate()
                        eval_dict_list.append(eval_dict)

                    eval_dict = self.average_dict_values(eval_dict_list)

                    # Manual logging
                    self.write_log(eval_dict, step=step, eval_log=True)
                    self.write_image(
                        traj_plot,
                        step=step,
                        logdir=f"{self.policy.name}",
                        name="traj_plot",
                    )

                    self.last_reward_mean.append(
                        eval_dict[f"{self.policy.name}/eval/rew_mean"]
                    )
                    self.last_reward_std.append(
                        eval_dict[f"{self.policy.name}/eval/rew_std"]
                    )

                    self.save_model(step)

            torch.cuda.empty_cache()

        self.logger.print(
            "total PPO training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

    def evaluate(self):
        """
        Given one ref, show tracking performance
        """
        dimension = self.env.pos_dimension
        assert dimension in [2, 3], "Dimension must be 2 or 3"

        # Set subplot parameters based on dimension
        subplot_kw = {"projection": "3d"} if dimension == 3 else {}
        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2, subplot_kw=subplot_kw, figsize=(10, 6)
        )

        # Dynamically create the coordinate list and plot the reference trajectory
        coords = [self.env.xref[:, i] for i in range(dimension)]
        first_point = [c[0] for c in coords]
        ax1.scatter(
            *first_point,
            marker="*",
            alpha=0.7,
            c="black",
            s=80.0,
        )
        ax1.plot(*coords, linestyle="--", c="black", label="Reference")

        error_norm_trajs = []
        error_trajs = []
        tref_trajs = []
        ep_buffer = []
        for num_episodes in range(self.eval_episodes):
            ep_reward, ep_tracking_error, ep_control_effort = 0, 0, 0
            auc_list = []

            # Env initialization
            options = {"replace_x_0": True}
            obs, infos = self.env.reset(seed=self.seed, options=options)

            trajectory = [infos["x"][:dimension]]
            error_norm_trajectory = [np.linalg.norm(self.env.xref[0] - infos["x"])]
            error_trajectory = [np.linalg.norm(self.env.xref[0] - infos["x"])]
            tref_trajectory = [self.env.time_steps]

            for t in range(1, self.env.episode_len + 1):
                with torch.no_grad():
                    a, _ = self.policy(obs, deterministic=True)
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                next_obs, rew, term, trunc, infos = self.env.step(a)
                trajectory.append(infos["x"][:dimension])  # Store trajectory point
                error_norm_trajectory.append(
                    np.linalg.norm(self.env.xref[t] - infos["x"])
                    / self.env.init_tracking_error
                )
                error_trajectory.append(np.linalg.norm(self.env.xref[t] - infos["x"]))
                tref_trajectory.append(self.env.time_steps)
                done = term or trunc

                obs = next_obs
                ep_reward += rew
                ep_tracking_error += infos["tracking_error"]
                ep_control_effort += infos["control_effort"]

                auc_list.append(infos["relative_tracking_error"])

                if done:
                    auc = np.trapezoid(auc_list, dx=self.env.dt)
                    ep_buffer.append(
                        {
                            "reward": ep_reward,
                            "auc": auc,
                            "tracking_error": ep_tracking_error,
                            "control_effort": ep_control_effort,
                        }
                    )

                    trajectory = np.array(trajectory)
                    coords = [trajectory[:, i] for i in range(dimension)]
                    first_point = [c[0] for c in coords]
                    ax1.scatter(
                        *first_point,
                        marker="*",
                        alpha=0.7,
                        c=COLORS[str(num_episodes)],
                        s=80.0,
                    )
                    ax1.plot(
                        *coords,
                        linestyle="-",
                        alpha=0.7,
                        c=COLORS[str(num_episodes)],
                        label=str(num_episodes),
                    )

                    error_norm_trajs.append(error_norm_trajectory)
                    error_trajs.append(error_trajectory)
                    tref_trajs.append(tref_trajectory)

                    break

        # Optional: Add axis labels
        ax1.set_xlabel("X", labelpad=10)
        ax1.set_ylabel("Y", labelpad=10)
        if dimension == 3:
            ax1.set_zlabel("Z", labelpad=10)
            # Set a nice viewing angle for 3D
            ax1.view_init(elev=25, azim=45)

        # calculate the mean and std of the traj norm error to make plot
        for t, traj in zip(tref_trajs, error_norm_trajs):
            ax2.plot(t, traj)

        plt.tight_layout()

        # Convert figure to a NumPy array
        # Render the figure to update the canvas
        fig.canvas.draw()

        # Extract image from the figure as a NumPy array (RGBA format)
        image_array = np.array(fig.canvas.renderer.buffer_rgba())

        # Close the figure to free memory
        plt.close(fig)

        rew_list = [ep_info["reward"] for ep_info in ep_buffer]
        auc_list = [ep_info["auc"] for ep_info in ep_buffer]
        trk_list = [ep_info["tracking_error"] for ep_info in ep_buffer]
        ctr_list = [ep_info["control_effort"] for ep_info in ep_buffer]

        rew_mean, rew_std = np.mean(rew_list), np.std(rew_list)
        auc_mean, auc_std = np.mean(auc_list), np.std(auc_list)
        trk_mean, trk_std = np.mean(trk_list), np.std(trk_list)
        ctr_mean, ctr_std = np.mean(ctr_list), np.std(ctr_list)

        convergence_rate = self.compute_convergence_rate(tref_trajs, error_trajs)

        eval_dict = {
            f"{self.policy.name}/eval/rew_mean": rew_mean,
            f"{self.policy.name}/eval/rew_std": rew_std,
            f"{self.policy.name}/eval/auc_mean": auc_mean,
            f"{self.policy.name}/eval/auc_std": auc_std,
            f"{self.policy.name}/eval/trk_error_mean": trk_mean,
            f"{self.policy.name}/eval/trk_error_std": trk_std,
            f"{self.policy.name}/eval/ctr_effort_mean": ctr_mean,
            f"{self.policy.name}/eval/ctr_effort_std": ctr_std,
            f"{self.policy.name}/eval/convergence_rate": convergence_rate,
        }

        return eval_dict, image_array

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, image: np.ndarray, step: int, logdir: str, name: str):
        image_list = [image]
        path_image_path = os.path.join(logdir, name)
        self.logger.write_images(step=step, images=image_list, logdir=path_image_path)

    def save_model(self, e):
        # save checkpoint
        name = f"model_{e}.p"
        path = os.path.join(self.logger.checkpoint_dir, name)
        pickle.dump(
            (self.policy),
            open(path, "wb"),
        )
        # self.policy.save_model(self.logger.checkpoint_dir, e)

        # save the best model
        if (
            np.mean(self.last_reward_mean) > self.last_max_reward
            and np.mean(self.last_reward_std) <= self.std_limit
        ):
            name = f"best_model_{e}.p"
            path = os.path.join(self.logger.log_dir, name)
            pickle.dump(
                (self.policy),
                open(path, "wb"),
            )

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

    def compute_convergence_rate(self, tref_trajs, error_trajs, alpha=0.05):
        """
        Given a list of error trajectories:
        - tref_trajs: list of lists (or arrays) for time,
        - error_trajs: list of NumPy arrays for errors,
        1) Use the first trajectory to find (lambda*, C*) that minimizes area under C e^{-lambda t}.
        2) For each trajectory, fix C*, solve for the largest lambda_i such that e_i(t) <= C* e^{-lambda_i t}.
        3) Compute ratio r_i = lambda_i / C*, gather them across all trajectories.
        4) Return the (1 - alpha)-quantile of {r_i}.

        :param tref_trajs: list of time sequences, each can be a Python list or a NumPy array
                        e.g., [ [0, 0.1, 0.2, ...], [0, 0.05, 0.1, ...], ... ]
        :param error_trajs: list of NumPy arrays for the corresponding error values
                            e.g., [array([...]), array([...]), ...]
        :param alpha: float, significance level (default=0.05).
        :return: float, the (1 - alpha)-quantile of the ratio (lambda_i / C*) across trajectories.
        """

        # -----------------------------------------------------------
        # Step A: Find (lambda*, C*) that minimize the AUC on the FIRST trajectory
        # -----------------------------------------------------------
        # Convert first trajectory time to a NumPy array (if it's not already)
        t_ref = np.array(tref_trajs[0], dtype=np.float64)
        e_ref = error_trajs[0]  # assumed to be a NumPy array

        # The horizon T is the max time
        T = np.max(t_ref)

        def area_of_lambda(lmbd):
            """Compute A(lambda) = C(lambda)*(1 - e^{-lmbd*T}) / lmbd, with C(lambda) >= 1."""
            if lmbd <= 0:
                return np.inf
            # C(lmbd) = max(1, max_j [ e_ref[j] * exp(lmbd * t_ref[j]) ])
            temp = np.clip(lmbd * t_ref, None, 700)  # no larger than 700
            c_val = max(1.0, np.max(e_ref * np.exp(temp)))
            return c_val * (1.0 - np.exp(-lmbd * T)) / lmbd

        # Simple log-spaced grid search for lambda in [1e-4, 1e2]
        lambdas = np.logspace(-4, 2, 200, dtype=np.float64)
        areas = [area_of_lambda(l) for l in lambdas]
        idx_min = np.argmin(areas)
        lambda_star = lambdas[idx_min]

        # Now compute C_star
        C_star = max(1.0, np.max(e_ref * np.exp(lambda_star * t_ref)))

        # -----------------------------------------------------------
        # Step B: For each trajectory, find the largest lambda_i s.t. e(t) <= C_star e^{-lambda_i t}
        # -----------------------------------------------------------
        def feasible_lambda_for_trajectory(t, e, c_val):
            """
            Solve for the largest lambda_i satisfying
                e[j] <= c_val * exp(-lambda_i * t[j]) for all j.
            =>  e[j] * exp(lambda_i * t[j]) <= c_val
            =>  lambda_i <= (1/t[j]) * ln(c_val / e[j])  for e[j]>0, t[j]>0
            """
            mask = (e > 0) & (t > 0)
            if not np.any(mask):
                # If e is zero for all t or t=0, no constraint => lambda_i can be "infinite"
                return np.inf

            e_nonzero = e[mask]
            t_nonzero = t[mask]
            bounds = (
                np.log(c_val / e_nonzero) / t_nonzero
            )  # might be inf if e_nonzero < c_val
            return np.min(bounds)

        lambda_ratios = []
        # Convert each t_i to float array if needed, e_i is already a NumPy array
        for t_i, e_i in zip(tref_trajs, error_trajs):
            t_i = np.array(t_i, dtype=float)
            e_i = np.array(e_i, dtype=float)
            lam_i = feasible_lambda_for_trajectory(t_i, e_i, C_star)
            ratio_i = lam_i / C_star
            lambda_ratios.append(ratio_i)

        # -----------------------------------------------------------
        # Step C: Compute the (1 - alpha)-quantile of these ratios
        # -----------------------------------------------------------
        # e.g., alpha=0.05 => 95th percentile
        lambda_ratio_quantile = np.quantile(lambda_ratios, 1 - alpha)

        return lambda_ratio_quantile
