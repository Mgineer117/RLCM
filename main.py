import numpy as np
import torch
import uuid
import random
import datetime
import json
import argparse

from envs.car import CarEnv
from utils.rl import get_policy
from utils.misc import setup_logger
from utils.sampler import OnlineSampler
from trainer.online_trainer import Trainer

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--project", type=str, default="Exp", help="WandB project classification"
)
parser.add_argument(
    "--logdir", type=str, default="log/train_log", help="name of the logging folder"
)
parser.add_argument(
    "--group",
    type=str,
    default=None,
    help="Global folder name for experiments with multiple seed tests.",
)
parser.add_argument(
    "--name",
    type=str,
    default=None,
    help='Seed-specific folder name in the "group" folder.',
)
parser.add_argument("--task", type=str, default="car", help="Name of the model.")
parser.add_argument("--algo-name", type=str, default="ppo", help="Disable cuda.")
parser.set_defaults(use_cuda=True)
parser.add_argument("--seed", type=int, default=42, help="Batch size.")
parser.add_argument(
    "--num_runs", type=int, default=10, help="Number of samples for training."
)  # 4096 * 32
parser.add_argument(
    "--actor-lr", type=float, default=0.0001, help="Base learning rate."
)
parser.add_argument(
    "--critic-lr", type=float, default=0.0003, help="Base learning rate."
)
parser.add_argument(
    "--actor-dim", type=list, default=[64, 64], help="Base learning rate."
)
parser.add_argument(
    "--critic-dim", type=list, default=[128, 128], help="Base learning rate."
)

parser.add_argument(
    "--timesteps", type=int, default=1e7, help="Number of training epochs."
)
parser.add_argument(
    "--log-intervals", type=int, default=10, help="Number of training epochs."
)
parser.add_argument(
    "--eval_episodes", type=int, default=10, help="Number of training epochs."
)
parser.add_argument(
    "--eval-num", type=int, default=10, help="Number of training epochs."
)

parser.add_argument("--num-minibatch", type=int, default=10, help="")
parser.add_argument("--minibatch-size", type=int, default=512, help="")
parser.add_argument("--K-epochs", type=int, default=3, help="")
parser.add_argument("--eps", type=float, default=0.2, help="Convergence rate: lambda")
parser.add_argument(
    "--target-kl",
    type=float,
    default=0.02,
    help="Upper bound of the eigenvalue of the dual metric.",
)
parser.add_argument(
    "--gae",
    type=float,
    default=0.95,
    help="Lower bound of the eigenvalue of the dual metric.",
)
parser.add_argument(
    "--entropy-scaler", type=float, default=0.001, help="Base learning rate."
)
parser.add_argument("--gamma", type=float, default=0.99, help="Base learning rate.")
parser.add_argument(
    "--load-pretrained-model",
    action="store_true",
    help="Path to a directory for storing the log.",
)

parser.add_argument("--gpu-idx", type=int, default=0, help="Number of training epochs.")

init_args = parser.parse_args()


def override_args(env_name: str | None = None):
    args = parser.parse_args()
    file_path = f"config/{args.algo_name}/{args.task}.json"
    current_params = load_hyperparams(file_path=file_path, task=args.task)

    # use pre-defined params if no pram given as args
    for k, v in current_params.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    return args


def load_hyperparams(file_path, task):
    """Load hyperparameters for a specific environment from a JSON file."""
    try:
        with open(file_path, "r") as f:
            hyperparams = json.load(f)
            return hyperparams.get(task, {})
    except FileNotFoundError:
        print(
            f"No file found at {file_path}. Returning default empty dictionary for {task}."
        )
        return {}


def select_device(gpu_idx=0, verbose=True):
    if verbose:
        print(
            "============================================================================================"
        )
        # set device to cpu or cuda
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")
        print(
            "============================================================================================"
        )
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
    return device


def run(args, seed, unique_id, exp_time):
    # get env
    env = CarEnv()
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.episode_len = env.episode_len
    policy = get_policy(args)
    sampler = OnlineSampler(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        episode_len=args.episode_len,
        batch_size=int(args.minibatch_size * args.num_minibatch),
    )
    logger, writer = setup_logger(args, unique_id, exp_time, seed)

    # get policy
    trainer = Trainer(
        env=env,
        policy=policy,
        sampler=sampler,
        logger=logger,
        writer=writer,
        timesteps=args.timesteps,
        log_interval=args.log_intervals,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )

    trainer.train()


if __name__ == "__main__":
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    random.seed(init_args.seed)
    seeds = [random.randint(1, 10_000) for _ in range(init_args.num_runs)]
    print(f"-------------------------------------------------------")
    print(f"      Running ID: {unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"      Time Begun   : {exp_time}")
    print(f"-------------------------------------------------------")

    for seed in seeds:
        args = override_args()
        args.seed = seed
        args.device = select_device(args.gpu_idx, True)

        run(args, seed, unique_id, exp_time)
    # concat_csv_columnwise_and_delete(folder_path=args.logdir)
