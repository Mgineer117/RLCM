import numpy as np
import uuid
import random
import datetime
import json
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--task', type=str,
                        default='car', help='Name of the model.')
parser.add_argument('--algo-name', type=str, help='Disable cuda.')
parser.set_defaults(use_cuda=True)
parser.add_argument('--seed', type=int, default=42, help='Batch size.')
parser.add_argument('--num_runs', type=int, default=10, help='Number of samples for training.') # 4096 * 32
parser.add_argument('--policy-lr', type=float, default=0.001, help='Base learning rate.')
parser.add_argument('--critic-lr', type=float, default=0.001, help='Base learning rate.')
parser.add_argument('--policy-dim', type=list, default=[64, 64], help='Base learning rate.')
parser.add_argument('--critic-dim', type=list, default=[64, 64], help='Base learning rate.')

parser.add_argument('--timesteps', type=int, default=15, help='Number of training epochs.')
parser.add_argument('--log-intervals', type=int, default=10, help='Number of training epochs.')
parser.add_argument('--eval-num', type=int, default=10, help='Number of training epochs.')

parser.add_argument('--num-minibatch', type=int, default=10, help='')
parser.add_argument('--minibatch-size', type=int, default=128, help='')
parser.add_argument('--K-epochs', type=int, default=5, help='')
parser.add_argument('--eps-clip', type=float, default=0.2, help='Convergence rate: lambda')
parser.add_argument('--target-kl', type=float, default=0.02, help='Upper bound of the eigenvalue of the dual metric.')
parser.add_argument('--gae', type=float, default=0.95, help='Lower bound of the eigenvalue of the dual metric.')
parser.add_argument('--load-pretrained-model', action="store_true", help='Path to a directory for storing the log.')

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
    
def run():



if __name__ == "__main__":
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    random.set_seed(init_args.seed)
    seeds = [random.randint(1, 10_000) for _ in range(init_args.num_runs)]
    print(f"-------------------------------------------------------")
    print(f"      Running ID: {unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"      Time Begun   : {exp_time}")
    print(f"-------------------------------------------------------")

    for seed in seeds:
        args = override_args()
        run(args, seed, unique_id, exp_time)
    # concat_csv_columnwise_and_delete(folder_path=args.logdir)