import torch
import argparse


def get_args():
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
    parser.add_argument(
        "--quality", type=str, default="expert", help="Name of the model."
    )
    parser.add_argument("--algo-name", type=str, default="mrl", help="Disable cuda.")
    parser.set_defaults(use_cuda=True)
    parser.add_argument("--seed", type=int, default=42, help="Batch size.")
    parser.add_argument(
        "--num-runs", type=int, default=10, help="Number of samples for training."
    )  # 4096 * 32
    parser.add_argument(
        "--actor-lr", type=float, default=None, help="Base learning rate."
    )
    parser.add_argument(
        "--critic-lr", type=float, default=None, help="Base learning rate."
    )
    parser.add_argument("--W-lr", type=float, default=None, help="Base learning rate.")
    parser.add_argument("--u-lr", type=float, default=None, help="Base learning rate.")
    parser.add_argument("--w-ub", type=float, default=None, help="Base learning rate.")
    parser.add_argument("--w-lb", type=float, default=None, help="Base learning rate.")
    parser.add_argument(
        "--eps-clip", type=float, default=None, help="Base learning rate."
    )
    parser.add_argument("--eps", type=float, default=None, help="Base learning rate.")
    parser.add_argument("--lbd", type=float, default=None, help="Base learning rate.")
    parser.add_argument(
        "--actor-dim", type=list, default=None, help="Base learning rate."
    )
    parser.add_argument(
        "--critic-dim", type=list, default=None, help="Base learning rate."
    )

    parser.add_argument(
        "--timesteps", type=int, default=None, help="Number of training epochs."
    )
    parser.add_argument(
        "--log-interval", type=int, default=None, help="Number of training epochs."
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=None, help="Number of training epochs."
    )
    parser.add_argument(
        "--eval-num", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--sigma", type=float, default=None, help="Number of training epochs."
    )
    parser.add_argument("--num-minibatch", type=int, default=None, help="")
    parser.add_argument("--minibatch-size", type=int, default=None, help="")
    parser.add_argument("--K-epochs", type=int, default=None, help="")
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Upper bound of the eigenvalue of the dual metric.",
    )
    parser.add_argument(
        "--gae",
        type=float,
        default=0.95,
        help="Lower bound of the eigenvalue of the dual metric.",
    )
    parser.add_argument(
        "--entropy-scaler", type=float, default=None, help="Base learning rate."
    )
    parser.add_argument("--gamma", type=float, default=None, help="Base learning rate.")
    parser.add_argument(
        "--load-pretrained-model",
        action="store_true",
        help="Path to a directory for storing the log.",
    )

    parser.add_argument(
        "--gpu-idx", type=int, default=0, help="Number of training epochs."
    )

    args = parser.parse_args()
    args.device = select_device(args.gpu_idx)

    return args


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
