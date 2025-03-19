import os
import pickle
import numpy as np
import torch.nn as nn

from envs.car import CarEnv
from envs.pvtol import PvtolEnv
from envs.quadrotor import QuadRotorEnv
from envs.neurallander import NeuralLanderEnv


from utils.get_args import get_args
from utils.misc import seed_all
from utils.sampler import OnlineSampler


def save_npz(data_dict, filename="data.npz"):
    """
    Save a dictionary of NumPy arrays to a single .npz file.
    """
    np.savez_compressed(filename, **data_dict)


def load_npz(filename="data.npz"):
    """
    Load the dictionary of NumPy arrays from a .npz file.
    Returns a dict with the same keys as saved.
    """
    loaded = np.load(filename)
    # Convert the NpzFile object to a regular dictionary
    data = {key: loaded[key] for key in loaded}
    avg_rewards = np.sum(data["rewards"]) / np.sum(data["terminals"])

    print(f"data keys: {data.keys()}")
    print(f"data length: {len(data['rewards'])}")
    print(f"data performance: {avg_rewards:2f}")

    return data


def call_env(task: str, sigma: float):
    if task == "car":
        env = CarEnv(sigma=sigma)
    elif task == "pvtol":
        env = PvtolEnv(sigma=sigma)
    elif task == "quadrotor":
        env = QuadRotorEnv(sigma=sigma)
    elif task == "neurallander":
        env = NeuralLanderEnv(sigma=sigma)
    else:
        raise NotImplementedError(f"{task} is not implemented.")

    return env


def get_random_policy(state_dim: int, action_dim: int):
    from policy.ppo import PPO
    from policy.layers.ppo_networks import PPO_Actor, PPO_Critic

    actor = PPO_Actor(state_dim, hidden_dim=[64, 64], a_dim=action_dim)
    critic = PPO_Critic(state_dim, hidden_dim=[128, 128])
    policy = PPO(actor=actor, critic=critic)

    return policy


def get_policy(task: str, quality: str | int, sigma: float):
    # Map string qualities to numeric scores
    quality_map = {"random": 0.0, "medium": 0.5, "expert": 1.0}

    # Determine the numeric quality_score
    if isinstance(quality, str):
        quality_score = quality_map.get(quality, 0.0)
    else:
        quality_score = float(quality)

    # Build the file path
    policy_path = os.path.join("model", task, str(sigma), f"model_{quality_score}.p")

    # Load and return the policy
    with open(policy_path, "rb") as f:
        policy = pickle.load(f)
    return policy


def concat_batches(batches):
    """
    Given a list of batch dictionaries, each containing arrays under the same keys,
    concatenate them along the first dimension.
    """
    # Assume there's at least one batch in the list
    if len(batches) == 1:
        return batches[0]
    else:
        # Each batch is a dict like {"states": [...], "actions": [...], ...}
        # We'll create a final_data dict that holds concatenated arrays
        final_data = {}

        # Get the keys from the first batch (e.g., 'states', 'actions', 'terminations')
        keys = batches[0].keys()

        # For each key, concatenate the corresponding arrays from all batches
        for k in keys:
            final_data[k] = np.concatenate([batch[k] for batch in batches], axis=0)

        return final_data


def data_loop(env, policy: nn.Module, size: int):
    ALLOWBLE_DATA = ["states", "actions", "next_states", "rewards", "terminals"]

    sampler = OnlineSampler(
        state_dim=state_dim,
        action_dim=action_dim,
        episode_len=episode_len,
        batch_size=size + episode_len,
        verbose=False,
    )

    batches = []
    count = 0
    current_size = 0
    while current_size <= size:
        batch, _ = sampler.collect_samples(
            env=env, policy=policy, seed=seed, deterministic=True
        )

        # Find all indices where the episode terminates
        done_indices = np.where(batch["terminals"] == 1)[0]
        last_done_idx = done_indices[-1]

        # Slice all arrays up to and including the last terminal index
        truncated_batch = {}
        for k, arr in batch.items():
            if k in ALLOWBLE_DATA:
                truncated_batch[k] = arr[: last_done_idx + 1]

        batches.append(truncated_batch)
        current_size += len(truncated_batch["rewards"])
        count += 1

    print(f"Terminating data loop with size: {current_size} and loop: {count}")
    data = concat_batches(batches)
    return data


def generate_dataset(task: str, quality: str, sigma: float, size: int):
    # fix seed
    global seed
    seed = 1_234_567
    seed_all(seed)

    env = call_env(task, sigma)

    global state_dim, action_dim, episode_len
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    episode_len = env.episode_len

    # generate data according to given quality
    if quality == "random":
        policy = get_random_policy(state_dim=state_dim, action_dim=action_dim)
        data = data_loop(env, policy, size)
        data_list = [data]
    else:
        if quality == "medium-replay":
            policies = [get_random_policy(state_dim=state_dim, action_dim=action_dim)]
            for quality in [0.1, 0.2, 0.3, 0.4, 0.5]:
                policies.append(get_policy(task, quality, sigma))

            num_policies = len(policies)
            minibatch = size // num_policies

            data_list = []
            for policy in policies:
                data = data_loop(env, policy, minibatch)
                data_list.append(data)

        elif quality == "medium-expert":
            policies = [get_policy(task, quality, sigma) for quality in [0.5, 1.0]]
            num_policies = len(policies)
            minibatch = size // num_policies

            data_list = []
            for policy in policies:
                data = data_loop(env, policy, minibatch)
                data_list.append(data)

        else:
            policy = get_policy(task, quality, sigma)
            data = data_loop(env, policy, size)
            data_list = [data]

    temp_data = concat_batches(data_list)
    data = {}
    for k, arr in temp_data.items():
        data[k] = arr[:size]

    data_name = f"{task}_{quality}_{sigma}.npz"
    save_npz(data, filename=f"{data_name}")
    load_npz(data_name)


# generate training and test separately
# this will be different task parameters, different spawn locations

if __name__ == "__main__":
    ################
    # data params
    ################
    # quality_list = ["random", "medium", "medium-replay", "medium-expert", "expert"]
    task = "car"
    quality = "medium-replay"
    sigma = 0.0
    size = 100_000

    generate_dataset(task, quality, sigma, size)
