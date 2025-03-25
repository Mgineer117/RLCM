import time
import os
import pickle
import torch
import torch.nn as nn
from torch import matmul, inverse, transpose
import numpy as np

# from utils.torch import get_flat_grad_from, get_flat_params_from, set_flat_params_to
from utils.rl import estimate_advantages

# from actor.layers.building_blocks import MLP
from policy.base import Base

# from models.layers.ppo_networks import PPO_Policy, PPO_Critic


class MRL(Base):
    def __init__(
        self,
        x_dim: int,
        effective_indices: list,
        W_func: nn.Module,
        actor: nn.Module,
        critic: nn.Module,
        W_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        w_ub: float = 1e-2,
        lbd: float = 1e-2,
        eps: float = 1e-2,
        eps_clip: float = 0.2,
        entropy_scaler: float = 1e-3,
        l2_reg: float = 1e-5,
        target_kl: float = 0.03,
        gamma: float = 0.99,
        gae: float = 0.9,
        K: int = 5,
        nupdates: int = 0,
        dt: float = 0.03,
        device: str = "cpu",
    ):
        super(MRL, self).__init__()

        # constants
        self.name = "MRL"
        self.device = device

        self.x_dim = x_dim
        self.action_dim = actor.action_dim
        self.effective_x_dim = len(effective_indices)
        self.effective_indices = effective_indices

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size
        self._entropy_scaler = entropy_scaler
        self.eps = eps
        self.gamma = gamma
        self.gae = gae
        self.K = K
        self.l2_reg = l2_reg
        self.target_kl = target_kl
        self.lbd = lbd
        self.eps_clip = eps_clip
        self.w_ub = w_ub
        self.dt = dt

        self._forward_steps = 0
        self.nupdates = nupdates
        self.current_update = 0

        # trainable networks
        self.W_func = W_func
        self.actor = actor
        self.critic = critic

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.W_func.parameters(), "lr": W_lr},
                {"params": self.actor.parameters(), "lr": actor_lr},
                {"params": self.critic.parameters(), "lr": critic_lr},
            ]
        )

        #
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        self._forward_steps += 1
        state = torch.from_numpy(state).to(self._dtype).to(self.device)

        a, metaData = self.actor(state, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
        }

    def loss_pos_matrix_random_sampling(self, A: torch.Tensor):
        # A: n x d x d
        # z: K x d
        n, A_dim, _ = A.shape

        z = torch.randn((n, A_dim)).to(self.device)
        z = z / z.norm(dim=-1, keepdim=True)
        z = z.unsqueeze(-1)
        zT = transpose(z, 1, 2)

        # K x d @ d x d = n x K x d
        zTAz = matmul(matmul(zT, A), z)

        negative_index = zTAz.detach().cpu().numpy() < 0
        if negative_index.sum() > 0:
            negative_zTAz = zTAz[negative_index]
            return -1.0 * (negative_zTAz.mean())
        else:
            return torch.tensor(0.0).type(z.type()).requires_grad_()

    def trim_state(self, state: torch.Tensor):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        # state trimming
        x = state[:, : self.x_dim]
        xref = state[:, self.x_dim : -self.action_dim]
        uref = state[:, -self.action_dim :]

        # require grad for x only
        # x = x.requires_grad_()

        x_trim = x[:, self.effective_indices]
        xref_trim = xref[:, self.effective_indices]

        return x, xref, uref, x_trim, xref_trim

    def get_rewards(self, states):
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)

        with torch.no_grad():
            W = self.W_func(x, xref, uref, x_trim, xref_trim)
            M = torch.inverse(W)

        error = (x - xref).unsqueeze(-1)
        errorT = transpose(error, 1, 2)

        rewards = (1 / (errorT @ M @ error + 1)).squeeze(-1)
        return rewards

    def contraction_loss(
        self, states: torch.Tensor, next_states: torch.Tensor, detach: bool
    ):
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        next_x, next_xref, next_uref, next_x_trim, next_xref_trim = self.trim_state(
            next_states
        )

        W = self.W_func(x, xref, uref, x_trim, xref_trim)
        M = torch.inverse(W)

        if detach:
            W = W.detach()
            M = M.detach()

        with torch.no_grad():
            next_W = self.W_func(
                next_x, next_xref, next_uref, next_x_trim, next_xref_trim
            )
            next_M = torch.inverse(next_W)

        error = (x - xref).unsqueeze(-1)
        errorT = transpose(error, 1, 2)

        # dot_x_approx = (next_x - x) / self.dt
        dot_M_approx = (next_M - M) / self.dt
        ABK_approx = error @ errorT

        # print(M.shape, ABK_approx.shape)
        MABK = matmul(M, ABK_approx)
        sym_MABK = MABK + transpose(MABK, 1, 2)

        C_u = dot_M_approx + sym_MABK + 2 * self.lbd * M  # .detach()

        pd_loss = self.loss_pos_matrix_random_sampling(
            -C_u - self.eps * torch.eye(C_u.shape[-1]).to(self.device)
        )
        overshoot_loss = self.loss_pos_matrix_random_sampling(
            self.w_ub * torch.eye(W.shape[-1]).to(self.device) - W
        )

        C_eig, _ = torch.linalg.eig(C_u)
        C_eig = torch.real(C_eig)
        C_eig_contraction = ((C_eig >= 0).sum(dim=-1) == 0).cpu().detach().numpy()

        return pd_loss, overshoot_loss, C_eig_contraction.mean()

    def learn(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        if self.current_update >= int(0.3 * self.nupdates):
            detach = True
        else:
            detach = False

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        actions = to_tensor(batch["actions"])
        next_states = to_tensor(batch["next_states"])
        original_rewards = to_tensor(batch["rewards"])
        rewards = self.get_rewards(states)
        terminals = to_tensor(batch["terminals"])
        old_logprobs = to_tensor(batch["logprobs"])

        # Compute advantages and returns
        with torch.no_grad():
            values = self.critic(states)
            advantages, returns = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self.gamma,
                gae=self.gae,
                device=self.device,
            )

        # Mini-batch training
        batch_size = states.size(0)

        # List to track actor loss over minibatches
        losses = []
        actor_losses = []
        value_losses = []
        entropy_losses = []
        pd_losses = []
        overshoot_losses = []
        C_eig_contraction_losses = []

        clip_fractions = []
        target_kl = []
        grad_dicts = []

        for k in range(self.K):
            for n in range(self.num_minibatch):
                indices = torch.randperm(batch_size)[: self.minibatch_size]
                mb_states, mb_next_states, mb_actions = (
                    states[indices],
                    next_states[indices],
                    actions[indices],
                )
                mb_old_logprobs, mb_returns = old_logprobs[indices], returns[indices]

                # advantages
                mb_advantages = advantages[indices]
                mb_advantages = (
                    mb_advantages - mb_advantages.mean()
                ) / mb_advantages.std()

                # 1. Critic Update (with optional regularization)
                mb_values = self.critic(mb_states)
                value_loss = self.mse_loss(mb_values, mb_returns)
                l2_reg = (
                    sum(param.pow(2).sum() for param in self.critic.parameters())
                    * self.l2_reg
                )
                value_loss += l2_reg

                # Track value loss for logging
                value_losses.append(value_loss.item())

                # 2. actor Update
                _, metaData = self.actor(mb_states)
                logprobs = self.actor.log_prob(metaData["dist"], mb_actions)
                entropy = self.actor.entropy(metaData["dist"])
                ratios = torch.exp(logprobs - mb_old_logprobs)

                surr1 = ratios * mb_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * mb_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = self._entropy_scaler * entropy.mean()

                # Track actor loss for logging
                actor_losses.append(actor_loss.item())
                entropy_losses.append(entropy_loss.item())

                pd_loss, overshoot_loss, C_eig_contraction = self.contraction_loss(
                    mb_states, mb_next_states, detach
                )
                pd_losses.append(pd_loss.item())
                overshoot_losses.append(overshoot_loss.item())
                C_eig_contraction_losses.append(C_eig_contraction)

                # Total loss
                loss = (
                    actor_loss
                    + 0.5 * value_loss
                    - entropy_loss
                    + pd_loss
                    + overshoot_loss
                )
                losses.append(loss.item())

                # Compute clip fraction (for logging)
                clip_fraction = torch.mean(
                    (torch.abs(ratios - 1) > self.eps_clip).float()
                ).item()
                clip_fractions.append(clip_fraction)

                # Check if KL divergence exceeds target KL for early stopping
                kl_div = torch.mean(mb_old_logprobs - logprobs)
                target_kl.append(kl_div.item())
                if kl_div.item() > self.target_kl:
                    break

                # Update critic parameters
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                grad_dict = self.compute_gradient_norm(
                    [self.W_func, self.actor, self.critic],
                    ["W_func", "actor", "critic"],
                    dir=f"{self.name}",
                    device=self.device,
                )
                grad_dicts.append(grad_dict)
                self.optimizer.step()

            if kl_div.item() > self.target_kl:
                break

        # Logging
        loss_dict = {
            f"{self.name}/loss/loss": np.mean(losses),
            f"{self.name}/loss/actor_loss": np.mean(actor_losses),
            f"{self.name}/loss/value_loss": np.mean(value_losses),
            f"{self.name}/loss/entropy_loss": np.mean(entropy_losses),
            f"{self.name}/loss/pd_loss": np.mean(pd_losses),
            f"{self.name}/loss/overshoot_loss": np.mean(overshoot_losses),
            f"{self.name}/analytics/clip_fraction": np.mean(clip_fractions),
            f"{self.name}/analytics/C_eig_contraction": np.mean(C_eig_contraction),
            f"{self.name}/analytics/klDivergence": target_kl[-1],
            f"{self.name}/analytics/K-epoch": k + 1,
            f"{self.name}/analytics/avg_rewards": torch.mean(original_rewards).item(),
        }
        grad_dict = self.average_dict_values(grad_dicts)
        norm_dict = self.compute_weight_norm(
            [self.W_func, self.actor, self.critic],
            ["W_func", "actor", "critic"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        self.current_update += 1

        # Cleanup
        del states, actions, rewards, terminals, old_logprobs
        self.eval()

        timesteps = self.num_minibatch * self.minibatch_size
        update_time = time.time() - t0
        return loss_dict, timesteps, update_time

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values and counts for each key
        sum_dict = {}
        count_dict = {}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                if key not in sum_dict:
                    sum_dict[key] = 0
                    count_dict[key] = 0
                sum_dict[key] += value
                count_dict[key] += 1

        # Calculate the average for each key
        avg_dict = {key: sum_val / count_dict[key] for key, sum_val in sum_dict.items()}

        return avg_dict
