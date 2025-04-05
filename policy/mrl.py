import time
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import matmul, inverse, transpose
from torch.linalg import matrix_norm
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from typing import Callable
from copy import deepcopy

# from utils.torch import get_flat_grad_from, get_flat_params_from, set_flat_params_to
from utils.functions import estimate_advantages

# from actor.layers.building_blocks import MLP
from policy.base import Base


# from models.layers.ppo_networks import PPO_Policy, PPO_Critic


class MRL(Base):
    def __init__(
        self,
        x_dim: int,
        effective_indices: list,
        W_func: nn.Module,
        f_func: Callable,
        B_func: Callable,
        Bbot_func: Callable,
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
        W_entropy_scaler: float = 1e-3,
        entropy_scaler: float = 1e-3,
        control_scaler: float = 0.0,
        l2_reg: float = 1e-8,
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
        self.W_entropy_scaler = W_entropy_scaler
        self._entropy_scaler = entropy_scaler
        self.control_scaler = control_scaler
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

        self.f_func = f_func
        self.B_func = B_func
        if Bbot_func is None:
            self.Bbot_func = self.B_null
        else:
            self.Bbot_func = Bbot_func

        self.nupdates = nupdates
        self.num_outer_update = 0
        self.num_inner_update = 0

        # trainable networks
        self.W_func = W_func
        self.W_optimizer = torch.optim.Adam(params=self.W_func.parameters(), lr=W_lr)

        self.actor = actor
        self.critic = critic

        self.ppo_optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": actor_lr},
                {"params": self.critic.parameters(), "lr": critic_lr},
            ]
        )

        self.lr_scheduler = LambdaLR(self.W_optimizer, lr_lambda=self.lr_lambda)

        self.to(self.device)

    def lr_lambda(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        x, xref, uref, x_trim, xref_trim = self.trim_state(state)
        a, metaData = self.actor(
            x, xref, uref, x_trim, xref_trim, deterministic=deterministic
        )

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
        }

    def learn(self, batch):
        detach = True if self.num_outer_update <= int(0.1 * self.nupdates) else False

        loss_dict, timesteps, update_time = self.learn_ppo(batch)

        if self.num_inner_update % 5 == 0:
            W_loss_dict, W_update_time = self.learn_W(batch, detach)

            loss_dict.update(W_loss_dict)
            update_time += W_update_time

            self.num_outer_update += 1
            self.lr_scheduler.step()
        else:
            loss_dict = {}
            timesteps = 0
            update_time = 0

        self.num_inner_update += 1

        return loss_dict, timesteps, update_time

    def learn_W(self, batch: dict, detach: bool):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        rewards = to_tensor(batch["rewards"])

        # List to track actor loss over minibatches
        states = states.requires_grad_()
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)

        if self.W_entropy_scaler > 0:
            W, infos = self.W_func(states)
        else:
            W, infos = self.W_func(states, deterministic=True)
        M = inverse(W)

        f = self.f_func(x).to(self.device)  # n, x_dim
        B = self.B_func(x).to(self.device)  # n, x_dim, action

        DfDx = self.Jacobian(f, x)  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x)  # n, x_dim, x_dim, b_dim
        Bbot = self.Bbot_func(x).to(self.device)  # n, x_dim, state - action dim

        u, _ = self.actor(x, xref, uref, x_trim, xref_trim, deterministic=True)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        u = u.detach()
        K = K.detach()

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.action_dim)
            ]
        )

        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = self.weighted_gradients(M, dot_x, x, detach)
        ABK = A + matmul(B, K)
        if detach:
            MABK = matmul(M.detach(), ABK)
            sym_MABK = MABK + transpose(MABK, 1, 2)
            C_u = dot_M + sym_MABK + 2 * self.lbd * M.detach()
        else:
            MABK = matmul(M, ABK)
            sym_MABK = MABK + transpose(MABK, 1, 2)
            C_u = dot_M + sym_MABK + 2 * self.lbd * M

        # C1
        DfW = self.weighted_gradients(W, f, x, detach)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = DfDxW + transpose(DfDxW, 1, 2)

        # this has to be a negative definite matrix
        C1_inner = -DfW + sym_DfDxW + 2 * self.lbd * W
        C1 = matmul(matmul(transpose(Bbot, 1, 2), C1_inner), Bbot)

        C2_inners = []
        C2s = []
        for j in range(self.action_dim):
            DbW = self.weighted_gradients(W, B[:, :, j], x, detach)
            DbDxW = matmul(DBDx[:, :, :, j], W)
            sym_DbDxW = DbDxW + transpose(DbDxW, 1, 2)
            C2_inner = DbW - sym_DbDxW
            C2 = matmul(matmul(transpose(Bbot, 1, 2), C2_inner), Bbot)
            C2_inners.append(C2_inner)
            C2s.append(C2)

        #### COMPUTE LOSS ####
        pd_loss = self.loss_pos_matrix_random_sampling(
            -C_u - self.eps * torch.eye(C_u.shape[-1]).to(self.device)
        )
        c1_loss = self.loss_pos_matrix_random_sampling(
            -C1 - self.eps * torch.eye(C1.shape[-1]).to(self.device)
        )
        # c2_loss = sum([C2.sum().mean() for C2 in C2s])
        c2_loss = sum([(matrix_norm(C2) ** 2).mean() for C2 in C2s])
        overshoot_loss = self.loss_pos_matrix_random_sampling(
            self.w_ub * torch.eye(W.shape[-1]).to(self.device) - W
        )

        ############# entropy loss ################
        mean_penalty = torch.exp(-rewards.mean())
        mean_entropy = infos["entropy"].mean()

        entropy_loss = self.W_entropy_scaler * mean_penalty * mean_entropy

        loss = pd_loss + c1_loss + c2_loss + overshoot_loss - entropy_loss

        self.W_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.W_func.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.W_func],
            ["W_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        self.W_optimizer.step()

        ### for loggings
        with torch.no_grad():
            dot_M_pos_eig, dot_M_neg_eig = self.get_matrix_eig(dot_M)
            sym_MABK_pos_eig, sym_MABK_neg_eig = self.get_matrix_eig(sym_MABK)
            M_pos_eig, M_neg_eig = self.get_matrix_eig(M)

            C_pos_eig, C_neg_eig = self.get_matrix_eig(C_u)
            C1_pos_eig, C1_neg_eig = self.get_matrix_eig(C1)

        # Logging
        loss_dict = {
            f"{self.name}/W_loss/loss": loss.item(),
            f"{self.name}/W_loss/pd_loss": pd_loss.item(),
            f"{self.name}/W_loss/c1_loss": c1_loss.item(),
            f"{self.name}/W_loss/c2_loss": c2_loss.item(),
            f"{self.name}/W_loss/overshoot_loss": overshoot_loss.item(),
            f"{self.name}/W_loss/entropy_loss": entropy_loss.item(),
            f"{self.name}/W_loss/mean_penalty": mean_penalty.item(),
            f"{self.name}/W_loss/mean_entropy": mean_entropy.item(),
            f"{self.name}/C_analytics/C_pos_eig": C_pos_eig.item(),
            f"{self.name}/C_analytics/C_neg_eig": C_neg_eig.item(),
            f"{self.name}/C_analytics/C1_pos_eig": C1_pos_eig.item(),
            f"{self.name}/C_analytics/C1_neg_eig": C1_neg_eig.item(),
            f"{self.name}/C_analytics/dot_M_pos_eig": dot_M_pos_eig.item(),
            f"{self.name}/C_analytics/dot_M_neg_eig": dot_M_neg_eig.item(),
            f"{self.name}/C_analytics/sym_MABK_pos_eig": sym_MABK_pos_eig.item(),
            f"{self.name}/C_analytics/sym_MABK_neg_eig": sym_MABK_neg_eig.item(),
            f"{self.name}/C_analytics/M_pos_eig": M_pos_eig.item(),
            f"{self.name}/C_analytics/M_neg_eig": M_neg_eig.item(),
            f"{self.name}/learning_rate/W_lr": self.W_optimizer.param_groups[0]["lr"],
        }
        norm_dict = self.compute_weight_norm(
            [self.W_func],
            ["W_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states
        self.eval()

        update_time = time.time() - t0
        return loss_dict, update_time

    def learn_ppo(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        actions = to_tensor(batch["actions"])
        original_rewards = to_tensor(batch["rewards"])
        rewards = self.get_rewards(states, actions)
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

        clip_fractions = []
        target_kl = []
        grad_dicts = []

        for k in range(self.K):
            for n in range(self.num_minibatch):
                indices = torch.randperm(batch_size)[: self.minibatch_size]
                mb_states, mb_actions = states[indices], actions[indices]
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
                x, xref, uref, x_trim, xref_trim = self.trim_state(mb_states)
                _, metaData = self.actor(x, xref, uref, x_trim, xref_trim)
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

                # Total loss
                loss = actor_loss + 0.5 * value_loss - entropy_loss
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
                self.ppo_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                grad_dict = self.compute_gradient_norm(
                    [self.actor, self.critic],
                    ["actor", "critic"],
                    dir=f"{self.name}",
                    device=self.device,
                )
                grad_dicts.append(grad_dict)
                self.ppo_optimizer.step()

            if kl_div.item() > self.target_kl:
                break

        # Logging
        loss_dict = {
            f"{self.name}/loss/loss": np.mean(losses),
            f"{self.name}/loss/actor_loss": np.mean(actor_losses),
            f"{self.name}/loss/value_loss": np.mean(value_losses),
            f"{self.name}/loss/entropy_loss": np.mean(entropy_losses),
            f"{self.name}/analytics/clip_fraction": np.mean(clip_fractions),
            f"{self.name}/analytics/klDivergence": target_kl[-1],
            f"{self.name}/analytics/K-epoch": k + 1,
            f"{self.name}/analytics/avg_rewards": torch.mean(original_rewards).item(),
            f"{self.name}/analytics/corrected_avg_rewards": torch.mean(rewards).item(),
        }
        grad_dict = self.average_dict_values(grad_dicts)
        norm_dict = self.compute_weight_norm(
            [self.actor, self.critic],
            ["actor", "critic"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, actions, rewards, terminals, old_logprobs
        self.eval()

        timesteps = self.num_minibatch * self.minibatch_size
        update_time = time.time() - t0
        return loss_dict, timesteps, update_time

    def get_rewards(self, states: torch.Tensor, actions: torch.Tensor):
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        x = x.requires_grad_()

        with torch.no_grad():
            ### Compute the main rewards
            W, infos = self.W_func(states, deterministic=True)
            M = torch.inverse(W)

            error = (x - xref).unsqueeze(-1)
            errorT = transpose(error, 1, 2)

            rewards = (1 / (errorT @ M @ error + 1)).squeeze(-1)

        ### Compute the aux rewards ###
        fuel_efficiency = 1 / (torch.linalg.norm(actions, dim=-1, keepdim=True) + 1)
        rewards = rewards + self.control_scaler * fuel_efficiency

        return rewards


class MRL_Approximation(Base):
    def __init__(
        self,
        x_dim: int,
        effective_indices: list,
        W_func: nn.Module,
        Dynamic_func: nn.Module,
        f_func: Callable,
        B_func: Callable,
        Bbot_func: Callable,
        actor: nn.Module,
        critic: nn.Module,
        W_lr: float = 3e-4,
        Dynamic_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        w_ub: float = 1e-2,
        lbd: float = 1e-2,
        eps: float = 1e-2,
        eps_clip: float = 0.2,
        W_entropy_scaler: float = 1e-3,
        entropy_scaler: float = 1e-3,
        control_scaler: float = 1e-3,
        l2_reg: float = 1e-5,
        target_kl: float = 0.03,
        gamma: float = 0.99,
        gae: float = 0.9,
        K: int = 5,
        nupdates: int = 0,
        dt: float = 0.03,
        device: str = "cpu",
    ):
        super(MRL_Approximation, self).__init__()

        # constants
        self.name = "MRL_Approximation"
        self.device = device

        self.x_dim = x_dim
        self.action_dim = actor.action_dim
        self.effective_x_dim = len(effective_indices)
        self.effective_indices = effective_indices

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size
        self.W_entropy_scaler = W_entropy_scaler
        self._entropy_scaler = entropy_scaler
        self.control_scaler = control_scaler
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

        self.f_func = f_func
        self.B_func = B_func
        if Bbot_func is None:
            self.Bbot_func = self.B_null
        else:
            self.Bbot_func = Bbot_func

        self.nupdates = nupdates
        self.num_outer_update = 0
        self.num_inner_update = 0

        # trainable networks
        self.W_func = W_func
        self.Dynamic_func = Dynamic_func

        self.actor = actor
        self.critic = critic

        self.W_optimizer = torch.optim.Adam(params=self.W_func.parameters(), lr=W_lr)
        self.Dynamic_optimizer = torch.optim.Adam(
            params=self.Dynamic_func.parameters(), lr=Dynamic_lr
        )
        self.ppo_optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": actor_lr},
                {"params": self.critic.parameters(), "lr": critic_lr},
            ]
        )

        self.tau = 1.0  # for soft update of target networks

        self.W_lr_scheduler = LambdaLR(self.W_optimizer, lr_lambda=self.W_lr_fn)
        self.D_lr_scheduler = LambdaLR(self.Dynamic_optimizer, lr_lambda=self.D_lr_fn)

        #
        self.to(self.device)

    def W_lr_fn(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def D_lr_fn(self, step):
        return 0.999**step

    def PPO_lr_fn(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        x, xref, uref, x_trim, xref_trim = self.trim_state(state)
        a, metaData = self.actor(
            x, xref, uref, x_trim, xref_trim, deterministic=deterministic
        )

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
        }

    def get_true_metrics(self, states: torch.Tensor):
        #### COMPUTE THE REAL DYNAMICS TO MEASURE ERRORS ####
        states = states.requires_grad_()
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)

        with torch.no_grad():
            W, _ = self.W_func(states, deterministic=True)
            M = inverse(W)

        f = self.f_func(x).to(self.device)  # n, x_dim
        B = self.B_func(x).to(self.device)  # n, x_dim, action

        DfDx = self.Jacobian(f, x).detach()  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x).detach()  # n, x_dim, x_dim, b_dim
        Bbot = self.Bbot_func(x).detach().to(self.device)

        u, _ = self.actor(x, xref, uref, x_trim, xref_trim, deterministic=True)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        u = u.detach()
        K = K.detach()

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.action_dim)
            ]
        )

        dot_x = f + matmul(B, u.unsqueeze(-1)).squeeze(-1)
        dot_M = self.weighted_gradients(M, dot_x, x, True)

        ABK = A + matmul(B, K)
        MABK = matmul(M.detach(), ABK)
        sym_MABK = MABK + transpose(MABK, 1, 2)

        return {
            "dot_x_true": dot_x.detach(),
            "dot_M_true": dot_M.detach(),
            "ABK_true": ABK.detach(),
            "sym_MABK_true": sym_MABK.detach(),
            "Bbot_true": Bbot.detach(),
            "f_true": f.detach(),
            "B_true": B.detach(),
        }

    def learn(self, batch):
        if self.num_inner_update <= int(0.1 * self.nupdates):
            loss_dict, update_time = self.learn_Dynamics(batch)
            loss_dict = {}
            timesteps = 0
            update_time = 0
            self.num_inner_update += 1
        else:
            detach = (
                True if self.num_outer_update <= int(0.1 * self.nupdates) else False
            )

            loss_dict, timesteps, update_time = self.learn_ppo(batch)

            if self.num_inner_update % 5 == 0:
                D_loss_dict, D_update_time = self.learn_Dynamics(batch)
                W_loss_dict, W_update_time = self.learn_W(batch, detach)

                loss_dict.update(W_loss_dict)
                loss_dict.update(D_loss_dict)
                update_time += W_update_time
                update_time += D_update_time

                self.num_outer_update += 1
                self.W_lr_scheduler.step()
                self.D_lr_scheduler.step()
            else:
                loss_dict = {}
                timesteps = 0
                update_time = 0

            self.num_inner_update += 1

        return loss_dict, timesteps, update_time

    def learn_Dynamics(self, batch: dict):
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        actions = to_tensor(batch["actions"])

        true_dict = self.get_true_metrics(states)

        x, _, _, _, _ = self.trim_state(states)
        f_approx, B_approx, Bbot_approx = self.Dynamic_func(x)

        f = self.f_func(x).to(self.device)  # n, x_dim
        B = self.B_func(x).to(self.device)  # n, x_dim, action

        dot_x_true = f + matmul(B, actions.unsqueeze(-1)).squeeze(-1)
        dot_x_approx = f_approx + matmul(B_approx, actions.unsqueeze(-1)).squeeze(-1)

        fB_loss = F.mse_loss(dot_x_true, dot_x_approx)

        ortho_loss = torch.mean(
            (matrix_norm(matmul(Bbot_approx.transpose(1, 2), B_approx.detach())))
        )

        loss = fB_loss + ortho_loss  # + adversarial_loss  # + cont_loss
        with torch.no_grad():
            f_error = F.l1_loss(true_dict["f_true"], f_approx)
            B_error = F.l1_loss(true_dict["B_true"], B_approx)

        self.Dynamic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Dynamic_func.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.Dynamic_func],
            ["Dynamic_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        self.Dynamic_optimizer.step()

        norm_dict = self.compute_weight_norm(
            [self.Dynamic_func],
            ["Dynamic_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict = {
            f"{self.name}/Dynamic_loss/loss": loss.item(),
            f"{self.name}/Dynamic_loss/fB_loss": fB_loss.item(),
            f"{self.name}/Dynamic_loss/ortho_loss": ortho_loss.item(),
            # f"{self.name}/Dynamic_loss/adversarial_loss": adversarial_loss.item(),
            f"{self.name}/Dynamic_analytics/f_error": f_error.item(),
            f"{self.name}/Dynamic_analytics/B_error": B_error.item(),
            f"{self.name}/learning_rate/D_lr": self.Dynamic_optimizer.param_groups[0][
                "lr"
            ],
        }
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, x
        self.eval()

        update_time = time.time() - t0
        return loss_dict, update_time

    def learn_W(self, batch: dict, detach: bool):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        rewards = to_tensor(batch["rewards"])

        true_dict = self.get_true_metrics(states)

        # List to track actor loss over minibatches
        states = states.requires_grad_()
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)

        if self.W_entropy_scaler > 0:
            W, infos = self.W_func(states)
        else:
            W, infos = self.W_func(states, deterministic=True)
        M = inverse(W)

        f_approx, B_approx, Bbot_approx = self.Dynamic_func(x)
        Bbot_approx = self.compute_B_perp_batch(
            B_approx.detach(), self.x_dim - self.action_dim
        )

        DfDx = self.Jacobian(f_approx, x).detach()  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B_approx, x).detach()  # n, x_dim, x_dim, b_dim

        f_approx = f_approx.detach()
        B_approx = B_approx.detach()
        Bbot_approx = Bbot_approx.detach()

        u, _ = self.actor(x, xref, uref, x_trim, xref_trim, deterministic=True)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        u = u.detach()
        K = K.detach()

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.action_dim)
            ]
        )

        # contraction condition
        dot_M_approx = self.weighted_gradients(M, true_dict["dot_x_true"], x, detach)
        ABK_approx = A + matmul(B_approx, K)
        if detach:
            MABK_approx = matmul(M.detach(), ABK_approx)
            sym_MABK_approx = MABK_approx + transpose(MABK_approx, 1, 2)
            C_u = dot_M_approx + sym_MABK_approx + 2 * self.lbd * M.detach()
        else:
            MABK_approx = matmul(M, ABK_approx)
            sym_MABK_approx = MABK_approx + transpose(MABK_approx, 1, 2)
            C_u = dot_M_approx + sym_MABK_approx + 2 * self.lbd * M

        # C1
        DfW = self.weighted_gradients(W, f_approx, x, detach)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = DfDxW + transpose(DfDxW, 1, 2)

        # this has to be a negative definite matrix
        C1_inner = -DfW + sym_DfDxW + 2 * self.lbd * W
        C1 = matmul(matmul(transpose(Bbot_approx, 1, 2), C1_inner), Bbot_approx)

        C2_inners = []
        C2s = []
        for j in range(self.action_dim):
            DbW = self.weighted_gradients(W, B_approx[:, :, j], x, detach)
            DbDxW = matmul(DBDx[:, :, :, j], W)
            sym_DbDxW = DbDxW + transpose(DbDxW, 1, 2)
            C2_inner = DbW - sym_DbDxW
            C2 = matmul(matmul(transpose(Bbot_approx, 1, 2), C2_inner), Bbot_approx)
            C2_inners.append(C2_inner)
            C2s.append(C2)

        #### COMPUTE LOSS ####
        pd_loss = self.loss_pos_matrix_random_sampling(
            -C_u - self.eps * torch.eye(C_u.shape[-1]).to(self.device)
        )
        c1_loss = self.loss_pos_matrix_random_sampling(
            -C1 - self.eps * torch.eye(C1.shape[-1]).to(self.device)
        )
        # c2_loss = sum([C2.sum().mean() for C2 in C2s])
        c2_loss = sum([(matrix_norm(C2) ** 2).mean() for C2 in C2s])
        overshoot_loss = self.loss_pos_matrix_random_sampling(
            self.w_ub * torch.eye(W.shape[-1]).to(self.device) - W
        )

        ############# entropy loss ################
        mean_penalty = torch.exp(-rewards.mean())
        mean_entropy = infos["entropy"].mean()

        entropy_loss = self.W_entropy_scaler * mean_penalty * mean_entropy

        loss = pd_loss + c1_loss + c2_loss + overshoot_loss - entropy_loss

        self.W_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.W_func.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.W_func],
            ["W_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        self.W_optimizer.step()

        ### for loggings
        with torch.no_grad():
            dot_M_pos_eig, dot_M_neg_eig = self.get_matrix_eig(true_dict["dot_M_true"])
            sym_MABK_pos_eig, sym_MABK_neg_eig = self.get_matrix_eig(
                true_dict["sym_MABK_true"]
            )
            M_pos_eig, M_neg_eig = self.get_matrix_eig(M)

            C_pos_eig, C_neg_eig = self.get_matrix_eig(C_u)
            C1_pos_eig, C1_neg_eig = self.get_matrix_eig(C1)

            dot_M_error = matrix_norm(
                true_dict["dot_M_true"] - dot_M_approx, ord="fro"
            ).mean()
            ABK_error = matrix_norm(
                true_dict["ABK_true"] - ABK_approx, ord="fro"
            ).mean()
            Bbot_error = matrix_norm(
                true_dict["Bbot_true"] - Bbot_approx, ord="fro"
            ).mean()

        # Logging
        loss_dict = {
            f"{self.name}/W_loss/loss": loss.item(),
            f"{self.name}/W_loss/pd_loss": pd_loss.item(),
            f"{self.name}/W_loss/c1_loss": c1_loss.item(),
            f"{self.name}/W_loss/c2_loss": c2_loss.item(),
            f"{self.name}/W_loss/overshoot_loss": overshoot_loss.item(),
            f"{self.name}/W_loss/entropy_loss": entropy_loss.item(),
            f"{self.name}/W_loss/mean_penalty": mean_penalty.item(),
            f"{self.name}/W_loss/mean_entropy": mean_entropy.item(),
            f"{self.name}/C_analytics/C_pos_eig": C_pos_eig.item(),
            f"{self.name}/C_analytics/C_neg_eig": C_neg_eig.item(),
            f"{self.name}/C_analytics/C1_pos_eig": C1_pos_eig.item(),
            f"{self.name}/C_analytics/C1_neg_eig": C1_neg_eig.item(),
            f"{self.name}/C_analytics/dot_M_pos_eig": dot_M_pos_eig.item(),
            f"{self.name}/C_analytics/dot_M_neg_eig": dot_M_neg_eig.item(),
            f"{self.name}/C_analytics/sym_MABK_pos_eig": sym_MABK_pos_eig.item(),
            f"{self.name}/C_analytics/sym_MABK_neg_eig": sym_MABK_neg_eig.item(),
            f"{self.name}/C_analytics/M_pos_eig": M_pos_eig.item(),
            f"{self.name}/C_analytics/M_neg_eig": M_neg_eig.item(),
            f"{self.name}/C_analytics/dot_M_error": dot_M_error.item(),
            f"{self.name}/C_analytics/ABK_error": ABK_error.item(),
            f"{self.name}/C_analytics/Bbot_error": Bbot_error.item(),
            f"{self.name}/learning_rate/W_lr": self.W_optimizer.param_groups[0]["lr"],
        }
        norm_dict = self.compute_weight_norm(
            [self.W_func],
            ["W_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states
        self.eval()

        update_time = time.time() - t0
        return loss_dict, update_time

    def learn_ppo(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        actions = to_tensor(batch["actions"])
        original_rewards = to_tensor(batch["rewards"])
        rewards = self.get_rewards(states, actions)
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

        clip_fractions = []
        target_kl = []
        grad_dicts = []

        for k in range(self.K):
            for n in range(self.num_minibatch):
                indices = torch.randperm(batch_size)[: self.minibatch_size]
                mb_states, mb_actions = states[indices], actions[indices]
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
                x, xref, uref, x_trim, xref_trim = self.trim_state(mb_states)
                _, metaData = self.actor(x, xref, uref, x_trim, xref_trim)
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

                # Total loss
                loss = actor_loss + 0.5 * value_loss - entropy_loss
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
                self.ppo_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                grad_dict = self.compute_gradient_norm(
                    [self.actor, self.critic],
                    ["actor", "critic"],
                    dir=f"{self.name}",
                    device=self.device,
                )
                grad_dicts.append(grad_dict)
                self.ppo_optimizer.step()

            if kl_div.item() > self.target_kl:
                break

        # Logging
        loss_dict = {
            f"{self.name}/loss/loss": np.mean(losses),
            f"{self.name}/loss/actor_loss": np.mean(actor_losses),
            f"{self.name}/loss/value_loss": np.mean(value_losses),
            f"{self.name}/loss/entropy_loss": np.mean(entropy_losses),
            f"{self.name}/analytics/clip_fraction": np.mean(clip_fractions),
            f"{self.name}/analytics/klDivergence": target_kl[-1],
            f"{self.name}/analytics/K-epoch": k + 1,
            f"{self.name}/analytics/avg_rewards": torch.mean(original_rewards).item(),
            f"{self.name}/analytics/corrected_avg_rewards": torch.mean(rewards).item(),
            f"{self.name}/learning_rate/actor_lr": self.ppo_optimizer.param_groups[0][
                "lr"
            ],
            f"{self.name}/learning_rate/critic_lr": self.ppo_optimizer.param_groups[1][
                "lr"
            ],
        }
        grad_dict = self.average_dict_values(grad_dicts)
        norm_dict = self.compute_weight_norm(
            [self.actor, self.critic],
            ["actor", "critic"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, actions, rewards, terminals, old_logprobs
        self.eval()

        timesteps = self.num_minibatch * self.minibatch_size
        update_time = time.time() - t0
        return loss_dict, timesteps, update_time

    def get_rewards(self, states: torch.Tensor, actions: torch.Tensor):
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)

        with torch.no_grad():
            ### Compute the main rewards
            W, _ = self.W_func(states, deterministic=True)
            M = torch.inverse(W)

            error = (x - xref).unsqueeze(-1)
            errorT = transpose(error, 1, 2)

            rewards = (1 / (errorT @ M @ error + 1)).squeeze(-1)

        ### Compute the aux rewards ###
        fuel_efficiency = 1 / (torch.linalg.norm(actions, dim=-1, keepdim=True) + 1)
        rewards = rewards + self.control_scaler * fuel_efficiency

        return rewards
