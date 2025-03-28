import time
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch import matmul, inverse, transpose
from torch.linalg import matrix_norm
import numpy as np
from typing import Callable

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

        self.f_func = f_func
        self.B_func = B_func
        if Bbot_func is None:
            self.Bbot_func = self.B_null
        else:
            self.Bbot_func = Bbot_func

        self._forward_steps = 0
        self.nupdates = nupdates
        self.num_outer_update = 0
        self.num_inner_update = 0

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
        self.dummy = torch.tensor(1e-5).to(self.device)
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        self._forward_steps += 1
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
        # state trimming
        x = state[:, : self.x_dim]
        xref = state[:, self.x_dim : -self.action_dim]
        uref = state[:, -self.action_dim :]

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

    def B_null(self, x: torch.Tensor):
        n = x.shape[0]
        Bbot = torch.cat(
            (
                torch.eye(self.x_dim - self.action_dim, self.x_dim - self.action_dim),
                torch.zeros(self.action_dim, self.x_dim - self.action_dim),
            ),
            dim=0,
        )
        Bbot.unsqueeze(0).to(self.device)
        return Bbot.repeat(n, 1, 1)

    def Jacobian(self, f: torch.Tensor, x: torch.Tensor):
        # NOTE that this function assume that data are independent of each other
        f = f + 0.0 * x.sum()  # to avoid the case that f is independent of x

        n = x.shape[0]
        f_dim = f.shape[-1]
        x_dim = x.shape[-1]

        J = torch.zeros(n, f_dim, x_dim).to(self.device)  # .to(x.type())
        for i in range(f_dim):
            J[:, i, :] = grad(f[:, i].sum(), x, create_graph=True)[0]  # [0]
        return J

    def Jacobian_Matrix(self, M: torch.Tensor, x: torch.Tensor):
        # NOTE that this function assume that data are independent of each other
        M = M + 0.0 * x.sum()  # to avoid the case that f is independent of x

        n = x.shape[0]
        matrix_dim = M.shape[-1]
        x_dim = x.shape[-1]

        J = torch.zeros(n, matrix_dim, matrix_dim, x_dim).to(self.device)
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=True)[0]

        return J

    def B_Jacobian(self, B: torch.Tensor, x: torch.Tensor):
        n = x.shape[0]
        x_dim = x.shape[-1]

        DBDx = torch.zeros(n, x_dim, x_dim, self.action_dim).to(self.device)
        for i in range(self.action_dim):
            DBDx[:, :, :, i] = self.Jacobian(B[:, :, i].unsqueeze(-1), x)
        return DBDx

    def weighted_gradients(
        self, W: torch.Tensor, v: torch.Tensor, x: torch.Tensor, detach: bool
    ):
        # v, x: bs x n x 1
        # DWDx: bs x n x n x n
        assert v.size() == x.size()
        bs = x.shape[0]
        if detach:
            return (self.Jacobian_Matrix(W, x).detach() * v.view(bs, 1, 1, -1)).sum(
                dim=3
            )
        else:
            return (self.Jacobian_Matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)

    def contraction_loss(
        self,
        states: torch.Tensor,
        detach: bool,
    ):
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        x = x.requires_grad_()

        W = self.W_func(x, xref, uref, x_trim, xref_trim)
        M = inverse(W)

        # x = x.requires_grad_()
        f = self.f_func(x).to(self.device)  # n, x_dim
        B = self.B_func(x).to(self.device)  # n, x_dim, action

        DfDx = self.Jacobian(f, x)  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x)  # n, x_dim, x_dim, b_dim
        Bbot = self.Bbot_func(x).to(self.device)  # n, x_dim, state - action dim

        u, _ = self.actor(x, xref, uref, x_trim, xref_trim)
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

        # contraction condition
        if detach:
            ABK = A + matmul(B, K)
            MABK = matmul(M.detach(), ABK)
            sym_MABK = MABK + transpose(MABK, 1, 2)
            C_u = dot_M + sym_MABK + 2 * self.lbd * M.detach()
        else:
            ABK = A + matmul(B, K)
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
        c2_loss = sum([C2.sum().mean() for C2 in C2s])
        overshoot_loss = self.loss_pos_matrix_random_sampling(
            self.w_ub * torch.eye(W.shape[-1]).to(self.device) - W
        )

        loss = pd_loss + c1_loss + c2_loss + overshoot_loss

        ### for loggings
        with torch.no_grad():
            dot_M_norm = torch.linalg.norm(dot_M)
            sym_MABK_norm = torch.linalg.norm(sym_MABK)
            M_norm = torch.linalg.norm(M)

        C_eig, _ = torch.linalg.eig(C_u)
        C1_eig, _ = torch.linalg.eig(C1)

        C_eig = torch.real(C_eig)
        C1_eig = torch.real(C1_eig)

        C_eig_contraction = ((C_eig >= 0).sum(dim=-1) == 0).cpu().detach().numpy()
        C1_eig_contraction = ((C1_eig >= 0).sum(dim=1) == 0).cpu().detach().numpy()

        return (
            loss,
            {
                "pd_loss": pd_loss.item(),
                "C1_loss": c1_loss.item(),
                "C2_loss": c2_loss.item(),
                "overshoot_loss": overshoot_loss.item(),
                "C_eig_contraction": C_eig_contraction.mean(),
                "C1_eig_contraction": C1_eig_contraction.mean(),
                "dot_M_norm": dot_M_norm.item(),
                "sym_MABK_norm": sym_MABK_norm.item(),
                "M_norm": M_norm.item(),
            },
        )

    def learn(self, batch):
        if self.num_outer_update <= int(0.2 * self.nupdates):
            detach = (
                True if self.num_outer_update <= int(0.1 * self.nupdates) else False
            )
            W_loss_dict, W_update_time = self.learn_W(batch, detach)

            if self.num_inner_update % 3 == 0:
                loss_dict, timesteps, update_time = self.learn_ppo(batch)
                loss_dict.update(W_loss_dict)
                update_time += W_update_time

                self.num_outer_update += 1
                self.num_inner_update += 1
            else:
                loss_dict = {}
                timesteps = 0
                update_time = 0

                self.num_inner_update += 1
        else:
            loss_dict, timesteps, update_time = self.learn_ppo(batch)
            self.num_outer_update += 1

        return loss_dict, timesteps, update_time

    def learn_W(
        self, batch: dict, detach: bool, avoid_update_and_collect_log: bool = False
    ):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        rewards = to_tensor(batch["rewards"])

        # List to track actor loss over minibatches
        loss, infos = self.contraction_loss(states, detach)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.W_func.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.W_func],
            ["W_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        if not avoid_update_and_collect_log:
            self.optimizer.step()

        # Logging
        loss_dict = {
            f"{self.name}/W_loss/loss": loss.item(),
            f"{self.name}/W_loss/pd_loss": infos["pd_loss"],
            f"{self.name}/W_loss/C1_loss": infos["C1_loss"],
            f"{self.name}/W_loss/C2_loss": infos["C2_loss"],
            f"{self.name}/W_loss/overshoot_loss": infos["overshoot_loss"],
            f"{self.name}/Contraction_analytics/C_eig_contraction": infos[
                "C_eig_contraction"
            ],
            f"{self.name}/Contraction_analytics/C1_eig_contraction": infos[
                "C1_eig_contraction"
            ],
            f"{self.name}/Contraction_analytics/dot_M_norm": infos["dot_M_norm"],
            f"{self.name}/Contraction_analytics/sym_MABK_norm": infos["sym_MABK_norm"],
            f"{self.name}/Contraction_analytics/M_norm": infos["M_norm"],
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
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                grad_dict = self.compute_gradient_norm(
                    [self.actor, self.critic],
                    ["actor", "critic"],
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
        entropy_scaler: float = 1e-3,
        l2_reg: float = 1e-5,
        target_kl: float = 0.03,
        gamma: float = 0.99,
        gae: float = 0.9,
        K: int = 5,
        nupdates: int = 0,
        dt: float = 0.03,
        ABK_scheme: str = "local",
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

        self.ABK_scheme = ABK_scheme

        self.f_func = f_func
        self.B_func = B_func
        if Bbot_func is None:
            self.Bbot_func = self.B_null
        else:
            self.Bbot_func = Bbot_func

        self._forward_steps = 0
        self.nupdates = nupdates
        self.num_outer_update = 0
        self.num_inner_update = 0

        # trainable networks
        self.W_func = W_func
        self.Dynamic_func = Dynamic_func
        self.actor = actor
        self.critic = critic

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.W_func.parameters(), "lr": W_lr},
                {"params": self.Dynamic_func.parameters(), "lr": Dynamic_lr},
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
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        x = x.requires_grad_()

        W = self.W_func(x, xref, uref, x_trim, xref_trim)
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
        dot_M = self.weighted_gradients(M, dot_x, x, True)

        ABK = A + matmul(B, K)

        return dot_x, dot_M, ABK, Bbot, f, B

    def contraction_loss(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        terminals: torch.Tensor,
        detach: bool,
    ):
        dot_x_true, dot_M_true, ABK_true, Bbot_true, _, _ = self.get_true_metrics(
            states
        )

        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        x = x.requires_grad_()

        W = self.W_func(x, xref, uref, x_trim, xref_trim)
        M = inverse(W)

        f_approx, B_approx, Bbot_approx = self.Dynamic_func(x)
        # Bbot_approx = self.B_null(x).to(self.device)
        Bbot_approx = self.compute_B_perp_batch(B_approx, self.x_dim - self.action_dim)

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
        dot_M_approx = self.weighted_gradients(M, dot_x_true, x, detach)
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
        overshoot_loss = self.loss_pos_matrix_random_sampling(
            self.w_ub * torch.eye(W.shape[-1]).to(self.device) - W
        )
        c1_loss = self.loss_pos_matrix_random_sampling(
            -C1 - self.eps * torch.eye(C1.shape[-1]).to(self.device)
        )
        # c2_loss = sum([C2.sum().mean() for C2 in C2s])
        c2_loss = sum([(matrix_norm(C2) ** 2).mean() for C2 in C2s])

        loss = pd_loss + overshoot_loss + c1_loss + c2_loss

        C_eig, _ = torch.linalg.eig(C_u)
        C1_eig, _ = torch.linalg.eig(C1)

        C_eig = torch.real(C_eig)
        C1_eig = torch.real(C1_eig)

        C_eig_contraction = ((C_eig >= 0).sum(dim=-1) == 0).cpu().detach().numpy()
        C1_eig_contraction = ((C1_eig >= 0).sum(dim=1) == 0).cpu().detach().numpy()

        ### for loggings
        with torch.no_grad():
            dot_M_error = torch.linalg.matrix_norm(
                dot_M_true - dot_M_approx, ord="fro"
            ).mean()
            ABK_error = torch.linalg.matrix_norm(
                ABK_true - ABK_approx, ord="fro"
            ).mean()
            Bbot_error = torch.linalg.matrix_norm(
                Bbot_true - Bbot_approx, ord="fro"
            ).mean()

        return (
            loss,
            {
                "pd_loss": pd_loss.item(),
                "overshoot_loss": overshoot_loss.item(),
                "c1_loss": c1_loss.item(),
                "c2_loss": c2_loss.item(),
                "C_eig_contraction": C_eig_contraction.mean(),
                "C1_eig_contraction": C1_eig_contraction.mean(),
                "dot_M_error": dot_M_error.item(),
                "ABK_error": ABK_error.item(),
                "Bbot_error": Bbot_error.item(),
            },
        )

    def learn(self, batch):
        if self.num_outer_update <= int(0.2 * self.nupdates):
            detach = (
                True if self.num_outer_update <= int(0.1 * self.nupdates) else False
            )

            D_loss_dict, D_update_time = self.learn_Dynamics(batch)

            if self.num_inner_update % 3 == 0:
                W_loss_dict, W_update_time = self.learn_W(batch, detach)
            if self.num_inner_update % 6 == 0:
                loss_dict, timesteps, update_time = self.learn_ppo(batch)
                loss_dict.update(D_loss_dict)
                loss_dict.update(W_loss_dict)
                update_time += D_update_time
                update_time += W_update_time

                self.num_outer_update += 1
                self.num_inner_update += 1
            else:
                loss_dict = {}
                timesteps = 0
                update_time = 0

                self.num_inner_update += 1
        else:
            loss_dict, timesteps, update_time = self.learn_ppo(batch)
            self.num_outer_update += 1

        return loss_dict, timesteps, update_time

    def learn_Dynamics(self, batch: dict):
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        actions = to_tensor(batch["actions"])

        dot_x, _, _, _, f, B = self.get_true_metrics(states)

        x, _, _, _, _ = self.trim_state(states)
        f_approx, B_approx, Bbot_approx = self.Dynamic_func(x)

        dot_x_approx = f_approx + matmul(B_approx, actions.unsqueeze(-1)).squeeze(-1)

        fB_loss = F.mse_loss(dot_x, dot_x_approx)

        ortho_loss = torch.mean(
            (matrix_norm(matmul(Bbot_approx.transpose(1, 2), B_approx.detach())))
        )

        #### find adversarial loss ####
        # x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        # x = x.requires_grad_()

        # with torch.no_grad():
        #     W = self.W_func(x, xref, uref, x_trim, xref_trim)

        # DfDx = self.Jacobian(f_approx, x).detach()
        # DfW = self.weighted_gradients(W, f_approx, x, True).detach()
        # DfDxW = matmul(DfDx, W)
        # sym_DfDxW = DfDxW + transpose(DfDxW, 1, 2)
        # C1_inner = -DfW + sym_DfDxW + 2 * self.lbd * W
        # C1 = matmul(matmul(transpose(Bbot_approx, 1, 2), C1_inner), Bbot_approx)
        # # adversarial_loss = -self.loss_pos_matrix_random_sampling(
        # #     -C1 - self.eps * torch.eye(C1.shape[-1]).to(self.device)
        # # )

        # this has to be a negative definite matrix

        # cont_loss = torch.mean(
        #     (
        #         matmul(
        #             Bbot_approx.transpose(1, 2),
        #             (dot_x - f_approx).unsqueeze(-1).detach(),
        #         )
        #         ** 2
        #     )
        # )
        # I = torch.eye(Bbot_approx.shape[-1], device=Bbot_approx.device)
        # identity_loss = torch.mean(
        #     matrix_norm(matmul(Bbot_approx.transpose(1, 2), Bbot_approx) - I)
        # )

        loss = fB_loss + ortho_loss  # + adversarial_loss  # + cont_loss
        with torch.no_grad():
            f_error = F.l1_loss(f, f_approx)
            B_error = F.l1_loss(B, B_approx)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Dynamic_func.parameters(), max_norm=5.0)
        grad_dict = self.compute_gradient_norm(
            [self.Dynamic_func],
            ["Dynamic_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        self.optimizer.step()

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
        }
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, x
        self.eval()

        update_time = time.time() - t0
        return loss_dict, update_time

    def B_null(self, x: torch.Tensor):
        n = x.shape[0]
        Bbot = torch.cat(
            (
                torch.eye(self.x_dim - self.action_dim, self.x_dim - self.action_dim),
                torch.zeros(self.action_dim, self.x_dim - self.action_dim),
            ),
            dim=0,
        )
        Bbot.unsqueeze(0).to(self.device)
        return Bbot.repeat(n, 1, 1)

    def learn_W(self, batch: dict, detach: bool):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        next_states = to_tensor(batch["next_states"])
        terminals = to_tensor(batch["terminals"])

        # List to track actor loss over minibatches
        loss, infos = self.contraction_loss(states, next_states, terminals, detach)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.W_func.parameters(), max_norm=5.0)
        grad_dict = self.compute_gradient_norm(
            [self.W_func],
            ["W_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        self.optimizer.step()

        # Logging
        loss_dict = {
            f"{self.name}/W_loss/loss": loss.item(),
            f"{self.name}/W_loss/pd_loss": infos["pd_loss"],
            f"{self.name}/W_loss/overshoot_loss": infos["overshoot_loss"],
            f"{self.name}/W_loss/c1_loss": infos["c1_loss"],
            f"{self.name}/W_loss/c2_loss": infos["c2_loss"],
            f"{self.name}/Contraction_analytics/C_eig_contraction": infos[
                "C_eig_contraction"
            ],
            f"{self.name}/Contraction_analytics/C1_eig_contraction": infos[
                "C1_eig_contraction"
            ],
            f"{self.name}/Contraction_analytics/dot_M_error": infos["dot_M_error"],
            f"{self.name}/Contraction_analytics/ABK_error": infos["ABK_error"],
            f"{self.name}/Contraction_analytics/Bbot_error": infos["Bbot_error"],
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
        del states, next_states, terminals
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
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                grad_dict = self.compute_gradient_norm(
                    [self.actor, self.critic],
                    ["actor", "critic"],
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
        # state trimming
        x = state[:, : self.x_dim]
        xref = state[:, self.x_dim : -self.action_dim]
        uref = state[:, -self.action_dim :]

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

    def B_null(self, x: torch.Tensor):
        n = x.shape[0]
        Bbot = torch.cat(
            (
                torch.eye(self.x_dim - self.action_dim, self.x_dim - self.action_dim),
                torch.zeros(self.action_dim, self.x_dim - self.action_dim),
            ),
            dim=0,
        )
        Bbot.unsqueeze(0).to(self.device)
        return Bbot.repeat(n, 1, 1)

    def Jacobian(self, f: torch.Tensor, x: torch.Tensor):
        # NOTE that this function assume that data are independent of each other
        f = f + 0.0 * x.sum()  # to avoid the case that f is independent of x

        n = x.shape[0]
        f_dim = f.shape[-1]
        x_dim = x.shape[-1]

        J = torch.zeros(n, f_dim, x_dim).to(self.device)  # .to(x.type())
        for i in range(f_dim):
            J[:, i, :] = grad(f[:, i].sum(), x, create_graph=True)[0]  # [0]
        return J

    def Jacobian_Matrix(self, M: torch.Tensor, x: torch.Tensor):
        # NOTE that this function assume that data are independent of each other
        M = M + 0.0 * x.sum()  # to avoid the case that f is independent of x

        n = x.shape[0]
        matrix_dim = M.shape[-1]
        x_dim = x.shape[-1]

        J = torch.zeros(n, matrix_dim, matrix_dim, x_dim).to(self.device)
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=True)[0]

        return J

    def B_Jacobian(self, B: torch.Tensor, x: torch.Tensor):
        n = x.shape[0]
        x_dim = x.shape[-1]

        DBDx = torch.zeros(n, x_dim, x_dim, self.action_dim).to(self.device)
        for i in range(self.action_dim):
            DBDx[:, :, :, i] = self.Jacobian(B[:, :, i].unsqueeze(-1), x)
        return DBDx

    def weighted_gradients(
        self, W: torch.Tensor, v: torch.Tensor, x: torch.Tensor, detach: bool
    ):
        # v, x: bs x n x 1
        # DWDx: bs x n x n x n
        assert v.size() == x.size()
        bs = x.shape[0]
        if detach:
            return (self.Jacobian_Matrix(W, x).detach() * v.view(bs, 1, 1, -1)).sum(
                dim=3
            )
        else:
            return (self.Jacobian_Matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)

    def extract_trajectories(self, x: torch.Tensor, terminals: torch.Tensor) -> list:
        traj_x_list = []
        x_list = []

        terminals = terminals.squeeze().tolist()

        for i in range(x.shape[0]):
            x_list.append(x[i])
            if terminals[i]:
                # Terminal state encountered: finalize current trajectory.
                x_tensor = torch.stack(x_list, dim=0)
                traj_x_list.append(x_tensor)
                x_list = []

        # If there are remaining states not ended by a terminal flag, add them as well.
        if len(x_list) > 0:
            traj_x_list.append(torch.stack(x_list, dim=0))

        return traj_x_list

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

    def compute_B_perp_batch(self, B, B_perp_dim, method="qr", threshold=1e-1):
        """
        Compute the nullspace basis B_perp for each sample (or a single matrix) from B,
        using either SVD or QR decomposition, and return a tensor of shape
        (batch, x_dim, B_perp_dim) or (x_dim, B_perp_dim) for single input.

        Parameters:
            B: Tensor of shape (batch, x_dim, x_dim) or (x_dim, x_dim).
            B_perp_dim: Desired number of nullspace vectors (output columns).
            method: "svd" or "qr".
            threshold: Threshold below which singular values or R diagonals are considered zero.

        Returns:
            B_perp_tensor: Tensor of shape (batch, x_dim, B_perp_dim) or (x_dim, B_perp_dim).
        """
        # Handle single matrix input by unsqueezing to batch size 1
        batch_size, x_dim, _ = B.shape
        B_perp_list = []

        for i in range(batch_size):
            B_i = B[i]

            if method.lower() == "svd":
                U, S, _ = torch.linalg.svd(B_i)
                null_indices = (S < threshold).nonzero(as_tuple=True)[0]
                B_perp = (
                    U[:, null_indices]
                    if null_indices.numel() > 0
                    else torch.empty(x_dim, 0, device=B.device, dtype=B.dtype)
                )

            elif method.lower() == "qr":
                Q, R = torch.linalg.qr(B_i, mode="complete")
                diag_R = torch.abs(torch.diag(R))
                null_indices = (diag_R < threshold).nonzero(as_tuple=True)[0]
                B_perp = (
                    Q[:, null_indices]
                    if null_indices.numel() > 0
                    else torch.empty(x_dim, 0, device=B.device, dtype=B.dtype)
                )
            else:
                raise ValueError("Method must be either 'svd' or 'qr'.")

            # Pad or truncate to fixed B_perp_dim
            padded = torch.zeros(x_dim, B_perp_dim, device=B.device, dtype=B.dtype)
            m = B_perp.shape[1]
            if m > 0:
                padded[:, : min(m, B_perp_dim)] = B_perp[:, :B_perp_dim]
            B_perp_list.append(padded)

        B_perp_tensor = torch.stack(B_perp_list, dim=0)  # (batch, x_dim, B_perp_dim)

        return B_perp_tensor

    def get_dot_x(
        self,
        x: torch.Tensor,
        next_x: torch.Tensor,
        terminals: torch.Tensor,
    ):
        with torch.no_grad():
            traj_x_list = self.extract_trajectories(x, terminals)
            dot_x_list = []
            total_num = 0
            for traj_x in traj_x_list:
                temp_x_list = []
                for i in range(traj_x.shape[0]):
                    if traj_x.shape[0] > 2:
                        if i == 0:
                            temp_x_list.append(
                                (
                                    (-1 / 2) * traj_x[2]
                                    + 2 * traj_x[1]
                                    - (3 / 2) * traj_x[0]
                                )
                                / self.dt
                            )
                        elif i == traj_x.shape[0] - 1:
                            temp_x_list.append(
                                (
                                    (3 / 2) * traj_x[-1]
                                    - 2 * traj_x[-2]
                                    + (1 / 2) * traj_x[-3]
                                )
                                / self.dt
                            )
                        else:
                            temp_x_list.append(
                                ((1 / 2) * traj_x[i + 1] - (1 / 2) * traj_x[i - 1])
                                / self.dt
                            )
                    else:
                        temp_x_list.append(
                            (next_x[total_num + i] - traj_x[i]) / self.dt
                        )
                dot_x_list.append(torch.stack(temp_x_list))
                total_num += traj_x.shape[0]

            dot_x_approx = torch.concatenate(dot_x_list, dim=0)

        return dot_x_approx
