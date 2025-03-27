import time
import os
import pickle
import torch
import torch.nn as nn
from torch.autograd import grad
from torch import matmul, inverse, transpose
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
        self.dummy = torch.tensor(1e-5).to(self.device)
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
        states = states.requires_grad_()
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        # next_x, next_xref, next_uref, next_x_trim, next_xref_trim = self.trim_state(
        #     next_states
        # )

        W = self.W_func(x, xref, uref, x_trim, xref_trim)
        M = inverse(W)

        # x = x.requires_grad_()
        f = self.f_func(x).to(self.device)  # n, x_dim
        B = self.B_func(x).to(self.device)  # n, x_dim, action

        DfDx = self.Jacobian(f, x)  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x)  # n, x_dim, x_dim, b_dim
        Bbot = self.Bbot_func(x).to(self.device)  # n, x_dim, state - action dim

        u, _ = self.actor(states)
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
        dot_W = self.weighted_gradients(W, dot_x, x, detach)

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
        if self.current_update <= int(0.25 * self.nupdates):
            detach = True
        else:
            detach = False

        loss_dict, timesteps, update_time = self.learn_ppo(batch)

        if self.current_update <= int(0.5 * self.nupdates):
            W_loss_dict, _, W_update_time = self.learn_W(batch, detach)
            loss_dict.update(W_loss_dict)
            update_time += W_update_time

        self.current_update += 1

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

        timesteps = rewards.shape[0]
        update_time = time.time() - t0
        return loss_dict, timesteps, update_time

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
