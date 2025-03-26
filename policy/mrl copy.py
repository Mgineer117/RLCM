import time
import os
import pickle
import torch
import torch.nn as nn
from torch.autograd import grad
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
        numerical_ord: int = 2,
        M_scheme: str = "fd",
        ABK_scheme: str = "local",
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

        self.numerical_ord = numerical_ord
        self.M_scheme = M_scheme
        self.ABK_scheme = ABK_scheme

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
        x = x.requires_grad_()

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

    def weighted_gradients(self, W: torch.Tensor, v: torch.Tensor, x: torch.Tensor):
        # v, x: bs x n x 1
        # DWDx: bs x n x n x n
        assert v.size() == x.size()
        bs = x.shape[0]
        return (self.Jacobian_Matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)

    def extract_trajectories(
        self, x: torch.Tensor, M: torch.Tensor, terminals: torch.Tensor
    ) -> list:
        traj_x_list = []
        traj_M_list = []

        x_list = []
        M_list = []

        terminals = terminals.squeeze().tolist()

        for i in range(x.shape[0]):
            x_list.append(x[i])
            M_list.append(M[i])
            if terminals[i]:
                # print(i)
                # Terminal state encountered: finalize current trajectory.
                x_tensor = torch.stack(x_list, dim=0)
                M_tensor = torch.stack(M_list, dim=0)

                traj_x_list.append(x_tensor)
                traj_M_list.append(M_tensor)

                x_list = []
                M_list = []

        # If there are remaining states not ended by a terminal flag, add them as well.
        if len(x_list) > 0:
            traj_x_list.append(torch.stack(x_list, dim=0))
            traj_M_list.append(torch.stack(M_list, dim=0))

        return traj_x_list, traj_M_list

    def compute_samplewise_ABK(
        self, x: torch.Tensor, dot_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute a local estimate of the Jacobian (A+BK) at each sample by performing
        a local regression using neighboring states.

        Parameters:
        x: Tensor of shape (T, n) representing T time steps (or samples) of the state.
        dot_x: Tensor of shape (T, n) representing the corresponding time derivative
                approximations at each sample.

        Returns:
        ABK_local: Tensor of shape (T, n, n) where each slice ABK_local[i] is the estimated
                    Jacobian at sample i.
        """
        T, n = x.shape
        ABK_local = torch.zeros(T, n, n, device=x.device, dtype=x.dtype)

        for i in range(T):
            # Collect indices of neighboring samples
            neighbor_indices = []
            if i - 1 >= 0:
                neighbor_indices.append(i - 1)
            if i + 1 < T:
                neighbor_indices.append(i + 1)

            # If we don't have any neighbors, we cannot estimate a local Jacobian
            if len(neighbor_indices) == 0:
                ABK_local[i] = torch.zeros(n, n, device=x.device, dtype=x.dtype)
                continue

            # Form the local differences relative to sample i.
            # We use differences both in state and in the approximate derivative.
            local_delta_x = []
            local_delta_dot = []
            for j in neighbor_indices:
                local_delta_x.append((x[j] - x[i]).unsqueeze(0))  # shape (1, n)
                local_delta_dot.append(
                    (dot_x[j] - dot_x[i]).unsqueeze(0)
                )  # shape (1, n)

            # Stack the local differences: X_local shape (k, n) and Y_local shape (k, n),
            # where k is the number of neighbors (typically 2).
            X_local = torch.cat(local_delta_x, dim=0)
            Y_local = torch.cat(local_delta_dot, dim=0)

            # Solve for the local Jacobian J_i such that: X_local @ J_i ≈ Y_local.
            # If there are enough neighbors (k >= n), we use lstsq. Otherwise, we fall back to a pseudo-inverse.
            residuals = 0
            num = 0

            if X_local.shape[0] >= n:
                J_local, res, rank, s = torch.linalg.lstsq(X_local, Y_local, rcond=None)
                num += 1
            else:
                J_local = torch.linalg.pinv(X_local) @ Y_local
                num = 1

            ABK_local[i] = J_local

        return ABK_local, residuals / num

    def compute_groupwise_ABK(
        self, x: torch.Tensor, dot_x: torch.Tensor, group_size: int
    ):
        """
        Compute a single Jacobian (A+BK) for each group of time steps.
        For each group (of size group_size), solve:
            x_group @ J ≈ dot_x_group,
        then assign the same J to every time step in that group.

        Parameters:
        x: Tensor of shape (T, n) representing T time steps of state.
        dot_x: Tensor of shape (T, n) representing the corresponding time derivative.
        group_size: The number of time steps in each group.

        Returns:
        ABK_groupwise: Tensor of shape (T, n, n) where each group of time steps share the same estimated Jacobian.
        avg_residual: Average least-squares residual computed across all groups.
        """
        T, n = x.shape
        ABK_groupwise = torch.zeros(T, n, n, device=x.device, dtype=x.dtype)
        total_residual = 0.0
        group_count = 0

        # Process complete groups.
        num_groups = T // group_size
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            x_group = x[start:end]  # shape: (group_size, n)
            dot_x_group = dot_x[start:end]  # shape: (group_size, n)

            # Solve the least-squares problem for this group.
            J_group, residuals, rank, s = torch.linalg.lstsq(
                x_group, dot_x_group, rcond=None
            )
            # Assign the computed Jacobian to all time steps in this group.
            ABK_groupwise[start:end] = J_group.unsqueeze(0).expand(group_size, n, n)

            # Accumulate residuals.
            if residuals.numel() > 0:
                total_residual += residuals.sum().item()
            else:
                # In underdetermined cases, compute the squared norm of the error.
                err = torch.matmul(x_group, J_group) - dot_x_group
                total_residual += torch.linalg.norm(err).item() ** 2
            group_count += 1

        # Process any remaining time steps (if T is not exactly divisible by group_size).
        if T % group_size != 0:
            start = num_groups * group_size
            x_group = x[start:]
            dot_x_group = dot_x[start:]
            J_group, residuals, rank, s = torch.linalg.lstsq(
                x_group, dot_x_group, rcond=None
            )
            ABK_groupwise[start:] = J_group.unsqueeze(0).expand(T - start, n, n)
            if residuals.numel() > 0:
                total_residual += residuals.sum().item()
            else:
                err = torch.matmul(x_group, J_group) - dot_x_group
                total_residual += torch.linalg.norm(err).item() ** 2
            group_count += 1

        avg_residual = total_residual / group_count if group_count > 0 else 0.0
        return ABK_groupwise, avg_residual

    def get_dot_x_M(
        self,
        x: torch.Tensor,
        next_x: torch.Tensor,
        M: torch.Tensor,
        next_M: torch.Tensor,
        terminals: torch.Tensor,
        ord: int = 1,
        M_scheme: str = "fd",
    ):
        if ord == 1 and M_scheme == "fd":
            dot_x_approx = (next_x - x) / self.dt
            dot_M_approx = (next_M - M) / self.dt
        elif ord == 1 and M_scheme == "pj":
            dot_x_approx = (next_x - x) / self.dt
            dot_M_approx = self.weighted_gradients(M, dot_x_approx, x)
        elif ord == 2 and M_scheme == "fd":
            # trajectory
            traj_x_list, traj_M_list = self.extract_trajectories(x, M, terminals)
            dot_x_list = []
            dot_M_list = []
            total_num = 0
            for traj_x, traj_M in zip(traj_x_list, traj_M_list):
                temp_x_list = []
                temp_M_list = []
                for i in range(traj_x.shape[0]):
                    if traj_x.shape[0] > 2:
                        if i == 0:
                            # print((traj_x[2] - 2 * traj_x[1] + traj_x[0]) / self.dt**2)
                            temp_x_list.append(
                                (
                                    traj_x[2].detach()
                                    - 2 * traj_x[1].detach()
                                    + traj_x[0]
                                )
                                / self.dt**2
                            )
                            temp_M_list.append(
                                (
                                    traj_M[2].detach()
                                    - 2 * traj_M[1].detach()
                                    + traj_M[0]
                                )
                                / self.dt**2
                            )
                        elif i == traj_x.shape[0] - 1:
                            temp_x_list.append(
                                (
                                    traj_x[-1]
                                    - 2 * traj_x[-2].detach()
                                    + traj_x[-3].detach()
                                )
                                / self.dt**2
                            )
                            temp_M_list.append(
                                (
                                    traj_M[-1]
                                    - 2 * traj_M[-2].detach()
                                    + traj_M[-3].detach()
                                )
                                / self.dt**2
                            )
                        else:
                            temp_x_list.append(
                                (
                                    traj_x[i + 1].detach()
                                    - 2 * traj_x[i]
                                    + traj_x[i - 1].detach()
                                )
                                / self.dt**2
                            )
                            temp_M_list.append(
                                (
                                    traj_M[i + 1].detach()
                                    - 2 * traj_M[i]
                                    + traj_M[i - 1].detach()
                                )
                                / self.dt**2
                            )
                    else:
                        temp_x_list.append(
                            (next_x[total_num + i] - traj_x[i]) / self.dt
                        )
                        temp_M_list.append(
                            (next_M[total_num + i] - traj_M[i]) / self.dt
                        )

                dot_x_list.append(torch.stack(temp_x_list))
                dot_M_list.append(torch.stack(temp_M_list))
                total_num += traj_x.shape[0]
            dot_x_approx = torch.concatenate(dot_x_list, dim=0)
            dot_M_approx = torch.concatenate(dot_M_list, dim=0)

        elif ord == 2 and M_scheme == "pj":
            # trajectory
            traj_x_list, _ = self.extract_trajectories(x, M, terminals)
            dot_x_list = []
            total_num = 0
            for traj_x in traj_x_list:
                temp_x_list = []
                for i in range(traj_x.shape[0]):
                    if traj_x.shape[0] > 2:
                        if i == 0:
                            temp_x_list.append(
                                (
                                    traj_x[2].detach()
                                    - 2 * traj_x[1].detach()
                                    + traj_x[0]
                                )
                                / self.dt**2
                            )
                        elif i == traj_x.shape[0] - 1:
                            temp_x_list.append(
                                (
                                    traj_x[-1]
                                    - 2 * traj_x[-2].detach()
                                    + traj_x[-3].detach()
                                )
                                / self.dt**2
                            )
                        else:
                            temp_x_list.append(
                                (
                                    traj_x[i + 1].detach()
                                    - 2 * traj_x[i]
                                    + traj_x[i - 1].detach()
                                )
                                / self.dt**2
                            )
                    else:
                        temp_x_list.append(
                            (next_x[total_num + i] - traj_x[i]) / self.dt
                        )
                dot_x_list.append(torch.stack(temp_x_list))
                total_num += traj_x.shape[0]
            dot_x_approx = torch.concatenate(dot_x_list, dim=0)
            dot_M_approx = self.weighted_gradients(M, dot_x_approx, x)

        return dot_x_approx, dot_M_approx

    def contraction_loss(
        self, states: torch.Tensor, next_states: torch.Tensor, terminals: torch.Tensor
    ):
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        next_x, next_xref, next_uref, next_x_trim, next_xref_trim = self.trim_state(
            next_states
        )

        W = self.W_func(x, xref, uref, x_trim, xref_trim)
        M = inverse(W)

        ### Numerical dot_M
        with torch.no_grad():
            next_W = self.W_func(
                next_x, next_xref, next_uref, next_x_trim, next_xref_trim
            )
            next_M = inverse(next_W)

        dot_x_approx, dot_M_approx = self.get_dot_x_M(
            x=x,
            next_x=next_x,
            M=M,
            next_M=next_M,
            terminals=terminals,
            ord=self.numerical_ord,
            M_scheme=self.M_scheme,
        )

        with torch.no_grad():
            if self.ABK_scheme == "local":
                # ABK_approx, residuals = self.compute_samplewise_ABK(x, dot_x_approx)
                ABK_approx, residuals = self.compute_groupwise_ABK(
                    x, dot_x_approx, group_size=5
                )
            elif self.ABK_scheme == "global":
                ABK_approx, residuals, _, _ = torch.linalg.lstsq(
                    x, dot_x_approx, rcond=None
                )
                residuals = residuals.mean().item()

        MABK = matmul(M, ABK_approx)
        sym_MABK = MABK + transpose(MABK, 1, 2)

        C_u = dot_M_approx + sym_MABK + 2 * self.lbd * M  # .detach()

        with torch.no_grad():
            dot_M_norm = torch.linalg.norm(dot_M_approx)
            sym_MABK_norm = torch.linalg.norm(sym_MABK)
            M_norm = torch.linalg.norm(M)

        pd_loss = self.loss_pos_matrix_random_sampling(
            -C_u - self.eps * torch.eye(C_u.shape[-1]).to(self.device)
        )
        overshoot_loss = self.loss_pos_matrix_random_sampling(
            self.w_ub * torch.eye(W.shape[-1]).to(self.device) - W
        )

        C_eig, _ = torch.linalg.eig(C_u)
        C_eig = torch.real(C_eig)
        C_eig_contraction = ((C_eig >= 0).sum(dim=-1) == 0).cpu().detach().numpy()

        return (
            pd_loss,
            overshoot_loss,
            C_eig_contraction.mean(),
            {
                "dot_M_norm": dot_M_norm.item(),
                "sym_MABK_norm": sym_MABK_norm.item(),
                "M_norm": M_norm.item(),
                "lstsq_residuals": residuals,
            },
        )

    def learn(self, batch):
        if self.current_update <= int(0.3 * self.nupdates):
            loss_dict, timesteps, update_time = self.learn_W(batch)
            ppo_loss_dict, timesteps, ppo_update_time = self.learn_ppo(batch)

            loss_dict.update(ppo_loss_dict)
            update_time += ppo_update_time
        else:
            loss_dict, timesteps, update_time = self.learn_ppo(batch)

        self.current_update += 1

        return loss_dict, timesteps, update_time

    def learn_W(self, batch):
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
        pd_loss, overshoot_loss, C_eig_contraction, contraction_norms = (
            self.contraction_loss(states, next_states, terminals)
        )
        loss = pd_loss + overshoot_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.W_func.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.W_func],
            ["W_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        self.optimizer.step()

        # Logging
        loss_dict = {
            # f"{self.name}/loss/loss": np.mean(losses),
            f"{self.name}/W_loss/pd_loss": pd_loss.item(),
            f"{self.name}/W_loss/overshoot_loss": overshoot_loss.item(),
            f"{self.name}/Contraction_analytics/C_eig_contraction": C_eig_contraction,
            f"{self.name}/Contraction_analytics/dot_M_norm": contraction_norms[
                "dot_M_norm"
            ],
            f"{self.name}/Contraction_analytics/sym_MABK_norm": contraction_norms[
                "sym_MABK_norm"
            ],
            f"{self.name}/Contraction_analytics/M_norm": contraction_norms["M_norm"],
            f"{self.name}/Contraction_analytics/lstsq_residuals": contraction_norms[
                "lstsq_residuals"
            ],
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
        del states, next_states
        self.eval()

        timesteps = 0
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
