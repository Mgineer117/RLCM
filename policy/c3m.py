import time
import os
import torch.nn.functional as F
import pickle
import torch
import torch.nn as nn
from torch.autograd import grad
from torch import matmul, inverse, transpose
from torch.linalg import matrix_norm
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from typing import Callable

# from utils.torch import get_flat_grad_from, get_flat_params_from, set_flat_params_to
from utils.functions import estimate_advantages

# from actor.layers.building_blocks import MLP
from policy.base import Base

# from models.layers.ppo_networks import PPO_Policy, PPO_Critic


class C3M(Base):
    def __init__(
        self,
        x_dim: int,
        effective_indices: list,
        action_dim: int,
        W_func: nn.Module,
        u_func: nn.Module,
        f_func: Callable,
        B_func: Callable,
        Bbot_func: Callable,
        W_lr: float = 3e-4,
        u_lr: float = 3e-4,
        lbd: float = 1e-2,
        eps: float = 1e-2,
        w_ub: float = 1e-2,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        nupdates: int = 0,
        device: str = "cpu",
    ):
        super(C3M, self).__init__()

        # constants
        self.name = "C3M"
        self.device = device

        self.x_dim = x_dim
        self.effective_x_dim = len(effective_indices)
        self.effective_indices = effective_indices
        self.action_dim = action_dim

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size

        self._forward_steps = 0
        self.nupdates = nupdates
        self.current_update = 0

        # trainable networks
        self.W_func = W_func
        self.u_func = u_func
        self.f_func = f_func
        self.B_func = B_func
        if Bbot_func is None:
            self.Bbot_func = self.B_null
        else:
            self.Bbot_func = Bbot_func
        self.lbd = lbd
        self.eps = eps
        self.w_ub = w_ub

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.W_func.parameters(), "lr": W_lr},
                {"params": self.u_func.parameters(), "lr": u_lr},
            ]
        )

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        #
        self.dummy = torch.tensor(1e-5)
        self.to(self.device)

    def lr_lambda(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        self._forward_steps += 1
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        x, xref, uref, x_trim, xref_trim = self.trim_state(state)
        a = self.u_func(x, xref, uref, x_trim, xref_trim)

        return a, {
            "probs": self.dummy,  # dummy for code consistency
            "logprobs": self.dummy,
            "entropy": self.dummy,
        }

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

        J = torch.zeros(n, f_dim, x_dim).to(self.device)
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
            return torch.tensor(0.0).to(self.device).requires_grad_()

    def trim_state(self, state: torch.Tensor):
        # state trimming
        x = state[:, : self.x_dim]
        xref = state[:, self.x_dim : -self.action_dim]
        uref = state[:, -self.action_dim :]

        x_trim = x[:, self.effective_indices]
        xref_trim = xref[:, self.effective_indices]

        return x, xref, uref, x_trim, xref_trim

    def get_matrix_eig(self, A: torch.Tensor):
        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(A)  # (batch, dim), real symmetric
            pos_eigvals = torch.relu(eigvals)
            neg_eigvals = torch.relu(-eigvals)
        return pos_eigvals.mean(dim=1).mean(), neg_eigvals.mean(dim=1).mean()

    def learn(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        detach = True if self.current_update <= int(0.1 * self.nupdates) else False

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        rewards = to_tensor(batch["rewards"])

        #### COMPUTE INGREDIENTS ####
        # grad tracking state elements
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        x = x.requires_grad_()

        W = self.W_func(x, xref, uref, x_trim, xref_trim)  # n, x_dim, x_dim
        M = inverse(W)  # n, x_dim, x_dim

        f = self.f_func(x).to(self.device)  # n, x_dim
        B = self.B_func(x).to(self.device)  # n, x_dim, action

        DfDx = self.Jacobian(f, x)  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x)  # n, x_dim, x_dim, b_dim
        Bbot = self.Bbot_func(x).to(self.device)  # n, x_dim, state - action dim

        # since online we do not do below
        u = self.u_func(x, xref, uref, x_trim, xref_trim)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        #  DBDx[:, :, :, i]: n, x_dim, x_dim
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
        # c2_loss = sum([C2.sum().mean() for C2 in C2s])
        c2_loss = sum([(matrix_norm(C2) ** 2).mean() for C2 in C2s])
        overshoot_loss = self.loss_pos_matrix_random_sampling(
            self.w_ub * torch.eye(W.shape[-1]).to(self.device) - W
        )

        loss = pd_loss + c1_loss + c2_loss + overshoot_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.W_func, self.u_func],
            ["W_func", "u_func"],
            dir="C3M",
            device=self.device,
        )
        self.optimizer.step()

        with torch.no_grad():
            dot_M_pos_eig, dot_M_neg_eig = self.get_matrix_eig(dot_M)
            sym_MABK_pos_eig, sym_MABK_neg_eig = self.get_matrix_eig(sym_MABK)
            M_pos_eig, M_neg_eig = self.get_matrix_eig(M)

            C_pos_eig, C_neg_eig = self.get_matrix_eig(C_u)
            C1_pos_eig, C1_neg_eig = self.get_matrix_eig(C1)

        corrected_rewards = self.get_rewards(states)
        # Logging
        loss_dict = {
            "C3M/loss/loss": loss.item(),
            "C3M/loss/pd_loss": pd_loss.item(),
            "C3M/loss/c1_loss": c1_loss.item(),
            "C3M/loss/c2_loss": c2_loss.item(),
            "C3M/loss/overshoot_loss": overshoot_loss.item(),
            "C3M/analytics/C_pos_eig": C_pos_eig.item(),
            "C3M/analytics/C_neg_eig": C_neg_eig.item(),
            "C3M/analytics/C1_pos_eig": C1_pos_eig.item(),
            "C3M/analytics/C1_neg_eig": C1_neg_eig.item(),
            "C3M/analytics/avg_rewards": torch.mean(rewards).item(),
            "C3M/analytics/dot_M_pos_eig": dot_M_pos_eig.item(),
            "C3M/analytics/dot_M_neg_eig": dot_M_neg_eig.item(),
            "C3M/analytics/sym_MABK_pos_eig": sym_MABK_pos_eig.item(),
            "C3M/analytics/sym_MABK_neg_eig": sym_MABK_neg_eig.item(),
            "C3M/analytics/M_pos_eig": M_pos_eig.item(),
            "C3M/analytics/M_neg_eig": M_neg_eig.item(),
            "C3M/analytics/corrected_rewards": corrected_rewards.mean().item(),
        }
        norm_dict = self.compute_weight_norm(
            [self.W_func, self.u_func],
            ["W_func", "u_func"],
            dir="C3M",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, u, rewards
        self.eval()
        self.current_update += 1

        timesteps = self.num_minibatch * self.minibatch_size
        update_time = time.time() - t0
        self.lr_scheduler.step()

        return loss_dict, timesteps, update_time

    def reawrd_pos_matrix_random_sampling(self, A: torch.Tensor):
        # A: n x d x d
        # z: K x d
        n, A_dim, _ = A.shape

        z = torch.randn((n, A_dim)).to(self.device)
        z = z / z.norm(dim=-1, keepdim=True)
        z = z.unsqueeze(-1)
        zT = transpose(z, 1, 2)

        # K x d @ d x d = n x K x d
        zTAz = matmul(matmul(zT, A), z).squeeze(-1)

        positive_index = zTAz.detach().cpu().numpy() > 0
        if positive_index.sum() > 0:
            zTAz[positive_index] = 0.0
            return zTAz
        else:
            return torch.zeros(n, 1).to(self.device)

    def get_rewards(self, states):
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        x = x.requires_grad_()

        with torch.no_grad():
            ### Compute the main rewards
            W = self.W_func(x, xref, uref, x_trim, xref_trim)
            M = torch.inverse(W)

            error = (x - xref).unsqueeze(-1)
            errorT = transpose(error, 1, 2)

            rewards = (1 / (errorT @ M @ error + 1)).squeeze(-1)

        ### Compute the aux rewards ###
        f = self.f_func(x).to(self.device)  # n, x_dim
        B = self.B_func(x).to(self.device)  # n, x_dim, action

        DfDx = self.Jacobian(f, x).detach()  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x).detach()  # n, x_dim, x_dim, b_dim

        u = self.u_func(x, xref, uref, x_trim, xref_trim)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        u = u.detach()
        K = K.detach()

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.action_dim)
            ]
        )

        ABK = A + matmul(B, K)
        MABK = matmul(M, ABK)
        sym_MABK = MABK + transpose(MABK, 1, 2)

        C_u_only = -sym_MABK - self.eps * torch.eye(sym_MABK.shape[-1]).to(self.device)

        aux_rewards = torch.linalg.eigvalsh(C_u_only).mean(dim=1).unsqueeze(-1)

        pos_indices = aux_rewards > 0
        neg_indices = aux_rewards <= 0

        aux_rewards[pos_indices] = torch.tanh(aux_rewards[pos_indices] / 30)
        aux_rewards[neg_indices] = -1.0

        alpha = 0.5
        rewards = alpha * rewards + (1 - alpha) * aux_rewards

        return rewards


class C3M_Approximation(Base):
    def __init__(
        self,
        x_dim: int,
        effective_indices: list,
        action_dim: int,
        W_func: nn.Module,
        u_func: nn.Module,
        Dynamic_func: nn.Module,
        f_func: Callable,
        B_func: Callable,
        Bbot_func: Callable,
        W_lr: float = 3e-4,
        u_lr: float = 3e-4,
        Dynamic_lr: float = 3e-4,
        lbd: float = 1e-2,
        eps: float = 1e-2,
        w_ub: float = 1e-2,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        nupdates: int = 0,
        dt: float = 0.03,
        device: str = "cpu",
    ):
        super(C3M_Approximation, self).__init__()

        # constants
        self.name = "C3M_Approximation"
        self.device = device

        self.x_dim = x_dim
        self.action_dim = action_dim
        self.effective_x_dim = len(effective_indices)
        self.effective_indices = effective_indices

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size

        self.eps = eps

        self.lbd = lbd

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
        self.u_func = u_func
        self.Dynamic_func = Dynamic_func

        self.W_u_optimizer = torch.optim.Adam(
            [
                {"params": self.W_func.parameters(), "lr": W_lr},
                {"params": self.u_func.parameters(), "lr": u_lr},
            ]
        )
        self.Dynamic_optimizer = torch.optim.Adam(
            params=self.Dynamic_func.parameters(), lr=Dynamic_lr
        )

        self.W_lr_scheduler = LambdaLR(self.W_u_optimizer, lr_lambda=self.W_lr_fn)
        self.D_lr_scheduler = LambdaLR(self.Dynamic_optimizer, lr_lambda=self.D_lr_fn)

        #
        self.dummy = torch.tensor(1e-5)
        self.to(self.device)

    def W_lr_fn(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def D_lr_fn(self, step):
        return 0.999**step

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        self._forward_steps += 1
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        x, xref, uref, x_trim, xref_trim = self.trim_state(state)
        a = self.u_func(x, xref, uref, x_trim, xref_trim)

        return a, {
            "probs": self.dummy,
            "logprobs": self.dummy,
            "entropy": self.dummy,
        }

    def contraction_loss(
        self,
        states: torch.Tensor,
        detach: bool,
    ):
        true_dict = self.get_true_metrics(states)

        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        x = x.requires_grad_()

        W = self.W_func(x, xref, uref, x_trim, xref_trim)
        M = inverse(W)

        f_approx, B_approx, Bbot_approx = self.Dynamic_func(x)
        Bbot_approx = self.compute_B_perp_batch(B_approx, self.x_dim - self.action_dim)

        DfDx = self.Jacobian(f_approx, x).detach()  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B_approx, x).detach()  # n, x_dim, x_dim, b_dim

        f_approx = f_approx.detach()
        B_approx = B_approx.detach()
        Bbot_approx = Bbot_approx.detach()

        u = self.u_func(x, xref, uref, x_trim, xref_trim)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

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
        overshoot_loss = self.loss_pos_matrix_random_sampling(
            self.w_ub * torch.eye(W.shape[-1]).to(self.device) - W
        )
        c1_loss = self.loss_pos_matrix_random_sampling(
            -C1 - self.eps * torch.eye(C1.shape[-1]).to(self.device)
        )
        # c2_loss = sum([C2.sum().mean() for C2 in C2s])
        c2_loss = sum([(matrix_norm(C2) ** 2).mean() for C2 in C2s])

        loss = pd_loss + overshoot_loss + c1_loss + c2_loss

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

        return (
            loss,
            {
                "pd_loss": pd_loss.item(),
                "overshoot_loss": overshoot_loss.item(),
                "c1_loss": c1_loss.item(),
                "c2_loss": c2_loss.item(),
                "C_pos_eig": C_pos_eig.item(),
                "C_neg_eig": C_neg_eig.item(),
                "C1_pos_eig": C1_pos_eig.item(),
                "C1_neg_eig": C1_neg_eig.item(),
                "dot_M_pos_eig": dot_M_pos_eig.item(),
                "dot_M_neg_eig": dot_M_neg_eig.item(),
                "sym_MABK_pos_eig": sym_MABK_pos_eig.item(),
                "sym_MABK_neg_eig": sym_MABK_neg_eig.item(),
                "M_pos_eig": M_pos_eig.item(),
                "M_neg_eig": M_neg_eig.item(),
                "dot_M_error": dot_M_error.item(),
                "ABK_error": ABK_error.item(),
                "Bbot_error": Bbot_error.item(),
            },
        )

    def get_matrix_eig(self, A: torch.Tensor):
        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(A)  # (batch, dim), real symmetric
            pos_eigvals = torch.relu(eigvals)
            neg_eigvals = torch.relu(-eigvals)
        return pos_eigvals.mean(dim=1).mean(), neg_eigvals.mean(dim=1).mean()

    def learn(self, batch):
        if self.num_inner_update <= int(0.1 * self.nupdates):
            loss_dict, update_time = self.learn_Dynamics(batch)
            loss_dict = {}
            timesteps = 0
            update_time = 0
            # timesteps = batch["rewards"].shape[0]
            self.num_inner_update += 1
        else:
            detach = (
                True if self.num_outer_update <= int(0.1 * self.nupdates) else False
            )

            loss_dict, timesteps, update_time = self.learn_W(batch, detach)
            D_loss_dict, D_update_time = self.learn_Dynamics(batch)

            loss_dict.update(D_loss_dict)
            update_time += D_update_time

            self.num_outer_update += 1
            self.W_lr_scheduler.step()
            self.D_lr_scheduler.step()

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

        loss = fB_loss + ortho_loss  # + cont_loss

        with torch.no_grad():
            f_error = F.l1_loss(true_dict["f_true"], f_approx)
            B_error = F.l1_loss(true_dict["B_true"], B_approx)

        self.Dynamic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Dynamic_func.parameters(), max_norm=5.0)
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
            # f"{self.name}/Dynamic_loss/cont_loss": cont_loss.item(),
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

    def learn_W(self, batch: dict, detach: bool):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        actions = to_tensor(batch["actions"])
        next_states = to_tensor(batch["next_states"])
        rewards = to_tensor(batch["rewards"])
        terminals = to_tensor(batch["terminals"])

        # List to track actor loss over minibatches
        loss, infos = self.contraction_loss(states, detach)

        self.W_u_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.W_func.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.u_func.parameters(), max_norm=10.0)
        grad_dict = self.compute_gradient_norm(
            [self.W_func, self.u_func],
            ["W_func", "u_func"],
            dir=f"{self.name}",
            device=self.device,
        )
        self.W_u_optimizer.step()
        norm_dict = self.compute_weight_norm(
            [self.W_func, self.u_func],
            ["W_func", "u_func"],
            dir=f"{self.name}",
            device=self.device,
        )

        # Logging
        loss_dict = {
            f"{self.name}/loss/loss": loss.item(),
            f"{self.name}/loss/pd_loss": infos["pd_loss"],
            f"{self.name}/loss/overshoot_loss": infos["overshoot_loss"],
            f"{self.name}/loss/c1_loss": infos["c1_loss"],
            f"{self.name}/loss/c2_loss": infos["c2_loss"],
            f"{self.name}/C_analytics/C_pos_eig": infos["C_pos_eig"],
            f"{self.name}/C_analytics/C_neg_eig": infos["C_neg_eig"],
            f"{self.name}/C_analytics/C1_pos_eig": infos["C1_pos_eig"],
            f"{self.name}/C_analytics/C1_neg_eig": infos["C1_neg_eig"],
            f"{self.name}/C_analytics/dot_M_pos_eig": infos["dot_M_pos_eig"],
            f"{self.name}/C_analytics/dot_M_neg_eig": infos["dot_M_neg_eig"],
            f"{self.name}/C_analytics/sym_MABK_pos_eig": infos["sym_MABK_pos_eig"],
            f"{self.name}/C_analytics/sym_MABK_neg_eig": infos["sym_MABK_neg_eig"],
            f"{self.name}/C_analytics/M_pos_eig": infos["M_pos_eig"],
            f"{self.name}/C_analytics/M_neg_eig": infos["M_neg_eig"],
            f"{self.name}/C_analytics/dot_M_error": infos["dot_M_error"],
            f"{self.name}/C_analytics/ABK_error": infos["ABK_error"],
            f"{self.name}/C_analytics/Bbot_error": infos["Bbot_error"],
            f"{self.name}/analytics/avg_rewards": torch.mean(rewards).item(),
        }

        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        timesteps = terminals.shape[0]

        # Cleanup
        del states, actions, next_states, terminals
        self.eval()

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
            return torch.tensor(0.0).to(self.device).requires_grad_()

    def trim_state(self, state: torch.Tensor):
        # state = state.requires_grad_()

        # state trimming
        x = state[:, : self.x_dim]
        xref = state[:, self.x_dim : -self.action_dim]
        uref = state[:, -self.action_dim :]

        x_trim = x[:, self.effective_indices]
        xref_trim = xref[:, self.effective_indices]

        return x, xref, uref, x_trim, xref_trim

    def get_rewards(self, states):
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        x = x.requires_grad_()

        with torch.no_grad():
            ### Compute the main rewards
            W = self.W_func(x, xref, uref, x_trim, xref_trim)
            M = torch.inverse(W)

            error = (x - xref).unsqueeze(-1)
            errorT = transpose(error, 1, 2)

            rewards = (1 / (errorT @ M @ error + 1)).squeeze(-1)

        ### Compute the aux rewards ###
        f = self.f_func(x).to(self.device)  # n, x_dim
        B = self.B_func(x).to(self.device)  # n, x_dim, action

        DfDx = self.Jacobian(f, x).detach()  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x).detach()  # n, x_dim, x_dim, b_dim

        u = self.u_func(x, xref, uref, x_trim, xref_trim)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        u = u.detach()
        K = K.detach()

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.action_dim)
            ]
        )

        ABK = A + matmul(B, K)
        MABK = matmul(M, ABK)
        sym_MABK = MABK + transpose(MABK, 1, 2)

        C_u_only = -sym_MABK - self.eps * torch.eye(sym_MABK.shape[-1]).to(self.device)

        aux_rewards = torch.linalg.eigvalsh(C_u_only).mean(dim=1).unsqueeze(-1)

        pos_indices = aux_rewards > 0
        neg_indices = aux_rewards <= 0

        aux_rewards[pos_indices] = torch.tanh(aux_rewards[pos_indices] / 30)
        aux_rewards[neg_indices] = -1.0

        alpha = 0.5
        rewards = alpha * rewards + (1 - alpha) * aux_rewards

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

        J = torch.zeros(n, f_dim, x_dim).to(self.device)
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

    def get_true_metrics(self, states: torch.Tensor):
        #### COMPUTE THE REAL DYNAMICS TO MEASURE ERRORS ####
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)
        x = x.requires_grad_()

        with torch.no_grad():
            W = self.W_func(x, xref, uref, x_trim, xref_trim)
            M = inverse(W)

        f = self.f_func(x).to(self.device)  # n, x_dim
        B = self.B_func(x).to(self.device)  # n, x_dim, action

        DfDx = self.Jacobian(f, x).detach()  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x).detach()  # n, x_dim, x_dim, b_dim
        Bbot = self.Bbot_func(x).detach().to(self.device)

        u = self.u_func(x, xref, uref, x_trim, xref_trim)
        K = self.Jacobian(u, x)  # n, f_dim, x_dim

        u = u.detach()
        K = K.detach()

        A = DfDx + sum(
            [
                u[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.action_dim)
            ]
        )

        dot_x = (f + matmul(B, u.unsqueeze(-1)).squeeze(-1)).detach()
        dot_M = self.weighted_gradients(M, dot_x, x, True)

        ABK = A + matmul(B, K)
        MABK = matmul(M.detach(), ABK)
        sym_MABK = MABK + transpose(MABK, 1, 2)

        return {
            "dot_x_true": dot_x,
            "dot_M_true": dot_M,
            "ABK_true": ABK,
            "sym_MABK_true": sym_MABK,
            "Bbot_true": Bbot,
            "f_true": f,
            "B_true": B,
        }

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

    def compute_B_perp_batch(self, B, B_perp_dim, method="svd", threshold=1e-1):
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
