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


class C3M(Base):
    def __init__(
        self,
        x_dim: int,
        state_dim: int,
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
        self.state_dim = state_dim
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

        #
        self.dummy = torch.tensor(1e-5)
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        self._forward_steps += 1
        state = torch.from_numpy(state).to(self._dtype).to(self.device)

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

    def learn(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        if self.current_update >= int(0.3 * self.nupdates):
            detach = False
        else:
            detach = True

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        rewards = to_tensor(batch["rewards"])
        n = rewards.shape[0]

        #### COMPUTE INGREDIENTS ####
        # grad tracking state elements
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)

        W = self.W_func(x, xref, uref, x_trim, xref_trim)  # n, x_dim, x_dim
        M = inverse(W)  # n, x_dim, x_dim

        f = self.f_func(x)  # n, x_dim
        B = self.B_func(x)  # n, x_dim, action

        DfDx = self.Jacobian(f, x)  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x)  # n, x_dim, x_dim, b_dim
        Bbot = self.Bbot_func(x)  # n, x_dim, state - action dim

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
        c2_loss = sum([C2.sum().mean() for C2 in C2s])
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

        C_eig, _ = torch.linalg.eig(C_u)
        C1_eig, _ = torch.linalg.eig(C1)

        C_eig = torch.real(C_eig)
        C1_eig = torch.real(C1_eig)

        C_eig_contraction = ((C_eig >= 0).sum(dim=-1) == 0).cpu().detach().numpy()
        C1_eig_contraction = ((C1_eig >= 0).sum(dim=1) == 0).cpu().detach().numpy()

        # Logging
        loss_dict = {
            "C3M/loss/loss": loss.item(),
            "C3M/loss/pd_loss": pd_loss.item(),
            "C3M/loss/c1_loss": c1_loss.item(),
            "C3M/loss/c2_loss": c2_loss.item(),
            "C3M/loss/overshoot_loss": overshoot_loss.item(),
            "C3M/analytics/C_eig_contraction": C_eig_contraction.mean(),
            "C3M/analytics/C1_eig_contraction": C1_eig_contraction.mean(),
            "C3M/analytics/avg_rewards": torch.mean(rewards).item(),
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
        return loss_dict, timesteps, update_time
