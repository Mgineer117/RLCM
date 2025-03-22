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

    def B_null(self, states):
        n = states.shape[0]
        Bbot = torch.cat(
            (
                torch.eye(
                    self.state_dim - self.action_dim, self.state_dim - self.action_dim
                ),
                torch.zeros(self.action_dim, self.state_dim - self.action_dim),
            ),
            dim=0,
        )
        Bbot.unsqueeze(0).to(self.device)
        return Bbot.repeat(n, 1, 1)

    def Jacobian(self, f, states):
        # NOTE that this function assume that data are independent of each other
        print(f.shape, states.shape)
        f = f + 0.0 * states.sum()  # to avoid the case that f is independent of x

        n = states.shape[0]
        f_dim = f.shape[-1]
        state_dim = states.shape[-1]

        J = torch.zeros(n, f_dim, state_dim).to(states.type())
        for i in range(f_dim):
            J[:, i, :] = grad(f[:, i].sum(), states, create_graph=True)[0]
        return J

    def Jacobian_Matrix(self, M, states):
        # NOTE that this function assume that data are independent of each other
        n = states.shape[0]
        matrix_dim = M.shape[-1]
        state_dim = states.shape[-1]

        J = torch.zeros(n, matrix_dim, matrix_dim, state_dim).to(states.type())
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                J[:, i, j, :] = grad(M[:, i, j].sum(), states, create_graph=True)[0]

        return J

    def B_Jacobian(self, B, states):
        n = states.shape[0]
        b_dim = B.shape[-1]
        state_dim = states.shape[-1]

        DBDx = torch.zeros(n, state_dim, state_dim, b_dim).to(states.type())
        for i in range(b_dim):
            DBDx[:, :, :, i] = self.Jacobian(B[:, :, i], states)
        return DBDx

    def weighted_gradients(self, W, v, x):
        # v, x: bs x n x 1
        # DWDx: bs x n x n x n
        assert v.size() == x.size()
        bs = x.shape[0]
        return (self.Jacobian_Matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)

    def loss_pos_matrix_random_sampling(self, A):
        # A: n x d x d
        # z: K x d
        K = 1024
        zT = torch.randn((K, A.size(-1))).to(self.device)
        zT = zT / zT.norm(dim=-1, keepdim=True)
        z = transpose(zT)

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

    def learn(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        actions = to_tensor(batch["actions"])  # n, action_dim
        rewards = to_tensor(batch["rewards"])
        terminals = to_tensor(batch["terminals"])

        #### COMPUTE INGREDIENTS ####
        x, xref, uref, x_trim, xref_trim = self.trim_state(states)

        W = self.W_func(x, xref, uref, x_trim, xref_trim)  # n, x_dim, x_dim
        M = inverse(W)  # n, x_dim, x_dim

        f = self.f_func(x)  # n, x_dim
        B = self.B_func(x)  # n, x_dim, action

        DfDx = self.Jacobian(f, x)  # n, f_dim, x_dim
        DBDx = self.B_Jacobian(B, x)  # n, x_dim, x_dim, b_dim
        Bbot = self.Bbot_func(x)  # n, x_dim, state - action dim

        # since online we do not do below
        # u = u_func(x, x - xref, uref)  # u: bs x m x 1 # TODO: x - xref
        K = self.Jacobian(actions, x)  # n, f_dim, x_dim

        #  actions[:, i]: n, 1, 1
        #  DBDx[:, :, :, i]: n, x_dim, x_dim
        A = DfDx + sum(
            [
                actions[:, i].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i]
                for i in range(self.action_dim)
            ]
        )

        dot_x = f + matmul(B, actions)
        dot_M = self.weighted_gradients(M, dot_x, states)
        dot_W = self.weighted_gradients(W, dot_x, states)

        # contraction condition
        ABK = A + matmul(B, K)
        MABK = matmul(M, ABK)
        sym_MABK = MABK + transpose(MABK)
        C_u = dot_M + sym_MABK + 2 * self.lbd * M

        # C1
        DfW = self.weighted_gradients(W, f, states)
        DfDxW = matmul(DfDx, W)
        sym_DfDxW = DfDxW + transpose(DfDxW)

        # this has to be a negative definite matrix
        C1_inner = -DfW + sym_DfDxW + 2 * self.lbd * M
        C1 = matmul(matmul(transpose(Bbot), C1_inner), Bbot)

        C2_inner = []
        C2s = []
        for j in range(self.action_dim):
            DbW = self.weighted_gradients(W, B[:, :, j], states)
            DbDxW = matmul(DBDx, W)
            sym_DbDxW = DbDxW + transpose(DbDxW)
            C2_inner = DbW - DbDxW
            C2 = matmul(matmul(transpose(Bbot), C2_inner), Bbot)

            C2_inner.append(C2_inner)
            C2s.append(C2)

        #### COMPUTE LOSS ####
        pd_loss = self.loss_pos_matrix_random_sampling(
            -C_u - self.eps * torch.eye(C_u.shape[-1]).to(C_u.type())
        )
        c1_loss = self.loss_pos_matrix_random_sampling(
            -C1 - self.eps * torch.eye(C1.shape[-1]).to(C1.type())
        )
        c2_loss = sum([C2.sum().mean() for C2 in C2s])
        overshoot_loss = self.loss_pos_matrix_random_sampling(
            self.w_ub * torch.eye(W.shape[-1]).to(W.type()) - W
        )

        loss = pd_loss + c1_loss + c2_loss + overshoot_loss

        self.optimizer.zero_grad()
        loss.backward()
        grad_dict = self.compute_gradient_norm(
            [self.W_func, self.u_func],
            ["W_func", "u_func"],
            dir="C3M",
            device=self.device,
        )
        self.optimizer.step()

        # Logging
        loss_dict = {
            "C3M/loss/loss": np.mean(loss),
            "C3M/loss/pd_loss": np.mean(pd_loss),
            "C3M/loss/c1_loss": np.mean(c1_loss),
            "C3M/loss/c2_loss": np.mean(c2_loss),
            "C3M/loss/overshoot_loss": np.mean(overshoot_loss),
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
        del states, actions, rewards, terminals
        self.eval()

        timesteps = self.num_minibatch * self.minibatch_size
        update_time = time.time() - t0
        return loss_dict, timesteps, update_time
