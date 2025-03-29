import time
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import grad
from torch import matmul, inverse, transpose
from torch.linalg import solve
from scipy.linalg import solve_continuous_are
import numpy as np
from typing import Callable
from policy.base import Base


class LQR(Base):
    def __init__(
        self,
        x_dim: int,
        effective_indices: list,
        action_dim: int,
        f_func: Callable,
        B_func: Callable,
        Bbot_func: Callable,
        Q_scaler: float = 1.0,
        R_scaler: float = 1.0,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        nupdates: int = 0,
        device: str = "cpu",
    ):
        super(LQR, self).__init__()

        """
        Do not use Multiprocessor => use less batch
        """
        # constants
        self.name = "LQR"
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

        self.Q_scaler = Q_scaler
        self.R_scaler = R_scaler
        self.f_func = f_func
        self.B_func = B_func
        if Bbot_func is None:
            self.Bbot_func = self.B_null
        else:
            self.Bbot_func = Bbot_func

        #
        self.dummy = torch.tensor(1e-5)
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        self._forward_steps += 1
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # shape: (1, state_dim)

        # Avoid multiprocessing + autograd threading conflict by disabling gradients when not needed
        # and using autograd context only when computing Jacobians

        # Decompose state
        x, xref, uref, x_trim, xref_trim = self.trim_state(state)

        # Safely create leaf tensor for gradient tracking
        xref = xref.requires_grad_()

        # Compute Jacobians inside enable_grad context
        with torch.enable_grad():
            # Compute f and B
            f_xref = self.f_func(xref)
            B_xref = self.B_func(xref)
            DfDx = self.Jacobian(f_xref, xref)  # shape: (1, x_dim, x_dim)
            DBDx = self.B_Jacobian(B_xref, xref)  # shape: (1, x_dim, x_dim, u_dim)

        # Compute A matrix: A = DfDx + sum_j uref_j * dB_j/dx
        A = DfDx.clone().squeeze(0)  # shape: (x_dim, x_dim)
        for j in range(self.action_dim):
            A += uref[0, j] * DBDx[0, :, :, j]  # shape: (x_dim, x_dim)

        B = B_xref.squeeze(0)  # shape: (x_dim, u_dim)

        # Solve Riccati equation: A^T P + P A - P B R^-1 B^T P + Q = -Q
        Q = (self.Q_scaler + 1e-5) * torch.eye(self.x_dim, device=self.device)
        R = (self.R_scaler + 1e-5) * torch.eye(self.action_dim, device=self.device)

        # Use SciPy solver for CARE
        A_np = A.detach().cpu().numpy()
        B_np = B.detach().cpu().numpy()
        Q_np = Q.detach().cpu().numpy()
        R_np = R.detach().cpu().numpy()
        P_np = solve_continuous_are(A_np, B_np, Q_np, R_np)
        P = torch.from_numpy(P_np).to(A)

        # Compute feedback gain: K = R^-1 B^T P
        K = solve(R, B.T @ P)  # shape: (u_dim, x_dim)

        # Compute LQR control law: u = uref - K @ e
        e = x - xref  # shape: (1, x_dim)
        u = uref - (K @ e.unsqueeze(-1)).squeeze(-1)

        # Return
        return u, {
            "probs": self.dummy,
            "logprobs": self.dummy,
            "entropy": self.dummy,
        }

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

    def trim_state(self, state: torch.Tensor):
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

        loss_dict = {
            f"{self.name}/analytics/avg_rewards": np.mean(batch["rewards"]).item()
        }
        timesteps = self.num_minibatch * self.minibatch_size
        update_time = time.time() - t0
        return loss_dict, timesteps, update_time


class LQR_Approximation(Base):
    def __init__(
        self,
        x_dim: int,
        effective_indices: list,
        action_dim: int,
        Dynamic_func: nn.Module,
        Dynamic_lr: float,
        f_func: Callable,
        B_func: Callable,
        Bbot_func: Callable,
        Q_scaler: float = 1.0,
        R_scaler: float = 1.0,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        nupdates: int = 0,
        dt: float = 0.03,
        device: str = "cpu",
    ):
        super(LQR_Approximation, self).__init__()

        """
        Do not use Multiprocessor => use less batch
        """
        # constants
        self.name = "LQR_Approximation"
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

        self.Dynamic_func = Dynamic_func
        self.optimizer = torch.optim.Adam(
            params=Dynamic_func.parameters(), lr=Dynamic_lr
        )

        self.Q_scaler = Q_scaler
        self.R_scaler = R_scaler
        self.f_func = f_func
        self.B_func = B_func
        if Bbot_func is None:
            self.Bbot_func = self.B_null
        else:
            self.Bbot_func = Bbot_func

        #
        self.num_update = 0
        self.dt = dt
        self.dummy = torch.tensor(1e-5)
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        self._forward_steps += 1
        state = torch.from_numpy(state).to(self._dtype).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # shape: (1, state_dim)

        # Decompose state
        x, xref, uref, x_trim, xref_trim = self.trim_state(state)
        xref = xref.requires_grad_()

        # Compute Jacobians inside enable_grad context
        with torch.enable_grad():
            # Compute f and B
            f_xref, B_xref, _ = self.Dynamic_func(xref)
            DfDx = self.Jacobian(f_xref, xref)  # shape: (1, x_dim, x_dim)
            DBDx = self.B_Jacobian(B_xref, xref)  # shape: (1, x_dim, x_dim, u_dim)

        # Compute A matrix: A = DfDx + sum_j uref_j * dB_j/dx
        A = DfDx.clone().squeeze(0)  # shape: (x_dim, x_dim)
        for j in range(self.action_dim):
            A += uref[0, j] * DBDx[0, :, :, j]  # shape: (x_dim, x_dim)

        B = B_xref.squeeze(0)  # shape: (x_dim, u_dim)

        # Solve Riccati equation: A^T P + P A - P B R^-1 B^T P + Q = -Q
        Q = (self.Q_scaler + 1e-5) * torch.eye(self.x_dim, device=self.device)
        R = (self.R_scaler + 1e-5) * torch.eye(self.action_dim, device=self.device)

        # Use SciPy solver for CARE
        A_np = A.detach().cpu().numpy()
        B_np = B.detach().cpu().numpy()
        Q_np = Q.detach().cpu().numpy()
        R_np = R.detach().cpu().numpy()
        P_np = solve_continuous_are(A_np, B_np, Q_np, R_np)
        P = torch.from_numpy(P_np).to(A)

        # Compute feedback gain: K = R^-1 B^T P
        K = solve(R, B.T @ P)  # shape: (u_dim, x_dim)

        # Compute LQR control law: u = uref - K @ e
        e = x - xref  # shape: (1, x_dim)
        u = uref - (K @ e.unsqueeze(-1)).squeeze(-1)

        # Return
        return u, {
            "probs": self.dummy,
            "logprobs": self.dummy,
            "entropy": self.dummy,
        }

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

    def trim_state(self, state: torch.Tensor):
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

        if self.num_update <= int(0.5 * self.nupdates):
            # Ingredients: Convert batch data to tensors
            def to_tensor(data):
                return torch.from_numpy(data).to(self._dtype).to(self.device)

            states = to_tensor(batch["states"])
            actions = to_tensor(batch["actions"])
            next_states = to_tensor(batch["next_states"])
            terminals = to_tensor(batch["terminals"])

            x, xref, uref, x_trim, xref_trim = self.trim_state(states)
            # next_x, next_xref, next_uref, next_x_trim, next_xref_trim = self.trim_state(
            #     next_states
            # )

            # dot_x_2nd = self.get_dot_x(
            #     x=x,
            #     next_x=next_x,
            #     terminals=terminals,
            # )

            f = self.f_func(x).to(self.device)  # n, x_dim
            B = self.B_func(x).to(self.device)  # n, x_dim, action
            dot_x = f + matmul(B, actions.unsqueeze(-1)).squeeze(-1)

            f_approx, B_approx, _ = self.Dynamic_func(x)
            dot_x_approx = f_approx + matmul(B_approx, actions.unsqueeze(-1)).squeeze(
                -1
            )

            fB_loss = F.mse_loss(dot_x, dot_x_approx)

            self.optimizer.zero_grad()
            fB_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
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

            with torch.no_grad():
                f = self.f_func(x)
                B = self.B_func(x)
                dot_x = f + matmul(B, actions.unsqueeze(-1)).squeeze(-1)

                f_error = F.l1_loss(f, f_approx)
                B_error = F.l1_loss(B, B_approx)
                dot_x_error = F.l1_loss(dot_x, dot_x_approx)

            loss_dict = {
                f"{self.name}/loss/fB_loss": fB_loss.item(),
                f"{self.name}/loss/f_error": f_error.item(),
                f"{self.name}/loss/B_error": B_error.item(),
                f"{self.name}/loss/dot_x_error": dot_x_error.item(),
                f"{self.name}/analytics/avg_rewards": np.mean(batch["rewards"]).item(),
            }
            loss_dict.update(grad_dict)
            loss_dict.update(norm_dict)

            timesteps = self.num_minibatch * self.minibatch_size
            update_time = time.time() - t0
        else:
            loss_dict = {}
            timesteps = self.num_minibatch * self.minibatch_size
            update_time = 0
        return loss_dict, timesteps, update_time

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
