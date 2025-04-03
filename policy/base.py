import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch import matmul, inverse, transpose
import os


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()

        self._dtype = torch.float32
        self.device = torch.device("cpu")

        # utils
        self.l1_loss = F.l1_loss
        self.mse_loss = F.mse_loss
        self.huber_loss = F.smooth_l1_loss

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
        x = state[:, : self.x_dim].requires_grad_()
        xref = state[:, self.x_dim : -self.action_dim].requires_grad_()
        uref = state[:, -self.action_dim :].requires_grad_()

        x_trim = x[:, self.effective_indices].requires_grad_()
        xref_trim = xref[:, self.effective_indices].requires_grad_()

        return x, xref, uref, x_trim, xref_trim

    def get_matrix_eig(self, A: torch.Tensor):
        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(A)  # (batch, dim), real symmetric
            pos_eigvals = torch.relu(eigvals)
            neg_eigvals = torch.relu(-eigvals)
        return pos_eigvals.mean(dim=1).mean(), neg_eigvals.mean(dim=1).mean()

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

    def compute_gradient_norm(self, models, names, device, dir="None", norm_type=2):
        grad_dict = {}
        for i, model in enumerate(models):
            if model is not None:
                total_norm = torch.tensor(0.0, device=device)
                try:
                    for param in model.parameters():
                        if (
                            param.grad is not None
                        ):  # Only consider parameters that have gradients
                            param_grad_norm = torch.norm(param.grad, p=norm_type)
                            total_norm += param_grad_norm**norm_type
                except:
                    try:
                        param_grad_norm = torch.norm(model.grad, p=norm_type)
                    except:
                        param_grad_norm = torch.tensor(0.0)
                    total_norm += param_grad_norm**norm_type

                total_norm = total_norm ** (1.0 / norm_type)
                grad_dict[dir + "/grad/" + names[i]] = total_norm.item()

        return grad_dict

    def compute_weight_norm(self, models, names, device, dir="None", norm_type=2):
        norm_dict = {}
        for i, model in enumerate(models):
            if model is not None:
                total_norm = torch.tensor(0.0, device=device)
                try:
                    for param in model.parameters():
                        param_norm = torch.norm(param, p=norm_type)
                        total_norm += param_norm**norm_type
                except:
                    param_norm = torch.norm(model, p=norm_type)
                    total_norm += param_norm**norm_type
                total_norm = total_norm ** (1.0 / norm_type)
                norm_dict[dir + "/weight/" + names[i]] = total_norm.item()

        return norm_dict

    def learn(self):
        pass
