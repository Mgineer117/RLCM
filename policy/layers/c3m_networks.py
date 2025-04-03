import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, Categorical
from policy.layers.building_blocks import MLP


def get_W_model(task, x_dim, effective_x_dim, action_dim):
    if task in ("car", "neurallander", "pvtol", "segway"):
        model_W = torch.nn.Sequential(
            torch.nn.Linear(effective_x_dim, 128, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(128, x_dim * x_dim, bias=False),
        )
        model_Wbot = torch.nn.Sequential(
            torch.nn.Linear(
                1,
                128,
                bias=True,
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(128, (x_dim - action_dim) ** 2, bias=False),
        )
    elif task == "quadrotor":
        model_W = torch.nn.Sequential(
            torch.nn.Linear(effective_x_dim, 128, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(128, x_dim * x_dim, bias=False),
        )
        model_Wbot = torch.nn.Sequential(
            torch.nn.Linear(
                effective_x_dim - action_dim,
                128,
                bias=True,
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(128, (x_dim - action_dim) ** 2, bias=False),
        )

    return model_W, model_Wbot


class C3M_W_Gaussian(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        x_dim: int,
        state_dim: int,
        hidden_dim: list,
        w_lb: float,
        activation: nn.Module = nn.Tanh(),
        device: str = "cpu",
    ):
        super(C3M_W_Gaussian, self).__init__()

        self.x_dim = x_dim
        self.state_dim = state_dim

        self.device = device

        self.w_lb = w_lb

        self.model = MLP(
            input_dim=state_dim, hidden_dims=hidden_dim, activation=activation
        )
        self.mu = nn.Linear(hidden_dim[-1], x_dim**2, bias=True)
        self.logstd = nn.Linear(hidden_dim[-1], x_dim**2, bias=True)
        nn.init.constant_(self.logstd.bias, 0.0)

    def forward(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
    ):
        n = states.shape[0]
        logits = self.model(states)  # .view(n, self.x_dim, self.x_dim)
        mu = self.mu(logits)  # .view(n, self.x_dim, self.x_dim)
        logstd = self.logstd(logits).clamp(min=-1, max=2)
        var = torch.exp(logstd) ** 2
        # clip var such that entropy is positive always

        if deterministic:
            W = mu  # .view(n, self.x_dim, self.x_dim)
            dist = None
            logprobs = torch.zeros_like(mu[:, 0:1])
            probs = torch.ones_like(logprobs)  # log(1) = 0
            entropy = torch.zeros_like(logprobs)
        else:
            covariance_matrix = torch.diag_embed(var)  # Variance is std^2
            dist = MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)

            W = dist.rsample()  # .view(n, self.x_dim, self.x_dim)

            logprobs = dist.log_prob(W).unsqueeze(-1)
            probs = torch.exp(logprobs)
            entropy = dist.entropy()

            # print(logstd, entropy)

        W = W.view(n, self.x_dim, self.x_dim)
        W = W.transpose(1, 2).matmul(W)
        W = W + self.w_lb * torch.eye(self.x_dim).to(self.device).view(
            1, self.x_dim, self.x_dim
        )

        return W, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def log_prob(self, dist: torch.distributions, W: torch.Tensor):
        """
        Actions must be tensor
        """
        W = W.view(W.shape[0], -1)
        logprobs = dist.log_prob(W).unsqueeze(-1)
        return logprobs

    def entropy(self, dist: torch.distributions):
        """
        For code consistency
        """
        return dist.entropy().unsqueeze(-1)


class C3M_W(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        x_dim: int,
        state_dim: int,
        effective_indices: list,
        action_dim: int,
        w_lb: float,
        task: str,
        device: str = "cpu",
    ):
        super(C3M_W, self).__init__()

        self.x_dim = x_dim
        self.state_dim = state_dim
        self.effective_x_dim = len(effective_indices)
        self.effective_indices = effective_indices
        self.action_dim = action_dim

        self.device = device

        self.w_lb = w_lb

        self.task = task

        model_W, model_Wbot = get_W_model(
            self.task, self.x_dim, self.effective_x_dim, self.action_dim
        )
        self.model_W = model_W
        self.model_Wbot = model_Wbot

    def forward(
        self,
        x: torch.Tensor,
        xref: torch.Tensor,
        uref: torch.Tensor,
        x_trim: torch.Tensor,
        xref_trim: torch.Tensor,
    ):
        if self.task == "car":
            n = x_trim.shape[0]

            W = self.model_W(x_trim).view(n, self.x_dim, self.x_dim)
            Wbot = self.model_Wbot(torch.ones(n, 1).to(self.device)).view(
                n, self.x_dim - self.action_dim, self.x_dim - self.action_dim
            )
            W[
                :,
                : self.x_dim - self.action_dim,
                : self.x_dim - self.action_dim,
            ] = Wbot
            W[
                :,
                self.x_dim - self.action_dim : :,
                : self.x_dim - self.action_dim,
            ] = 0

            W = W.transpose(1, 2).matmul(W)
            W = W + self.w_lb * torch.eye(self.x_dim).to(self.device).view(
                1, self.x_dim, self.x_dim
            )
        elif self.task == "neurallander":
            n = x_trim.shape[0]

            W = self.model_W(x_trim).view(n, self.x_dim, self.x_dim)
            Wbot = self.model_Wbot(x[:, 2:3]).view(
                n, self.x_dim - self.action_dim, self.x_dim - self.action_dim
            )
            W[
                :,
                : self.x_dim - self.action_dim,
                : self.x_dim - self.action_dim,
            ] = Wbot
            W[
                :,
                self.x_dim - self.action_dim : :,
                : self.x_dim - self.action_dim,
            ] = 0

            W = W.transpose(1, 2).matmul(W)
            W = W + self.w_lb * torch.eye(self.x_dim).to(self.device).view(
                1, self.x_dim, self.x_dim
            )
        elif self.task in ("pvtol", "segway"):
            n = x_trim.shape[0]

            W = self.model_W(x_trim).view(n, self.x_dim, self.x_dim)

            W = W.transpose(1, 2).matmul(W)
            W = W + self.w_lb * torch.eye(self.x_dim).to(self.device).view(
                1, self.x_dim, self.x_dim
            )
        elif self.task == "quadrotor":
            n = x_trim.shape[0]

            W = self.model_W(x_trim).view(n, self.x_dim, self.x_dim)
            Wbot = self.model_Wbot(
                x[:, self.effective_indices[: -self.action_dim]]
            ).view(n, self.x_dim - self.action_dim, self.x_dim - self.action_dim)
            W[
                :,
                : self.x_dim - self.action_dim,
                : self.x_dim - self.action_dim,
            ] = Wbot
            W[
                :,
                self.x_dim - self.action_dim : :,
                : self.x_dim - self.action_dim,
            ] = 0

            W = W.transpose(1, 2).matmul(W)
            W = W + self.w_lb * torch.eye(self.x_dim).to(self.device).view(
                1, self.x_dim, self.x_dim
            )
        return W


def get_u_model(task, x_dim, effective_x_dim, action_dim):
    input_dim = 2 * effective_x_dim
    c = 3 * x_dim

    w1 = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, c * x_dim, bias=True),
    )

    w2 = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, c * action_dim, bias=True),
    )

    return w1, w2


class C3M_U(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        x_dim: int,
        state_dim: int,
        effective_indices: list,
        action_dim: int,
        task: str,
    ):
        super(C3M_U, self).__init__()
        """
        
        """
        self.x_dim = x_dim
        self.state_dim = state_dim
        self.effective_x_dim = len(effective_indices)
        self.action_dim = action_dim
        self.effective_indices = effective_indices

        self.task = task

        self.w1, self.w2 = get_u_model(
            self.task, x_dim, self.effective_x_dim, self.action_dim
        )

    def trim_state(self, state: torch.Tensor):
        # state trimming
        x = state[:, : self.x_dim]
        xref = state[:, self.x_dim : -self.action_dim]
        uref = state[:, -self.action_dim :]

        x_trim = x[:, self.effective_indices]
        xref_trim = xref[:, self.effective_indices]

        return x, xref, uref, x_trim, xref_trim

    def forward(
        self,
        x: torch.Tensor,
        xref: torch.Tensor,
        uref: torch.Tensor,
        x_trim: torch.Tensor,
        xref_trim: torch.Tensor,
        deterministic: bool = False,
    ):
        n = x.shape[0]

        x_xref_trim = torch.cat((x_trim, xref_trim), axis=-1)
        e = (x - xref).unsqueeze(-1)

        w1 = self.w1(x_xref_trim).reshape(n, -1, self.x_dim)
        w2 = self.w2(x_xref_trim).reshape(n, self.action_dim, -1)

        l1 = F.tanh(torch.matmul(w1, e))
        u = torch.matmul(w2, l1).squeeze(-1)

        # a = u + uref

        return u
