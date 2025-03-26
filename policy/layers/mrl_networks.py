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


class MRL_W(nn.Module):
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
    ):
        super(MRL_W, self).__init__()

        self.x_dim = x_dim
        self.state_dim = state_dim
        self.effective_x_dim = len(effective_indices)
        self.effective_indices = effective_indices
        self.action_dim = action_dim

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
            Wbot = self.model_Wbot(torch.ones(n, 1).type(x_trim.type())).view(
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
            W = W + self.w_lb * torch.eye(self.x_dim).view(
                1, self.x_dim, self.x_dim
            ).type(x_trim.type())
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
            W = W + self.w_lb * torch.eye(self.x_dim).view(
                1, self.x_dim, self.x_dim
            ).type(x_trim.type())
        elif self.task in ("pvtol", "segway"):
            n = x_trim.shape[0]

            W = self.model_W(x_trim).view(n, self.x_dim, self.x_dim)

            W = W.transpose(1, 2).matmul(W)
            W = W + self.w_lb * torch.eye(self.x_dim).view(
                1, self.x_dim, self.x_dim
            ).type(x_trim.type())
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
            W = W + self.w_lb * torch.eye(self.x_dim).view(
                1, self.x_dim, self.x_dim
            ).type(x_trim.type())
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


class MRL_Actor(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        x_dim: int,
        effective_indices: list,
        action_dim: int,
        task: str,
    ):
        super(MRL_Actor, self).__init__()
        """
        
        """
        self.x_dim = x_dim
        self.effective_x_dim = len(effective_indices)
        self.action_dim = action_dim
        self.effective_indices = effective_indices

        self.task = task

        self.w1, self.w2 = get_u_model(
            self.task, x_dim, self.effective_x_dim, self.action_dim
        )

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

        # Concatenate trimmed state and reference features.
        # x_xref = torch.cat((x, xref), axis=-1)
        x_xref_trim = torch.cat((x_trim, xref_trim), axis=-1)

        # Compute the error between x and xref.
        e = (x - xref).unsqueeze(-1)  # shape: (n, x_dim, 1)

        # Process concatenated features through two linear layers (or conv layers, etc.).
        w1 = self.w1(x_xref_trim).reshape(n, -1, self.x_dim)
        w2 = self.w2(x_xref_trim).reshape(n, self.action_dim, -1)

        # Compute intermediate non-linear features.
        l1 = torch.tanh(torch.matmul(w1, e))
        mu = torch.matmul(w2, l1).squeeze(-1)

        # The deterministic mean is the offset from the reference action.
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)

        # Form a diagonal covariance matrix.
        covariance_matrix = torch.diag_embed(std**2)
        dist = MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)

        # Sample actions stochastically (or choose deterministic actions).
        if deterministic:
            a = mu
        else:
            a = dist.rsample()

        # Compute log-probabilities and entropy for PPO loss.
        logprobs = dist.log_prob(a).unsqueeze(-1)  # shape: (n, 1)
        probs = torch.exp(logprobs)
        entropy = dist.entropy()  # shape: (n,)

        u = a + uref

        return u, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def log_prob(self, dist: torch.distributions, actions: torch.Tensor):
        """
        Actions must be tensor
        """
        actions = actions.squeeze() if actions.shape[-1] > 1 else actions
        logprobs = dist.log_prob(actions).unsqueeze(-1)
        return logprobs

    def entropy(self, dist: torch.distributions):
        """
        For code consistency
        """
        return dist.entropy().unsqueeze(-1)


class MRL_Critic(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self, input_dim: int, hidden_dim: list, activation: nn.Module = nn.Tanh()
    ):
        super(MRL_Critic, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self.model = MLP(
            input_dim, hidden_dim, 1, activation=self.act, initialization="critic"
        )

    def forward(self, x: torch.Tensor):
        value = self.model(x)
        return value
