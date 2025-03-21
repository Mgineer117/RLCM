import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, Categorical
from policy.layers.building_blocks import MLP


class C3M_W(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self, state_dim: int, effective_indices: list, action_dim: int, w_lb: float
    ):
        super(C3M_W, self).__init__()

        self.state_dim = state_dim
        self.effective_state_dim = len(effective_indices)
        self.effective_indices = effective_indices
        self.action_dim = action_dim

        self.w_lb = w_lb

        self.model_Wbot = torch.nn.Sequential(
            torch.nn.Linear(
                self.effective_state_dim - self.action_dim,
                128,
                bias=True,
            ),
            torch.nn.Tanh(),
            torch.nn.Linear(128, (self.state_dim - self.action_dim) ** 2, bias=False),
        )

        self.model_W = torch.nn.Sequential(
            torch.nn.Linear(self.effective_state_dim, 128, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(128, self.state_dim * self.action_dim, bias=False),
        )

    def trim_state(self, state: torch.Tensor):
        # state trimming
        x_xref = state[:, : -self.action_dim]
        uref = state[:, -self.action_dim :]

        n = x_xref.shape[0] // 2
        x = x_xref[:, :n]
        xref = x_xref[:, n:]

        x = x[:, self.effective_indices]
        xref = xref[:, self.effective_indices]

        x = state[:, : self.state_dim]
        xref = state[:, self.state_dim : 2 * self.state_dim]
        uref = state[:, -self.action_dim :]

        return x, xref, uref

    def forward(self, state: torch.Tensor):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        x, _, _ = self.trim_state(state)
        n = x.shape[0]

        W = self.model_W(x[:, self.effective_indices]).view(
            n, self.state_dim, self.state_dim
        )
        Wbot = self.model_Wbot(x[:, self.effective_indices[: -self.action_dim]]).view(
            n, self.state_dim - self.action_dim, self.state_dim - self.action_dim
        )
        W[
            :,
            0 : self.state_dim - self.action_dim,
            0 : self.state_dim - self.action_dim,
        ] = Wbot
        W[
            :,
            self.state_dim - self.action_dim : :,
            0 : self.state_dim - self.action_dim,
        ] = 0

        W = W.transpose(1, 2).matmul(W)
        W = W + self.w_lb * torch.eye(self.state_dim).view(
            1, self.state_dim, self.state_dim
        ).type(x.type())

        return W

    def log_prob(self, dist: torch.distributions, actions: torch.Tensor):
        """
        Actions must be tensor
        """
        actions = actions.squeeze() if actions.shape[-1] > 1 else actions

        if self.is_discrete:
            logprobs = dist.log_prob(torch.argmax(actions, dim=-1)).unsqueeze(-1)
        else:
            logprobs = dist.log_prob(actions).unsqueeze(-1)
        return logprobs

    def entropy(self, dist: torch.distributions):
        """
        For code consistency
        """
        return dist.entropy().unsqueeze(-1)


class C3M_U(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        state_dim: int,
        effective_indices: list,
        action_dim: int,
    ):
        super(C3M_U, self).__init__()
        """
        
        """
        self.state_dim = state_dim
        self.effective_state_dim = len(effective_indices)
        self.action_dim = action_dim
        self.effective_indices = effective_indices

        input_dim = 2 * self.effective_state_dim
        c = 3 * self.state_dim

        self.w1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(128, c * self.state_dim, bias=True),
        )

        self.w2 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(128, c * self.action_dim, bias=True),
        )

    def trim_state(self, state: torch.Tensor):
        # state trimming
        x_xref = state[:, : -self.action_dim]
        uref = state[:, -self.action_dim :]

        n = x_xref.shape[0] // 2
        x = x_xref[:, :n]
        xref = x_xref[:, n:]

        x = x[:, self.effective_indices]
        xref = xref[:, self.effective_indices]

        x = state[:, : self.state_dim]
        xref = state[:, self.state_dim : 2 * self.state_dim]
        uref = state[:, -self.action_dim :]

        return x, xref, uref

    def forward(self, state: torch.Tensor):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        n = state.shape[0]
        x, xref, uref = self.trim_state(state)
        e = x - xref

        w1 = self.w1(state).reshape(n, -1, self.state_dim)
        w2 = self.w2(state).reshape(n, self.action_dim, -1)

        l1 = F.tanh(torch.matmul(w1, e))
        l2 = torch.matmul(w2, l1)

        u = l2 + uref
        return u
