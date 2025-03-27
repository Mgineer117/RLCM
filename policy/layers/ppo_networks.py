import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, Categorical
from policy.layers.building_blocks import MLP


class PPO_Actor(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        a_dim: int,
        activation: nn.Module = nn.Tanh(),
        is_discrete: bool = False,
    ):
        super(PPO_Actor, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self.action_dim = a_dim
        self.is_discrete = is_discrete

        self.model = MLP(
            input_dim, hidden_dim, a_dim, activation=self.act, initialization="actor"
        )

    def forward(self, state: torch.Tensor, deterministic: bool = False):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        state = state.reshape(state.shape[0], -1)

        logits = self.model(state)

        if self.is_discrete:
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            if deterministic:
                a_argmax = torch.argmax(probs, dim=-1)  # .to(self._dtype)
            else:
                a_argmax = dist.sample()
            a = F.one_hot(a_argmax, num_classes=self._a_dim)

            logprobs = dist.log_prob(a_argmax).unsqueeze(-1)
            probs = torch.sum(probs * a, dim=-1)

        else:
            ### Shape the output as desired
            mu = logits
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)

            covariance_matrix = torch.diag_embed(std**2)  # Variance is std^2
            dist = MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)

            if deterministic:
                a = mu
            else:
                a = dist.rsample()

            logprobs = dist.log_prob(a).unsqueeze(-1)
            probs = torch.exp(logprobs)

        entropy = dist.entropy()

        return a, {
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


class Manual_PPO_Actor(nn.Module):
    def __init__(
        self,
        x_dim: int,
        effective_indices: list,
        action_dim: int,
        task: str,
    ):
        super(Manual_PPO_Actor, self).__init__()
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

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ):
        x, xref, uref, x_trim, xref_trim = self.trim_state(state)
        n = x.shape[0]

        # Concatenate trimmed state and reference features.
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

        return a, {
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


class PPO_Critic(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self, input_dim: int, hidden_dim: list, activation: nn.Module = nn.Tanh()
    ):
        super(PPO_Critic, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self.model = MLP(
            input_dim, hidden_dim, 1, activation=self.act, initialization="critic"
        )

    def forward(self, x: torch.Tensor):
        value = self.model(x)
        return value
