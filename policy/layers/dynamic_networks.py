import torch
import torch.nn as nn
from policy.layers.building_blocks import MLP


class DynamicLearner(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        x_dim: int,
        action_dim: int,
        hidden_dim: list,
        drop_out: float | None = None,
        activation: nn.Module = nn.Tanh(),
    ):
        super(DynamicLearner, self).__init__()

        self.x_dim = x_dim
        self.action_dim = action_dim
        self.activation = activation

        self.f = MLP(
            x_dim, hidden_dim, x_dim, activation=self.activation, dropout_rate=drop_out
        )
        self.B = MLP(
            x_dim,
            hidden_dim,
            x_dim * action_dim,
            activation=self.activation,
            dropout_rate=drop_out,
        )
        self.Bbot = MLP(
            x_dim,
            hidden_dim,
            x_dim * (x_dim - action_dim),
            activation=self.activation,
            dropout_rate=drop_out,
        )

    def forward(self, x: torch.Tensor):
        n = x.shape[0]

        f = self.f(x)
        B = self.B(x).reshape(n, self.x_dim, self.action_dim)
        Bbot = self.Bbot(x).reshape(n, self.x_dim, (self.x_dim - self.action_dim))

        return f, B, Bbot
