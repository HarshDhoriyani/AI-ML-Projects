"""
model.py
--------
Defines the Q-Network: a simple feed-forward MLP that maps a state
(8-dim vector for LunarLander) to Q-values for each of the 4 discrete
actions.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Feed-forward network approximating Q(s, a) for all actions a."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(128, 128)):
        super().__init__()

        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (batch, state_dim) -> returns (batch, action_dim) Q-values."""
        return self.net(state)
