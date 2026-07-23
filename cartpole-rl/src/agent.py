"""
agent.py
--------
A linear-policy agent trained with the Hill Climbing algorithm
(steepest ascent + adaptive Gaussian noise).

CartPole-v1's observation space is 4 continuous values and the action
space is 2 discrete actions. A single 4x2 weight matrix is enough to
solve the environment (max score = 500) in a handful of episodes,
which is why this simple approach is used instead of a full Deep
Q-Network -- it converges far faster for this environment.
"""

import numpy as np


class HillClimbingAgent:
    """Linear softmax policy optimized with Hill Climbing."""

    def __init__(self, state_size: int, action_size: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.w = 1e-4 * self.rng.random((state_size, action_size))

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Return action probabilities for a given state."""
        x = np.dot(state, self.w)
        exp_x = np.exp(x - np.max(x))  # numerically stable softmax
        return exp_x / np.sum(exp_x)

    def act(self, state: np.ndarray) -> int:
        """Return the greedy (best) action for a given state."""
        probs = self.forward(state)
        return int(np.argmax(probs))

    def get_weights(self) -> np.ndarray:
        return self.w.copy()

    def set_weights(self, w: np.ndarray) -> None:
        self.w = w.copy()

    def perturb(self, noise_scale: float) -> None:
        """Add random noise to the current weights (exploration step)."""
        self.w = self.w + noise_scale * self.rng.random((self.state_size, self.action_size))
