"""
replay_buffer.py
----------------
Fixed-size experience replay buffer that stores (state, action, reward,
next_state, done) transitions and supports random-batch sampling, which
breaks correlation between consecutive samples and stabilizes DQN training.
"""

import random
from collections import deque, namedtuple

import numpy as np
import torch

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        batch = self.rng.sample(self.buffer, batch_size)

        states = torch.as_tensor(
            np.array([t.state for t in batch]), dtype=torch.float32, device=device
        )
        actions = torch.as_tensor(
            np.array([t.action for t in batch]), dtype=torch.int64, device=device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            np.array([t.reward for t in batch]), dtype=torch.float32, device=device
        ).unsqueeze(1)
        next_states = torch.as_tensor(
            np.array([t.next_state for t in batch]), dtype=torch.float32, device=device
        )
        dones = torch.as_tensor(
            np.array([t.done for t in batch]), dtype=torch.float32, device=device
        ).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
