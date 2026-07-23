"""
agent.py
--------
DQNAgent wraps:
  - the online Q-network (trained every step)
  - a target Q-network (updated slowly, stabilizes learning targets)
  - epsilon-greedy action selection
  - the learning step (Bellman update)
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim, action_dim, config, device):
        self.action_dim = action_dim
        self.device = device
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.batch_size = config["batch_size"]
        self.min_buffer_before_train = config["min_buffer_before_train"]

        self.epsilon = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]

        hidden_sizes = config["hidden_sizes"]

        self.q_network = QNetwork(state_dim, action_dim, hidden_sizes).to(device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_sizes).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config["learning_rate"])
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss, more robust than MSE

        self.replay_buffer = ReplayBuffer(config["buffer_size"], seed=config["seed"])
        self.learn_step_counter = 0
        self.target_update_every = config["target_update_every"]

        self.rng = random.Random(config["seed"])

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Epsilon-greedy action selection. `evaluate=True` disables exploration."""
        if (not evaluate) and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_dim)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def can_learn(self) -> bool:
        return len(self.replay_buffer) >= max(self.batch_size, self.min_buffer_before_train)

    def learn(self):
        """One gradient step on a random minibatch from the replay buffer."""
        if not self.can_learn():
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # Current Q estimate for the action actually taken
        q_values = self.q_network(states).gather(1, actions)

        # Target: r + gamma * max_a' Q_target(s', a')   (0 if terminal)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_every == 0:
            self._soft_update_target()

        return loss.item()

    def _soft_update_target(self):
        """Polyak averaging: target = tau*online + (1-tau)*target."""
        for target_param, online_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
