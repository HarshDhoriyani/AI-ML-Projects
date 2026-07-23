"""
utils.py
--------
Small shared helpers: config loading, global seeding, reward plotting.
"""

import os
import random

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_rewards(rewards, rolling_window: int, save_path: str, solve_score: float = 200.0):
    """Plots per-episode reward + rolling average, saves to disk."""
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rewards = np.array(rewards)
    rolling_avg = np.array(
        [
            rewards[max(0, i - rolling_window + 1): i + 1].mean()
            for i in range(len(rewards))
        ]
    )

    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label="Episode reward")
    plt.plot(rolling_avg, linewidth=2, label=f"Rolling avg ({rolling_window} ep)")
    plt.axhline(y=solve_score, color="green", linestyle="--", label=f"Solve threshold ({solve_score})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("LunarLander-v3 DQN Training Progress")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
