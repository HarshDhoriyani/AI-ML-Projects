"""
evaluate.py
-----------
Load a trained checkpoint and run/render it in the environment to watch
the lander land (or crash).

Usage:
    python src/evaluate.py --model models/dqn_lunarlander.pth --episodes 5 --render
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import DQNAgent
from utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN LunarLander agent")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="models/dqn_lunarlander.pth")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="Open a window and render gameplay")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    render_mode = "human" if args.render else None
    env = gym.make(config["env_name"], render_mode=render_mode)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    agent = DQNAgent(state_dim, action_dim, config, device)
    agent.load(args.model)

    scores = []
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)  # no exploration
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        scores.append(total_reward)
        print(f"Episode {episode}: reward = {total_reward:.2f}")

    print(f"\nAverage reward over {args.episodes} episodes: {np.mean(scores):.2f}")
    env.close()


if __name__ == "__main__":
    main()
