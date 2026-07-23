"""
evaluate.py
-----------
Loads a trained policy (weights saved by train.py) and runs it on
CartPole-v1, optionally rendering the environment so you can watch
the agent balance the pole.

Usage:
    python src/evaluate.py --episodes 5 --render
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent import HillClimbingAgent  # noqa: E402


def evaluate(weights_path: str, episodes: int = 5, render: bool = True):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = HillClimbingAgent(state_size, action_size)
    weights = np.load(weights_path)
    agent.set_weights(weights)

    scores = []
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        for _ in range(500):
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        scores.append(total_reward)
        print(f"Episode {episode}: Score = {total_reward}")

    env.close()
    print(f"\nAverage score over {episodes} episodes: {np.mean(scores):.2f}")
    return scores


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CartPole agent")
    parser.add_argument("--weights", type=str, default="models/best_weights.npy")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.weights, episodes=args.episodes, render=args.render)
