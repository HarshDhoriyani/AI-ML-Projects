"""
train.py
--------
Trains a HillClimbingAgent on Gymnasium's CartPole-v1 environment.

Goal: reach the maximum score of 500 within 20 episodes.

Usage:
    python src/train.py
    python src/train.py --episodes 20 --seed 0 --render
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent import HillClimbingAgent  # noqa: E402


def run_episode(env, agent, max_t: int = 500, render: bool = False) -> float:
    """Run a single episode and return the total reward."""
    state, _ = env.reset()
    total_reward = 0.0
    for _ in range(max_t):
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if render:
            env.render()
        if terminated or truncated:
            break
    return total_reward


def train(
    episodes: int = 20,
    target_score: float = 500.0,
    seed: int = 0,
    noise_scale_init: float = 1e-2,
    model_out: str = "models/best_weights.npy",
    results_out: str = "results/scores.csv",
    render: bool = False,
):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = HillClimbingAgent(state_size, action_size, seed=seed)

    best_reward = -np.inf
    best_weights = agent.get_weights()
    noise_scale = noise_scale_init
    scores = []

    print(f"{'Episode':>7} | {'Score':>6} | {'Best':>6} | {'Noise':>8}")
    print("-" * 40)

    for episode in range(1, episodes + 1):
        reward = run_episode(env, agent, render=render)
        scores.append(reward)

        if reward >= best_reward:
            # Improvement (or tie): keep weights, shrink noise, explore locally
            best_reward = reward
            best_weights = agent.get_weights()
            noise_scale = max(1e-3, noise_scale / 2)
            agent.perturb(noise_scale)
        else:
            # No improvement: revert to best weights, widen noise
            agent.set_weights(best_weights)
            noise_scale = min(2.0, noise_scale * 2)
            agent.perturb(noise_scale)

        print(f"{episode:>7} | {reward:>6.0f} | {best_reward:>6.0f} | {noise_scale:>8.4f}")

        if best_reward >= target_score:
            print(f"\nTarget score of {target_score} reached at episode {episode}!")

    env.close()

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    os.makedirs(os.path.dirname(results_out), exist_ok=True)
    np.save(model_out, best_weights)
    np.savetxt(results_out, scores, delimiter=",", header="score", comments="")

    print(f"\nBest weights saved to:  {model_out}")
    print(f"Scores saved to:        {results_out}")
    print(f"Best score achieved:    {best_reward}")

    return scores, best_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Hill Climbing agent on CartPole-v1")
    parser.add_argument("--episodes", type=int, default=20, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--target-score", type=float, default=500.0, help="Score to reach")
    parser.add_argument("--render", action="store_true", help="Render the environment while training")
    parser.add_argument("--model-out", type=str, default="models/best_weights.npy")
    parser.add_argument("--results-out", type=str, default="results/scores.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        episodes=args.episodes,
        target_score=args.target_score,
        seed=args.seed,
        model_out=args.model_out,
        results_out=args.results_out,
        render=args.render,
    )
