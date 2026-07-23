"""
train.py
--------
Entry point for training the DQN agent on LunarLander-v3.

Usage:
    python src/train.py                          # use configs/config.yaml defaults
    python src/train.py --episodes 1000
    python src/train.py --episodes 20 --no-render   # quick smoke test only
"""

import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import DQNAgent
from utils import load_config, set_global_seed, plot_rewards


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent on LunarLander-v3")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--episodes", type=int, default=None, help="Override config episode count")
    parser.add_argument("--no-render", action="store_true", help="Force no rendering (default anyway)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.episodes is not None:
        config["episodes"] = args.episodes

    set_global_seed(config["seed"])

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = gym.make(config["env_name"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, config, device)

    all_rewards = []
    solve_window = config["solve_window"]
    solve_score = config["solve_score"]

    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["plot_path"]), exist_ok=True)

    start_time = time.time()

    for episode in range(1, config["episodes"] + 1):
        state, _ = env.reset(seed=config["seed"] + episode)
        episode_reward = 0.0

        for step in range(config["max_steps_per_episode"]):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            episode_reward += reward

            if done:
                break

        agent.decay_epsilon()
        all_rewards.append(episode_reward)

        rolling_avg = np.mean(all_rewards[-solve_window:])

        print(
            f"Episode {episode:4d} | Reward: {episode_reward:8.2f} | "
            f"Rolling Avg({min(len(all_rewards), solve_window)}): {rolling_avg:8.2f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

        # Early stop once solved (needs full window of data)
        if len(all_rewards) >= solve_window and rolling_avg >= solve_score:
            elapsed = time.time() - start_time
            print(
                f"\n✅ Solved in {episode} episodes! "
                f"Rolling avg {rolling_avg:.2f} >= {solve_score}. "
                f"Elapsed: {elapsed / 60:.1f} min."
            )
            break
    else:
        print(
            f"\n⚠️ Did not reach solve threshold within {config['episodes']} episodes. "
            f"Best rolling avg: {np.max([np.mean(all_rewards[max(0,i-solve_window+1):i+1]) for i in range(len(all_rewards))]):.2f}. "
            "Try more episodes or tune hyperparameters in configs/config.yaml."
        )

    agent.save(config["save_path"])
    print(f"Model saved to {config['save_path']}")

    plot_rewards(all_rewards, solve_window, config["plot_path"], solve_score)
    print(f"Training plot saved to {config['plot_path']}")

    env.close()


if __name__ == "__main__":
    main()
