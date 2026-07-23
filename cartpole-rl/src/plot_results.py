"""
plot_results.py
---------------
Plots the per-episode scores saved by train.py.

Usage:
    python src/plot_results.py --results results/scores.csv --out results/training_curve.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def plot_scores(results_path: str, out_path: str, target_score: float = 500.0):
    scores = np.loadtxt(results_path, delimiter=",", skiprows=1)
    scores = np.atleast_1d(scores)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(scores) + 1), scores, marker="o", label="Episode score")
    plt.axhline(y=target_score, color="r", linestyle="--", label=f"Target ({int(target_score)})")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("CartPole-v1 Training Progress (Hill Climbing)")
    plt.xticks(range(1, len(scores) + 1))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training scores")
    parser.add_argument("--results", type=str, default="results/scores.csv")
    parser.add_argument("--out", type=str, default="results/training_curve.png")
    parser.add_argument("--target-score", type=float, default=500.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_scores(args.results, args.out, args.target_score)
