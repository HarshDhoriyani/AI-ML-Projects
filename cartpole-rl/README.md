# CartPole-v1 — Reinforcement Learning with Hill Climbing 🎯

Solve OpenAI Gymnasium's classic **CartPole-v1** environment: balance a
pole on a moving cart and reach the **maximum score of 500** in
**20 episodes or fewer**.

This project uses **Hill Climbing** with a simple linear softmax
policy. Because CartPole is a low-dimensional environment (4 state
values, 2 actions), a full Deep Q-Network is overkill — Hill Climbing
converges to a perfect score in a handful of episodes, comfortably
meeting the 20-episode target.

## How it works

1. The policy is a single 4×2 weight matrix mapping the observation
   (cart position, cart velocity, pole angle, pole angular velocity)
   to action probabilities via softmax.
2. Each episode, the agent plays with its current weights and
   observes the total reward.
3. **If the score improved (or matched the best so far):** keep the
   weights, shrink the exploration noise, and take a small step
   (steepest ascent).
4. **If the score got worse:** revert to the best known weights and
   widen the exploration noise (encourages a bigger jump next try).
5. Repeat for 20 episodes — the adaptive noise schedule lets the
   agent lock onto a great policy quickly.

## Project structure

```
cartpole-rl/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore
├── configs/
│   └── config.yaml            # Hyperparameters (episodes, seed, target score...)
├── src/
│   ├── __init__.py
│   ├── agent.py                # HillClimbingAgent: linear softmax policy
│   ├── train.py                 # Training loop (main entry point)
│   ├── evaluate.py              # Load a trained policy and watch it play
│   └── plot_results.py          # Plot the training score curve
├── models/                     # Saved weights land here (best_weights.npy)
├── results/                    # scores.csv + training_curve.png land here
└── .github/
    └── workflows/
        └── train.yml            # CI: runs training headlessly on every push
```

## Setup

```bash
git clone <your-repo-url>
cd cartpole-rl
pip install -r requirements.txt
```

## Usage

### Train the agent (20 episodes, target score 500)

```bash
python src/train.py --episodes 20 --seed 0
```

Example output:

```
Episode |  Score |   Best |    Noise
----------------------------------------
      1 |    152 |    152 |   0.0050
      2 |     66 |    152 |   0.0100
      ...
     14 |    500 |    500 |   0.0010
     15 |    500 |    500 |   0.0005
     ...
     20 |    500 |    500 |   0.0005

Target score of 500.0 reached at episode 14!
Best weights saved to:  models/best_weights.npy
Scores saved to:        results/scores.csv
```

> Note: Hill Climbing uses randomized exploration, so the exact
> episode where 500 is first reached can vary by seed — pass
> `--seed` to reproduce a specific run. Seed `0` reliably reaches 500
> well before episode 20.

### Watch the trained agent play

```bash
python src/evaluate.py --weights models/best_weights.npy --episodes 5 --render
```

### Plot the training curve

```bash
python src/plot_results.py --results results/scores.csv --out results/training_curve.png
```

## Configuration

Edit `configs/config.yaml` to change the number of episodes, seed,
target score, or output paths. (Command-line flags on `train.py`
override the config values if passed.)

## Requirements

- Python 3.9+
- gymnasium
- numpy
- matplotlib
- pyyaml

Install everything with:

```bash
pip install -r requirements.txt
```

## License

MIT — feel free to use and adapt this project.
