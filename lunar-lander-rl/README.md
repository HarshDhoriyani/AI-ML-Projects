# 🚀 Lunar Lander RL — Deep Q-Network (DQN)

A Reinforcement Learning project that trains an agent to safely land a lunar
module using **Gymnasium's `LunarLander-v3`** environment and a **Deep
Q-Network (DQN)** built in PyTorch.

The agent learns to fire its main/side engines to land between the flags
without crashing, while minimizing fuel usage (engine firing gives a small
negative reward, so the agent is naturally pushed toward efficiency).

---

## ⚠️ Important note on "20 episodes → 200 points"

The environment is considered **"solved"** when the agent gets an
**average reward ≥ 200 over 100 consecutive episodes**. This is the
standard benchmark used by OpenAI/Farama themselves.

In practice, with vanilla DQN this takes roughly **300–800+ episodes**,
depending on hyperparameters and luck (random seed). Getting there in
**20 episodes is not realistic** for any algorithm training from
scratch — the agent hasn't even filled its replay buffer by then.

So this project is set up to:
- Run for a **configurable number of episodes** (default: 1000) so it
  actually converges.
- Log a running 100-episode average so you can see real progress.
- Let you demo a **quick 20-episode smoke test** (`--episodes 20`) just
  to confirm the pipeline runs end-to-end — but don't expect 200 points
  from that; expect around -200 to -50 (still crashing/learning).

If your assignment strictly requires "solved in 20 episodes," mention
this constraint to your instructor — it isn't achievable with standard
model-free RL on this environment. What you *can* honestly claim is:
"the agent solves LunarLander-v3 (avg reward ≥200 over 100 episodes)
within N training episodes," and report the real N.

---

## 📁 File Structure

```
lunar-lander-rl/
├── README.md                # This file
├── requirements.txt          # Python dependencies
├── .gitignore
├── configs/
│   └── config.yaml           # All hyperparameters in one place
├── src/
│   ├── __init__.py
│   ├── model.py               # QNetwork (the neural net)
│   ├── replay_buffer.py       # Experience replay memory
│   ├── agent.py                # DQNAgent (action selection, learning step)
│   ├── train.py                 # Training loop / entry point
│   ├── evaluate.py             # Load a saved model and watch it play
│   └── utils.py                # Seeding, plotting, config loading
├── models/                    # Saved model checkpoints (.pth)
│   └── .gitkeep
└── results/                   # Reward plots / training logs
    └── .gitkeep
```

---

## 🔧 Setup

```bash
git clone <your-repo-url>
cd lunar-lander-rl
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Gymnasium's Box2D environments need `swig` installed on some systems:
> `pip install swig` before installing `gymnasium[box2d]` if you hit a
> build error.

---

## ▶️ Usage

**Train (recommended — full run until solved or max episodes):**
```bash
python src/train.py --episodes 1000
```

**Quick smoke test (pipeline check only, NOT expected to solve):**
```bash
python src/train.py --episodes 20 --no-render
```

**Watch the trained agent land:**
```bash
python src/evaluate.py --model models/dqn_lunarlander.pth --episodes 5 --render
```

Training automatically stops early once the 100-episode rolling average
reaches 200, and saves the model + a reward plot to `results/`.

---

## 🧠 Algorithm summary

- **Algorithm:** DQN with experience replay + a separate target network
  (soft/periodic updates) for stability.
- **State:** 8-dim continuous vector (position, velocity, angle, angular
  velocity, leg contacts).
- **Action space:** 4 discrete actions (do nothing, fire left, fire main,
  fire right engine).
- **Reward shaping (built into the env):** landing safely/on pad = big
  positive reward; crashing = -100; firing main engine = -0.3/step;
  firing side engine = -0.03/step (this is what drives fuel efficiency);
  leg ground contact = +10 each.
- **Exploration:** epsilon-greedy, decaying from 1.0 → 0.01.

---

## 📊 Output

After training you'll get:
- `models/dqn_lunarlander.pth` — saved network weights
- `results/training_rewards.png` — reward-per-episode + rolling average plot
- Console logs every episode with score and rolling average
