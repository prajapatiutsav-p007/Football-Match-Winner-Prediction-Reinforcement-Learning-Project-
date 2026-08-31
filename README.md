# ⚽ Football Match Winner Prediction — Reinforcement Learning Project

A **Q-learning based Reinforcement Learning agent** that learns, purely
through trial-and-reward feedback, to predict football match outcomes
(Home Win / Draw / Away Win) — trained on **6,169 real international
football matches (2020–2026)**.

> **PBL Category:** Complex Problem-Solving / Mini Project
> **SDG Alignment:** SDG 9 — Industry, Innovation and Infrastructure
> (application of AI-based decision-support systems to real-world,
> data-driven prediction problems), and SDG 3 — Good Health and
> Well-Being (sports analytics supports informed, safer decision-making
> around athlete/team performance evaluation).

---

## 🧠 Why This Is Reinforcement Learning (not plain classification)

| RL Concept | In this project |
|---|---|
| **Agent** | The Q-learning predictor |
| **Environment** | A stream of real historical football matches |
| **State** | Match situation, discretized from 4 features: Elo rating gap, recent form gap, head-to-head win rate, neutral-venue flag |
| **Action** | Predict `Home Win`, `Draw`, or `Away Win` |
| **Reward** | `+1` if prediction is correct, `-1` if wrong |
| **Policy** | **Epsilon-greedy** — starts by exploring randomly (epsilon=1.0) and gradually shifts to exploiting its learned Q-table (epsilon=0.05) |
| **Learning rule** | Classic Q-learning update: `Q(s,a) ← Q(s,a) + α(reward − Q(s,a))` |

The agent starts with **zero knowledge** — its Q-table is all zeros. It has
no idea what "Home Win" even means. It only learns which action earns the
most reward in each state by playing through thousands of real matches,
exactly like an RL agent learning to play a game through experience.

---

## 📁 Project Structure
```
football_rl_project/
├── src/
│   ├── config.py                 # central path config (portable — no hardcoded paths)
│   ├── 01_get_dataset.py         # downloads real match data
│   ├── 02_feature_engineering.py # builds Elo/form/head-to-head features
│   ├── 03_train_rl_agent.py      # trains the Q-learning agent
│   └── 04_visualize_results.py   # generates the results charts
├── data/
│   ├── matches_raw.csv           # raw real match data (6,169 rows)
│   ├── matches_features.csv      # RL-ready features
│   ├── rl_training_results.csv   # reward log, step by step
│   ├── summary.csv               # final accuracy numbers
│   └── q_table.npy               # the trained Q-table
├── charts/
│   └── rl_training_overview.png  # learning curve, epsilon decay, accuracy, result distribution
└── README.md
```

## ▶️ How to Run (VS Code / terminal)
Open the `football_rl_project` folder in VS Code (not a subfolder) and run
each script from inside `src/` in order — every script resolves its own
paths relative to the project root via `config.py`, so it works regardless
of which folder you launch it from:

```bash
cd src
pip install pandas numpy matplotlib seaborn
python 01_get_dataset.py          # step 1: get real data (needs internet)
python 02_feature_engineering.py  # step 2: build features
python 03_train_rl_agent.py       # step 3: train the RL agent
python 04_visualize_results.py    # step 4: generate charts
```
If you don't have internet access when running step 1, `data/matches_raw.csv`
is already included in this project, so you can start directly from step 2.

## 📊 Dataset
- **Source:** [martj42/international_results](https://github.com/martj42/international_results)
  on GitHub — the same continuously-updated data that powers the popular
  Kaggle dataset *"International football results from 1872 to 2017"*.
- **6,169 real matches**, played 2020 onward, across all FIFA member nations.
- Result split: **Home 47.6% / Away 29.4% / Draw 23.0%** — realistic, matches
  known home-advantage patterns in international football.

## 🔧 Features (State Representation)
| Feature | Meaning |
|---|---|
| `elo_diff` | Elo rating gap between home and away team (updates after every match) |
| `form_diff` | Difference in points won over each team's last 5 matches |
| `h2h_home_rate` | Historical head-to-head win rate for the home team |
| `neutral_venue` | Whether the match was played at a neutral venue (common in tournaments) |

Each continuous feature is discretized into 5 quantile buckets so the
Q-table has a finite, manageable state space (~1,250 states × 3 actions).

## 📈 Results
| | Accuracy |
|---|---|
| **RL Agent (Q-learning)** | **58.0%** |
| Baseline (always predict majority class) | 48.1% |

- The agent is strong at predicting **Home Wins (77.6% recall)** and
  **Away Wins (66.1% recall)**.
- Like almost every football-prediction system (including professional
  ones), it **struggles with Draws (5.1% recall)** — draws are the
  hardest outcome to forecast in football because they arise from close,
  low-signal matches.
- The learning curve (see `charts/rl_training_overview.png`) shows the
  agent's rolling-average reward climbing from negative (mostly wrong,
  during exploration) to consistently positive (mostly correct) as
  epsilon decays and the Q-table converges.

# Football-Match-Winner-Prediction-Reinforcement-Learning-Project-
