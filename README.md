
# 🕹️ Deep Reinforcement Learning – Atari Breakout

![Breakout Gameplay](./assets/atari_main.gif)

This project implements and compares various Deep Reinforcement Learning (DRL) algorithms to train an agent to play the classic Atari game **Breakout**, based on the paper [*Human-level control through deep reinforcement learning*](https://www.nature.com/articles/nature14236).

The project is a part of WPI's *Deep Learning for Advanced Analytics (DS 551)* course and explores how different DQN-based models perform in a high-dimensional, pixel-based environment.

## 📌 What Was Done

- Implemented **Deep Q-Network (DQN)** with:
  - Experience Replay
  - Frame stacking
  - Epsilon-greedy exploration
- Extended to advanced variants:
  - **Double DQN**
  - **Dueling DQN**
- Compared performance of each model over training episodes
- Visualized agent's gameplay and Q-value estimation
- Achieved stable training and improved scores through hyperparameter tuning and architecture enhancements

## 🧠 Tech Stack

- Python
- PyTorch
- OpenAI Gym
- NumPy
- Matplotlib

## 🚀 Running Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/harshv01/RL_atari_breakout.git
cd RL_atari_breakout
```

### 2. Create and Activate Virtual Environment (Optional but Recommended)
```bash
python3 -m venv venv
source venv/bin/activate    # For Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Atari Dependencies
```bash
pip install gym[atari,accept-rom-license]
```

### 5. Train the Agent
Choose your algorithm and run:

```bash
# For standard DQN
python dqn.py

# For Double DQN
python double_dqn.py

# For Dueling DQN
python dueling_dqn.py
```

### 6. Watch the Trained Agent
Use the playback script (if implemented), or modify your training script to render the environment during evaluation.

---

## 📈 Results

- Dueling DQN showed more stable and higher performance compared to standard DQN
- Plots for rewards per episode and loss curves are available in the `results/` directory

## 📂 Directory Structure

```
RL_atari_breakout/
├── dqn.py
├── double_dqn.py
├── dueling_dqn.py
├── models/
├── utils/
├── results/
└── README.md
```

## 🧠 References

- [DeepMind Nature paper (2015)](https://www.nature.com/articles/nature14236)
- [WPI DS551 Project Description](https://github.com/UrbanIntelligence/WPI-DS551-Fall24/tree/main/Project3)
