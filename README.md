
# 🕹️ Deep Reinforcement Learning – Atari Breakout

![Breakout Gameplay](./assets/atari_main.gif)

This project implements and compares various Deep Reinforcement Learning (DRL) algorithms to train an agent to play the classic Atari game **Breakout**, based on the paper [*Human-level control through deep reinforcement learning*](https://www.nature.com/articles/nature14236).

The project is a part of WPI's *Reinforcement Learning (CS 525)* course and explores how different DQN-based models perform in a high-dimensional, pixel-based environment.

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

```bash
python main.py --train_dqn
```

### 6. Watch the Trained Agent
Test by:
```bash
python main.py --test_dqn
```

Testing DQN while recording a video (recording video takes time, so usually you use this option when the number of testing episodes is small):

```bash
python main.py --test_dqn --record_video
```

## 📈 Results

- Dueling DQN showed more stable and higher performance compared to standard DQN
- Plots for rewards per episode and loss curves are available in the `results/` directory



## 🧠 References

- [DeepMind Nature paper (2015)](https://www.nature.com/articles/nature14236)
- [WPI CS525 Project Description](https://github.com/UrbanIntelligence/WPI-DS551-Fall24/tree/main/Project3)
