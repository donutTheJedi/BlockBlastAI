# Block Blast AI

A reinforcement learning agent that learns to play Block Blast, a Tetris-like puzzle game where you place pieces on an 8×8 board to clear rows and columns.

## Overview

This project implements a Deep Q-Network (DQN) from scratch using only NumPy to train an AI agent to play Block Blast. The agent learns to strategically place pieces to maximize line clears while minimizing holes on the board.

## Features

- **Custom Neural Network**: 4-layer fully connected network implemented in pure NumPy with ReLU activations
- **DQN with Target Network**: Stabilized training using a separate target network updated periodically
- **Experience Replay**: Stores transitions in a replay buffer for more efficient learning
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation with decaying epsilon
- **Reward Shaping**: Custom reward function that encourages line clears and penalizes holes
- **Pygame Visualization**: Interactive game interface and replay functionality for watching trained agents

## How It Works

### State Representation
The agent encodes the game state as a 128-dimensional vector:
- 64 values for the board state (flattened 8×8 grid)
- 64 values for the action mask (where the piece would be placed)

### Reward Structure
- Line clears: exponential bonus `(2^cleared - 1) * 10`
- Hole creation: -4 per new hole
- Hole removal: +2 per filled hole
- Adjacency bonus: +0.5 per touching block
- Board density penalty: discourages cluttered boards
- Set completion bonus: +5 for placing all 3 pieces

### Training
The agent trains over thousands of episodes, gradually reducing exploration (epsilon decay from 1.0 → 0.07) while learning optimal piece placements through Q-learning updates.

## Results

After training, the agent achieves scores of 200-350 points per game, clearing 30-50+ moves before game over.

## Dependencies

```
numpy
pygame
numba
matplotlib
```

## Usage

Run the Jupyter notebook to:
1. Play the game manually (interactive Pygame window)
2. Train the DQN agent
3. Visualize training metrics (reward, steps, lines cleared, holes)
4. Replay the best AI runs

```python
# Train the agent
# (training loop runs automatically in the notebook)

# Find and replay best performance
best_trace = find_best_trace(nn, trials=100)
replay_trace(best_trace, delay=0.7)
```

## Project Structure

- `BlockBlastAI.ipynb` - Main notebook containing:
  - Game environment (Pygame-based Block Blast)
  - Neural network implementation
  - DQN training loop
  - Visualization and replay utilities

## Future Improvements

- Implement Double DQN to reduce overestimation
- Add convolutional layers for spatial feature extraction
- Experiment with prioritized experience replay
- Save/load trained model weights