# Super Mario Bros Reinforcement Learning with ResNet-DQN

This repository contains an implementation of a Deep Q-Network (DQN) agent enhanced with residual connections for playing Super Mario Bros. The agent is designed to learn directly from pixel inputs, using deep convolutional neural networks with residual connections to improve gradient flow and representation learning.

## Overview

The project explores the application of deep reinforcement learning techniques to the challenging domain of Super Mario Bros, with a focus on improving the stability and performance of DQN through architectural innovations. Specifically, the implementation incorporates ResNet-style skip connections to enhance feature extraction from high-dimensional visual inputs.

## Features

- **ResNet-Enhanced DQN**: Implementation of a DQN agent with residual connections for improved gradient flow
- **Environment Wrappers**: Custom wrappers for the `gym-super-mario-bros` environment with frame preprocessing
- **Experience Replay**: Efficient storage and sampling of agent experiences
- **Training Visualization**: Tools for visualizing agent performance and learning progress
- **Model Checkpointing**: Save and load agent models to continue training or evaluation

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/maxenceleguery/amlaa-project.git
cd amlaa-project
conda env create -f environment.yaml
```

## How to use ?

- To run our DQN implementation, use the scripts main.py and eval.py.
- To run Stable Baselines 3 PPO implementation, use the scripts with "sbl" in their names.