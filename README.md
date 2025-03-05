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

## Requirements

- Python 3.8+
- PyTorch 1.9+
- gym-super-mario-bros
- OpenAI Gym
- NumPy
- Matplotlib (for visualization)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/username/super-mario-rl.git
cd super-mario-rl
onda env create -f environment.yaml
```

## Project Structure

```
super-mario-rl/
├── agent.py          # DQN agent implementation
├── models.py         # Neural network architectures (standard and ResNet-DQN)
├── env.py            # Environment wrappers and preprocessing
├── main.py           # Training and evaluation script
├── record.py         # Utilities for recording agent gameplay
├── utils.py          # Helper functions
└── configs/          # Configuration files for different experiments
```

## Usage

### Training

To train the agent using the ResNet-enhanced DQN architecture:

```bash
python main.py --mode train --model resnet --level 1-1 --episodes 50
```

Options:
- `--model`: Choose between 'standard' DQN and 'resnet' DQN
- `--level`: Specify the Super Mario Bros level (e.g., '1-1', '1-2')
- `--episodes`: Number of training episodes
- `--lr`: Learning rate
- `--batch_size`: Batch size for training
- `--replay_size`: Size of the replay buffer
- `--target_update`: Frequency of target network updates

### Evaluation

To evaluate a trained agent:

```bash
python main.py --mode eval --model resnet --level 1-1 --checkpoint path/to/model.pt
```

### Recording Gameplay

To record the agent's gameplay:

```bash
python record.py --model path/to/model.pt --level 1-1 --output mario_gameplay.mp4
```

## Model Architecture

The ResNet-enhanced DQN architecture consists of:

1. An initial convolutional layer (32 filters, 8×8 kernel, stride 4)
2. Two residual blocks with skip connections
3. Max pooling layers after each residual block
4. A fully-connected layer with 512 units
5. An output layer with 5 units (one per action)

The residual blocks improve gradient flow during training, addressing the vanishing gradient problem in deep networks.

## Experimental Results

The ResNet-enhanced DQN demonstrates improved performance compared to the standard DQN architecture in several metrics:

- Higher average reward
- Better completion rate
- More stable training
- Improved transfer learning to unseen levels

For detailed results, see the `results/` directory.

## Future Work

- Hyperparameter optimization using Bayesian methods
- Exploration of alternative network architectures
- Implementation of additional DQN improvements (Dueling DQN, Prioritized Experience Replay)
- Curriculum learning approaches for more efficient training
- Multi-objective reinforcement learning for optimizing multiple game aspects

## Citation

If you use this code in your research, please cite:

```
@misc{super-mario-rl,
  author = {Your Name},
  title = {Super Mario Bros Reinforcement Learning with ResNet-DQN},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/username/super-mario-rl}}
}
```

## License

MIT License

## Acknowledgments

- The implementation builds upon the gym-super-mario-bros environment by Christian Kauten
- The DQN algorithm is based on the original paper by Mnih et al. (2015)
- ResNet architecture is inspired by He et al. (2015)
