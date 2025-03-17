import gym_super_mario_bros
import os
import numpy as np

from models import DQNSolverResNet
from env import make_env
from agent import DQNAgent
from main import eval_all

env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0', apply_api_compatibility=True, render_mode='rgb_array')
env = make_env(env)#, 'training_videos')
observation_space = env.observation_space.shape
action_space = env.action_space.n
agent = DQNAgent(state_space=observation_space, action_space=action_space, lr=1e-4, batch_size=128, exploration_decay=0.999, model=DQNSolverResNet)
env.close()

if os.path.exists('mario_model.pth'):
    agent.load('mario_model.pth')

mean_reward = eval_all(agent)
print(f"Average Reward over all stages: {np.mean(mean_reward)}")