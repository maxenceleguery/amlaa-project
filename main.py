import warnings
warnings.filterwarnings('ignore')

import gym
import time
import matplotlib.pyplot as plt
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import torch
from tqdm import tqdm
import os

from models import DQNSolver, DQNSolverResNet
from agent import DQNAgent

from env import make_env

# ## Testing cell

"""
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = make_env(env)

c = 0
images = None
# run 1 episode
env.reset()
while True:
    action = env.action_space.sample()
    action = 0
    state, reward, done, _, info = env.step(action)
    if c==50:
        images = state
    #time.sleep(1/30)
    if done or info['time'] < 380:
        break
    c+=1
env.close()

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    axes[i].imshow(images[i], cmap='gray')
plt.show()
"""


def evaluate_agent(agent: DQNAgent, env: gym.Env, num_episodes=1, show: bool = False) -> float:
    """Runs the trained agent in the environment without training."""
    
    total_rewards = []
    
    for ep_num in range(num_episodes):
        state, info = env.reset()
        state = torch.Tensor(state[0].__array__() if isinstance(state, tuple) else state.__array__())
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state, evaluate=True)  # Use deterministic policy
            state_next, reward, done, _, info = env.step(int(action.item()))
            if show:
                time.sleep(1/30)
            done = done or info['time'] < 250
            
            total_reward += reward
            state = torch.Tensor(state_next[0].__array__() if isinstance(state_next, tuple) else state_next.__array__())
        
        total_rewards.append(total_reward)
        if show:
            print(f"Evaluation Episode {ep_num + 1}: Total Reward = {total_reward}")
    if show:
        print(f"Average Reward over {num_episodes} episodes: {np.mean(total_rewards)}")
    return np.mean(total_rewards)

def eval_all(agent: DQNAgent, levels = None, verbose: bool = True) -> float:
    rewards = []

    if levels is None:
        worlds = list(range(1, 9))
        stages = list(range(1, 5))
        levels = []
        for world in worlds:
            for stage in stages:
                levels.append(f"{world}-{stage}")


    for level in levels:
        env = gym_super_mario_bros.make(f'SuperMarioBros-{level}-v0', apply_api_compatibility=True, render_mode='rgb_array')
        env = make_env(env)
        rewards.append(evaluate_agent(agent, env))
        env.close()
        if verbose:
            print("Stage {}: Reward = {}".format(level, rewards[-1]))

    return np.mean(rewards)

import matplotlib.pyplot as plt

def train(agent: DQNAgent, env: gym.Env, num_episodes: int = 10, eval_step: int = 10, levels = None) -> DQNAgent:    
    total_rewards = [0]
    eval_rewards = [0]
    eval_ep = []
    
    for ep_num in (pbar := tqdm(range(num_episodes))):
        state, info = env.reset()

        # State is a LazyFrame
        state = torch.Tensor(state[0].__array__() if isinstance(state, tuple) else state.__array__())

        total_reward = 0
        while True:
            pbar.set_postfix_str(f"Step {agent.step}, Train Reward {total_rewards[-1]}, Eval Reward {eval_rewards[-1]}")

            action = agent.act(state, sample=True)
            
            state_next, reward, terminal, trunc, info = env.step(int(action.item()))
            total_reward += reward
            
            state_next = torch.Tensor(state_next[0].__array__() if isinstance(state_next, tuple) else state_next.__array__())
            reward = torch.tensor([reward])#.unsqueeze(0)
            
            terminal = torch.tensor([int(terminal)])#.unsqueeze(0)
            agent.remember(state, action, reward, state_next, terminal)
            agent.experience_replay(num_replay=10)
            
            state = state_next
            if terminal:
                break

        agent.scheduler.step()
        
        total_rewards.append(total_reward)
        agent.save('mario_model.pth')

        if ep_num % eval_step == 0:
            eval_rewards.append(eval_all(agent, levels=levels, verbose=False))
            eval_ep.append(ep_num)

    return agent, eval_ep, eval_rewards[1:], total_rewards

if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0', apply_api_compatibility=True, render_mode='rgb_array')
    env = make_env(env)#, 'training_videos')
    observation_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = DQNAgent(state_space=observation_space, action_space=action_space, lr=1e-4, batch_size=128, exploration_decay=0.99 , model=DQNSolverResNet)
    env.close()

    if os.path.exists('mario_model.pth'):
        agent.load('mario_model.pth')

    levels = ['1-1']#, '1-2', '1-3', '1-4']
    env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0', stages=levels, apply_api_compatibility=True, render_mode='rgb_array')
    #env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='rgb_array')
    env = make_env(env)#, 'training_videos')

    print(f"Start training on {agent.device}")
    agent, eval_ep, eval_rewards, total_rewards = train(agent, env, num_episodes=3, levels=levels)
    env.close()
    agent.save('mario_model.pth')

    os.makedirs("./plots", exist_ok=True)
    os.makedirs("./training", exist_ok=True)

    plt.plot(total_rewards, label="Training Reward")
    plt.plot(eval_ep, eval_rewards, label="Evaluation Reward on the world 1")
    plt.title("Mario Total Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig(f"./plots/training-{time.ctime()}.png")
    agent.save(f'./training/mario_model-{time.ctime()}.pth')
    plt.show()

    if os.path.exists('mario_model.pth'):
        agent.load('mario_model.pth')

    mean_reward = eval_all(agent)
    print(f"Average Reward over all stages: {np.mean(mean_reward)}")

    exit(0)

    env = gym_super_mario_bros.make(f'SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='human')
    env = make_env(env)
    evaluate_agent(agent, env, show=True)
    env.close()

    #from record import record_mario_gameplay
    # This will play and record Mario with random actions\n"
    #gif_path = record_mario_gameplay()
    #print(f"GIF saved to: {gif_path}")


