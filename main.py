import warnings
warnings.filterwarnings('ignore')

import argparse
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import logging
import wandb
from gym.wrappers import TimeLimit
import gym_super_mario_bros

from env import make_env
from agent import DQNAgent, PolicyGradientAgent
from models import DQNSolverResNet
from record import save_video

logging.basicConfig(level=logging.INFO)

def evaluate_agent(agent, env, num_episodes=5, max_steps=4000, show=False):
    total_rewards = []
    for _ in range(num_episodes):
        state, info = env.reset()
        if isinstance(agent, PolicyGradientAgent):
            state = torch.tensor(state.__array__(), dtype=torch.float32, device=agent.device).unsqueeze(0)
        else:
            state = torch.tensor(state.__array__(), dtype=torch.float32, device=agent.device)
        total_reward = 0
        for _ in range(max_steps):
            a = agent.act(state, evaluate=True)
            if isinstance(agent, PolicyGradientAgent):
                a, _ = a
            action = int(a.item()) if torch.is_tensor(a) else int(a)
            next_state, reward, done, trunc, info = env.step(action)
            total_reward += reward
            if show:
                time.sleep(1/30)
            if isinstance(agent, PolicyGradientAgent):
                state = torch.tensor(next_state.__array__(), dtype=torch.float32, device=agent.device).unsqueeze(0)
            else:
                state = torch.tensor(next_state.__array__(), dtype=torch.float32, device=agent.device)
            if done or trunc:
                break
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

def eval_all(agent, levels=None, verbose=True):
    if levels is None:
        worlds = range(1, 9)
        stages = range(1, 5)
        levels = [f"{w}-{s}" for w in worlds for s in stages]
    rewards = []
    for lvl in levels:
        env = gym_super_mario_bros.make(f"SuperMarioBros-{lvl}-v0", apply_api_compatibility=True, render_mode='rgb_array')
        env = TimeLimit(env, max_episode_steps=4000)
        env = make_env(env)
        r = evaluate_agent(agent, env, max_steps=4000, show=False)
        if verbose:
            print(f"Stage {lvl}: Reward = {r:.2f}")
        rewards.append(r)
        env.close()
    return np.mean(rewards)

def train(agent, env, num_episodes=10, eval_step=5, levels=None, max_steps=4000, use_wandb=False, num_replay=5, update_freq=300):
    total_rewards, eval_rewards, eval_ep = [], [], []
    for ep_num in range(num_episodes):
        state, info = env.reset()
        if isinstance(agent, PolicyGradientAgent):
            state_t = torch.tensor(state.__array__(), dtype=torch.float32, device=agent.device).unsqueeze(0)
        else:
            state_t = torch.tensor(state.__array__(), dtype=torch.float32, device=agent.device)
        total_reward = 0
        for _ in range(max_steps):
            if isinstance(agent, DQNAgent):
                action = agent.act(state_t, sample=True)
                next_state, reward, done, trunc, info = env.step(int(action.item()))
                next_state_t = torch.tensor(next_state.__array__(), dtype=torch.float32, device=agent.device)
                agent.remember(state_t, action, torch.tensor([reward], device=agent.device), next_state_t, torch.tensor([done or trunc], device=agent.device))
                agent.experience_replay(num_replay=num_replay)
                if agent.step % update_freq == 0:
                    agent.copy_model()
                state_t = next_state_t
            else:
                a, logprob = agent.act(state_t, evaluate=False)
                next_state, reward, done, trunc, info = env.step(a)
                agent.store_step(state_t, logprob, reward)
                next_state_t = torch.tensor(next_state.__array__(), dtype=torch.float32, device=agent.device).unsqueeze(0)
                state_t = next_state_t
            total_reward += reward
            if done or trunc:
                break
        if isinstance(agent, DQNAgent):
            agent.scheduler.step()
        else:
            agent.update_policy()
        if use_wandb:
            wandb.log({"training_reward": total_reward, "episode": ep_num})
        else:
            logging.info("Episode %d, training_reward: %f", ep_num, total_reward)
        total_rewards.append(total_reward)
        if ep_num % eval_step == 0 and ep_num > 0:
            eval_r = eval_all(agent, levels=levels, verbose=True)
            eval_rewards.append(eval_r)
            eval_ep.append(ep_num)
            if use_wandb:
                wandb.log({"eval_reward": eval_r, "episode": ep_num})
            else:
                logging.info("Episode %d, eval_reward: %f", ep_num, eval_r)
        if hasattr(agent, "save"):
            agent.save("mario_model.pth")
    return agent, eval_ep, eval_rewards, total_rewards

def main():
    parser = argparse.ArgumentParser(description="Mario RL Training (No HPO)")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--agent-type", choices=["dqn","pg"], default="dqn")
    parser.add_argument("--eval", action="store_true", help="Evaluation of one checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Loading a checkpoint")
    args = parser.parse_args()

    if args.use_wandb:
        run = wandb.init(project="MarioDQN-NoHPO", name="run_no_hpo", reinit=True)

    lr = 1e-4
    batch_size = 64
    exploration_decay = 0.995
    gamma = 0.95
    num_replay = 5
    target_update_freq = 500

    if args.use_wandb:
        wandb.config.update({
            "lr": lr,
            "batch_size": batch_size,
            "exploration_decay": exploration_decay,
            "gamma": gamma,
            "num_replay": num_replay,
            "target_update_freq": target_update_freq
        })
    else:
        logging.info("Hyperparameters: lr=%f, batch_size=%d, exploration_decay=%f, gamma=%f, num_replay=%d, target_update_freq=%d",
                     lr, batch_size, exploration_decay, gamma, num_replay, target_update_freq)

    levels = ['1-1']
    raw_env = gym_super_mario_bros.make(
        'SuperMarioBrosRandomStages-v0',
        stages=levels,
        apply_api_compatibility=True,
        render_mode='rgb_array'
    )
    raw_env = TimeLimit(raw_env, max_episode_steps=args.max_steps)
    env = make_env(raw_env)

    obs_shape = env.observation_space.shape
    act_space = env.action_space.n

    if args.agent_type == "dqn":
        from models import DQNSolver
        agent = DQNAgent(
            state_space=obs_shape,
            action_space=act_space,
            lr=lr,
            batch_size=batch_size,
            exploration_decay=exploration_decay,
            gamma=gamma,
            model=DQNSolverResNet
        )
    else:
        agent = PolicyGradientAgent(
            state_space=obs_shape,
            action_space=act_space,
            lr=lr,
            gamma=gamma
        )

    if args.checkpoint is not None:
        if not os.path.exists(args.checkpoint):
            raise ValueError(f"Checkpoint does not exist : {args.checkpoint}")
        agent.load(args.eval)

    if args.eval:
        mean_reward = eval_all(agent, levels=None, verbose=False)
        print(f"Average reward across all stages: {mean_reward:.2f}")
        exit(0)

    agent, eval_ep, eval_rewards, total_rewards = train(
        agent,
        env,
        num_episodes=args.episodes,
        eval_step=5,
        levels=levels,
        max_steps=args.max_steps,
        use_wandb=args.use_wandb,
        num_replay=num_replay
    )

    video_path = save_video(env, agent, video_dir_path='my_best_videos', max_steps=args.max_steps)
    if args.use_wandb:
        wandb.log({"final_video": wandb.Video(video_path)})
    else:
        logging.info("final_video: %s", video_path)

    env.close()
    if hasattr(agent, "save"):
        agent.save('best_mario_model.pth')

    plt.figure()
    plt.plot(total_rewards, label="Training Reward")
    plt.plot(eval_ep, eval_rewards, label="Evaluation Reward")
    plt.title("Total Rewards - Final Training (No HPO)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/training-{str(time.time()).replace(' ', '_')}.png")
    plt.show()

    mean_reward = eval_all(agent, levels=None, verbose=False)
    print(f"Average reward across all stages: {mean_reward:.2f}")
    if args.use_wandb:
        wandb.log({"final_mean_reward_all_stages": mean_reward})
        wandb.finish()
    else:
        logging.info("final_mean_reward_all_stages: %f", mean_reward)

if __name__ == "__main__":
    main()
