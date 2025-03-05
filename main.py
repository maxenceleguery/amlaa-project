import warnings
warnings.filterwarnings('ignore')

import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import optuna
import wandb
import torch

from gym.wrappers import TimeLimit
import gym_super_mario_bros

from env import make_env
from agent import DQNAgent
from models import DQNSolverResNet
from record import save_video


def evaluate_agent(agent, env, num_episodes=1, max_steps=4000, show=False):
    total_rewards = []
    for _ in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state.__array__(), dtype=torch.float, device=agent.device)
        total_reward = 0
        for step in range(max_steps):
            action = agent.act(state, evaluate=True)
            next_state, reward, done, trunc, info = env.step(int(action.item()))
            total_reward += reward
            if show:
                time.sleep(1/30)
            state = torch.tensor(next_state.__array__(), dtype=torch.float, device=agent.device)
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
        env = gym_super_mario_bros.make(f"SuperMarioBros-{lvl}-v0",
                                        apply_api_compatibility=True,
                                        render_mode='rgb_array')
        env = TimeLimit(env, max_episode_steps=4000)
        env = make_env(env)
        r = evaluate_agent(agent, env, max_steps=4000, show=False)
        if verbose:
            print(f"Stage {lvl}: Reward = {r:.2f}")
        rewards.append(r)
        env.close()
    return np.mean(rewards)

def train(agent, env, num_episodes=10, eval_step=5, levels=None, max_steps=4000):
    total_rewards, eval_rewards, eval_ep = [], [], []
    for ep_num in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state.__array__(), dtype=torch.float, device=agent.device)
        total_reward = 0
        for step in range(max_steps):
            action = agent.act(state, sample=True)
            next_state, reward, done, trunc, info = env.step(int(action.item()))
            next_state = torch.tensor(next_state.__array__(), dtype=torch.float, device=agent.device)

            agent.remember(state, action, reward, next_state, done or trunc)
            agent.experience_replay()

            # Mise à jour du réseau cible au besoin
            if agent.step % agent.target_update_freq == 0:
                agent.update_target_network()

            state = next_state
            total_reward += reward
            if done or trunc:
                break

        agent.scheduler.step()

        wandb.log({"training_reward": total_reward, "episode": ep_num})

        total_rewards.append(total_reward)

        if ep_num % eval_step == 0 and ep_num > 0:
            eval_r = eval_all(agent, levels=levels, verbose=False)
            eval_rewards.append(eval_r)
            eval_ep.append(ep_num)
            wandb.log({"eval_reward": eval_r, "episode": ep_num})

        agent.save("mario_model.pth")

    return agent, eval_ep, eval_rewards, total_rewards



def objective(trial):
    child_run = wandb.init(
        project="MarioDQN-Optuna",
        name=f"trial_{trial.number}",
        reinit=True
    )

    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    exploration_decay = trial.suggest_float('exploration_decay', 0.95, 0.9999, step=0.0005)
    gamma = trial.suggest_float('gamma', 0.90, 0.99, step=0.01)
    num_replay = trial.suggest_categorical('num_replay', [1, 5, 10, 20])
    target_update_freq = trial.suggest_categorical('target_update_freq', [250, 500, 1000, 2000])

    wandb.config.update({
        "trial_number": trial.number,
        "lr": lr,
        "batch_size": batch_size,
        "exploration_decay": exploration_decay,
        "gamma": gamma,
        "num_replay": num_replay,
        "target_update_freq": target_update_freq
    })

    levels = ['1-1']
    raw_env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0',
                                        stages=levels,
                                        apply_api_compatibility=True,
                                        render_mode='rgb_array')
    raw_env = TimeLimit(raw_env, max_episode_steps=2000)
    env = make_env(raw_env)

    obs_shape = env.observation_space.shape
    act_space = env.action_space.n

    agent = DQNAgent(
        state_space=obs_shape,
        action_space=act_space,
        lr=lr,
        batch_size=batch_size,
        exploration_decay=exploration_decay,
        gamma=gamma,
        num_replay=num_replay,
        target_update_freq=target_update_freq,
        model_class=DQNSolverResNet
    )

    agent, _, _, _ = train(agent, env, num_episodes=50, eval_step=5, levels=levels, max_steps=2000)

    mean_reward = eval_all(agent, levels=levels, verbose=False)

    video_path = save_video(env, agent, video_dir_path='trial_videos', max_steps=1000)
    wandb.log({"final_video": wandb.Video(video_path)})

    env.close()
    child_run.log({"final_mean_reward": mean_reward})
    child_run.finish()

    return mean_reward



def mother_callback(study, trial):

    wandb.log({
        "trial_number": trial.number,
        "final_eval_reward": trial.value
    })
    for k, v in trial.params.items():
        wandb.log({f"params/{k}": v})


if __name__ == "__main__":
    mother_run = wandb.init(
        project="MarioDQN-Optuna",
        name="mother_run",
        reinit=True
    )

    # Lancement de l'optimisation
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=5,     # ajustez le nombre de trials selon vos ressources
        n_jobs=1,
        callbacks=[mother_callback]
    )

    print("Meilleurs hyperparamètres:")
    print(study.best_params)
    wandb.log({"best_reward": study.best_value})

    print("Meilleurs hyperparamètres:")
    print(study.best_params)
    wandb.config.update(study.best_params)

    best_params = study.best_params
    levels = ['1-1']
    raw_env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0', stages=levels, apply_api_compatibility=True, render_mode='rgb_array')
    raw_env = TimeLimit(raw_env, max_episode_steps=10000)
    env = make_env(raw_env)

    obs_shape = env.observation_space.shape
    act_space = env.action_space.n

    agent = DQNAgent(
        state_space=obs_shape,
        action_space=act_space,
        lr=best_params['lr'],
        batch_size=best_params['batch_size'],
        exploration_decay=best_params['exploration_decay'],
        gamma=best_params['gamma'],
        num_replay=best_params['num_replay'],
        target_update_freq=best_params['target_update_freq'],
        model_class=DQNSolverResNet
    )

    agent, eval_ep, eval_rewards, total_rewards = train(
        agent, env, num_episodes=10000, eval_step=5, levels=levels, max_steps=2000
    )

    video_path = save_video(env, agent, video_dir_path='my_best_videos', max_steps=3000)
    wandb.log({"final_video": wandb.Video(video_path)})

    env.close()
    agent.save('best_mario_model.pth')

    plt.figure()
    plt.plot(total_rewards, label="Training Reward")
    plt.plot(eval_ep, eval_rewards, label="Evaluation Reward")
    plt.title("Total Rewards - Final Training")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/training-{time.time()}.png")
    plt.show()

    mean_reward = eval_all(agent, levels=None, verbose=False)
    print(f"Reward moyen sur tous les stages: {mean_reward:.2f}")
    wandb.log({"final_mean_reward_all_stages": mean_reward})
    wandb.finish()
    mother_run.finish()