import warnings
warnings.filterwarnings('ignore')

import argparse
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


def evaluate_agent(agent, env, num_episodes=5, max_steps=4000, show=False):
    total_rewards = []
    for _ in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state.__array__(), dtype=torch.float, device=agent.device)
        total_reward = 0
        for _ in range(max_steps):
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
        for _ in range(max_steps):
            action = agent.act(state, sample=True)
            next_state, reward, done, trunc, info = env.step(int(action.item()))
            next_state = torch.tensor(next_state.__array__(), dtype=torch.float, device=agent.device)

            agent.remember(state, action, reward, next_state, done or trunc)
            agent.experience_replay()

            if agent.step % agent.target_update_freq == 0:
                agent.update_target_network()

            state = next_state
            total_reward += reward
            if done or trunc:
                break

        agent.scheduler.step()

        # Log local (run en cours, soit run enfant si HPO)
        wandb.log({"training_reward": total_reward, "episode": ep_num})
        total_rewards.append(total_reward)

        # Évaluation périodique
        if ep_num % eval_step == 0 and ep_num > 0:
            eval_r = eval_all(agent, levels=levels, verbose=False)
            eval_rewards.append(eval_r)
            eval_ep.append(ep_num)
            wandb.log({"eval_reward": eval_r, "episode": ep_num})

        agent.save("mario_model.pth")

    return agent, eval_ep, eval_rewards, total_rewards



def objective(trial):
    # Run enfant : on crée un nouveau run pour ce trial
    child_run = wandb.init(
        project="MarioDQN-Optuna",
        name=f"trial_{trial.number}",
        reinit=True
    )

    # Hyperparamètres suggérés
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    exploration_decay = trial.suggest_float('exploration_decay', 0.95, 0.9999, step=0.0005)
    gamma = trial.suggest_float('gamma', 0.90, 0.99, step=0.01)
    num_replay = trial.suggest_categorical('num_replay', [1, 5, 10, 20])
    target_update_freq = trial.suggest_categorical('target_update_freq', [250, 500, 1000, 2000])

    # On log ces HP dans le run enfant
    wandb.config.update({
        "trial_number": trial.number,
        "lr": lr,
        "batch_size": batch_size,
        "exploration_decay": exploration_decay,
        "gamma": gamma,
        "num_replay": num_replay,
        "target_update_freq": target_update_freq
    })

    # Environnement
    levels = ['1-1']
    raw_env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0',
                                        stages=levels,
                                        apply_api_compatibility=True,
                                        render_mode='rgb_array')
    raw_env = TimeLimit(raw_env, max_episode_steps=4000)
    env = make_env(raw_env)

    obs_shape = env.observation_space.shape
    act_space = env.action_space.n

    # Agent
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

    # Entraînement rapide sur quelques épisodes
    agent, _, _, _ = train(agent, env, num_episodes=50, eval_step=5, levels=levels, max_steps=4000)

    # Évaluation
    mean_reward = eval_all(agent, levels=levels, verbose=False)

    # Enregistrement vidéo
    video_path = save_video(env, agent, video_dir_path='trial_videos', max_steps=4000)
    wandb.log({"final_video": wandb.Video(video_path)})

    env.close()
    child_run.log({"final_mean_reward": mean_reward})
    child_run.finish()
    return mean_reward


##################################################
# Callback “mère” : on stocke la perf du trial
##################################################

# Pour éviter le bug "W&B n'est pas initialisé" dans la callback,
# on ne loggue pas directement dans la mother_run ici
# (puisque l'init du run enfant écrase le run mère).
# On collecte juste la valeur, et on la logguera APRÈS l'optimisation.
mother_results = []

def mother_callback(study, trial):
    mother_results.append((trial.number, trial.value, dict(trial.params)))




def main():
    parser = argparse.ArgumentParser(description="Mario RL with optional HPO")
    parser.add_argument("--hpo", type=bool, default=False, help="Run HPO with Optuna if True")
    parser.add_argument("--n-trials", type=int, default=5, help="Number of Optuna trials")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel workers for Optuna")

    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes for final training")
    parser.add_argument("--max-steps", type=int, default=4000, help="Max steps per episode")

    args = parser.parse_args()

    if args.hpo:
        # Run mère
        mother_run = wandb.init(project="MarioDQN-Optuna", name="mother_run", reinit=True)

        # Optuna
        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            callbacks=[mother_callback]
        )

        # Après la fin de la HPO, on loggue les résultats dans la run mère
        best_reward = study.best_value
        best_params = study.best_params
        wandb.log({"best_reward": best_reward})
        wandb.config.update(best_params)

        for trial_num, value, params in mother_results:
            wandb.log({
                "mother/trial_number": trial_num,
                "mother/final_eval_reward": value
            })

        print("=== Meilleurs hyperparamètres ===")
        print(best_params)

        mother_run.finish()
    else:
        run = wandb.init(project="MarioDQN-Optuna", name="no_hpo_run", reinit=True)

        lr = 1e-4
        batch_size = 64
        exploration_decay = 0.995
        gamma = 0.95
        num_replay = 5
        target_update_freq = 500

        wandb.config.update({
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
        raw_env = TimeLimit(raw_env, max_episode_steps=args.max_steps)
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

        # Training final
        agent, eval_ep, eval_rewards, total_rewards = train(
            agent, env, num_episodes=args.episodes, eval_step=5, levels=levels, max_steps=args.max_steps
        )

        # Enregistrement vidéo
        video_path = save_video(env, agent, video_dir_path='my_best_videos', max_steps=args.max_steps)
        wandb.log({"final_video": wandb.Video(video_path)})

        env.close()
        agent.save('best_mario_model.pth')

        # Plot
        plt.figure()
        plt.plot(total_rewards, label="Training Reward")
        plt.plot(eval_ep, eval_rewards, label="Evaluation Reward")
        plt.title("Total Rewards - Final Training (No HPO)")
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


if __name__ == "__main__":
    main()
