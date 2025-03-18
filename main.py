import warnings
warnings.filterwarnings('ignore')

import argparse
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
import logging
import wandb
from gym.wrappers import TimeLimit
import gym_super_mario_bros

from env import make_env
from agent import DQNAgent, PolicyGradientAgent
from models import DQNSolverResNet
from record import save_video

logging.basicConfig(level=logging.INFO, filename=f"./log/mario_rl_train_{time.time()}.log")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

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

def eval_all(agent, levels=None, verbose=False):
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

def train(agent, env, num_episodes=10, eval_step=20, levels=None, max_steps=4000, use_wandb=False, num_replay=5, update_freq=300):
    total_rewards, eval_rewards, eval_ep = [], [], []
    best_reward = 0
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
            agent.scheduler.step(total_reward, ep_num)
        else:
            agent.update_policy()

        if use_wandb:
            wandb.log({"training_reward": total_reward, "episode": ep_num})
        else:
            logging.info("Episode %d, training_reward: %f", ep_num, total_reward)

        total_rewards.append(total_reward)

        if ep_num % eval_step == 0 and ep_num > 0:
            eval_r = eval_all(agent, levels=levels, verbose=False)
            eval_rewards.append(eval_r)
            eval_ep.append(ep_num)

            if eval_r > best_reward:
                best_reward = eval_r
                if hasattr(agent, "save"):
                    agent.save(f"best_mario_model_{eval_r:.2f}.pth")

            if use_wandb:
                wandb.log({"eval_reward": eval_r, "episode": ep_num, "lr": agent.scheduler.get_last_lr()[-1]})
            else:
                logging.info("Episode %d, eval_reward: %f, lr %f", ep_num, eval_r, agent.scheduler.get_last_lr()[-1])

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

    lr = 1e-2
    batch_size = 128
    exploration_decay = 0.9999
    gamma = 0.95
    num_replay = 5
    target_update_freq = 10000

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
<<<<<<< HEAD
    raw_env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0',
                                        stages=levels,
                                        apply_api_compatibility=True,
                                        render_mode='rgb_array')
    raw_env = TimeLimit(raw_env, max_episode_steps=4000)
=======
    raw_env = gym_super_mario_bros.make(
        'SuperMarioBrosRandomStages-v0',
        stages=levels,
        apply_api_compatibility=True,
        render_mode='rgb_array'
    )
    raw_env = TimeLimit(raw_env, max_episode_steps=args.max_steps)
>>>>>>> 68d89c09809e9a51262f3da93c3b610d2873ce28
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

<<<<<<< HEAD
    agent, _, _, _ = train(agent, env, num_episodes=50, eval_step=5, levels=levels, max_steps=4000)

    mean_reward = eval_all(agent, levels=levels, verbose=False)

    video_path = save_video(env, agent, video_dir_path='trial_videos', max_steps=4000)
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
        agent, env, num_episodes=10000, eval_step=5, levels=levels, max_steps=4000
    )

    video_path = save_video(env, agent, video_dir_path='my_best_videos', max_steps=4000)
    wandb.log({"final_video": wandb.Video(video_path)})
=======
    if args.checkpoint is not None:
        if not os.path.exists(args.checkpoint):
            raise ValueError(f"Checkpoint does not exist : {args.checkpoint}")
        agent.load(args.checkpoint)

    if args.eval:
        mean_reward = eval_all(agent, levels=None, verbose=True)
        print(f"Average reward across all stages: {mean_reward:.2f}")
        exit(0)

    agent, eval_ep, eval_rewards, total_rewards = train(
        agent,
        env,
        num_episodes=args.episodes,
        levels=levels,
        max_steps=args.max_steps,
        use_wandb=args.use_wandb,
        num_replay=num_replay,
        update_freq=target_update_freq,
    )

    video_path = save_video(env, agent, video_dir_path='my_best_videos', max_steps=args.max_steps)
    if args.use_wandb:
        wandb.log({"final_video": wandb.Video(video_path)})
    else:
        logging.info("final_video: %s", video_path)
>>>>>>> 68d89c09809e9a51262f3da93c3b610d2873ce28

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
