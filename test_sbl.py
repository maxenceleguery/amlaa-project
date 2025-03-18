import argparse
from stable_baselines3 import PPO
import gym_super_mario_bros
import pandas as pd
import numpy as np
import os

from train_sbl import JoypadSpace, CustomRewardAndDoneEnv, SkipFrame, GrayScaleObservation, ResizeEnv, DummyVecEnv, VecFrameStack

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mario RL Training (No HPO)")
    parser.add_argument("--name", type=str, default=None, help="Name of the training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Loading a checkpoint")
    args = parser.parse_args()

    if args.name is not None and args.checkpoint is not None:
        names = [args.name]
        checkpoints = [args.checkpoint]
    elif args.name is None and args.checkpoint is None:
        names = ["level_1-1", "world1_v1", "world1_v2"]
        checkpoints = ["best_model_ppo/level1-1_v2.zip", "best_model_ppo/world1.zip", "best_model_ppo/world1_v2.zip"]
    else:
        print("Provide --name and --checkpoint or none of them.")
        exit(0)

    for name, checkpoint in zip(names, checkpoints):
        model = PPO.load(checkpoint)

        if not os.path.exists("best_model_ppo/win_rates.csv"):
            os.makedirs("./best_model_ppo", exist_ok=True)
            data = {
                "Levels" : [f"{i+1}-{j+1}" for i in range(8) for j in range(4)]
            }
            df = pd.DataFrame(data)
        else:
            df = pd.read_csv("best_model_ppo/win_rates.csv")
        win_rates = []
        rewards = []

        for stage in df["Levels"]:
            MOVEMENT = [['left', 'A'], ['right', 'B'], ['right', 'A', 'B']]
            env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0', stages=[stage], apply_api_compatibility=True, render_mode='rgb_array')
            env = JoypadSpace(env, MOVEMENT)
            env = CustomRewardAndDoneEnv(env)
            env = SkipFrame(env, skip=4)
            env = GrayScaleObservation(env, keep_dim=True)
            env = ResizeEnv(env, size=84)
            env = DummyVecEnv([lambda: env])
            env = VecFrameStack(env, 4, channels_order='last')

            state = env.reset()
            done = False
            plays = 0
            wins = 0
            total_reward = 0
            while plays < 100:
                if done:
                    state = env.reset() 
                    if info[0]["flag_get"]:
                        wins += 1
                    plays += 1
                action, _ = model.predict(state)
                state, reward, done, info = env.step(action)
                total_reward += np.sum(reward)
            print(f"Model win rate on stage {stage} : {wins} % (Mean total reward : {total_reward/100.:.2f})")

            win_rates.append(wins/100.)
            rewards.append(total_reward/100.)

        df[f"{name}_win_rates"] = win_rates
        df[f"{name}_mean_reward"] = rewards
        df.to_csv("best_model_ppo/win_rates.csv")