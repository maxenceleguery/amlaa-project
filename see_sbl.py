import pandas as pd
import os
import argparse
from pathlib import Path
from stable_baselines3 import PPO
import gym_super_mario_bros
from record import save_video_stablebaseline

from train_sbl import JoypadSpace, CustomRewardAndDoneEnv, SkipFrame, GrayScaleObservation, ResizeEnv, DummyVecEnv, VecFrameStack, STAGE_NAME

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mario RL Training (No HPO)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Loading a checkpoint")
    args = parser.parse_args()

    MOVEMENT = [['left', 'A'], ['right', 'B'], ['right', 'A', 'B']]
    env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0', stages=["1-3"], apply_api_compatibility=True, render_mode='rgb_array')
    env = JoypadSpace(env, MOVEMENT)
    env = CustomRewardAndDoneEnv(env)
    env = SkipFrame(env, skip=4, render=True)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeEnv(env, size=84)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

    if args.checkpoint is None:
        save_dir = Path('./test_sbl_resnet')
        reward_log = pd.read_csv(save_dir / "reward_log.csv", index_col='timesteps')
        print(reward_log.dtypes)
        reward_log.astype({'reward': 'float32', 'best_reward': 'float32'})
        best_epoch = reward_log['reward'].idxmax()
        print('best epoch:', best_epoch)

        best_model_path = os.path.join(save_dir, 'best_model_{}'.format(best_epoch))
    else:
        best_model_path = args.checkpoint
    model = PPO.load(best_model_path)

    save_video_stablebaseline(env, model, "videos_stablebaseline")
