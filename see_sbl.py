import pandas as pd
import os
from pathlib import Path
from stable_baselines3 import PPO
import gym_super_mario_bros

from test_sbl import JoypadSpace, CustomRewardAndDoneEnv, SkipFrame, GrayScaleObservation, ResizeEnv, DummyVecEnv, VecFrameStack, STAGE_NAME

if __name__ == "__main__":
    MOVEMENT = [['left', 'A'], ['right', 'B'], ['right', 'A', 'B']]
    env = gym_super_mario_bros.make(STAGE_NAME)
    env = JoypadSpace(env, MOVEMENT)
    env = CustomRewardAndDoneEnv(env)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeEnv(env, size=84)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

    save_dir = Path('./test_sbl')

    reward_log = pd.read_csv("reward_log_Standar.csv", index_col='timesteps')
    best_epoch = reward_log['reward'].idxmax()
    print('best epoch:', best_epoch)

    best_model_path = os.path.join(save_dir, 'best_model_{}'.format(best_epoch))
    model = PPO.load(best_model_path)

    state = env.reset()
    done = True
    plays = 0
    wins = 0
    while plays < 100:
        if done:
            state = env.reset() 
            if info[0]["flag_get"]:
                wins += 1
            plays += 1
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
    print("Model win rate: " + str(wins) + "%")

