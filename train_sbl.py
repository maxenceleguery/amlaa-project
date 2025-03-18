from stable_baselines3 import PPO
import torch
from torch import nn

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gym_super_mario_bros
STAGE_NAME = 'SuperMarioBros-1-1-v0'
#STAGE_NAME = 'SuperMarioBros-1-1-v3'

from nes_py.wrappers import JoypadSpace

import gym
import numpy as np
import cv2
import os
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from pathlib import Path

from models import ResidualBlock

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip, render=False):
        super().__init__(env)
        self._skip = skip
        self._render = render

    def step(self, action):
        total_reward = 0.0
        done = False
        frames = []
        for i in range(self._skip):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if self._render:
                frames.append(np.array(self.env.render()))
            if done:
                break
        info[f"frames"] = frames
        return obs, total_reward, done, trunc, info


class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, size):
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, oldc) = env.observation_space.shape
        newshape = (size, size, oldc)
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=newshape, dtype=np.uint8)

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:,:,None]
        return frame

class CustomRewardAndDoneEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(CustomRewardAndDoneEnv, self).__init__(env)
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0

    def reset(self, **kwargs):
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        return self.env.reset()

    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)
        reward += max(0, info['x_pos'] - self.max_x)
        if (info['x_pos'] - self.current_x) == 0:
            self.current_x_count += 1
        else:
            self.current_x_count = 0
        if info["flag_get"]:
            reward += 500
            done = True
            print("GOAL")
        if info["life"] < 2:
            reward -= 500
            done = True
        self.current_score = info["score"]
        self.max_x = max(self.max_x, self.current_x)
        self.current_x = info["x_pos"]
        return state, reward / 10., done, trunc, info

class MarioNet(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(MarioNet, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    
class MarioResNet(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(MarioResNet, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=8, stride=4),
            nn.ReLU(),
            ResidualBlock(8, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(16, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = (save_dir / 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

            total_reward = [0] * EPISODE_NUMBERS
            total_time = [0] * EPISODE_NUMBERS
            best_reward = 0

            for i in range(EPISODE_NUMBERS):
                state = env.reset()  # reset for each new trial
                done = False
                total_reward[i] = 0
                total_time[i] = 0
                while not done and total_time[i] < MAX_TIMESTEP_TEST:
                    action, _ = model.predict(state)
                    state, reward, done, *_ = env.step(action)
                    total_reward[i] += reward[0]
                    total_time[i] += 1

                if total_reward[i] > best_reward:
                    best_reward = total_reward[i]
                    best_epoch = self.n_calls

                state = env.reset()  # reset for each new trial

            print('time steps:', self.n_calls, '/', TOTAL_TIMESTEP_NUMB)
            print('average reward:', (sum(total_reward) / EPISODE_NUMBERS),
                  'average time:', (sum(total_time) / EPISODE_NUMBERS),
                  'best_reward:', best_reward)

            with open(reward_log_path, 'a') as f:
                print(self.n_calls, ',', sum(total_reward) / EPISODE_NUMBERS, ',', best_reward, file=f)

        return True

if __name__ == "__main__":
    MOVEMENT = [['left', 'A'], ['right', 'B'], ['right', 'A', 'B']]
    levels = ["1-1", "1-2", "1-3", "1-4"]
    env = gym_super_mario_bros.make(
        'SuperMarioBrosRandomStages-v0',
        stages=levels,
        apply_api_compatibility=True,
        render_mode='rgb_array'
    )
    env = JoypadSpace(env, MOVEMENT)
    env = CustomRewardAndDoneEnv(env)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeEnv(env, size=84)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

    # Model Param
    CHECK_FREQ_NUMB = 10000
    TOTAL_TIMESTEP_NUMB = 5000000
    LEARNING_RATE = 0.0001
    GAE = 1.0
    ENT_COEF = 0.01
    N_STEPS = 512
    GAMMA = 0.9
    BATCH_SIZE = 64
    N_EPOCHS = 10

    # Test Param
    EPISODE_NUMBERS = 20
    MAX_TIMESTEP_TEST = 1000

    policy_kwargs = dict(
        features_extractor_class=MarioResNet,
        features_extractor_kwargs=dict(features_dim=512),
    )

    save_dir = Path('./test_sbl_resnet')
    save_dir.mkdir(parents=True, exist_ok=True)
    reward_log_path = (save_dir / 'reward_log.csv')

    with open(reward_log_path, 'a') as f:
        print('timesteps,reward,best_reward', file=f)
    
    callback = TrainAndLoggingCallback(check_freq=CHECK_FREQ_NUMB, save_path=save_dir)

    model = PPO('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs, learning_rate=LEARNING_RATE, n_steps=N_STEPS,
                batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=GAE, ent_coef=ENT_COEF)

    #model.set_parameters("best_model_ppo/world1.zip")

    model.learn(total_timesteps=TOTAL_TIMESTEP_NUMB, callback=callback)