import gym
import numpy as np
import torch
from gym import spaces
from gym.wrappers import RecordVideo, FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from torchvision import transforms as T
import time
#env
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
    def step(self, action):
        total_reward = 0.0
        done, trunc = False, False
        for _ in range(self._skip):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done or trunc:
                break
        return obs, total_reward/(4.*15.), done, trunc, info
        # Normalizing reward between -1 and 1

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.transform = T.Grayscale()

    def observation(self, observation):
        obs_t = torch.tensor(observation.transpose((2, 0, 1)).copy(), dtype=torch.float)
        obs_t = self.transform(obs_t)
        obs_t /= 255
        return obs_t

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape=84):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        obs_shape = self.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.float32)
        self.transforms = T.Compose([
            T.Resize(self.shape, antialias=True),
        ])

    def observation(self, observation):
        return self.transforms(observation).squeeze(0)

def make_env(env, video_dir_path=None):
    env = JoypadSpace(env, RIGHT_ONLY)
    if video_dir_path is not None:
        env = RecordVideo(
            env,
            video_folder=video_dir_path,
            episode_trigger=lambda episode_id: True,
            name_prefix=f"mario-video-{time.time()}"
        )
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 84)
    env = FrameStack(env, num_stack=4)
    return env
