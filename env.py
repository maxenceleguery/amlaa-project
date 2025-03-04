import gym
from gym import spaces
import gym_super_mario_bros
from gym.wrappers import RecordVideo, FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

import numpy as np
import torch
from torchvision import transforms as T

import time

video_dir_path = 'mario_videos'



class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info
    
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.transform = T.Grayscale()

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        observation = self.transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )

    def observation(self, observation):
        observation = self.transforms(observation).squeeze(0)
        return observation

def make_env(env, video_dir_path=None):
    """Apply a series of wrappers to the environment."""
    env = JoypadSpace(env, RIGHT_ONLY)  # Reduce action space
    
    if video_dir_path is not None:
        env = RecordVideo(
            env,
            video_folder=video_dir_path,
            episode_trigger=lambda episode_id: True,
            name_prefix='mario-video-{}'.format(time.ctime())
        )
    
    #env = RecordEpisodeStatistics(env)  # Track stats
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)  # Convert to grayscale
    env = ResizeObservation(env, 84)  # Resize to 84x84
    env = FrameStack(env, num_stack=4)  # Stack 4 frames
    return env