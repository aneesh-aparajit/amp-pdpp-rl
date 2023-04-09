import random
from typing import Tuple

import gym
import numpy as np


def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    pos_ind = dem != 0
    # print(pos_ind)
    # print(num.shape, dem.shape)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    return 100 * np.mean(smap)


class ParkinsonEnv(gym.Env):
    def __init__(
        self, 
        dataset: Tuple[np.ndarray, np.ndarray],
        samples_per_episode: int = 16, 
        random: bool = True
    ) -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1, ), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.infty, high=np.infty, shape=(dataset[0].shape[1], ), dtype=np.float32
        )

        self.samples_per_episode = samples_per_episode
        self.step_count = 0
        self.X, self.y = dataset[0], dataset[1]
        self.random = random
        self.dataset_idx = 0

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        done = False
        reward = smape(y_true=self.expected_action, y_pred=np.array(action))
        obs = self._next_obs()
        self.step_count += 1
        if self.step_count >= self.samples_per_episode:
            done = True
        return obs, -np.mean(reward), done, {}

    def reset(self):
        self.step_count = 0
        obs = self._next_obs()
        return obs
    
    def _next_obs(self):
        if self.random:
            next_obs_idx = random.randint(0, len(self.X) - 1)
            self.expected_action = np.array([self.y[next_obs_idx]]).astype(np.float32)
            obs = self.X[next_obs_idx]
        else:
            obs = self.X[self.dataset_idx]
            self.expected_action = np.array(self.y[self.dataset_idx]).astype(np.float32)
            self.dataset_idx += 1
            if self.dataset_idx >= len(self.X):
                raise StopIteration()
        return obs
