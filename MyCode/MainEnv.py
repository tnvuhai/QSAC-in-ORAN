import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

class ResourceAllocationEnv(gym.Env):
    def __init__(self, lte_file, nr_file, gamma=0.5, N_R=60.0, history_len=10, scaling=0.01, max_steps=1):
        super().__init__()
        self.gamma = gamma
        self.N_R = N_R
        self.scaling = scaling
        self.history_len = history_len
        self.max_steps = max_steps

        self.lte_demand = lte_file
        self.nr_demand = nr_file
        self.total_len = len(self.lte_demand)

        self.lte_min = np.min(self.lte_demand)
        self.lte_max = np.max(self.lte_demand)
        self.nr_min = np.min(self.nr_demand)
        self.nr_max = np.max(self.nr_demand)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2*history_len + 5,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.idx = self.history_len
        return self._get_obs(), {}

    def _get_obs(self):
        raw_lte_hist = self.lte_demand[self.idx - self.history_len + 1: self.idx + 1]
        raw_nr_hist = self.nr_demand[self.idx - self.history_len + 1: self.idx + 1]

        lte_hist = (raw_lte_hist - self.lte_min) / (self.lte_max - self.lte_min + 1e-8)
        nr_hist = (raw_nr_hist - self.nr_min) / (self.nr_max - self.nr_min + 1e-8)

        current_lte = (self.lte_demand[self.idx] - self.lte_min) / (self.lte_max - self.lte_min + 1e-8)
        current_nr = (self.nr_demand[self.idx] - self.nr_min) / (self.nr_max - self.nr_min + 1e-8)

        normalized_time = self.idx / self.total_len
        
        obs = np.concatenate([
            lte_hist,
            nr_hist,
            [current_lte],
            [current_nr],
            [self.gamma],
            [1.0],
            [normalized_time]
        ])
        return obs.astype(np.float32)

    def reward_function(self, observation, action):
        D_A = observation[0]
        D_B = observation[1]
        w_A = observation[2]
        w_B = 1 - w_A
        N_R = observation[3]
        N_A = action[0]
        N_B = action[1]

        if D_A + D_B - N_R > 0:
            if N_A + N_B - N_R > 0:
                reward = -(w_A * ((N_A - D_A) / D_A)**2 + w_B * ((N_B - D_B) / D_B)**2)
                reward -= 10 * (w_A * ((N_A - D_A) / D_A)**2 + w_B * ((N_B - D_B) / D_B)**2)
            elif N_A == 0 or N_B == 0:
                reward = 0
                reward -= 5 * (w_A * ((N_A - D_A) / D_A)**2 + w_B * ((N_B - D_B) / D_B)**2)
            elif (N_A + N_B > N_R - 0.03):
                reward = -(w_A * ((N_A - D_A) / D_A)**2 + w_B * ((N_B - D_B) / D_B)**2) + 10
            else:
                reward = -(w_A * ((N_A - D_A) / D_A)**2 + w_B * ((N_B - D_B) / D_B)**2) + 5
        else:
            if N_A + N_B - N_R > 0:
                reward = -(w_A * ((N_A - D_A) / D_A)**2 + w_B * ((N_B - D_B) / D_B)**2)
                reward -= 10 * (w_A * ((N_A - D_A) / D_A)**2 + w_B * ((N_B - D_B) / D_B)**2)
            else:
                reward = -(w_A * ((N_A - D_A) / D_A)**2 + w_B * ((N_B - D_B) / D_B)**2)
        return reward

    def step(self, action):
        action = np.clip(action, 0.01, 1.00)
        alloc = action * self.N_R
        total_alloc = np.sum(alloc)
        if total_alloc > self.N_R:
            alloc = (alloc / total_alloc) * self.N_R

        demand_lte = self.lte_demand[self.idx]
        demand_nr = self.nr_demand[self.idx]

        reward = self.reward_function([demand_lte, demand_nr, self.gamma, self.N_R], alloc)

        self.idx += 1
        self.step_count += 1

        terminated = self.idx >= self.total_len
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            "alloc": alloc,
            "demand": (demand_lte, demand_nr)
        }
    
def load_demand_data(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.xlsx' or ext == '.xls':
        data = pd.read_excel(file_path)
    elif ext == '.csv':
        data = pd.read_csv(file_path)
    elif ext == '.tsv':
        data = pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if isinstance(data, pd.DataFrame):
        series = data.select_dtypes(include=[np.number]).values.flatten()
    else:
        series = data.values.flatten()

    series = pd.to_numeric(series, errors='coerce')
    series = np.nan_to_num(series, nan=0.0)
    return series.astype(np.float32)