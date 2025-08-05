
"""
Reimplementation of RL_Resource_Allocation.slx in Python using Gym + Stable-Baselines3
This file replaces the Simulink environment with a custom Gym environment.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

class ResourceAllocationEnv(gym.Env):
    def __init__(self, lte_demand, nr_demand, gamma=0.5, N_R=60, num_past=10, max_steps=500):
        super().__init__()
        self.gamma = gamma
        self.N_R = N_R
        self.num_past = num_past
        self.max_steps = max_steps
        self.step_count = 0

        self.lte_demand_series = lte_demand
        self.nr_demand_series = nr_demand
        self.demand_len = len(lte_demand)

        self.lte_hist = np.zeros(num_past)
        self.nr_hist = np.zeros(num_past)

        self.observation_space = spaces.Box(low=0, high=100, shape=(2*num_past+2,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=N_R, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.lte_hist[:] = 25
        self.nr_hist[:] = 25
        obs = np.concatenate([self.lte_hist, self.nr_hist, [self.gamma], [self.N_R]])
        return obs, {}

    def step(self, action):
        lte_alloc, nr_alloc = action
        idx = self.step_count % self.demand_len
        lte_demand = self.lte_demand_series[idx]
        nr_demand = self.nr_demand_series[idx]

        self.lte_hist = np.roll(self.lte_hist, -1)
        self.nr_hist = np.roll(self.nr_hist, -1)
        self.lte_hist[-1] = lte_demand
        self.nr_hist[-1] = nr_demand

        surplus_lte = (lte_alloc - lte_demand) / (lte_demand + 1e-6)
        surplus_nr = (nr_alloc - nr_demand) / (nr_demand + 1e-6)
        fairness = 0.5 * ((lte_alloc + nr_alloc)**2) / ((lte_alloc**2 + nr_alloc**2 + 1e-6))
        reward = -abs(surplus_lte) - abs(surplus_nr) + fairness

        obs = np.concatenate([self.lte_hist, self.nr_hist, [self.gamma], [self.N_R]])
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

lte_demands = []
nr_demands = []

for n in range(50):
    lte_path = f"Data/LTE_Demand_{n}.xlsx"
    nr_path = f"Data/NR_Demand_{n}.xlsx"
    
    lte_df = pd.read_excel(lte_path)
    nr_df = pd.read_excel(nr_path)

    # Giả sử cột cần dùng là 'NRB'
    lte_series = lte_df['NRB'].values
    nr_series = nr_df['NRB'].values

    lte_demands.append(lte_series)
    nr_demands.append(nr_series)

env = ResourceAllocationEnv(lte_demands, nr_demands)
check_env(env, warn=True)

# Train or load agent
model_path = "DDPGAgent.pt"
do_training = False
if do_training:
    model = DDPG("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save(model_path)
else:
    model = DDPG.load(model_path)

# Simulate trained agent
obs = env.reset()
lte_alloc_log, nr_alloc_log = [], []
lte_demand_log, nr_demand_log = [], []
for _ in range(500):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    lte_alloc_log.append(action[0])
    nr_alloc_log.append(action[1])
    lte_demand_log.append(env.lte_hist[-1])
    nr_demand_log.append(env.nr_hist[-1])
    if done:
        break

plt.plot(lte_alloc_log, label="LTE Allocation")
plt.plot(lte_demand_log, label="LTE Demand")
plt.plot(nr_alloc_log, label="NR Allocation")
plt.plot(nr_demand_log, label="NR Demand")
plt.legend()
plt.title("Resource Allocation by Trained DDPG Agent")
plt.xlabel("Timestep")
plt.ylabel("Resources")
plt.grid(True)
plt.show()
