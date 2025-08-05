import pandas as pd
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

# --- Load dữ liệu ---
lte = pd.read_excel("Data/LTE_Demand_1.xlsx")
nr = pd.read_excel("Data/NR_Demand_1.xlsx")
assert len(lte) == len(nr), "Dữ liệu LTE và NR phải cùng số dòng"

DA = lte.iloc[:, 0].values
DB = nr.iloc[:, 0].values

# --- Môi trường chia phổ ---
class SpectrumSharingEnv(gym.Env):
    def __init__(self, DA, DB, Nr=60, zeta=0.5, history_len=3):
        super().__init__()
        self.DA = DA
        self.DB = DB
        self.N = len(DA)
        self.Nr = Nr
        self.zeta = zeta
        self.hist_len = history_len
        self.idx = history_len

        self.observation_space = spaces.Box(low=0, high=100, shape=(2 * history_len,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=self.Nr, shape=(1,), dtype=np.float32)

    def reset(self):
        self.idx = self.hist_len
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i in range(self.hist_len):
            obs.extend([self.DA[self.idx - i - 1], self.DB[self.idx - i - 1]])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        NA = float(np.clip(action[0], 0, self.Nr))
        NB = self.Nr - NA
        DA_t = self.DA[self.idx]
        DB_t = self.DB[self.idx]

        frac_err_A = (NA - DA_t) / max(DA_t, 1e-3)
        frac_err_B = (NB - DB_t) / max(DB_t, 1e-3)
        J = self.zeta * (frac_err_A**2) + (1 - self.zeta) * (frac_err_B**2)
        eta = 0  # nếu muốn phạt mạnh hơn, đổi eta > 0
        reward = -(1 + eta) * J

        self.idx += 1
        done = self.idx >= self.N
        return self._get_obs(), reward, done, {}

# --- Callback để log reward theo episode ---
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = 0
        self.episode_lengths = 0

    def _on_step(self) -> bool:
        self.current_rewards += self.locals["rewards"][0]
        self.episode_lengths += 1
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_rewards)
            print(f"📘 Episode {len(self.episode_rewards)}: reward = {self.current_rewards:.3f}, length = {self.episode_lengths}")
            self.current_rewards = 0
            self.episode_lengths = 0
        return True

# --- Khởi tạo môi trường và mô hình ---
env = SpectrumSharingEnv(DA, DB, Nr=60, zeta=0.5, history_len=3)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=0)
callback = RewardLoggerCallback()

# --- Huấn luyện ---
model.learn(total_timesteps=10000, callback=callback)

# --- Đánh giá mô hình ---
obs = env.reset()
rewards, NA_list, NB_list = [], [], []
while True:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    rewards.append(reward)
    NA_list.append(float(action[0]))
    NB_list.append(env.Nr - float(action[0]))
    if done:
        break

# --- Vẽ reward theo episode ---
plt.figure()
plt.plot(callback.episode_rewards)
plt.title("Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()

# --- Vẽ phân bổ PRBs và nhu cầu ---
plt.figure()
plt.plot(DA[env.hist_len:], label='DA (LTE Demand)')
plt.plot(DB[env.hist_len:], label='DB (NR Demand)')
plt.plot(NA_list, label='Allocated NA (LTE)')
plt.plot(NB_list, label='Allocated NB (NR)')
plt.legend()
plt.title("Demand vs Allocated PRBs")
plt.xlabel("Time")
plt.ylabel("PRBs")
plt.grid(True)
plt.show()
