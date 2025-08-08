import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# Giả sử đã có các biến sau được thu thập trong quá trình chạy model.predict()
# Ta mô phỏng quá trình đánh giá lại mô hình để thu thập chúng

# from resource_env import ResourceAllocationEnv
from stable_baselines3 import DDPG

# class ResourceAllocationEnv(gym.Env):
#     def __init__(self, lte_file, nr_file, gamma=0.5, N_R=100, history_len=20, scaling=0.01, max_steps=1):
#         super().__init__()
#         self.gamma = gamma
#         self.N_R = N_R
#         self.scaling = scaling
#         self.history_len = history_len
#         self.max_steps = max_steps

#         self.lte_demand = lte_file
#         self.nr_demand = nr_file
#         self.total_len = len(self.lte_demand)

#         self.observation_space = spaces.Box(low=0.1, high=50.0, shape=(2*history_len + 2,), dtype=np.float32)
#         self.action_space = spaces.Box(low=0.1, high=1.0, shape=(2,), dtype=np.float32)

#         self.reset()

#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
#         self.step_count = 0
#         self.idx = self.history_len
#         return self._get_obs(), {}

#     def _get_obs(self):
#         lte_hist = self.lte_demand[self.idx - self.history_len:self.idx]
#         nr_hist = self.nr_demand[self.idx - self.history_len:self.idx]
#         obs = np.concatenate([lte_hist, nr_hist, [self.gamma], [self.N_R]])
#         return obs.astype(np.float32)

#     def reward_function(self, observation, action):
#         D_A = observation[0]
#         D_B = observation[1]
#         w_A = observation[2]
#         w_B = 1 - w_A
#         N_R = observation[3]
#         N_A = action[0]
#         N_B = action[1]

#         if D_A + D_B - N_R > 0:
#             if N_A + N_B - N_R > 0:
#                 reward = -(w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
#                         - 10 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
#             elif N_A == 0 or N_B == 0:
#                 reward = 0
#                 if N_A == 0:
#                     reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
#                             + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
#                 if N_B == 0:
#                     reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
#                             + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
#             elif (N_A + N_B > N_R - 0.03) and (N_A + N_B - N_R < 0):
#                 reward = -(w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) + 10
#             else:
#                 reward = -(w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) + 5
#         else:
#             if N_A + N_B - N_R > 0:
#                 reward = -(w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
#                         - 10 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
#             elif (N_A - D_A < 0 or N_B - D_B < 0 or N_A - D_A > 0 or N_B - D_B > 0 or N_A == 0 or N_B == 0):
#                 reward = 0
#                 if N_A - D_A < 0:
#                     reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
#                             + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
#                 if N_B - D_B < 0:
#                     reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
#                             + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
#                 if N_A - D_A > 0:
#                     reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
#                             + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
#                 if N_B - D_B > 0:
#                     reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
#                             + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
#                 if N_A == 0:
#                     reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
#                             + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
#                 if N_B == 0:
#                     reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
#                             + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
#             else:
#                 reward = -(w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)

#         return reward

#     def step(self, action):
#         # Ensure no zero allocation and sum ≤ N_R
#         action = np.clip(action, 0.01, 1.0)
#         alloc = action * self.N_R
#         print(alloc)
#         total_alloc = np.sum(alloc)
#         if total_alloc > self.N_R:
#             alloc = (alloc / total_alloc) * self.N_R

#         demand_lte = self.lte_demand[self.idx]
#         demand_nr = self.nr_demand[self.idx]

#         obs = np.concatenate([
#             self.lte_demand[self.idx - self.history_len:self.idx],
#             self.nr_demand[self.idx - self.history_len:self.idx],
#             [self.gamma],
#             [self.N_R]
#         ])

#         reward = self.reward_function([demand_lte, demand_nr, self.gamma, self.N_R], alloc)

#         self.idx += 1
#         self.step_count += 1

#         terminated = self.idx >= self.total_len
#         truncated = self.step_count >= self.max_steps

#         return obs.astype(np.float32), reward, terminated, truncated, {
#             "alloc": alloc,
#             "demand": (demand_lte, demand_nr),
#         }


class ResourceAllocationEnv(gym.Env):
    def __init__(self, lte_file, nr_file, gamma=0.5, N_R=60.0, history_len=20, scaling=0.01, max_steps=1):
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
        # Lấy lịch sử + thời điểm hiện tại: từ (t - history_len + 1) đến t
        # lte_hist = self.lte_demand[self.idx - self.history_len + 1: self.idx + 1] / self.N_R
        # nr_hist = self.nr_demand[self.idx - self.history_len + 1: self.idx + 1] / self.N_R

        # Lấy lịch sử + thời điểm hiện tại
        raw_lte_hist = self.lte_demand[self.idx - self.history_len + 1: self.idx + 1]
        raw_nr_hist = self.nr_demand[self.idx - self.history_len + 1: self.idx + 1]

        # Chuẩn hóa theo Min-Max
        lte_hist = (raw_lte_hist - self.lte_min) / (self.lte_max - self.lte_min + 1e-8)
        nr_hist = (raw_nr_hist - self.nr_min) / (self.nr_max - self.nr_min + 1e-8)

        # Current demand
        current_lte = (self.lte_demand[self.idx] - self.lte_min) / (self.lte_max - self.lte_min + 1e-8)
        current_nr = (self.nr_demand[self.idx] - self.nr_min) / (self.nr_max - self.nr_min + 1e-8)


        normalized_time = self.idx / self.total_len
        
        obs = np.concatenate([
            lte_hist,               # Demand LTE: t-history_len+1 to t
            nr_hist,                # Demand NR: t-history_len+1 to t
            [current_lte],           # Demand LTE hiện tại
            [current_nr],            # Demand NR hiện tại
            [self.gamma],           # ζ (ưu tiên)
            [1.0],                   # N_R / N_R = 1.0 (chuẩn hóa)
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
                reward = -(w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
                        - 10 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
            elif N_A == 0 or N_B == 0:
                reward = 0
                if N_A == 0:
                    reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
                            + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
                if N_B == 0:
                    reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
                            + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
            elif (N_A + N_B > N_R - 0.03) and (N_A + N_B - N_R < 0):
                reward = -(w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) + 10
            else:
                reward = -(w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) + 5
        else:
            if N_A + N_B - N_R > 0:
                reward = -(w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
                        - 10 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
            elif (N_A - D_A < 0 or N_B - D_B < 0 or N_A - D_A > 0 or N_B - D_B > 0 or N_A == 0 or N_B == 0):
                reward = 0
                if N_A - D_A < 0:
                    reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
                            + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
                if N_B - D_B < 0:
                    reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
                            + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
                if N_A - D_A > 0:
                    reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
                            + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
                if N_B - D_B > 0:
                    reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
                            + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
                if N_A == 0:
                    reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
                            + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
                if N_B == 0:
                    reward -= (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2) \
                            + 5 * (w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)
            else:
                reward = -(w_A * ((N_A - D_A) / D_A) ** 2 + w_B * ((N_B - D_B) / D_B) ** 2)

        return reward

    def step(self, action):
        # Ensure no zero allocation and sum ≤ N_R
        action = np.clip(action, 0.0, 1.0)
        alloc = action * self.N_R
        total_alloc = np.sum(alloc)
        if total_alloc > self.N_R:
            alloc = (alloc / total_alloc) * self.N_R

        demand_lte = self.lte_demand[self.idx]
        demand_nr = self.nr_demand[self.idx]

        # obs = np.concatenate([
        #     self.lte_demand[self.idx - self.history_len:self.idx],
        #     self.nr_demand[self.idx - self.history_len:self.idx],
        #     [self.gamma],
        #     [self.N_R]
        # ])

        reward = self.reward_function([demand_lte, demand_nr, self.gamma, self.N_R], alloc)

        self.idx += 1
        self.step_count += 1

        terminated = self.idx >= self.total_len
        truncated = self.step_count >= self.max_steps

        # return obs.astype(np.float32), reward, terminated, truncated, {
        #     "alloc": alloc,
        #     "demand": (demand_lte, demand_nr),
        # }
    
        return self._get_obs(), reward, terminated, truncated, {
            "alloc": alloc,
            "demand": (demand_lte, demand_nr),
        }

# Đường dẫn đến dữ liệu Excel
def load_nrb_series(filepath):
    if filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)['NRB'].values
    elif filepath.endswith('.tsv'):
        return pd.read_csv(filepath, sep='\\t', engine='python')['NRB'].values
    else:
        return pd.read_csv(filepath)['NRB'].values
    
lte_file = load_nrb_series("Data/LTE_Demand_32.xlsx")
nr_file = load_nrb_series("Data/NR_Demand_32.xlsx")   
# Khởi tạo lại môi trường và mô hình đã huấn luyện
env = ResourceAllocationEnv(lte_file, nr_file, gamma=0.5, N_R=60)
model = DDPG.load("ddpg_resource_allocation_model.pt", env=env)

obs, _ = env.reset()
done = False
rewards = []
alloc_A, alloc_B = [], []
demand_A, demand_B = [], []
surplus_A, surplus_B = [], []

# while not done:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated
#     rewards.append(reward)

#     alloc = info["alloc"]
#     demand = info["demand"]
    
#     alloc_A.append(alloc[0])
#     alloc_B.append(alloc[1])
#     demand_A.append(demand[0])
#     demand_B.append(demand[1])
#     surplus_A.append(alloc[0] - demand[0])
#     surplus_B.append(alloc[1] - demand[1])

T = 100  # hoặc 20 nếu muốn
for t in range(T):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    rewards.append(reward)

    alloc = info["alloc"]
    demand = info["demand"]

    alloc_A.append(alloc[0])
    alloc_B.append(alloc[1])
    demand_A.append(demand[0])
    demand_B.append(demand[1])
    surplus_A.append(alloc[0] - demand[0])
    surplus_B.append(alloc[1] - demand[1])


# === Đánh giá hiệu suất theo công thức bài báo ===
D_A = np.array(demand_A)
D_B = np.array(demand_B)
N_A = np.array(alloc_A)
N_B = np.array(alloc_B)

# Tính surplus trung bình
SA = np.mean((N_A - D_A) / D_A)
SB = np.mean((N_B - D_B) / D_B)

# Tính Jain’s Fairness Index
numerator = (N_A + N_B) ** 2
denominator = 2 * (N_A ** 2 + N_B ** 2)
fairness = np.mean(numerator / denominator)

episodes = np.arange(len(rewards))
# Vẽ cả reward, surplus và fairness theo thời gian
fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex=True)
rel_surplus_A = ((np.array(alloc_A) - np.array(demand_A)) / np.array(demand_A)) * 100
rel_surplus_B = ((np.array(alloc_B) - np.array(demand_B)) / np.array(demand_B)) * 100

# 1. Reward
axs[0, 0].plot(episodes, rewards, label="Reward", color='blue')
axs[0, 0].set_ylabel("Reward")
axs[0, 0].set_title("Reward per Step")
axs[0, 0].grid(True)

# 2. Allocation vs Demand
axs[0, 1].plot(episodes, alloc_A, label="Alloc LTE", linestyle='--')
axs[0, 1].plot(episodes, demand_A, label="Demand LTE", linestyle='-')
axs[0, 1].plot(episodes, alloc_B, label="Alloc NR", linestyle='--')
axs[0, 1].plot(episodes, demand_B, label="Demand NR", linestyle='-')
axs[0, 1].set_ylabel("Resource")
axs[0, 1].set_title("Allocation vs Demand")
axs[0, 1].legend()
axs[0, 1].grid(True)

# 3. Surplus/Deficit
axs[1, 0].plot(episodes, surplus_A, label="Surplus/Deficit LTE")
axs[1, 0].plot(episodes, surplus_B, label="Surplus/Deficit NR")
axs[1, 0].set_ylabel("Surplus")
axs[1, 0].set_title("Surplus/Deficit over Time")
axs[1, 0].legend()
axs[1, 0].grid(True)

# 4. Jain's Fairness Index (theo step)
fairness_per_step = ((np.array(alloc_A) + np.array(alloc_B))**2) / (2 * (np.array(alloc_A)**2 + np.array(alloc_B)**2))
axs[1, 1].plot(episodes, fairness_per_step, label="Fairness", color='green')
axs[1, 1].set_xlabel("Step")
axs[1, 1].set_ylabel("Fairness")
axs[1, 1].set_title("Jain's Fairness Index per Step")
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plot_path = "allocation_demand_surplus_fairness_plot.png"
plt.savefig(plot_path)
plt.show()


# # Vẽ biểu đồ
# episodes = np.arange(len(rewards))

# plt.figure(figsize=(12, 8))

# plt.subplot(3, 1, 1)
# plt.plot(episodes, rewards, label="Reward", color='blue')
# plt.ylabel("Reward")
# plt.title("Reward per Step")
# plt.grid(True)

# plt.subplot(3, 1, 2)
# plt.plot(episodes, alloc_A, label="Alloc A", linestyle='--')
# plt.plot(episodes, demand_A, label="Demand A", linestyle='-')
# plt.plot(episodes, alloc_B, label="Alloc B", linestyle='--')
# plt.plot(episodes, demand_B, label="Demand B", linestyle='-')
# plt.ylabel("Resource")
# plt.title("Allocation vs Demand")
# plt.legend()
# plt.grid(True)

# plt.subplot(3, 1, 3)
# plt.plot(episodes, surplus_A, label="Surplus/Deficit A")
# plt.plot(episodes, surplus_B, label="Surplus/Deficit B")
# plt.xlabel("Step")
# plt.ylabel("Surplus")
# plt.title("Surplus/Deficit over Time")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig("allocation_demand_surplus_plot.png")
# plt.show()