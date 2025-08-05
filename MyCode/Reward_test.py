# File: resource_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class ResourceAllocationEnv(gym.Env):
    def __init__(self, lte_file, nr_file, gamma=0.5, N_R=60, history_len=20, scaling=0.01, max_steps=1):
        super().__init__()
        self.gamma = gamma
        self.N_R = N_R
        self.scaling = scaling
        self.history_len = history_len
        self.max_steps = max_steps

        self.lte_demand = lte_file
        self.nr_demand = nr_file
        self.total_len = len(self.lte_demand)

        self.observation_space = spaces.Box(low=0.0, high=60.0, shape=(2*history_len + 2,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.01, high=1.0, shape=(2,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.idx = self.history_len
        return self._get_obs(), {}

    def _get_obs(self):
        # Lấy lịch sử + thời điểm hiện tại: từ (t - history_len + 1) đến t
        lte_hist = self.lte_demand[self.idx - self.history_len + 1: self.idx + 1] #/ self.N_R
        nr_hist = self.nr_demand[self.idx - self.history_len + 1: self.idx + 1] #/ self.N_R
        normalized_time = self.idx / self.total_len

        obs = np.concatenate([
            lte_hist,               # Demand LTE: t-history_len+1 to t
            nr_hist,                # Demand NR: t-history_len+1 to t
            [self.gamma],           # ζ (ưu tiên)
            [60.0],                  # N_R / N_R = 1.0 (chuẩn hóa)
            #[normalized_time]       
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
        action = np.clip(action, 0.01, 1.0)
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

        # return obs.astype(np.float32) , reward, terminated, truncated, {
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

lte_file = load_nrb_series("Data/LTE_Demand_hourly.tsv")
nr_file = load_nrb_series("Data/NR_Demand_hourly.tsv")

# Tạo môi trường
env = ResourceAllocationEnv(lte_file=lte_file, nr_file=nr_file, gamma=0.5, N_R=60)

# Kiểm tra tính hợp lệ của môi trường
check_env(env, warn=True)

# Noise cho hành động DDPG
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), 
    sigma=0.1 * np.ones(n_actions)  # std = 0.1
)

# Khởi tạo mô hình DDPG
model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    learning_rate=1e-3,
    buffer_size=int(1e6),               # = ExperienceBufferLength
    learning_starts=1000,
    batch_size=48,                      # = MiniBatchSize
    tau=1e-4,                           # = TargetSmoothFactor
    gamma=0.99,                         # = DiscountFactor
    train_freq=(1, "step"),
    gradient_steps=1,
    verbose=1,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log="./ddpg_tensorboard/"
)

# model = DDPG(
#     "MlpPolicy",
#     env,
#     action_noise=action_noise,
#     verbose=1,
#     learning_rate=1e-3,
#     buffer_size=50000,
#     learning_starts=1000,
#     batch_size=64,
#     tau=0.005,
#     gamma=0.99,
#     train_freq=(1, "step"),
#     policy_kwargs=dict(net_arch=[256, 256]),
#     tensorboard_log="./ddpg_tensorboard/"
# )


# Huấn luyện agent
model.learn(total_timesteps=5_000)

# Lưu model
model.save("ddpg_resource_allocation_model.pt")

# Đánh giá agent đã huấn luyện
obs, _ = env.reset()
done = False 
rewards = []
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    rewards.append(reward)

print("Sum reward sau 1 episode kiểm tra:", sum(rewards))

# Lưu ra file để tiện sử dụng
# pipeline_path = Path("/mnt/data/train_ddpg_pipeline.py")
# pipeline_path.write_text(ddpg_pipeline_code)
# pipeline_path.name