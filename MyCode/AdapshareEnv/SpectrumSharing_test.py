import gym
import numpy as np
import pandas as pd
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import torch


def load_all_demand(folder_path):
    lte_list, nr_list = [], []
    for i in range(1, 50):
        lte_file = os.path.join(folder_path, f"LTE_Demand_{i}.xlsx")
        nr_file = os.path.join(folder_path, f"NR_Demand_{i}.xlsx")

        lte_d = pd.read_excel(lte_file).values.squeeze()
        nr_d = pd.read_excel(nr_file).values.squeeze()

        lte_list.append(lte_d)
        nr_list.append(nr_d)

    return lte_list, nr_list


class SpectrumSharingEnv(gym.Env):
    def __init__(self, D_LTE, D_NR, Nr=60, zeta=0.5, history_len=5):
        super(SpectrumSharingEnv, self).__init__()
        self.D_LTE = D_LTE  # Demand vector for LTE
        self.D_NR = D_NR    # Demand vector for NR
        self.Nr = Nr        # Total number of PRBs
        self.zeta = zeta    # Weighting factor
        self.history_len = history_len
        self.t = history_len  # Start after initial history
        self.max_t = len(D_LTE)

        # Observation: flattened history of LTE and NR demands
        self.observation_space = gym.spaces.Box(
            low=0, high=Nr,
            shape=(history_len * 2,),
            dtype=np.float32
        )

        # Action: allocation to LTE and NR, continuous
        self.action_space = gym.spaces.Box(
            low=0.0, high=float(Nr),
            shape=(2,),
            dtype=np.float32
        )

    def reset(self):
        self.t = self.history_len
        return self._get_observation()

    def _get_observation(self):
        obs = []
        for i in range(self.history_len):
            obs.append(self.D_LTE[self.t - i - 1])
            obs.append(self.D_NR[self.t - i - 1])
        return np.array(obs[::-1], dtype=np.float32)

    def step(self, action):
        A_LTE = float(np.clip(action[0], 0, self.Nr))
        A_NR = float(np.clip(action[1], 0, self.Nr - A_LTE))

        D_L = self.D_LTE[self.t]
        D_N = self.D_NR[self.t]

        loss = self.zeta * ((A_LTE - D_L) / D_L)**2 + \
               (1 - self.zeta) * ((A_NR - D_N) / D_N)**2

        reward = -loss
        done = self.t >= self.max_t - 1
        self.t += 1
        obs = self._get_observation()
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass  # Can be extended for visualization

def train_td3_single(lte, nr, save_path, steps=10000):
    env = SpectrumSharingEnv(lte, nr, Nr=60, zeta=0.5, history=3)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MlpPolicy", env, action_noise=action_noise,
                verbose=0, tensorboard_log="./td3_tensorboard/")
    model.learn(total_timesteps=steps)
    model.save(save_path)
    return model