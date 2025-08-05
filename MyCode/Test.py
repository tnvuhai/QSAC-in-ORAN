import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from gym import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

# ==== Load all LTE and NR demand files ====
def load_all_demand(folder_path):
    lte_list = []
    nr_list = []
    for i in range(1, 50):
        lte_file = os.path.join(folder_path, f"LTE_Demand_{i}.xlsx")
        nr_file = os.path.join(folder_path, f"NR_Demand_{i}.xlsx")

        lte_d = pd.read_excel(lte_file).values.squeeze()
        nr_d = pd.read_excel(nr_file).values.squeeze()

        lte_list.append(lte_d)
        nr_list.append(nr_d)

    return lte_list, nr_list

# ==== Environment for Spectrum Sharing ====
class SpectrumEnv(gym.Env):
    def __init__(self, lte_demand, nr_demand, Nr=60, zeta=0.5, history=3):
        super(SpectrumEnv, self).__init__()
        assert len(lte_demand) == len(nr_demand)
        self.lte = lte_demand
        self.nr = nr_demand
        self.Nr = Nr
        self.zeta = zeta
        self.history = history
        self.t = history
        self.T = len(lte_demand)

        self.observation_space = spaces.Box(low=0, high=Nr,
                                            shape=(history * 2,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=float(Nr),
                                       shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.history
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        obs = []
        for i in range(self.t - self.history, self.t):
            val_lte = float(np.ravel(self.lte[i])[0])
            val_nr = float(np.ravel(self.nr[i])[0])
            obs.append(val_lte)
            obs.append(val_nr)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        A_LTE = float(np.clip(action[0], 0, self.Nr))
        A_NR = float(np.clip(action[1], 0, self.Nr - A_LTE))

        D_L = float(np.ravel(self.lte[self.t])[0])
        D_N = float(np.ravel(self.nr[self.t])[0])

        loss = self.zeta * ((A_LTE - D_L) / D_L)**2 + (1 - self.zeta) * ((A_NR - D_N) / D_N)**2
        reward = float(-loss)

        self.t += 1
        done = self.t >= self.T
        terminated = self.t >= self.T
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        pass

# ==== Train TD3 model for one pair ====
def train_td3_single(lte, nr, save_path, steps=20000):
    env = SpectrumEnv(lte, nr, Nr=60, zeta=0.5, history=3)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MlpPolicy", env, action_noise=action_noise,
                verbose=0, tensorboard_log="./td3_tensorboard/")
    model.learn(total_timesteps=steps)
    model.save(save_path)
    return model

# ==== Evaluate trained model ====
def evaluate_model(model_path, lte_demand, nr_demand, Nr=60, zeta=0.5, history=3):
    model = TD3.load(model_path)
    T = len(lte_demand)
    surplus_LTE, surplus_NR = [], []
    alloc_LTE, alloc_NR = [], []

    obs = []
    for i in range(history):
        lte_val = float(np.ravel(lte_demand[i])[0])
        nr_val = float(np.ravel(nr_demand[i])[0])
        obs.extend([lte_val, nr_val])
    obs = np.array(obs, dtype=np.float32)

    for t in range(history, T):
        action, _ = model.predict(obs, deterministic=True)
        A_LTE = np.clip(action[0], 0, Nr)
        A_NR = np.clip(action[1], 0, Nr - A_LTE)

        D_L, D_N = lte_demand[t], nr_demand[t]

        
        surplus_LTE.append((A_LTE - D_L) / D_L)
        surplus_NR.append((A_NR - D_N) / D_N)

        alloc_LTE.append(A_LTE)
        alloc_NR.append(A_NR)

        obs = []
        for i in range(t - history + 1, t + 1):
            lte_val = float(np.ravel(lte_demand[i])[0])
            nr_val = float(np.ravel(nr_demand[i])[0])
            obs.extend([lte_val, nr_val])
        obs = np.array(obs, dtype=np.float32)

    fairness = np.mean((np.array(alloc_LTE) + np.array(alloc_NR))**2 /
                       (2 * (np.array(alloc_LTE)**2 + np.array(alloc_NR)**2 + 1e-8)))

    return np.mean(surplus_LTE), np.mean(surplus_NR), fairness

# ==== Main workflow ====
def main():
    folder = "./Data/"
    save_dir = "./TD3_models/"
    os.makedirs(save_dir, exist_ok=True)

    lte_all, nr_all = load_all_demand(folder)

    all_metrics = []

    for i in range(2):
        print(f"🔁 Training on pair {i+1}")
        lte = lte_all[i]
        nr = nr_all[i]
        save_path = os.path.join(save_dir, f"TD3_model_{i+1}")
        model = train_td3_single(lte, nr, save_path, steps=20000)

        surplus_L, surplus_N, fair = evaluate_model(save_path, lte, nr)
        print(f"✅ Model {i+1}: Surplus LTE={surplus_L:.3f}, NR={surplus_N:.3f}, Fairness={fair:.3f}")
        all_metrics.append((i+1, surplus_L, surplus_N, fair))

    df = pd.DataFrame(all_metrics, columns=["Model", "Surplus_LTE", "Surplus_NR", "Fairness"])
    df.to_csv("TD3_Evaluation_Metrics.csv", index=False)
    print("📊 Evaluation metrics saved to TD3_Evaluation_Metrics.csv")

if __name__ == "__main__":
    main()