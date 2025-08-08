import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from _SAC import SACAgent  # ensure import path is correct
from MainEnv import ResourceAllocationEnv

# -------------------------------
# Load Data
# -------------------------------
def load_nrb_series(filepath):
    if filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)['NRB'].values
    elif filepath.endswith('.tsv'):
        return pd.read_csv(filepath, sep='\t', engine='python')['NRB'].values
    else:
        return pd.read_csv(filepath)['NRB'].values

lte_file = load_nrb_series("Data/LTE_Demand_32.xlsx")
nr_file = load_nrb_series("Data/NR_Demand_32.xlsx")

# -------------------------------
# Load Env & Model
# -------------------------------
env = ResourceAllocationEnv(lte_file, nr_file, gamma=0.5, N_R=60.0, max_steps=100)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = 1.00


# -------------------------------
# Load Trained SAC Agent
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = SACAgent(state_dim, action_dim, max_action, device)
agent.actor.load_state_dict(torch.load("Model/sac_actor_model.pth", map_location=device))
agent.actor.eval()

# -------------------------------
# Inference Loop
# -------------------------------
obs, _ = env.reset()

NUM_STEPS = 100
all_rewards = []
alloc_lte_list = []
alloc_nr_list = []
demand_lte_list = []
demand_nr_list = []
fairness_list = []

for step in range(NUM_STEPS):
    action = agent.select_action(obs, evaluate=True)
    next_obs, reward, terminated, truncated, info = env.step(action)

    alloc_lte, alloc_nr = info["alloc"]
    demand_lte, demand_nr = info["demand"]
    fairness = (alloc_lte + alloc_nr)**2 / (2 * (alloc_lte**2 + alloc_nr**2 + 1e-8))

    obs = next_obs

    all_rewards.append(reward)
    alloc_lte_list.append(alloc_lte)
    alloc_nr_list.append(alloc_nr)
    demand_lte_list.append(demand_lte)
    demand_nr_list.append(demand_nr)
    fairness_list.append(fairness)

    if terminated or truncated:
        break

# -------------------------------
# Plot 4-panel metrics
# -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
timesteps = list(range(len(all_rewards)))

# Reward per step
axs[0, 0].plot(timesteps, all_rewards, label="Reward", color="blue")
axs[0, 0].set_title("Reward per Step")
axs[0, 0].set_xlabel("Step")
axs[0, 0].set_ylabel("Reward")
axs[0, 0].grid(True)

# Allocation vs Demand
axs[0, 1].plot(timesteps, alloc_lte_list, label="Alloc LTE", linestyle='--', color="steelblue")
axs[0, 1].plot(timesteps, demand_lte_list, label="Demand LTE", linestyle='-', color="orange")
axs[0, 1].plot(timesteps, alloc_nr_list, label="Alloc NR", linestyle='--', color="firebrick")
axs[0, 1].plot(timesteps, demand_nr_list, label="Demand NR", linestyle='-', color="green")
axs[0, 1].set_title("Allocation vs Demand")
axs[0, 1].set_xlabel("Step")
axs[0, 1].set_ylabel("Resource")
axs[0, 1].legend()
axs[0, 1].grid(True)

# Surplus/Deficit per network
surplus_lte = np.array(alloc_lte_list) - np.array(demand_lte_list)
surplus_nr = np.array(alloc_nr_list) - np.array(demand_nr_list)
axs[1, 0].plot(timesteps, surplus_lte, label="Surplus/Deficit LTE", color="dodgerblue")
axs[1, 0].plot(timesteps, surplus_nr, label="Surplus/Deficit NR", color="darkorange")
axs[1, 0].set_title("Surplus/Deficit over Time")
axs[1, 0].set_xlabel("Step")
axs[1, 0].set_ylabel("Surplus")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Jain's Fairness
axs[1, 1].plot(timesteps, fairness_list, label="Fairness", color="green")
axs[1, 1].set_title("Jain's Fairness Index per Step")
axs[1, 1].set_xlabel("Step")
axs[1, 1].set_ylabel("Fairness")
axs[1, 1].set_ylim(0.997, 1.0)
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig("sac_metrics_4panel.png")
plt.show()
