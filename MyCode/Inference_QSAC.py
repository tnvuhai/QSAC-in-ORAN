import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# Utilities
# -------------------------------
def load_nrb_series(filepath):
    if filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)['NRB'].values
    elif filepath.endswith('.tsv'):
        return pd.read_csv(filepath, sep='\t', engine='python')['NRB'].values
    else:
        return pd.read_csv(filepath)['NRB'].values


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# -------------------------------
# Config
# -------------------------------
# lte_path = "Data/LTE_Demand_32.xlsx"
# nr_path = "Data/NR_Demand_32.xlsx"
lte_path = "Data/LTE_Demand_hourly.tsv"
nr_path = "Data/NR_Demand_hourly.tsv"
gamma = 0.5
N_R = 60.0
max_steps = 100
arch = "qactor_classic_critic"  # choose: qactor_classic_critic, full_quantum, classical
ckpt = "Model/sac_qactor_actor_model.pth"
# arch = "classical"
# ckpt = "Model/sac_actor_model.pth"
# ckpt = "Model/td3_actor_model.pth"
device_str = "cuda" if torch.cuda.is_available() else "cpu"
NUM_STEPS = 100
save_plot = "sac_metrics_4panel_qsac.png"
save_csv = "sac_metrics_qsac.csv"
dump_model_info = True

# -------------------------------
# Build QSAC Agent
# -------------------------------
def build_qsac_agent(arch: str, state_dim: int, action_dim: int, max_action: float, device: torch.device):
    arch = arch.lower()
    print(arch)
    if arch in {"qactor_classic_critic", "quantum_actor_classic_critic", "qa_cc"}:
        from SAC_QonlyCritic_1 import SACAgent as QSACAgent
        agent = QSACAgent(state_dim, action_dim, max_action, device)
        return agent, "quantum_actor_classic_critic"
    elif arch in {"full_quantum", "fq"}:
        from SAC_quantum_full import SACAgent as QSACAgent
        agent = QSACAgent(state_dim, action_dim, max_action, device)
        return agent, "full_quantum"
    else:
        from _SAC import SACAgent as ClassicalSAC
        agent = ClassicalSAC(state_dim, action_dim, max_action, device)
        return agent, "classical"
        # from _TD3 import TD3Agent as ClassicalTD3
        # agent = ClassicalTD3(state_dim, action_dim, max_action, device)
        # return agent, "classical"

# -------------------------------
# Main
# -------------------------------
torch.set_default_dtype(torch.float32)
from MainEnv import ResourceAllocationEnv

lte_series = load_nrb_series(lte_path)
nr_series = load_nrb_series(nr_path)

env = ResourceAllocationEnv(lte_series, nr_series, gamma=gamma, N_R=N_R, max_steps=max_steps)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = 1.0

device = torch.device(device_str)

agent, resolved_arch = build_qsac_agent(arch, state_dim, action_dim, max_action, device)

if not os.path.isfile(ckpt):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
sd = torch.load(ckpt, map_location=device)
agent.actor.load_state_dict(sd)
agent.actor.eval()

if dump_model_info:
    os.makedirs("ModelName", exist_ok=True)
    info_path = os.path.join("ModelName", f"inference_qsac_model_info_{resolved_arch}.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"ARCH: {resolved_arch}\n\n")
        f.write("===== ACTOR ARCHITECTURE =====\n")
        f.write(str(agent.actor) + "\n\n")
        f.write(f"Actor params: {count_params(agent.actor):,}\n\n")

        f.write("===== CRITIC ARCHITECTURE =====\n")
        f.write(str(agent.critic) + "\n\n")
        f.write(f"Critic params: {count_params(agent.critic):,}\n\n")

        f.write("===== TOTAL TRAINABLE PARAMETERS =====\n")
        f.write(f"{count_params(agent.actor) + count_params(agent.critic):,}\n")
    print(f"[INFO] Model info written to {info_path}")

obs, _ = env.reset()

all_rewards = []
alloc_lte_list, alloc_nr_list = [], []
demand_lte_list, demand_nr_list = [], []
fairness_list = []

for step in range(NUM_STEPS):
    obs = np.asarray(obs, dtype=np.float32)
    action = agent.select_action(obs, evaluate=True)
    next_obs, reward, terminated, truncated, info = env.step(action)
    alloc_lte, alloc_nr = info["alloc"]
    demand_lte, demand_nr = info["demand"]
    fairness = (alloc_lte + alloc_nr)**2 / (2 * (alloc_lte**2 + alloc_nr**2 + 1e-8))

    obs = next_obs
    idx_dbg = info.get("t_idx", None)  # nếu env có field này
    if step < 5 or step % 20 == 0:
        print(f"[step={step}] idx={idx_dbg}, "
            f"obs0={float(obs[0]):.4f}, "
            f"action={action}, alloc={alloc_lte:.2f}/{alloc_nr:.2f}, "
            f"demand={demand_lte:.2f}/{demand_nr:.2f}")
    all_rewards.append(reward)
    alloc_lte_list.append(alloc_lte)
    alloc_nr_list.append(alloc_nr)
    demand_lte_list.append(demand_lte)
    demand_nr_list.append(demand_nr)
    fairness_list.append(fairness)

    if terminated or truncated:
        obs, _ = env.reset()
        break

metrics = pd.DataFrame({
    "step": np.arange(len(all_rewards)),
    "reward": all_rewards,
    "alloc_lte": alloc_lte_list,
    "alloc_nr": alloc_nr_list,
    "demand_lte": demand_lte_list,
    "demand_nr": demand_nr_list,
    "fairness": fairness_list,
})
metrics.to_csv(save_csv, index=False)
print(f"[INFO] Metrics saved to {save_csv}")

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
timesteps = list(range(len(all_rewards)))

axs[0, 0].plot(timesteps, all_rewards, label="Reward")
axs[0, 0].set_title("Reward per Step")
axs[0, 0].set_xlabel("Step")
axs[0, 0].set_ylabel("Reward")
axs[0, 0].grid(True)

axs[0, 1].plot(timesteps, alloc_lte_list, label="Alloc LTE", linestyle='--')
axs[0, 1].plot(timesteps, demand_lte_list, label="Demand LTE", linestyle='-')
axs[0, 1].plot(timesteps, alloc_nr_list, label="Alloc NR", linestyle='--')
axs[0, 1].plot(timesteps, demand_nr_list, label="Demand NR", linestyle='-')
axs[0, 1].set_title("Allocation vs Demand")
axs[0, 1].set_xlabel("Step")
axs[0, 1].set_ylabel("Resource")
axs[0, 1].legend()
axs[0, 1].grid(True)

surplus_lte = np.array(alloc_lte_list) - np.array(demand_lte_list)
surplus_nr = np.array(alloc_nr_list) - np.array(demand_nr_list)
axs[1, 0].plot(timesteps, surplus_lte, label="Surplus/Deficit LTE")
axs[1, 0].plot(timesteps, surplus_nr, label="Surplus/Deficit NR")
axs[1, 0].set_title("Surplus/Deficit over Time")
axs[1, 0].set_xlabel("Step")
axs[1, 0].set_ylabel("Surplus")
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].plot(timesteps, fairness_list, label="Fairness")
axs[1, 1].set_title("Jain's Fairness Index per Step")
axs[1, 1].set_xlabel("Step")
axs[1, 1].set_ylabel("Fairness")
axs[1, 1].set_ylim(0.997, 1.0)
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig(save_plot)
print(f"[INFO] Plot saved to {save_plot}")
plt.show()
