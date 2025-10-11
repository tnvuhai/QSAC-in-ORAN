# compare_four_models_plot.py
import os, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from Ultils import load_actor_from_checkpoint
from MainEnv import ResourceAllocationEnv, load_demand_data

# ===== Config =====
lte_path = "Data/LTE_Demand_32.xlsx"
nr_path  = "Data/NR_Demand_32.xlsx"

# lte_path = "Data/LTE_Demand_hourly.tsv"
# nr_path  = "Data/NR_Demand_hourly.tsv"
gamma, N_R, max_steps = 0.5, 60.0, 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "Inference_output"  # Thư mục để lưu tất cả các kết quả

CKPTS = {
    "QSAC": {"arch":"qactor_classic_critic", "ckpt":"Model/sac_qactor_actor_model.pth"},
    "SAC":  {"arch":"classical_sac",        "ckpt":"Model/sac_actor_model.pth"},
    "TD3":  {"arch":"classical_td3",        "ckpt":"Model/td3_actor_model.pth"},
    "DDPG": {"arch":"classical_ddpg",       "ckpt":"Model/ddpg_actor_model.pth"},
}

def build_agent(arch, state_dim, action_dim, max_action, device):
    arch = arch.lower()
    if arch.startswith("qactor"):
        from _QSAC import QSACAgent
        return QSACAgent(state_dim, action_dim, max_action, device)
    elif "sac" in arch:
        from _SAC import SACAgent;  return SACAgent(state_dim, action_dim, max_action, device)
    elif "td3" in arch:
        from _TD3 import TD3Agent;  return TD3Agent(state_dim, action_dim, max_action, device)
    elif "ddpg" in arch:
        from _DDPG import DDPGAgent; return DDPGAgent(state_dim, action_dim, max_action, device)
    else:
        raise ValueError(f"Unknown arch {arch}")

# ===== Load env once to read spaces =====
def load_nrb_series(filepath):
    if filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)['NRB'].values
    elif filepath.endswith('.tsv'):
        return pd.read_csv(filepath, sep='\t', engine='python')['NRB'].values
    else:
        return pd.read_csv(filepath)['NRB'].values

lte_series, nr_series = load_nrb_series(lte_path), load_nrb_series(nr_path)
_probe = ResourceAllocationEnv(lte_series, nr_series, gamma=gamma, N_R=N_R, max_steps=max_steps)
state_dim  = _probe.observation_space.shape[0]
action_dim = _probe.action_space.shape[0]
max_action = 1.0

# ===== Run one full episode (0..max_steps-1) and collect time-series =====
def run_model_timeseries(name, cfg):
    agent = build_agent(cfg["arch"], state_dim, action_dim, max_action, device)
    if not os.path.isfile(cfg["ckpt"]):
        raise FileNotFoundError(f"[{name}] checkpoint not found: {cfg['ckpt']}")
    load_actor_from_checkpoint(cfg["ckpt"], agent.actor, map_location=device)
    agent.actor.eval()

    env = ResourceAllocationEnv(lte_series, nr_series, gamma=gamma, N_R=N_R, max_steps=max_steps)
    obs, _ = env.reset()

    rows = []
    for step in range(max_steps):
        obs_np = np.asarray(obs, dtype=np.float32)
        action = agent.select_action(obs_np, evaluate=True)
        next_obs, reward, terminated, truncated, info = env.step(action)

        alloc_lte, alloc_nr   = info["alloc"]
        demand_lte, demand_nr = info["demand"]
        rows.append({
            "model": name, "step": step,
            "alloc_lte": float(alloc_lte), "alloc_nr": float(alloc_nr),
            "demand_lte": float(demand_lte), "demand_nr": float(demand_nr),
            "surplus_lte": float(alloc_lte - demand_lte),
            "surplus_nr":  float(alloc_nr  - demand_nr),
        })
        obs = next_obs
        if terminated or truncated:
            # We expect a full run from 0..99; if the env ends early, stop there.
            break

    return pd.DataFrame(rows)


def compute_metrics_from_timeseries(df: pd.DataFrame, N_R: float = 60.0,
                                    eps_list=(0.05, 0.10, 0.15)) -> pd.DataFrame:
    rows = []
    for m, sub in df.groupby("model"):
        sub = sub.sort_values("step").copy()

        # --- DSR (LTE/NR)
        dsr_lte = (np.minimum(sub["alloc_lte"].values, sub["demand_lte"].values) /
                   np.maximum(sub["demand_lte"].values, 1e-6))
        dsr_nr  = (np.minimum(sub["alloc_nr"].values,  sub["demand_nr"].values)  /
                   np.maximum(sub["demand_nr"].values,  1e-6))

        # --- Utilization
        util = (sub["alloc_lte"].values + sub["alloc_nr"].values) / max(N_R, 1e-6)

        # --- On-band rates
        ob = {}
        for eps in eps_list:
            ob[f"onband_lte_{int(eps*100)}%"] = np.mean(
                np.abs(sub["alloc_lte"].values - sub["demand_lte"].values)
                <= eps * np.maximum(sub["demand_lte"].values, 1e-6)
            )
            ob[f"onband_nr_{int(eps*100)}%"] = np.mean(
                np.abs(sub["alloc_nr"].values - sub["demand_nr"].values)
                <= eps * np.maximum(sub["demand_nr"].values, 1e-6)
            )

        row = {
            "model": m,
            "DSR_LTE_mean": float(np.mean(dsr_lte)),
            "DSR_NR_mean":  float(np.mean(dsr_nr)),
            "Util_mean":    float(np.mean(util)),
            **{k: float(v) for k, v in ob.items()},
            # (optional) add standard deviation for reference
            "DSR_LTE_std":  float(np.std(dsr_lte)),
            "DSR_NR_std":   float(np.std(dsr_nr)),
            "Util_std":     float(np.std(util)),
        }
        rows.append(row)
    return pd.DataFrame(rows)

# ===== Collect for all 4 models =====
dfs = []
for name, cfg in CKPTS.items():
    try:
        dfs.append(run_model_timeseries(name, cfg))
    except Exception as e:
        print(f"[ERROR] {name}: {e}")

if not dfs:
    raise SystemExit("[FATAL] No model ran successfully.")

df = pd.concat(dfs, ignore_index=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
timeseries_csv_path = os.path.join(OUTPUT_DIR, "compare_models_timeseries.csv")
df.to_csv(timeseries_csv_path, index=False)
print(f"[INFO] saved -> {timeseries_csv_path}")


# ====== Statistics =====
metrics_df = compute_metrics_from_timeseries(df, N_R=60.0, eps_list=(0.05,0.10,0.15))
metrics_csv_path = os.path.join(OUTPUT_DIR, "compare_models_metrics.csv")
metrics_xlsx_path = os.path.join(OUTPUT_DIR, "compare_models_metrics.xlsx")
metrics_df.to_csv(metrics_csv_path, index=False)
try:
    metrics_df.to_excel(metrics_xlsx_path, index=False)
except Exception as e:
    print("[WARN] cannot write xlsx:", e)
print(f"[INFO] saved metrics -> {metrics_csv_path} (and .xlsx)")
print(metrics_df)


# ====== PLOTS (6 subplots) ======
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.ravel()

# ---- 1–4: Allocation vs Demand (time-series) for each model ----
model_order = ["QSAC", "SAC", "DDPG", "TD3"]
for k, m in enumerate(model_order):
    sub = df[df["model"] == m].sort_values("step")
    ax = axes[k]
    x = sub["step"].values

    # LTE
    ax.plot(x, sub["alloc_lte"].values,  linestyle="--", marker=None, label="Alloc LTE")
    ax.plot(x, sub["demand_lte"].values, linestyle="--", marker=None, label="Demand LTE")
    # NR
    ax.plot(x, sub["alloc_nr"].values,   linestyle="-",  marker=None, label="Alloc NR")
    ax.plot(x, sub["demand_nr"].values,  linestyle="-",  marker=None, label="Demand NR")
    ax.set_title(f"{m}: Allocation vs Demand", fontsize=17)
    ax.set_xlabel("Step", fontsize=17); ax.set_ylabel("PRBs", fontsize=17)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    #ax.set_ylim(0, 40)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

# ---- 5: Surplus/Deficit LTE over time (4 lines - one for each model) ----
ax5 = axes[4]
for m in model_order:
    sub = df[df["model"] == m].sort_values("step")
    ax5.plot(sub["step"], sub["surplus_lte"], linestyle="--", label=m)
ax5.set_title("Surplus/Deficit over Time (LTE)", fontsize=17)
ax5.set_xlabel("Step",fontsize=17); ax5.set_ylabel("Surplus/Dificit",fontsize=17)
ax5.tick_params(axis='x', labelsize=10)
ax5.tick_params(axis='y', labelsize=10)
ax5.grid(True, alpha=0.3); ax5.legend()

# ---- 6: Surplus/Deficit NR over time (4 lines - one for each model) ----
ax6 = axes[5]
for m in model_order:
    sub = df[df["model"] == m].sort_values("step")
    ax6.plot(sub["step"], sub["surplus_nr"], linestyle="-", label=m)
ax6.set_title("Surplus/Deficit over Time (NR)", fontsize=17)
ax6.set_xlabel("Step",fontsize=17); ax6.set_ylabel("Surplus/Dificit",fontsize=17)
ax6.tick_params(axis='x', labelsize=12)
ax6.tick_params(axis='y', labelsize=12)
ax6.grid(True, alpha=0.3); ax6.legend()

plt.tight_layout()
out_png = os.path.join(OUTPUT_DIR, "compare_models_6subplots_timeseries.pdf")
plt.savefig(out_png, dpi=300)
print(f"[INFO] plot saved -> {out_png}")
plt.show()
