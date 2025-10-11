#!/usr/bin/env python3
# plot_compare_rl_fixed.py
# Read 4 CSVs (TD3, DDPG, SAC, QSAC) with 2 columns: episode, reward and draw a line chart

import os
import pandas as pd
import matplotlib.pyplot as plt

# ====== CONFIG ======
# Directory containing CSVs (priority "Result", fallback "result")
DIR_CANDIDATES = ["FigureReward"]

FILES = {
    "TD3":  "td3_training_log.csv",
    "DDPG": "ddpg_training_log.csv",
    "SAC":  "sac_training_log.csv",
    "QSAC": "sac_qactor_training_log.csv",
}

EPISODE_COL = "episode"
REWARD_COL  = "reward"

SMOOTHING_WINDOW = 100   # 0 or 1 = no smoothing; e.g., 10 for MA(10) smoothing
TRUNCATE_MIN_LENGTH = False
TITLE  = "TD3 vs DDPG vs SAC vs QSAC — Episode Reward"
XLABEL = "Episode"
YLABEL = "Reward"
OUTPUT_DIR = "FigureReward"  # Thư mục để lưu biểu đồ
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rl_compare.pdf")  # change to None to only display

# ====== HELPERS ======
def resolve_dir():
    for d in DIR_CANDIDATES:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"No directory found in {DIR_CANDIDATES}")

def load_xy(path):
    df = pd.read_csv(path)  # 2 columns: episode,reward (comma)
    # ensure correct types and sorting
    df = df[[EPISODE_COL, REWARD_COL]].dropna()
    df[EPISODE_COL] = pd.to_numeric(df[EPISODE_COL], errors="coerce")
    df[REWARD_COL]  = pd.to_numeric(df[REWARD_COL], errors="coerce")
    df = df.dropna().sort_values(EPISODE_COL).reset_index(drop=True)

    y = df[REWARD_COL]
    if SMOOTHING_WINDOW and SMOOTHING_WINDOW > 1:
        y = y.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    return df[EPISODE_COL], y

# ====== MAIN ======
def main():
    base_dir = resolve_dir()

    series = []
    for label, fname in FILES.items():
        fpath = os.path.join(base_dir, fname)
        if not os.path.isfile(fpath):
            print(f"[WARN] Skipping {label}: file not found {fpath}")
            continue
        x, y = load_xy(fpath)
        series.append((label, x, y))

    if not series:
        raise RuntimeError("No lines to draw. Check file names/paths.")

    if TRUNCATE_MIN_LENGTH:
        min_len = min(len(y) for _, _, y in series)
        series = [(label, x.iloc[:min_len], y.iloc[:min_len]) for (label, x, y) in series]

    plt.figure(figsize=(10, 6))
    for label, x, y in series:
        if len(x) == len(y):
            plt.plot(x, y, label=label)
        else:
            # fallback if lengths differ
            plt.plot(range(len(y)), y, label=label)

    plt.title(TITLE, fontsize=16)
    max_x = max(len(y) for _, _, y in series)
    plt.xticks(range(0, max_x+1, 5000), fontsize=17)
    plt.yticks(fontsize=17)
    plt.xlabel(XLABEL, fontsize=17)
    plt.ylabel(YLABEL, fontsize=17)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=17)
    plt.tight_layout()

    if OUTPUT_FILE:
        out_dir = os.path.dirname(OUTPUT_FILE)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(OUTPUT_FILE, dpi=200)
        print(f"Saved figure to: {OUTPUT_FILE}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
