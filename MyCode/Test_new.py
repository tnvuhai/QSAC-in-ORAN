import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt
import pandas as pd

# ===============================
# FULL-QUANTUM SAC (no MLP layers)
# - Actor & Critic are built only from variational quantum circuits (VQCs)
# - Keeping your data re-uploading principle exactly:
#     [StronglyEntanglingLayers] -> [AngleEmbedding(inputs * emb_w[i])]
#   repeated qnn_layers times, then a final entangling block, then quantum-only readouts.
# - Readouts are implemented by adding trainable post-rotations and measuring PauliZ
#   (no classical Linear layers / MLPs to map features to outputs).
# ===============================

import pennylane as qml
from typing import Tuple

from MainEnv import ResourceAllocationEnv, load_demand_data

# ---------------------------------------------
# 0) Core ReUploading VQC (shared building block)
# ---------------------------------------------
class ReUploadingVQC(nn.Module):
    def __init__(self, n_qubits: int, qnn_layers: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.qnn_layers = qnn_layers
        self.dev = qml.device("default.qubit.torch", wires=n_qubits, shots=None)

        # Trainable circuit weights
        ent_shape_single = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=n_qubits)
        self.entangling_weights = nn.Parameter(
            torch.randn((qnn_layers + 1,) + ent_shape_single) * 0.05
        )
        self.embedding_weights = nn.Parameter(torch.randn(qnn_layers, n_qubits) * 0.05)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def core(inputs, ent_w, emb_w):
            zero_state = torch.zeros(self.n_qubits, dtype=torch.int64)
            qml.BasisState(zero_state, wires=range(self.n_qubits))
            for i in range(self.qnn_layers):
                qml.StronglyEntanglingLayers(ent_w[i], wires=range(self.n_qubits))
                features = inputs * emb_w[i]
                qml.AngleEmbedding(features=features, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(ent_w[-1], wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        self.core = core

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(self.core(x, self.entangling_weights, self.embedding_weights))

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        if x_batch.dim() == 1:
            return self.forward_single(x_batch)
        outs = [self.forward_single(x_batch[i]) for i in range(x_batch.shape[0])]
        return torch.stack(outs, dim=0)


# -----------------------------------------------------
# 1) Quantum-only Readout Blocks (no classical Linear)
#    - We create K quantum "heads" by adding trainable post-rotations
#      then measuring PauliZ on a designated wire.
#    - All parameters are quantum circuit params; outputs come directly
#      from quantum expectation values.
# -----------------------------------------------------
class QuantumReadout(nn.Module):
    """Generic quantum readout that maps an n_qubits state to M outputs via
    head-specific post-rotations + Z measurements.

    Each head j has its own post-rotations (RX/RY/RZ per wire). This keeps everything
    inside the quantum graph; there is no classical linear/MLP mapping.
    """
    def __init__(self, n_qubits: int, n_heads: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_heads = n_heads
        self.dev = qml.device("default.qubit.torch", wires=n_qubits, shots=None)

        # Post-rotation parameters per head and per wire
        # shape: [n_heads, n_qubits, 3] for RX/RY/RZ angles
        self.post_angles = nn.Parameter(torch.randn(n_heads, n_qubits, 3) * 0.05)

        # We build one qnode that applies given post-angles for a single head
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def head_measure(prep_expvals, angles):
            # Re-prepare state from expvals is not directly possible; instead,
            # we reconstruct by using a small data-embedding layer so the readout is quantum-only.
            # We encode the incoming "prep_expvals" as angles on each qubit.
            zero_state = torch.zeros(self.n_qubits, dtype=torch.int64)
            qml.BasisState(zero_state, wires=range(self.n_qubits))
            qml.AngleEmbedding(features=prep_expvals, wires=range(self.n_qubits))
            # Apply head-specific post-rotations per wire
            for w in range(self.n_qubits):
                qml.RX(angles[w, 0], wires=w)
                qml.RY(angles[w, 1], wires=w)
                qml.RZ(angles[w, 2], wires=w)
            # Measure Z on wire 0..(n_qubits-1), then average to get a single scalar output
            # (Alternative: choose a fixed wire or a trainable selection; we average for stability.)
            expvals = [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]
            return tuple(expvals)

        self.head_measure = head_measure

    def forward(self, prep_expvals: torch.Tensor) -> torch.Tensor:
        """prep_expvals: [B, n_qubits] -> outputs [B, n_heads]
        We run a lightweight quantum readout per head using the same encoded prep_expvals.
        """
        if prep_expvals.dim() == 1:
            prep_expvals = prep_expvals.unsqueeze(0)
        B = prep_expvals.shape[0]
        outs = []
        for j in range(self.n_heads):
            angles = self.post_angles[j]
            # loop over batch
            head_out = []
            for i in range(B):
                exp_list = self.head_measure(prep_expvals[i], angles)
                # average across wires to produce one scalar output per head
                head_out.append(torch.mean(torch.stack(exp_list)))
            outs.append(torch.stack(head_out))  # [B]
        return torch.stack(outs, dim=1)  # [B, n_heads]


# ---------------------------------------------
# 2) Quantum Actor (FULL quantum)
#    - Input state is encoded directly (no linear layer)
#    - ReUploadingVQC produces n_qubits expvals
#    - Two quantum readout heads produce mean and log_std (action_dim each)
# ---------------------------------------------
class QuantumActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 n_qubits: int = 8, qnn_layers: int = 2):
        super().__init__()
        self.max_action = max_action
        self.n_qubits = n_qubits
        self.qnn_layers = qnn_layers

        # If state_dim != n_qubits, we tile/crop to match qubits purely as data-prep (no trainable MLP)
        self.state_dim = state_dim

        self.vqc = ReUploadingVQC(n_qubits=n_qubits, qnn_layers=qnn_layers)
        # Two quantum readouts: one for mean, one for log_std. Each returns action_dim scalars.
        self.readout_mean = QuantumReadout(n_qubits=n_qubits, n_heads=action_dim)
        self.readout_logstd = QuantumReadout(n_qubits=n_qubits, n_heads=action_dim)

    @staticmethod
    def _fit_to_qubits(x: torch.Tensor, n_qubits: int) -> torch.Tensor:
        # x shape: [B, state_dim] -> [B, n_qubits] via tile/crop (no trainable classical layer)
        B, D = x.shape
        if D == n_qubits:
            return x
        if D < n_qubits:
            reps = (n_qubits + D - 1) // D
            x_rep = x.repeat(1, reps)
            return x_rep[:, :n_qubits]
        else:
            return x[:, :n_qubits]

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._fit_to_qubits(state, self.n_qubits)
        x = torch.tanh(x)  # squash angles to a stable range (non-trainable, not an MLP)
        feats = self.vqc(x)               # [B, n_qubits]
        mean = self.readout_mean(feats)   # [B, action_dim]
        log_std = self.readout_logstd(feats).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


# ---------------------------------------------
# 3) Quantum Critic (FULL quantum twin critics)
#    - Input: concat(state, action) -> fit to n_qubits (tile/crop) -> VQC
#    - Quantum readout produces a single scalar Q for each twin
# ---------------------------------------------
class QuantumCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, n_qubits: int = 8, qnn_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.qnn_layers = qnn_layers

        self.vqc1 = ReUploadingVQC(n_qubits=n_qubits, qnn_layers=qnn_layers)
        self.vqc2 = ReUploadingVQC(n_qubits=n_qubits, qnn_layers=qnn_layers)
        self.readout1 = QuantumReadout(n_qubits=n_qubits, n_heads=1)
        self.readout2 = QuantumReadout(n_qubits=n_qubits, n_heads=1)

        self.state_dim = state_dim
        self.action_dim = action_dim

    @staticmethod
    def _fit_to_qubits(x: torch.Tensor, n_qubits: int) -> torch.Tensor:
        B, D = x.shape
        if D == n_qubits:
            return x
        if D < n_qubits:
            reps = (n_qubits + D - 1) // D
            x_rep = x.repeat(1, reps)
            return x_rep[:, :n_qubits]
        else:
            return x[:, :n_qubits]

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = self._fit_to_qubits(x, self.n_qubits)
        x = torch.tanh(x)
        feats = self.vqc1(x)
        q = self.readout1(feats)  # [B, 1]
        return q

    def q2(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = self._fit_to_qubits(x, self.n_qubits)
        x = torch.tanh(x)
        feats = self.vqc2(x)
        q = self.readout2(feats)  # [B, 1]
        return q

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return self.q1(state, action), self.q2(state, action)


# ---------------------------------------------
# 4) Replay Buffer (unchanged)
# ---------------------------------------------
class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(np.array(a), dtype=torch.float32)
        r = torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(1)
        s_ = torch.tensor(np.array(s_), dtype=torch.float32)
        d = torch.tensor(np.array(d), dtype=torch.float32).unsqueeze(1)
        return s, a, r, s_, d


# ---------------------------------------------
# 5) SAC Agent (same losses/updates, quantum nets inside)
# ---------------------------------------------
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, device,
                 n_qubits: int = 8, qnn_layers: int = 2,
                 actor_lr: float = 1e-4, critic_lr: float = 2e-3, alpha_lr: float = 3e-4):
        self.device = device
        self.actor = QuantumActor(state_dim, action_dim, max_action, n_qubits, qnn_layers).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = QuantumCritic(state_dim, action_dim, n_qubits, qnn_layers).to(device)
        self.critic_target = QuantumCritic(state_dim, action_dim, n_qubits, qnn_layers).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -action_dim

        self.max_action = max_action

        self.noise_std = 0.09
        self.noise_std_min = 0.0

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.max_action
        else:
            action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=48, gamma=1.0, tau=1e-4):
        s, a, r, s_, d = replay_buffer.sample(batch_size)
        s, a, r, s_, d = s.to(self.device), a.to(self.device), r.to(self.device), s_.to(self.device), d.to(self.device)

        with torch.no_grad():
            a_, log_pi = self.actor.sample(s_)
            q1_, q2_ = self.critic_target(s_, a_)
            q_target = r + (1 - d) * gamma * (torch.min(q1_, q2_) - torch.exp(self.log_alpha) * log_pi)

        q1, q2 = self.critic(s, a)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        a_pi, log_pi = self.actor.sample(s)
        q1_pi, q2_pi = self.critic(s, a_pi)
        actor_loss = (torch.exp(self.log_alpha) * log_pi - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# ---------------------------------------------
# 6) Training Loop (example)
# ---------------------------------------------
if __name__ == '__main__':
    lte_demand = load_demand_data("Data/LTE_Demand_hourly.tsv")
    nr_demand = load_demand_data("Data/NR_Demand_hourly.tsv")
    env = ResourceAllocationEnv(lte_demand, nr_demand)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_qubits = 4
    qnn_layers = 2

    agent = SACAgent(state_dim, action_dim, max_action, device,
                     n_qubits=n_qubits, qnn_layers=qnn_layers,
                     actor_lr=1e-4, critic_lr=2e-3, alpha_lr=3e-4)

    buffer = ReplayBuffer()

    episodes = 20000
    batch_size = 16
    rewards = []
    logs = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            if len(buffer.buffer) > batch_size:
                agent.train(buffer, batch_size)

        rewards.append(total_reward)
        logs.append({"episode": ep, "reward": total_reward})
        print(f"Episode {ep}, Reward: {total_reward:.2f}")

    torch.save(agent.actor.state_dict(), "Model/sac_q_fullquant_actor.pth")
    pd.DataFrame(logs).to_csv("Result/sac_q_fullquant_log.csv", index=False)
    smoothed_rewards = pd.Series(rewards).rolling(window=100, min_periods=1).mean()
    plt.plot(smoothed_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Full-Quantum SAC Training Reward")
    plt.grid(True)
    plt.savefig("FigureReward/sac_q_fullquant_plot.png")
    plt.show()
