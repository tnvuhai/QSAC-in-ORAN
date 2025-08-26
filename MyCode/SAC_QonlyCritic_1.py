import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt
import pandas as pd
from Ultils import * 

# ========= Quantum deps =========
import pennylane as qml
from typing import Tuple
torch.set_default_dtype(torch.float32)
from MainEnv import ResourceAllocationEnv, load_demand_data

# =============================================================
# 0) ReUploading VQC block (keeps your principle exactly)
#    - StronglyEntanglingLayers + AngleEmbedding(inputs * embedding_weights)
#      repeated qnn_layers times, plus a final entangling layer
#    - Returns expvals of PauliZ on each qubit (size = n_qubits)
# =============================================================
class ReUploadingVQC(nn.Module):
    def __init__(self, n_qubits: int, qnn_layers: int, device_name: str = "lightning.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.qnn_layers = qnn_layers
        self.dev = qml.device(device_name, wires=n_qubits, shots=None)

        # Learnable circuit params
        ent_shape_single = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=n_qubits)
        self.entangling_weights = nn.Parameter(
            torch.randn((qnn_layers + 1,) + ent_shape_single) * 0.05
        )
        self.embedding_weights = nn.Parameter(torch.randn(qnn_layers, n_qubits) * 0.05)

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(inputs, ent_w, emb_w):
            # Prepare |0...0| (use BasisState for modern PL; matches your BasisStatePreparation intent)
            zero_state = torch.zeros(self.n_qubits, dtype=torch.int64)
            qml.BasisState(zero_state, wires=range(self.n_qubits))
            for i in range(self.qnn_layers):
                qml.StronglyEntanglingLayers(ent_w[i], wires=range(self.n_qubits))
                features = inputs * emb_w[i]
                qml.AngleEmbedding(features=features, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(ent_w[-1], wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        self.circuit = circuit

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.stack(self.circuit(x, self.entangling_weights, self.embedding_weights))
        return out.to(x.dtype)   # <— ép về cùng dtype với input

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        if x_batch.dim() == 1:
            return self.forward_single(x_batch)
        outs = [self.forward_single(x_batch[i]) for i in range(x_batch.shape[0])]
        out = torch.stack(outs, dim=0)
        return out.to(x_batch.dtype)


# =============================================================
# 1) Quantum Actor (hybrid): state -> (fit_to_qubits) -> VQC -> Linear heads
#    - No MLP feature extractor; only Linear heads for mean/log_std
#    - Preserves your data re-uploading inside the VQC
# =============================================================
class QuantumActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 n_qubits: int = 6, qnn_layers: int = 2, device_name: str = "lightning.qubit"):
        super().__init__()
        self.max_action = max_action
        self.n_qubits = n_qubits
        self.qnn_layers = qnn_layers
        self.state_dim = state_dim

        self.vqc = ReUploadingVQC(n_qubits=n_qubits, qnn_layers=qnn_layers, device_name=device_name)
        # Small linear heads to map quantum features -> outputs
        self.mean_head = nn.Linear(n_qubits, action_dim)
        self.log_std_head = nn.Linear(n_qubits, action_dim)

    @staticmethod
    def _fit_to_qubits(x: torch.Tensor, n_qubits: int) -> torch.Tensor:
        # Non-trainable fit: tile/crop to n_qubits
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
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(torch.float32)     # <— đảm bảo float32
        x = self._fit_to_qubits(state, self.n_qubits)
        x = torch.tanh(x)
        feats = self.vqc(x)                 # giờ feats sẽ là float32
        mean = self.mean_head(feats)
        log_std = self.log_std_head(feats).clamp(-20, 2)
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


# =============================================================
# 2) Classical Critic (same as _SAC.py)
# =============================================================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)


# =============================================================
# 3) Replay Buffer (unchanged)
# =============================================================
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


# =============================================================
# 4) SAC Agent (quantum actor + classical critics)
# =============================================================
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, device,
                 n_qubits: int = 6, qnn_layers: int = 2, q_device_name: str = "lightning.qubit",
                 actor_lr: float = 1e-4, critic_lr: float = 2e-3, alpha_lr: float = 3e-4):
        self.device = device

        self.actor = QuantumActor(state_dim, action_dim, max_action,
                                   n_qubits=n_qubits, qnn_layers=qnn_layers, device_name=q_device_name).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
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


# =============================================================
# 5) Training Loop (same structure as _SAC.py)
# =============================================================
if __name__ == '__main__':
    # Load data & env
    lte_demand = load_demand_data("Data/LTE_Demand_hourly.tsv")
    nr_demand = load_demand_data("Data/NR_Demand_hourly.tsv")
    env = ResourceAllocationEnv(lte_demand, nr_demand)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Quantum hyperparams for actor
    n_qubits = 6      # reduce for speed if needed
    qnn_layers = 2
    q_device_name = "lightning.qubit"  # switch to "lightning.qubit" if installed for speed

    agent = SACAgent(state_dim, action_dim, max_action, device,
                     n_qubits=n_qubits, qnn_layers=qnn_layers, q_device_name=q_device_name,
                     actor_lr=1e-4, critic_lr=2e-3, alpha_lr=3e-4)
    with open("ModelInfo/model_info_QSAC_6qubits_2layers.txt", "w", encoding="utf-8") as f:
        f.write("===== ACTOR ARCHITECTURE =====\n")
        f.write(str(agent.actor) + "\n\n")
        actor_params = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
        f.write(f"Actor total parameters: {actor_params:,}\n\n")

        f.write("===== CRITIC ARCHITECTURE =====\n")
        f.write(str(agent.critic) + "\n\n")
        critic_params = sum(p.numel() for p in agent.critic.parameters() if p.requires_grad)
        f.write(f"Critic total parameters: {critic_params:,}\n\n")

        total_params = actor_params + critic_params
        f.write(f"===== TOTAL TRAINABLE PARAMETERS =====\n{total_params:,}\n")

    buffer = ReplayBuffer()

    episodes = 20000
    batch_size = 48
    rewards = []
    logs = []
    resume_path = None#"Checkpoint/sac_qsac_ep40.pth"
    if resume_path:
        start_ep = load_checkpoint(agent, buffer, resume_path, device=device)
    else:
        start_ep = 0

    # If you want to start
    #start_ep = load_checkpoint(agent, buffer, "Checkpoint/sac_qsac_ep5000.pth", device=device)
    for ep in range(episodes):

        state, _ = env.reset()
        total_reward = 0.00
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            if len(buffer.buffer) >= batch_size:
                agent.train(buffer, batch_size)

        rewards.append(total_reward)
        alloc_lte, alloc_nr = info["alloc"]
        demand_lte, demand_nr = info["demand"]
        logs.append({"episode": ep, "reward": total_reward, "alloc_lte":alloc_lte, "alloc_nr": alloc_nr, "demand_lte":demand_lte, "demand_nr":demand_nr})
        print(f"Episode {ep}, Reward: {total_reward:.2f}, | Demand=({demand_lte:.2f}, {demand_nr :.2f}) "
        f"| Alloc=({alloc_lte:.2f}, {alloc_nr:.2f})", flush=True)

        if (ep + 1) % 5000 == 0:
            save_checkpoint(agent, buffer, ep, f"Checkpoint/sac_qsac_ep{ep+1}.pth")
        

    # Save & plot
    torch.save(agent.actor.state_dict(), "Model/sac_qactor_actor_model.pth")
    pd.DataFrame(logs).to_csv("Result/sac_qactor_training_log.csv", index=False)
    smoothed_rewards = pd.Series(rewards).rolling(window=100, min_periods=1).mean()
    plt.plot(smoothed_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("SAC (Quantum Actor + Classic Critics) Training Reward")
    plt.grid(True)
    plt.savefig("FigureReward/sac_qactor_training_reward_plot.png")
    plt.show()
