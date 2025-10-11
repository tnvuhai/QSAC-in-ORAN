import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
from MainEnv import ResourceAllocationEnv, load_demand_data # Reuse
# =============================
# 1. Actor-Critic Networks
# =============================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() 
        y_t = torch.tanh(x_t)
        a01 = 0.5 * (y_t + 1.0)
        action = a01 * self.max_action

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


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


# =============================
# 2. Replay Buffer
# =============================
class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        s, a, r, s_, d = zip(*batch)
        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(np.array(a), dtype=torch.float32)
        r = torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(1)
        s_ = torch.tensor(np.array(s_), dtype=torch.float32)
        d = torch.tensor(np.array(d), dtype=torch.float32).unsqueeze(1)
        return s, a, r, s_, d


# =============================
# 3. SAC Agent
# =============================
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=2e-3)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim

        self.max_action = max_action
        self.device = device

        self.policy_freq = 2
        self.total_it = 0

        # Noise parameters (Gaussian)
        self.noise_std = 0.09
        self.noise_std_min = 0.0

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            mean, _ = self.actor(state)
            y = torch.tanh(mean)           # [-1,1]
            a01 = 0.5 * (y + 1.0)          # [0,1]
            action = a01 * self.max_action
        else:
            action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=48, gamma=1, tau=1e-4):
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

        if self.total_it % self.policy_freq == 0:
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

        self.total_it += 1
# =============================
# 4. Training Loop
# =============================
if __name__ == '__main__':
    lte_demand = load_demand_data("Data/LTE_Demand_hourly.tsv")
    nr_demand = load_demand_data("Data/NR_Demand_hourly.tsv")
    env = ResourceAllocationEnv(lte_demand, nr_demand)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = 1.00

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(state_dim, action_dim, max_action, device)
    buffer = ReplayBuffer()

    episodes = 20000
    batch_size = 48
    rewards = []
    logs = []
    os.makedirs("FigureReward", exist_ok=True)
    os.makedirs("Result", exist_ok=True)
    os.makedirs("Model", exist_ok=True)
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            if len(buffer.buffer) > batch_size:
                agent.train(buffer, batch_size)

        rewards.append(total_reward)
        alloc_lte, alloc_nr = info["alloc"]
        demand_lte, demand_nr = info["demand"]
        logs.append({"episode": ep, "reward": total_reward, "alloc_lte":alloc_lte, "alloc_nr": alloc_nr, "demand_lte":demand_lte, "demand_nr":demand_nr})
        print(f"Episode {ep}, Reward: {total_reward:.2f}, | Demand=({demand_lte:.2f}, {demand_nr :.2f}) "
              f"| Alloc=({alloc_lte:.2f}, {alloc_nr:.2f})")

    torch.save(agent.actor.state_dict(), "Model/sac_actor_model.pth")
    # torch.save(agent.critic.state_dict(), "Model/sac_critic_model.pth")
    pd.DataFrame(logs).to_csv("Result/sac_training_log.csv", index=False)
    smoothed_rewards = pd.Series(rewards).rolling(window=100, min_periods=1).mean()
    plt.plot(smoothed_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("SAC Training Reward")
    plt.grid(True)
    plt.savefig("FigureReward/sac_training_reward_plot.png")
    plt.show()
