import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
from MainEnv import ResourceAllocationEnv, load_demand_data

# =============================
# 1. Actor and Critic Networks
# =============================
class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)


class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)


# =============================
# 2. TD3 Agent
# =============================
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.actor = TD3Actor(state_dim, action_dim).to(device)
        self.actor_target = TD3Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3Critic(state_dim, action_dim).to(device)
        self.critic_target = TD3Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.device = device
        self.max_action = max_action
        self.tau = 1e-4
        self.gamma = 1
        self.policy_noise = 0.09
        self.noise_clip = 0.0
        self.policy_freq = 2
        self.total_it = 0

    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()
        action = action * self.max_action

        if not evaluate:
            noise = np.random.normal(0, self.policy_noise, size=action.shape)
            action += noise

        return np.clip(action, 0.0, self.max_action)

    def train(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return

        self.total_it += 1
        s, a, r, s_, d = replay_buffer.sample(batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(s_) + noise).clamp(-1, 1) * self.max_action
            target_q1, target_q2 = self.critic_target(s_, next_action)
            target_q = r + (1 - d) * self.gamma * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(s, a)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1(torch.cat([s, self.actor(s)], dim=1)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



# =============================
# 3. Replay Buffer
# =============================
class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(s_), np.array(d)

    def __len__(self):
        return len(self.buffer)


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
    agent = TD3Agent(state_dim, action_dim, max_action, device)
    buffer = ReplayBuffer()

    episodes = 20000
    batch_size = 48
    max_steps = 1
    logs = []
    os.makedirs("FigureReward", exist_ok=True)
    os.makedirs("Result", exist_ok=True)
    os.makedirs("Model", exist_ok=True)
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            agent.train(buffer, batch_size)

            if done:
                break
        
        alloc_lte, alloc_nr = info["alloc"]
        demand_lte, demand_nr = info["demand"]
        logs.append({"episode": ep, "reward": total_reward, "alloc_lte":alloc_lte, "alloc_nr": alloc_nr, "demand_lte":demand_lte, "demand_nr":demand_nr})
        print(f"Episode {ep}, Reward: {total_reward:.2f}, | Demand=({demand_lte:.2f}, {demand_nr :.2f}) "
              f"| Alloc=({alloc_lte:.2f}, {alloc_nr:.2f})")

    pd.DataFrame(logs).to_csv("Result/td3_training_log.csv", index=True)
    torch.save(agent.actor.state_dict(), "Model/td3_actor_model.pth")
    #torch.save(agent.critic.state_dict(), "Model/td3_critic_model.pth")

    rewards = [entry["reward"] for entry in logs]
    smoothed_rewards = pd.Series(rewards).rolling(window=100, min_periods=1).mean()
    plt.plot(smoothed_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("TD3 Training Reward")
    plt.grid(True)
    plt.savefig("FigureReward/td3_training_reward_plot.png")
    plt.show()