import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import random

from MainEnv import ResourceAllocationEnv, load_demand_data

# =============================
# 1. Actor and Critic Networks
# =============================
class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, state):
        return self.net(state)


class PPOCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.net(state)


# =============================
# 2. PPO Agent
# =============================
class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.actor = PPOActor(state_dim, action_dim).to(device)
        self.critic = PPOCritic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=2e-3)
        self.device = device
        self.max_action = max_action

        # Noise parameters
        self.noise_std = 0.09
        self.noise_std_min = 0.0

    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean = self.actor(state_tensor)

        if evaluate:
            action = mean
            log_prob = torch.zeros(1, device=self.device)  # no sampling, so log_prob is placeholder
        else:
            dist = torch.distributions.Normal(mean, self.noise_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)

        action = action.squeeze(0) * self.max_action
        action = torch.clamp(action, 0.0, self.max_action)
        return action.cpu().detach().numpy(), log_prob.item()

    def evaluate(self, states, actions):
        means = self.actor(states) * self.max_action
        dist = torch.distributions.Normal(means, self.noise_std)
        log_probs = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        values = self.critic(states).squeeze()
        return log_probs, entropy, values


# =============================
# 3. Replay Buffer (like SAC)
# =============================
class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, log_prob, reward, next_state, done):
        self.buffer.append((state, action, log_prob, reward, next_state, done))

    def sample_all(self):
        batch = list(self.buffer)
        states, actions, log_probs, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(log_probs),
                np.array(rewards), np.array(next_states), np.array(dones))

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


# =============================
# 4. PPO Training Loop
# =============================
if __name__ == '__main__':
    lte_demand = load_demand_data("Data/LTE_Demand_hourly.tsv")
    nr_demand = load_demand_data("Data/NR_Demand_hourly.tsv")
    env = ResourceAllocationEnv(lte_demand, nr_demand)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(state_dim, action_dim, max_action, device)
    buffer = ReplayBuffer()

    gamma = 1
    gae_lambda = 0.95
    eps_clip = 0.2
    K_epochs = 4
    episodes = 20000
    max_steps = 1
    batch_size = 48
    logs = []
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        buffer.clear()

        for _ in range(max_steps):
            action, log_prob = agent.select_action(state)
            clipped_action = np.clip(action, 0.01, 1.00)
            value = agent.critic(torch.FloatTensor(state).unsqueeze(0).to(device)).item()

            next_state, reward, terminated, truncated, _ = env.step(clipped_action)
            done = terminated or truncated
            buffer.add(state, action, log_prob, reward, next_state, done)

            state = next_state
            total_reward += reward
            if done:
                break
        
        if len(buffer.buffer) > batch_size:
            agent.train(buffer, batch_size)

        # Extract and compute returns and advantages
        states, actions, log_probs, rewards, next_states, dones = buffer.sample_all()
        values = agent.critic(torch.FloatTensor(states).to(device)).view(-1).detach().cpu().numpy()
        next_values = agent.critic(torch.FloatTensor(next_states).to(device)).view(-1).detach().cpu().numpy()

        returns = []
        advantages = []
        G = 0
        A = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * next_values[t] * (1 - dones[t])
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            A = delta + gamma * gae_lambda * A * (1 - dones[t])
            returns.insert(0, G)
            advantages.insert(0, A)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(log_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)

        if advantages.numel() < 2 or advantages.std().item() <= 1e-6:
            advantages = torch.zeros_like(advantages)
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(K_epochs):
            indices = torch.randperm(states.size(0))
            for start in range(0, states.size(0), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                with torch.no_grad():
                    if torch.isnan(batch_states).any():
                        continue

                log_probs, entropy, values = agent.evaluate(batch_states, batch_actions)
                if torch.isnan(log_probs).any() or torch.isnan(values).any():
                    continue

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values.view(-1), batch_returns.view(-1))
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                agent.actor_optimizer.zero_grad()
                agent.critic_optimizer.zero_grad()
                loss.backward()
                agent.actor_optimizer.step()
                agent.critic_optimizer.step()

        logs.append({"episode": ep, "reward": total_reward})
        print(f"Episode {ep}, Reward: {total_reward:.2f}")

    pd.DataFrame(logs).to_csv("Result/ppo_training_log.csv", index=True)
    torch.save(agent.actor.state_dict(), "Model/ppo_actor_model.pth")
    #torch.save(agent.critic.state_dict(), "Model/ppo_critic_model.pth")

    rewards = [entry["reward"] for entry in logs]
    smoothed_rewards = pd.Series(rewards).rolling(window=100, min_periods=1).mean()
    plt.plot(smoothed_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Reward")
    plt.grid(True)
    plt.savefig("ppo_training_reward_plot.png")
    plt.show()
