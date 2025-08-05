import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Đường dẫn đến log của SB3
log_dir = "./ddpg_tensorboard/DDPG_5"  # hoặc tên khác nếu bạn đặt lại
event_acc = EventAccumulator(log_dir)
event_acc.Reload()

# Lấy reward trung bình từ logs
rewards = event_acc.Scalars("rollout/ep_rew_mean")
episodes = [s.step for s in rewards]
reward_vals = [s.value for s in rewards]

# Vẽ biểu đồ
plt.figure(figsize=(10, 5))
plt.plot(episodes, reward_vals, label="Average Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DDPG: Reward per Episode")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()