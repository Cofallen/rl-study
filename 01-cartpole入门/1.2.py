import gymnasium as gym
import numpy as np
import torch.nn as nn


env = gym.make('CartPole-v1')
print("观测上限", env.observation_space.high)
print("观测下限", env.observation_space.low)
print('小车位置阈值', env.unwrapped.x_threshold)
print('杆子角度阈值', env.unwrapped.theta_threshold_radians)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )



# for _ in range(num_steps):
#     action, log_prob, value = model.