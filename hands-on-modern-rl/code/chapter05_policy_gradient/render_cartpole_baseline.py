"""
Render CartPole rollouts for Chapter 5.3.

This script trains two policies on CartPole-v1:
1. Vanilla REINFORCE
2. REINFORCE with a value baseline

It then renders deterministic rollouts as GIFs for the course notes.
"""

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        logits = self.network(x)
        return torch.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def compute_returns(rewards, gamma=0.99):
    returns = []
    g = 0.0
    for reward in reversed(rewards):
        g = reward + gamma * g
        returns.insert(0, g)
    return returns


def collect_episode(policy, env):
    state, _ = env.reset()
    states, actions, rewards = [], [], []
    terminated = truncated = False

    while not (terminated or truncated):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = policy(state_t)
        action = torch.distributions.Categorical(probs).sample().item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    return states, actions, rewards, float(sum(rewards))


def train_vanilla(seed, episodes=500, gamma=0.99, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    policy = PolicyNetwork(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    rewards_history = []

    for _ in range(episodes):
        states, actions, rewards, episode_reward = collect_episode(policy, env)
        returns = compute_returns(rewards, gamma)
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        returns_t = torch.FloatTensor(returns)

        probs = policy(states_t)
        action_probs = probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        log_probs = torch.log(action_probs + 1e-8)
        loss = -(log_probs * returns_t).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rewards_history.append(episode_reward)

    env.close()
    return policy, rewards_history


def train_with_baseline(seed, episodes=500, gamma=0.99, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)
    value_net = ValueNetwork(state_dim=state_dim)
    policy_optimizer = optim.Adam(policy.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
    rewards_history = []

    for _ in range(episodes):
        states, actions, rewards, episode_reward = collect_episode(policy, env)
        returns = compute_returns(rewards, gamma)
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        returns_t = torch.FloatTensor(returns)

        values = value_net(states_t)
        value_loss = nn.MSELoss()(values, returns_t)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        with torch.no_grad():
            advantages = returns_t - value_net(states_t)

        probs = policy(states_t)
        action_probs = probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        log_probs = torch.log(action_probs + 1e-8)
        policy_loss = -(log_probs * advantages).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        rewards_history.append(episode_reward)

    env.close()
    return policy, rewards_history


def render_policy(policy, output, seed=0, max_steps=500, fps=30):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0.0

    for _ in range(max_steps):
        frames.append(env.render())
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = int(torch.argmax(policy(state_t), dim=-1).item())
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            frames.append(env.render())
            break

    env.close()
    imageio.mimsave(output, frames, fps=fps)
    return total_reward, len(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/chapter05_policy_gradient/images"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    vanilla_policy, vanilla_rewards = train_vanilla(args.seed, args.episodes)
    baseline_policy, baseline_rewards = train_with_baseline(args.seed + 100, args.episodes)

    vanilla_return, vanilla_frames = render_policy(
        vanilla_policy,
        args.output_dir / "cartpole-vanilla-reinforce.gif",
        seed=args.seed,
    )
    baseline_return, baseline_frames = render_policy(
        baseline_policy,
        args.output_dir / "cartpole-reinforce-baseline.gif",
        seed=args.seed,
    )

    print("Vanilla final 50 mean:", float(np.mean(vanilla_rewards[-50:])))
    print("Baseline final 50 mean:", float(np.mean(baseline_rewards[-50:])))
    print("Vanilla GIF return:", vanilla_return, "frames:", vanilla_frames)
    print("Baseline GIF return:", baseline_return, "frames:", baseline_frames)


if __name__ == "__main__":
    main()
