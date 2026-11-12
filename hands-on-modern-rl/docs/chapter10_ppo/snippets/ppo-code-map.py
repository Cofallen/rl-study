import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, act_dim)
        self.critic_head = nn.Linear(hidden, 1)

    # [A] 策略和值函数：Actor 输出动作概率，Critic 输出状态价值
    def forward(self, obs):
        h = self.backbone(obs)
        logits = self.actor_head(h)
        action_probs = F.softmax(logits, dim=-1)
        value = self.critic_head(h).squeeze(-1)
        return action_probs, value

    # [B] 动作采样：根据策略分布选择动作，并记录 log_prob
    def act(self, obs):
        action_probs, value = self.forward(obs)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, obs, actions):
        action_probs, values = self.forward(obs)
        dist = Categorical(action_probs)
        new_logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return new_logprobs, values, entropy


# [C] 采样一批 on-policy 数据：这些数据来自"当前策略"
def collect_rollout(env, model, steps=2048, device="cpu"):
    obs, _ = env.reset()
    batch = {k: [] for k in ["states", "actions", "rewards", "dones", "old_logprobs", "values"]}

    for _ in range(steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action, old_logprob, value = model.act(obs_tensor)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        batch["states"].append(obs)
        batch["actions"].append(action.item())
        batch["rewards"].append(reward)
        batch["dones"].append(done)
        batch["old_logprobs"].append(old_logprob.item())
        batch["values"].append(value.item())

        obs = next_obs if not done else env.reset()[0]

    return {
        "states": torch.as_tensor(np.array(batch["states"]), dtype=torch.float32, device=device),
        "actions": torch.as_tensor(batch["actions"], dtype=torch.long, device=device),
        "rewards": torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device),
        "dones": torch.as_tensor(batch["dones"], dtype=torch.float32, device=device),
        "old_logprobs": torch.as_tensor(batch["old_logprobs"], dtype=torch.float32, device=device),
        "values": torch.as_tensor(batch["values"], dtype=torch.float32, device=device),
    }


# [D] 计算优势：GAE（Generalized Advantage Estimation）
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_advantage = 0.0
    next_value = 0.0

    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_advantage = delta + gamma * lam * mask * last_advantage
        advantages[t] = last_advantage
        next_value = values[t]

    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


# [E] PPO 损失函数（DeepSpeed-Chat / VeRL / OpenRLHF 风格）
def actor_loss_fn(new_logprobs, old_logprobs, advantages, clip_eps=0.2):
    ratio = torch.exp(new_logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    return policy_loss


def critic_loss_fn(new_values, old_values, returns, clip_eps_value=0.2):
    values_clipped = torch.clamp(
        new_values, old_values - clip_eps_value, old_values + clip_eps_value
    )
    vf_loss1 = (new_values - returns) ** 2
    vf_loss2 = (values_clipped - returns) ** 2
    vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
    return vf_loss


def ppo_update(model, optimizer, batch, advantages, returns,
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
               epochs=10, minibatch_size=64):
    states = batch["states"]
    actions = batch["actions"]
    old_logprobs = batch["old_logprobs"]
    old_values = batch["values"]
    batch_size = states.size(0)

    for _ in range(epochs):
        indices = torch.randperm(batch_size, device=states.device)
        for start in range(0, batch_size, minibatch_size):
            mb = indices[start:start + minibatch_size]

            new_logprobs, new_values, entropy = model.evaluate(states[mb], actions[mb])

            pg_loss = actor_loss_fn(new_logprobs, old_logprobs[mb], advantages[mb], clip_eps)
            vf_loss = critic_loss_fn(new_values, old_values[mb], returns[mb], clip_eps)
            entropy_bonus = entropy.mean()
            loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# [F] 训练循环：采样一批数据，再用这批数据更新多轮
device = "cuda" if torch.cuda.is_available() else "cpu"
env = gym.make("CartPole-v1")
model = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for update in range(100):
    batch = collect_rollout(env, model, steps=2048, device=device)
    advantages, returns = compute_gae(batch["rewards"], batch["values"], batch["dones"])
    ppo_update(model, optimizer, batch, advantages, returns)
