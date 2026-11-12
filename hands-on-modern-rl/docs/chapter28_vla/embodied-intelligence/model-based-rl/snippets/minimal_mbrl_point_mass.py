import random

import torch
import torch.nn as nn


torch.manual_seed(7)
random.seed(7)


def env_reset():
    return torch.randn(2) * torch.tensor([2.0, 0.5])


def env_step(state, action):
    action = action.clamp(-1.0, 1.0)
    a = action[..., 0] if action.shape[-1:] == (1,) else action

    x = state[..., 0]
    v = state[..., 1]
    next_v = 0.95 * v + 0.15 * torch.tanh(a)
    next_x = x + next_v
    next_state = torch.stack([next_x, next_v], dim=-1)

    reward = -(next_x.square() + 0.1 * next_v.square() + 0.001 * a.square())
    return next_state, reward


def collect_random_data(num_steps=3000):
    states, actions, rewards, next_states = [], [], [], []
    state = env_reset()

    for _ in range(num_steps):
        action = torch.empty(1).uniform_(-1.0, 1.0)
        next_state, reward = env_step(state, action)

        states.append(state.clone())
        actions.append(action.clone())
        rewards.append(reward.clone())
        next_states.append(next_state.clone())

        reset = abs(next_state[0].item()) > 5.0 or random.random() < 0.03
        state = env_reset() if reset else next_state.detach()

    return (
        torch.stack(states),
        torch.stack(actions),
        torch.stack(rewards),
        torch.stack(next_states),
    )


class DynamicsModel(nn.Module):
    def __init__(self, state_dim=2, action_dim=1, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim + 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


def train_model(model, data, updates=900, batch_size=256):
    states, actions, rewards, next_states = data
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    num_samples = states.shape[0]

    for update in range(updates):
        index = torch.randint(0, num_samples, (batch_size,))
        state = states[index]
        action = actions[index]
        reward = rewards[index].unsqueeze(-1)
        next_state = next_states[index]

        target = torch.cat([next_state - state, reward], dim=-1)
        pred = model(state, action)
        loss = ((pred - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if update % 300 == 0:
            print(f"update={update:04d} model_loss={loss.item():.6f}")


@torch.no_grad()
def score_action_sequences(model, state, action_sequences, discount=0.97):
    num_samples, horizon, _ = action_sequences.shape
    imagined_state = state.view(1, -1).expand(num_samples, -1).clone()
    returns = torch.zeros(num_samples)
    gamma = 1.0

    for t in range(horizon):
        pred = model(imagined_state, action_sequences[:, t])
        delta_state = pred[:, :2]
        reward = pred[:, 2]
        imagined_state = imagined_state + delta_state
        returns = returns + gamma * reward
        gamma *= discount

    return returns


@torch.no_grad()
def random_shooting_mpc(model, state, horizon=20, num_samples=2048):
    action_sequences = torch.empty(num_samples, horizon, 1).uniform_(-1.0, 1.0)
    scores = score_action_sequences(model, state, action_sequences)
    best = scores.argmax()
    return action_sequences[best, 0]


def rollout(policy, start_state=None, steps=60):
    state = torch.tensor([2.5, 0.0]) if start_state is None else start_state.clone()
    total_reward = 0.0

    for _ in range(steps):
        action = policy(state)
        state, reward = env_step(state, action)
        total_reward += reward.item()

    return total_reward, state


def main():
    data = collect_random_data()
    model = DynamicsModel()
    train_model(model, data)

    with torch.no_grad():
        states, actions, rewards, next_states = data
        target = torch.cat([next_states - states, rewards.unsqueeze(-1)], dim=-1)
        mse = ((model(states, actions) - target) ** 2).mean().item()

    random_return, random_final = rollout(lambda _: torch.empty(1).uniform_(-1.0, 1.0))
    mbrl_return, mbrl_final = rollout(lambda s: random_shooting_mpc(model, s))

    print(f"one_step_model_mse={mse:.6f}")
    print(f"random_policy_return={random_return:.2f}, final_state={random_final.tolist()}")
    print(f"mbrl_mpc_return={mbrl_return:.2f}, final_state={mbrl_final.tolist()}")


if __name__ == "__main__":
    main()
