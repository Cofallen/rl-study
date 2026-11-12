---
title: B.2 PPO and GAE
---

# B.2 PPO and GAE

PPO is the most frequently tested algorithm in LLM RL interviews. Interviewers typically ask you to write the **clipped policy loss**, and may follow up with the value loss and GAE.

---

## GAE (Generalized Advantage Estimation)

**Core problem**: combine multi-step future returns into a low-variance advantage estimate $\hat{A}_t$ that tells the actor how much better than average this step was.

**Core variables**:

- `delta_t`: TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, the one-step "surprise"
- `advantage_t`: accumulated advantage $\hat{A}_t$, computed backward
- `gamma` ($\gamma$): discount factor, controls how far ahead we look
- `lambda` ($\lambda$): bias-variance tradeoff; $\lambda=0$ reduces to TD(0), $\lambda=1$ to Monte-Carlo
- `done_t`: episode-end flag, cuts the recursion across episodes

### One-Line Memory

> Sweep backward: $\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$.

### Pseudocode

```
delta_t     = reward_t + gamma * value_{t+1} * (1 - done_t) - value_t
advantage_t = delta_t + gamma * lambda * (1 - done_t) * advantage_{t+1}
return_t    = advantage_t + value_t
```

$\lambda = 0$ uses only one-step TD (low variance, high bias); $\lambda = 1$ sums all $\delta$ (high variance, low bias).

### Python Implementation

```python
import numpy as np

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: [T]
    values:  [T+1]  (last element is the bootstrap value)
    dones:   [T]
    """
    T = len(rewards)
    advantages = np.zeros(T)
    last_adv = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
        advantages[t] = last_adv

    returns = advantages + values[:T]
    return advantages, returns
```

### PyTorch Implementation

```python
import torch

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: [B, T]
    values:  [B, T+1]
    dones:   [B, T]
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(B)

    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * values[:, t + 1] * (1 - dones[:, t]) - values[:, t]
        last_adv = delta + gamma * lam * (1 - dones[:, t]) * last_adv
        advantages[:, t] = last_adv

    returns = advantages + values[:, :T]
    return advantages, returns
```

---

## PPO Clipped Policy Loss

**Core problem**: make policy improvement within a "trust region," using clip to approximate the KL constraint so that a single update can reuse old data (importance sampling) without blowing up when the ratio is large.

**Core variables**:

- `ratio`: $r_t(\theta) = \pi_\theta(a_t\mid s_t) / \pi_{\theta_{old}}(a_t\mid s_t)$, the new/old policy ratio (denominator is the sampling policy)
- `advantage`: $\hat{A}_t$, from GAE
- `eps` ($\epsilon$): clip range, typically `0.1` or `0.2`
- `new_log_prob` / `old_log_prob`: log-probs under the current and sampling policies

### One-Line Memory

> Ratio $r_t = \pi_{new}/\pi_{old}$, clipped to $[1-\epsilon, 1+\epsilon]$; take min of the raw and clipped surrogate — trust only the more conservative one.

$$L^{CLIP} = -\min\big(r_t(\theta) \cdot A_t,\;\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t\big)$$

### Pseudocode

```
ratio = exp(new_log_prob - old_log_prob)
surr1 = ratio * advantage
surr2 = clip(ratio, 1-eps, 1+eps) * advantage
loss  = -min(surr1, surr2).mean()
```

When `advantage > 0` the upper bound of ratio is clipped; when `advantage < 0` the lower bound is clipped; `min` keeps the more conservative target.

### Python Implementation

```python
import numpy as np

def ppo_policy_loss(new_logp, old_logp, advantages, clip_eps=0.2):
    """
    new_logp:   [T]  log-probs under the current policy
    old_logp:   [T]  log-probs under the sampling policy
    advantages: [T]
    """
    ratio = np.exp(new_logp - old_logp)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -np.minimum(surr1, surr2).mean()
```

### PyTorch Implementation

```python
import torch

def ppo_policy_loss(new_logps, old_logps, advantages, clip_eps=0.2):
    """
    new_logps:   [B, T]
    old_logps:   [B, T]
    advantages:  [B, T]
    """
    ratio = torch.exp(new_logps - old_logps)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()
```

---

## PPO Value Loss

**Core problem**: train the critic to predict accumulated returns $V(s)$ accurately, and prevent it from drifting too far in a single update (value clipping).

**Core variables**:

- `value_pred`: critic's current prediction of $V(s_t)$
- `old_values`: critic's prediction at sampling time, used as the clip reference
- `returns`: $R_t = \hat{A}_t + V_{old}(s_t)$, regression target from GAE
- `eps`: clip range, shared with the policy loss

### One-Line Memory

> $(V_{pred} - R)^2$ averaged; optional clip: don't let the new prediction stray past $\epsilon$ from the old one.

### Pseudocode

```
value_clipped = old_values + clip(value_pred - old_values, -eps, eps)
loss1 = (value_pred    - returns)^2
loss2 = (value_clipped - returns)^2
loss  = 0.5 * max(loss1, loss2).mean()
```

### PyTorch Implementation

```python
def ppo_value_loss(values, old_values, returns, clip_eps=0.2):
    loss1 = (values - returns) ** 2
    values_clipped = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
    loss2 = (values_clipped - returns) ** 2
    return 0.5 * torch.max(loss1, loss2).mean()
```

---

## Total PPO Loss

```
total_loss = policy_loss + vf_coeff * value_loss - ent_coeff * entropy
```

| Component           | Purpose               | Typical coefficient |
| ------------------- | --------------------- | ------------------- |
| clipped policy loss | update the policy     | `1.0`               |
| value loss (MSE)    | update the critic     | `vf_coef=0.5`       |
| entropy bonus       | encourage exploration | `ent_coef=0.01`     |

A minus sign precedes entropy: maximizing entropy is equivalent to subtracting it from the loss.

---

## Common Pitfalls

| Pitfall                       | Explanation                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Using division for `ratio`    | Use `exp(logp_new - logp_old)`; it is more numerically stable.                                                |
| Wrong `ratio` denominator     | The denominator must be the sampling policy $\pi_{old}$, not the current policy.                              |
| Wrong clip range              | It is $[1-\epsilon, 1+\epsilon]$ around 1, not $[-\epsilon, \epsilon]$ around 0.                              |
| `min`/`max` confusion         | Policy loss takes `min` over the **two surrogates** (conservative); value loss takes `max` over the two MSEs. |
| Forgot to stop gradients      | `old_log_probs` and `old_values` should be `.detach()`'d.                                                     |
| Advantages not normalized     | In practice, advantages are usually normalized within a batch (mean 0, std 1).                                |
| GAE direction                 | Must be computed backward, starting from the last timestep.                                                   |
| Missing `done` masking in GAE | When `done=1`, cut the recursion: multiply by `gamma * lambda * (1-done)`.                                    |
| Missing bootstrap value       | `values` should have length `T+1`; the last value is the bootstrap.                                           |
| Wrong entropy sign            | `- ent_coeff * entropy` (entropy is positive; the minus sign in the loss encourages higher entropy).          |
