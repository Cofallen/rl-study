# B.2 PPO 与 GAE

PPO 是大模型 RL 面试中考查频率最高的算法。面试官通常要求写出 **clipped policy loss**，并追问 value loss 和 GAE。

---

## GAE（广义优势估计）

**核心问题**：把多步未来回报组合成一个低方差的优势估计 $\hat{A}_t$，告诉 actor 这一步比平均好多少。

**核心变量**：

- `delta_t`：TD 误差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，单步"惊喜"
- `advantage_t`：累积优势 $\hat{A}_t$，从后往前递推
- `gamma`（$\gamma$）：折扣因子，控制目光远近
- `lambda`（$\lambda$）：偏差-方差权衡参数，$\lambda=0$ 退化为 TD(0)，$\lambda=1$ 退化为 Monte-Carlo
- `done_t`：episode 结束标志，截断跨 episode 的递推

### 一句话记忆

> **从后往前扫：$\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$。**

### 伪代码

```
delta_t     = reward_t + gamma * value_{t+1} * (1 - done_t) - value_t
advantage_t = delta_t + gamma * lambda * (1 - done_t) * advantage_{t+1}
return_t    = advantage_t + value_t
```

$\lambda = 0$ 只看单步 TD（方差低、偏差高）；$\lambda = 1$ 累加所有 $\delta$（方差高、偏差低）。

### Python 实现

```python
import numpy as np

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: [T]
    values:  [T+1]  (最后一个为 bootstrap value)
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

### PyTorch 实现

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

**核心问题**：在"信任域"下做策略提升，用 clip 近似 KL 约束，让单步更新既能利用旧数据（重要性采样），又不会因 ratio 过大而崩溃。

**核心变量**：

- `ratio`：$r_t(\theta) = \pi_\theta(a_t\mid s_t) / \pi_{\theta_{old}}(a_t\mid s_t)$，新旧策略概率比（分母是采样时的旧策略）
- `advantage`：$\hat{A}_t$，来自 GAE
- `eps`（$\epsilon$）：clip 范围，典型 `0.1` 或 `0.2`
- `new_log_prob` / `old_log_prob`：当前策略与采样时策略的 log 概率

### 一句话记忆

> **比值 $r_t=\pi_{new}/\pi_{old}$，clip 到 $[1-\epsilon, 1+\epsilon]$；原始与裁剪两个 surrogate 取 min——只信更保守的那个。**

$$L^{CLIP} = -\min\big(r_t(\theta) \cdot A_t,\;\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t\big)$$

### 伪代码

```
ratio = exp(new_log_prob - old_log_prob)
surr1 = ratio * advantage
surr2 = clip(ratio, 1-eps, 1+eps) * advantage
loss  = -min(surr1, surr2).mean()
```

`advantage > 0` 时截断 ratio 的上界，`advantage < 0` 时截断下界；`min` 保证取更保守的目标。

### Python 实现

```python
import numpy as np

def ppo_policy_loss(new_logp, old_logp, advantages, clip_eps=0.2):
    """
    new_logp:   [T]  当前策略的 log 概率
    old_logp:   [T]  采样时策略的 log 概率
    advantages: [T]
    """
    ratio = np.exp(new_logp - old_logp)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -np.minimum(surr1, surr2).mean()
```

### PyTorch 实现

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

**核心问题**：训练 critic 让它准确预测累积回报 $V(s)$，并防止它在单次更新中漂移过大（value clipping）。

**核心变量**：

- `value_pred`：critic 当前对 $V(s_t)$ 的预测
- `old_values`：采样时 critic 的预测，用作 clip 参考
- `returns`：$R_t = \hat{A}_t + V_{old}(s_t)$，GAE 给出的回归目标
- `eps`：clip 范围，与 policy loss 共享

### 一句话记忆

> **$(V_{pred} - R)^2$ 求均值；可选 clip：预测别离旧预测超过 $\epsilon$。**

### 伪代码

```
value_clipped = old_values + clip(value_pred - old_values, -eps, eps)
loss1 = (value_pred    - returns)^2
loss2 = (value_clipped - returns)^2
loss  = 0.5 * max(loss1, loss2).mean()
```

### PyTorch 实现

```python
def ppo_value_loss(values, old_values, returns, clip_eps=0.2):
    loss1 = (values - returns) ** 2
    values_clipped = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
    loss2 = (values_clipped - returns) ** 2
    return 0.5 * torch.max(loss1, loss2).mean()
```

---

## PPO 总 Loss

```
total_loss = policy_loss + vf_coeff * value_loss - ent_coeff * entropy
```

| 组成                  | 作用        | 典型系数        |
| --------------------- | ----------- | --------------- |
| policy loss (clipped) | 更新策略    | `1.0`           |
| value loss (MSE)      | 更新 Critic | `vf_coef=0.5`   |
| entropy bonus         | 鼓励探索    | `ent_coef=0.01` |

entropy 前加负号：最大化 entropy 等价于在 loss 里减去它。

---

## 易错点

| 易错               | 说明                                                                              |
| ------------------ | --------------------------------------------------------------------------------- |
| ratio 用除法       | 用 `exp(logp_new - logp_old)`，数值更稳定                                         |
| ratio 分母用错     | 分母必须是采样时的旧策略 $\pi_{old}$，不是当前策略                                |
| clip 范围写错      | 是围绕 1 的 $[1-\epsilon, 1+\epsilon]$，不是围绕 0 的 $[-\epsilon, \epsilon]$     |
| min/max 混淆       | policy loss 对**两项 surrogate** 取 `min`（保守）；value loss 对两个 MSE 取 `max` |
| 忘了 stop gradient | `old_log_probs` 和 `old_values` 要 `.detach()`                                    |
| advantage 没归一化 | 工程中 advantage 通常做 batch 内归一化（mean 0, std 1）                           |
| GAE 方向           | 必须从后往前算（最后一个时步开始）                                                |
| GAE 的 done mask   | `done=1` 时截断递推：`gamma * lambda * (1-done) * next_adv`                       |
| value 的 bootstrap | `values` 长度是 T+1，最后一个位置是 bootstrap value                               |
| entropy 符号       | `- ent_coeff * entropy`（entropy 为正，加负号让 loss 减小 = 鼓励高熵）            |
