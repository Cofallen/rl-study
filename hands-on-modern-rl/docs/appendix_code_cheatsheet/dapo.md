# B.5 DAPO

DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）是 2025 年字节跳动提出的 GRPO 改进，针对长链推理 RL 训练中的四个痛点给出四把刀：解耦裁剪、动态采样、Token 级 loss、超长软惩罚。面试高频。

---

## 核心问题

GRPO 训练长链推理模型时遇到四个问题：对称裁剪把好动作的探索压得太死；全对/全错的 prompt 浪费采样算力却给不出梯度；序列级 loss 让长回答被低估；超长回答一刀切扣成 0 分，边界无梯度。DAPO 用四个独立改动分别应对。

## 核心变量

- `ratio`：新旧策略概率比 $r = \exp(\text{new\_logp} - \text{old\_logp})$
- `advantage`：组内 z-score $\hat{A}_{i,t} = (R_i - \bar R)/\mathrm{std}(R)$（与 GRPO 相同）
- `ε_{low}`、`ε_{high}`：解耦裁剪的下/上界（论文典型 0.2 / 0.28）
- `reward_std`：组内 reward 标准差，用于动态采样过滤
- `max_len`、`buffer_len`、`penalty_factor`：超长软惩罚的阈值、缓冲区宽度、最大罚分

## 一句话记忆

> **四刀：clip 上下用不同 ε、全对/全错的 prompt 跳过、loss 拉平 token 级、超长线性软扣。**

---

## 四个改动对比

| 改动      | GRPO                                | DAPO                                              |
| --------- | ----------------------------------- | ------------------------------------------------- |
| 裁剪      | 对称 `clip(r, 1-ε, 1+ε)`            | 解耦 `clip(r, 1-ε_{low}, 1+ε_{high})`，上界更松   |
| 采样      | 全部 prompt 都用                    | 过滤组内 reward 方差为 0（全对/全错）的 prompt    |
| loss 粒度 | 序列级（先 token-mean 再 seq-mean） | token 级（所有 token 直接求和再除以总 token 数）  |
| 超长回答  | 直接 reward=0（二元突变）           | 软惩罚：超出缓冲区线性扣分，封顶 `penalty_factor` |

---

## 解耦裁剪（Clip-Higher）

### 核心问题

对称裁剪 `clip(r, 1-ε, 1+ε)` 在两边用同一个 ε。但正 advantage 的好动作值得鼓励更大步幅，上界 ε 应该更大；负 advantage 的坏动作要稳一点，下界 ε 可以更小。DAPO 把上下界解耦，让两个方向独立调探索力度。

### 一句话记忆

> **PPO 同款 min-clipped，但上下用不同 $\varepsilon$：上界 $\varepsilon_{high}$ 宽、下界 $\varepsilon_{low}$ 紧。**

### 伪代码

```
ratio = exp(new_logp - old_logp)

# PPO 同款 min-clipped，但 clip 上下界用不同 ε
surr1   = ratio * advantage
surr2   = clip(ratio, 1 - eps_low, 1 + eps_high) * advantage
loss_t  = -min(surr1, surr2)            # per-token loss
```

> 注：当 advantage > 0 时下界不起作用（被 min 吃掉），advantage < 0 时上界不起作用，因此效果等价于"正优势只裁上界、负优势只裁下界"，但写法上直接照搬 PPO 的 min-clipped 更接近论文。

### Python 实现

```python
import numpy as np

def dapo_policy_loss(new_logp, old_logp, advantages,
                     eps_low=0.2, eps_high=0.28):
    """
    new_logp:   [T]
    old_logp:   [T]
    advantages: [T]
    返回 token 级 loss（求和除以 token 数）
    """
    ratio = np.exp(new_logp - old_logp)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - eps_low, 1 + eps_high) * advantages
    loss_per_token = -np.minimum(surr1, surr2)
    return loss_per_token.sum() / len(loss_per_token)
```

### PyTorch 实现

```python
import torch

def dapo_policy_loss(new_logps, old_logps, advantages,
                     eps_low=0.2, eps_high=0.28):
    """
    new_logps:  [B, seq_len]
    old_logps:  [B, seq_len]
    advantages: [B, seq_len]
    """
    ratio = torch.exp(new_logps - old_logps)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * advantages
    loss_per_token = -torch.minimum(surr1, surr2)
    return loss_per_token.sum() / loss_per_token.numel()
```

---

## 动态采样（Dynamic Sampling）

### 核心问题

GRPO 的 advantage 是组内 z-score。若某 prompt 下 G 条回答全对或全错，组内 reward 方差为 0，z-score 退化，这条 prompt 给不出任何梯度信号却白白消耗采样算力。DAPO 在数据层面提前过滤这些无效 prompt，并继续采样直到 batch 填满有效样本。

### 一句话记忆

> **组内 reward 方差为 0（全对/全错）→ 没梯度信号 → 跳过，继续采到填满 batch。**

### 伪代码

```
# 每个 prompt 采 G 条回答并打分
rewards = [get_reward(c) for c in group]      # [G]

# 组内 reward 全相同 → 跳过，继续采新 prompt 补位
if std(rewards) == 0:
    skip this prompt and resample
```

### PyTorch 实现

```python
def dynamic_sampling_filter(rewards):
    """
    rewards: [B, G]  B 个 prompt，每个 G 条回答的 reward
    返回 bool mask [B]，True = 保留
    """
    return rewards.std(dim=1) > 1e-6
```

---

## Token 级 Loss

### 核心问题

GRPO 默认按序列级聚合 loss：先对每条回答的 token 求平均得到序列 loss，再对序列求平均。这会让长回答（token 多）和短回答在最终 loss 里权重相同，长回答里的细节 token 被低估。DAPO 改成 token 级聚合：所有回答的所有 token 直接求和，再除以总 token 数，长回答自然贡献更多梯度。

### 一句话记忆

> **别先 seq-mean——所有 token 拉平求和除以总数，长回答不被压低。**

### 伪代码

```
loss_mat = -min(ratio*A, clip(ratio, 1-eps_low, 1+eps_high)*A)   # [B, T]

# GRPO: 先 token-mean 再 seq-mean  →  长序列被压平
seq_loss = mean(loss_mat, dim=token)        # 每条序列一个值
loss_grpo = mean(seq_loss)

# DAPO: 所有 token 拉平
loss_dapo = sum(loss_mat) / total_num_tokens
```

### PyTorch 实现

```python
def token_level_loss(loss_mat, loss_mask):
    """
    loss_mat:  [B, T]  per-token policy loss
    loss_mask: [B, T]  1 for valid token, 0 for padding
    返回 token 级聚合的 loss
    """
    return (loss_mat * loss_mask).sum() / loss_mask.sum()
```

---

## 超长软惩罚（Soft Overlong Punishment）

### 核心问题

GRPO 对超过最大长度的回答一刀切置零，边界处无梯度信号——策略只知道"被罚了"，不知道"短一点会更好"。DAPO 引入一段缓冲区：长度在 `[max_len - buffer_len, max_len]` 区间内不罚，超过 `max_len - buffer_len` 后线性扣分，扣到 `-penalty_factor` 封顶，给策略一个连续可微的方向信号。

### 一句话记忆

> **超过 `max_len − buffer_len` 后按比例线性扣，`-penalty_factor` 封顶——不一刀切零。**

### 伪代码

```
expected_len = max_len - buffer_len
exceed_len   = response_length - expected_len

if exceed_len > 0:
    # 线性罚分：超出越多扣越多，封顶 -penalty_factor（不会无限扣）
    penalty = max(-penalty_factor, -(exceed_len / buffer_len) * penalty_factor)
    reward = reward + penalty        # penalty ≤ 0
```

### Python 实现

```python
def soft_overlong_penalty(response_length, max_len,
                          buffer_len, penalty_factor=1.0):
    """返回罚分（≤0），加到原始 reward 上"""
    expected_len = max_len - buffer_len
    exceed_len = response_length - expected_len
    if exceed_len <= 0:
        return 0.0
    linear = -(exceed_len / buffer_len) * penalty_factor
    return max(-penalty_factor, linear)        # 封顶，不无限扣
```

---

## DAPO 总 Loss 草图

```
# 1. 组内 z-score 归一化（与 GRPO 相同）
advantages = (rewards - rewards.mean(dim=G)) / (rewards.std(dim=G) + eps)

# 2. 动态采样过滤
valid = dynamic_sampling_filter(rewards)        # 丢掉全对/全错的 prompt

# 3. 解耦裁剪 + token 级 loss
ratio = exp(new_logp - old_logp)
surr1 = ratio * advantages
surr2 = clip(ratio, 1 - eps_low, 1 + eps_high) * advantages
loss_mat = -minimum(surr1, surr2)               # per-token

# 4. token 级聚合（关键改动，长回答不被低估）
policy_loss = (loss_mat * mask)[valid].sum() / mask[valid].sum()

# 5. KL 惩罚（与 GRPO 相同）
kl = ((exp(ref_logp - new_logp) - 1) - (ref_logp - new_logp)).mean()

loss = policy_loss + kl_coeff * kl
```

---

## 易错点

| 易错                                         | 说明                                                                   |
| -------------------------------------------- | ---------------------------------------------------------------------- |
| 解耦裁剪 ≠ 取消裁剪                          | 仍然是 PPO 的 `min(r*A, clip(r,lo,hi)*A)`，只是上下界用不同 $\epsilon$ |
| 正/负 advantage 各自只受一个 $\epsilon$ 约束 | A>0 时下界无效、A<0 时上界无效（min 会吃掉），所以"正优裁上、负优裁下" |
| 动态采样的判定                               | 不是"reward 低于阈值"，而是"组内 reward **方差为 0**"                  |
| Token 级 loss 是第四个关键改动               | GRPO 是序列级聚合，DAPO 是 token 级聚合，长回答权重更高                |
| 超长软惩罚是线性不是指数                     | 简单的 `exceed_len / buffer_len`，封顶 `-penalty_factor`               |
| Advantage 仍是组内归一化                     | 这部分和 GRPO 完全一样，DAPO 没动                                      |
