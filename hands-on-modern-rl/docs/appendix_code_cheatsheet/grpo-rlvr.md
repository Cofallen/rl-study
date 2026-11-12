# B.4 GRPO 与奖励模型

---

## GRPO Loss

**核心问题**：PPO 需要训练一个和 policy 同样大的 Critic 来估计 $V(s)$ 提供 baseline。GRPO 利用"同一 prompt 采 G 条回答"天然形成的对照组，用组内 reward 的归一化直接当 advantage，砍掉 Critic。

**核心变量**：

- `rewards`：同一 prompt 下 G 条回答的奖励（来自 RM 或规则验证器），形状 `[G]`
- `advantages`：组内归一化后的优势，$A_i = (r_i - \bar r)/\mathrm{std}(r)$
- `old_log_probs` / `new_log_probs`：采样时策略与当前策略对回答的 log 概率
- `ref_log_probs`：参考策略的 log 概率，用于 KL 惩罚
- `clip_eps`、`kl_coeff`：PPO 同款超参

### 一句话记忆

> **同 prompt 采 G 条，组内 reward 做 z-score 当 advantage；抄 PPO 的 clip 和 KL，砍掉 Critic。**

### 伪代码

```
# 第 1 步 与 同一 prompt 采 G 条回答，逐条打分
rewards = [reward_fn(generate(prompt)) for _ in range(G)]   # [G]

# 第 2 步 与 组内归一化（减均值除标准差）当 advantage
advantages = (rewards - mean(rewards)) / (std(rewards) + eps)

# 第 3 步 与 PPO clipped loss（advantage 来自第 2 步，不是 Critic）
ratio = exp(new_logp - old_logp)
surr1 = ratio * advantages
surr2 = clip(ratio, 1-eps, 1+eps) * advantages
policy_loss = -min(surr1, surr2).mean()

# 第 4 步 与 k3 KL 惩罚（拉住，别离参考模型太远）
log_ratio = ref_logp - new_logp
kl = (exp(log_ratio) - 1 - log_ratio).mean()

# 第 5 步 与 总 loss
loss = policy_loss + kl_coeff * kl
```

### PPO vs GRPO 对比

|                | PPO                         | GRPO                           |
| -------------- | --------------------------- | ------------------------------ |
| Advantage 来源 | Critic 预测 $V(s)$ → GAE    | 组内 reward 归一化             |
| 模型数量       | 4（actor, critic, ref, rm） | 2~3（actor, ref, rm/verifier） |
| KL             | 可选                        | 几乎必加                       |
| 采样方式       | 单条 rollout                | 同 prompt 采 G 条              |

### Python 实现

```python
import numpy as np

def grpo_advantages(rewards):
    """
    rewards: [num_prompts, G]  每个 prompt 的 G 条回答的 reward
    返回组内 z-score 归一化的 advantage
    """
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    return (rewards - mean) / (std + 1e-8)

def grpo_policy_loss(new_logps, old_logps, advantages, clip_eps=0.2):
    """和 PPO clipped loss 完全相同"""
    ratio = np.exp(new_logps - old_logps)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -np.minimum(surr1, surr2).mean()
```

### PyTorch 实现

```python
import torch
import torch.nn.functional as F

def grpo_loss(log_probs, old_log_probs, ref_log_probs,
              rewards, clip_eps=0.2, kl_coeff=0.05):
    """
    log_probs:     [B, G]  当前策略对每条 completion 的序列级 log_prob
    old_log_probs: [B, G]  采样时策略
    ref_log_probs: [B, G]  参考策略
    rewards:       [B, G]  组内 reward
    B = num_prompts, G = group_size
    """
    # 1. 组内归一化（按 prompt 分组）
    advantages = (rewards - rewards.mean(dim=1, keepdim=True)) \
                 / (rewards.std(dim=1, keepdim=True) + 1e-8)   # [B, G]

    # 2. Clipped policy loss（与 PPO 完全相同）
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 3. KL 惩罚（k3 估计器：log_ratio = log(π_ref/π_θ)，样本来自 π_θ）
    log_ratio = ref_log_probs - log_probs
    kl = (torch.exp(log_ratio) - 1 - log_ratio).mean()

    return policy_loss + kl_coeff * kl
```

---

## Reward Model（Bradley-Terry 模型）

**核心问题**：RLHF-PPO 需要一个标量奖励 $r \in \mathbb{R}$ 指导策略优化，但人类偏好只有相对顺序（"A 比 B 好"）。RM 把偏好对压缩成绝对分数，让"好回答分数高于坏回答"的概率最大。

**核心变量**：

- `r_chosen` / `r_rejected`：RM 对好/坏回答打的标量分数
- Bradley-Terry 假设：$P(y_w \succ y_l) = \sigma(r_w - r_l)$，偏好概率正比于分数差的 sigmoid

### 一句话记忆

> **让好分高于坏分：`-log_sigmoid(r_chosen - r_rejected)`，一行完事。**

### 伪代码

```
# 第 1 步 与 RM 对两个回答各打一个标量分
r_w = reward_model(chosen_input)     # 好回答的分数
r_l = reward_model(rejected_input)   # 坏回答的分数

# 第 2 步 与 希望 r_w > r_l，对差值过 sigmoid 取负对数
loss = -log(sigmoid(r_w - r_l))
```

### Python 实现

```python
import numpy as np

def log_sigmoid(x):
    return -np.logaddexp(0, -x)   # 数值稳定的 log σ(x)

def reward_model_loss(r_chosen, r_rejected):
    """r_chosen, r_rejected: [B]"""
    return -log_sigmoid(r_chosen - r_rejected).mean()
```

### PyTorch 实现

```python
import torch.nn.functional as F

def reward_model_loss(r_chosen, r_rejected):
    """
    r_chosen:   [B]  RM 对 chosen 的打分
    r_rejected: [B]  RM 对 rejected 的打分
    """
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

---

## GRPO、PPO-RLHF 与 RLVR 的区别

|                | PPO-RLHF     | GRPO                      | RLVR                          |
| -------------- | ------------ | ------------------------- | ----------------------------- |
| Advantage 来源 | Critic + GAE | 组内 reward 归一化        | 组内 reward 归一化            |
| Critic         | 必须         | 不需要                    | 不需要                        |
| Reward 来源    | 训练好的 RM  | RM 或 verifier            | 规则验证器（数学/代码正确性） |
| 在线采样       | 必须         | 必须（同 prompt 采 G 条） | 必须（同 prompt 采 G 条）     |

RLVR 是 GRPO 的一个特例：reward 不来自学习到的 RM，而来自对答案的规则验证（数学题答案是否相等、代码是否通过测试），因此没有 reward hacking 风险。

---

## 易错点

| 易错                           | 说明                                                                         |
| ------------------------------ | ---------------------------------------------------------------------------- |
| GRPO 的 advantage 是组内归一化 | 不是全局归一化，只比较**同一 prompt** 的 G 条回答                            |
| GRPO 没有 value loss           | 没有 Critic，所以没有 value loss，这是和 PPO 的核心区别                      |
| 组内归一化的分母有两种实现     | 多数实现用 std（z-score）；也有的只用 `rewards - mean`（不减 std）。注意区分 |
| RM 在 policy 训练时要冻结      | 训练 RM 时 reward 参与梯度；训练 policy 时 RM 通常 detach 或冻结             |
| KL 是序列级                    | 通常先对每条 completion 的 token log_prob 求和，再算 KL，不是 token 级       |
| k3 KL 估计器的方向             | `log_ratio = log(π_ref/π_θ)`，样本来自当前策略 $\pi_\theta$                  |
| RLVR 场景                      | reward 来自规则验证器（代码执行、数学答案检查），不是 RM 打分                |
