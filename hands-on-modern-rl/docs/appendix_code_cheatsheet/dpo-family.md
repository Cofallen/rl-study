# B.3 DPO 方法家族

DPO Loss 是后训练岗位面试中考查频率**最高**的手写代码题，几乎每场都会考。本节给出 DPO、IPO、KTO、SimPO 四个变体的速记。

---

## DPO Loss

**核心问题**：绕开 RLHF 中"训 reward model + PPO"的整套流程，直接用偏好对 $(y_w, y_l)$ 监督微调策略，等价于在隐式 reward 上做 Bradley-Terry 最大似然，无需在线采样和 critic。

**核心变量**：

- `pi_chosen` / `pi_rejected`：当前策略 $\pi_\theta$ 对好/坏回答的 log 概率
- `ref_chosen` / `ref_rejected`：参考策略 $\pi_{ref}$ 对好/坏回答的 log 概率（需 `detach`）
- `log_ratio_w` / `log_ratio_l`：$\log\frac{\pi_\theta}{\pi_{ref}}$，每个回答的隐式奖励（除以 $\beta$）
- `beta`（$\beta$）：温度，控制策略偏离 reference 的强度；越大越敏感，典型 0.1~0.5

### 一句话记忆

> **4 条 logp（2 模型 × 2 回答）：每条算"当前 − 参考"；好减坏、乘 β、过 sigmoid、取负 log。**

$$\mathcal{L}_{DPO} = -\mathbb{E}\Big[\log\sigma\Big(\beta\Big(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\Big)\Big]$$

### 伪代码

```
log_ratio_w = log_pi_theta(y_w|x) - log_pi_ref(y_w|x)   # 好回答的隐式奖励
log_ratio_l = log_pi_theta(y_l|x) - log_pi_ref(y_l|x)   # 坏回答的隐式奖励
logits      = beta * (log_ratio_w - log_ratio_l)        # 好坏优势 × 温度
loss        = -log_sigmoid(logits)                       # 最大化优势 → 等价 BT 似然
```

### Python 实现

```python
import numpy as np

def log_sigmoid(x):
    return -np.logaddexp(0, -x)  # 数值稳定

def dpo_loss(logp_chosen, logp_rejected,
             logp_ref_chosen, logp_ref_rejected,
             beta=0.1):
    """
    所有参数: scalar 或 [B]，返回标量 loss
    """
    log_ratio_w = logp_chosen - logp_ref_chosen
    log_ratio_l = logp_rejected - logp_ref_rejected
    logits = beta * (log_ratio_w - log_ratio_l)
    return -log_sigmoid(logits).mean()
```

### PyTorch 实现

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta=0.1):
    """所有参数: [B]"""
    log_ratio_w = policy_chosen_logps - ref_chosen_logps
    log_ratio_l = policy_rejected_logps - ref_rejected_logps
    logits = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(logits).mean()
```

---

## IPO

**核心问题**：DPO 的 $-\log\sigma$ 损失在偏好差距很大时梯度饱和，且对噪声偏好过拟合。IPO 改用平方损失，把好坏优势推向固定目标 $\frac{1}{2\beta}$，惩罚双向偏离，训练更稳。

**核心变量**：

- `delta`：$\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$，不带 β 的好坏优势
- 目标间隔 $\frac{1}{2\beta}$，由最优性条件推出

### 一句话记忆

> **把 DPO 的 $-\log\sigma$ 换成 $(\Delta - \frac{1}{2\beta})^2$——饱和没了，目标变成定点。**

$$\mathcal{L}_{IPO} = \Big(\Delta - \frac{1}{2\beta}\Big)^2, \quad \Delta = \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$$

### 伪代码

```
delta = log_ratio_w - log_ratio_l     # 不带 β 的好坏优势
loss  = (delta - 1 / (2 * beta)) ** 2 # 推向固定目标
```

### PyTorch 实现

```python
def ipo_loss(log_ratio_w, log_ratio_l, beta=0.1):
    delta = log_ratio_w - log_ratio_l
    return ((delta - 1.0 / (2 * beta)) ** 2).mean()
```

---

## KTO

**核心问题**：DPO/IPO 必须有 chosen-rejected 配对，但实际业务常只有"用户点赞/点踩"这种**单条二元反馈**。KTO 借用前景理论改造为单样本损失，无需配对，且对坏样本加权更高（损失厌恶）。

**核心变量**：

- `log_ratio`：单条样本的 $\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$
- `z_ref`：基线，是 desirable 样本 logit 比的 EMA，即 $z_{ref} \approx \beta \mathbb{E}[\log\frac{\pi_\theta}{\pi_{ref}}]$，需 `detach`
- `lambda_D` / `lambda_U`：好/坏样本权重，通常 $\lambda_U > \lambda_D$（损失厌恶）

### 一句话记忆

> **无需配对：好样本 $\beta\log r$ 推过 $z_{ref}$，坏样本压到 $z_{ref}$ 下，各自 $-\log\sigma$。**

### 伪代码

```
logit = beta * log_ratio                  # 单条样本的隐式奖励 × 温度
loss_desirable   = -log_sigmoid(logit - z_ref)        # 好样本: logit 超过基线
loss_undesirable = -log_sigmoid(z_ref - logit)        # 坏样本: logit 低于基线
loss = lambda_D * loss_desirable + lambda_U * loss_undesirable
```

### PyTorch 实现

```python
import torch
import torch.nn.functional as F

def kto_loss(log_ratio, is_desirable, z_ref=0.0,
             beta=0.1, lambda_D=1.0, lambda_U=1.33):
    """
    log_ratio: [B] = log_pi(y|x) - log_ref(y|x)
    is_desirable: [B] bool; z_ref 已含 β（TRL 约定）
    """
    logit = beta * log_ratio
    loss = torch.zeros_like(log_ratio)
    d = is_desirable
    u = ~is_desirable
    if d.any():
        loss[d] = lambda_D * -F.logsigmoid(logit[d] - z_ref)
    if u.any():
        loss[u] = lambda_U * -F.logsigmoid(z_ref - logit[u])
    return loss.mean()
```

---

## SimPO

**核心问题**：DPO 需要在显存里同时驻留当前策略和参考策略，开销大；且长回答天然 log 概率更低，会被不公平地惩罚。SimPO 用**长度归一化**代替参考模型，去掉 ref 同时消除长度偏差。

**核心变量**：

- `chosen_logps` / `rejected_logps`：当前策略对好/坏回答的 log 概率（序列求和）
- `chosen_lengths` / `rejected_lengths`：好/坏回答的 token 数
- `beta`（$\beta$）：偏好差距放大系数（SimPO 典型 2.0，比 DPO 大）
- `gamma`（$\gamma$）：目标 margin，相当于可学的"好回答要比坏回答高出多少"阈值

### 一句话记忆

> **DPO 不要 ref：logp 除以回答长度当好减坏，乘 β 再减 margin $\gamma$。**

### 伪代码

```
logp_w = log_pi(chosen)  / len(chosen)   # 长度归一化
logp_l = log_pi(rejected) / len(rejected)
logits = beta * (logp_w - logp_l) - gamma # 减 margin
loss   = -log_sigmoid(logits)
```

### PyTorch 实现

```python
import torch.nn.functional as F

def simpo_loss(chosen_logps, rejected_logps,
               chosen_lengths, rejected_lengths,
               beta=2.0, gamma=0.5):
    logp_w = chosen_logps / chosen_lengths
    logp_l = rejected_logps / rejected_lengths
    logits = beta * (logp_w - logp_l) - gamma
    return -F.logsigmoid(logits).mean()
```

---

## DPO 家族对比速查

| 算法  | 需要 ref? | 需要配对?            | 核心区别                                           |
| ----- | --------- | -------------------- | -------------------------------------------------- |
| DPO   | 是        | 是 (chosen/rejected) | $-\log\sigma(\beta\Delta)$，经典版                 |
| IPO   | 是        | 是                   | 平方损失 $(\Delta - \frac{1}{2\beta})^2$，避免饱和 |
| KTO   | 是        | 否（好/坏标签）      | 单样本 $\pm$ sigmoid + 基线 $z_{ref}$，损失厌恶    |
| SimPO | **否**    | 是                   | 长度归一化 log-prob + margin $\gamma$              |

注：$\Delta$ 为好减坏的 log-ratio 差。

---

## 易错点

| 易错                   | 说明                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| 四条 log_prob 搞混     | 记住：每个模型出两条（chosen + rejected），一共四条；ref 要 `detach`                             |
| log-prob 用错          | DPO/IPO/KTO 用的是 $\log\frac{\pi_\theta}{\pi_{ref}}$，**不是** raw log-prob                     |
| `log_sigmoid` 数值溢出 | PyTorch 的 `F.logsigmoid` 内置处理；手写时用 `logaddexp`                                         |
| $\beta$ 含义           | $\beta$ 是温度，越大对偏好差距越敏感；不是学习率                                                 |
| IPO 没有 sigmoid       | IPO 用平方损失回归到 $\frac{1}{2\beta}$，不用 sigmoid                                            |
| KTO 的 $z_{ref}$ 约定  | TRL 中 $z_{ref}$ 已带 $\beta$（是 logit 的 EMA）；写成 $\beta \cdot \text{log\_ratio} - z_{ref}$ |
| SimPO 忘除长度         | 长度归一化是 SimPO 的核心，长回答 log-prob 更小要除掉                                            |
| chosen/rejected 反了   | 检查数据集：chosen 是人类偏好的那条                                                              |
