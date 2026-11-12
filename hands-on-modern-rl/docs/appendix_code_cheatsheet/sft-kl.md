# B.1 SFT 与 KL 散度

> 面试前 30 分钟翻一遍，每个算法记住一句话 + 一个公式就够了。

本附录覆盖大模型后训练岗位面试中最常被要求手写的算法代码，按考查频率排序。每个算法提供四种视角：

| 视角             | 用途                              |
| ---------------- | --------------------------------- |
| **一句话记忆**   | 上考场前默念的口诀                |
| **伪代码**       | 面试白板写的版本                  |
| **Python 实现**  | 用 numpy / 原生 Python 讲清楚逻辑 |
| **PyTorch 实现** | 面试中常问的工程版本              |

## 本附录目录

| 节                                           | 算法                                        | 考查频率 |
| -------------------------------------------- | ------------------------------------------- | -------- |
| [B.1 SFT Loss 与 KL 散度](./sft-kl)          | SFT 自回归 loss、shift right、KL 估计       | ★★★★     |
| [B.2 PPO 策略损失与 GAE](./ppo-gae)          | Clipped surrogate、value loss、GAE 逆向递推 | ★★★★★    |
| [B.3 DPO 及其变体](./dpo-family)             | DPO loss、IPO、KTO、SimPO                   | ★★★★★    |
| [B.4 GRPO 与 Reward Model](./grpo-rlvr)      | GRPO 组内归一化、Bradley-Terry RM           | ★★★★     |
| [B.6 Softmax 与 Cross-Entropy](./softmax-ce) | 数值稳定 softmax、log-sum-exp、CE loss      | ★★★★     |
| [B.7 Top-k / Top-p Sampling](./top-k-top-p)  | Temperature、Top-k、Top-p (Nucleus) 解码    | ★★★★     |
| [B.8 Attention / MHA / GQA](./attention-mha) | Scaled dot-product、多头注意力、MQA、GQA    | ★★★★★    |
| [B.5 DAPO](./dapo)                           | 解耦裁剪、动态采样、超长惩罚                | ★★★      |

## 使用建议

1. **先背一句话**。每个算法开头都有一句口诀，记住它就能推导出伪代码。
2. **伪代码为主**。面试白板场景下，写出伪代码 + 讲清楚变量含义即可过关。
3. **PyTorch 补细节**。如果面试官追问实现细节（如 `ignore_index`、`log_sum_exp`、`clamp`），翻到对应 PyTorch 代码段。
4. **易错点速查**。每个文件末尾列了高频踩坑项，面试前一晚过一遍。

## SFT Loss（自回归交叉熵）

**核心问题**：在每个位置预测下一个 token，且只在回答部分计算 loss。

**核心变量**：

- `logits`：模型输出，形状 `[B, seq_len, vocab_size]`，位置 $t$ 预测 $t+1$
- `labels`：真实 token 序列，prompt 部分标 `ignore_index=-100`
- `ignore_index`：交叉熵跳过该位置（默认 `-100`）

### 一句话记忆

> **logits 砍尾、labels 砍头：位置 $t$ 预测 $t+1$；prompt 标 `-100`，不进 loss。**

### 伪代码

```
logits = model(input_ids)                # 位置 t 预测 t+1
shift_logits = logits[:, :-1, :]         # 砍尾：句末无"下一个"
shift_labels = labels[:, 1:]             # 砍头：句首无人预测
loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

自回归模型在位置 $t$ 预测 $t+1$，故 logits 的第 $t$ 位对齐 labels 的第 $t+1$ 位。

### Python 实现

```python
import numpy as np

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)  # 先减 max，防溢出
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def sft_loss(logits, labels, ignore_index=-100):
    """
    logits: [seq_len, vocab_size]
    labels: [seq_len]  (未 shift)
    """
    shift_logits = logits[:-1]
    shift_labels = labels[1:]

    probs = softmax(shift_logits, axis=-1)
    total, count = 0.0, 0
    for t in range(len(shift_labels)):
        if shift_labels[t] == ignore_index:
            continue
        total += -np.log(probs[t, shift_labels[t]] + 1e-12)
        count += 1
    return total / max(count, 1)
```

### PyTorch 实现

```python
import torch
import torch.nn.functional as F

def sft_loss(logits, labels, ignore_index=-100):
    """
    logits: [B, seq_len, vocab_size]
    labels: [B, seq_len]
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
```

---

## KL 散度估计

**核心问题**：估计当前策略 $p$ 与参考策略 $q$ 的差异，用于 PPO / GRPO 的 KL 惩罚。

**核心变量**：

- `log_probs`：当前策略 $p$ 对采样 token 的 log 概率
- `ref_log_probs`：参考策略 $q$（通常冻结的 SFT 模型）对同一批 token 的 log 概率
- `log_ratio`：$\log(q/p)$，k3 的核心量

### 一句话记忆

> **k1：`mean(log_p − log_q)`，简单无偏但能负；k3：`mean(exp(Δ) − 1 − Δ)`，$\Delta=\log\frac{q}{p}$，恒非负。**

### 伪代码

```
# k1（PPO 常用） 与 直接平均，无偏但高方差，样本少时可能为负
kl = (log_probs - ref_log_probs).mean()

# k3（GRPO / trl 默认） 与 恒非负，ratio 方向 q/p
log_ratio = ref_log_probs - log_probs        # log(q/p)
kl = (exp(log_ratio) - 1 - log_ratio).mean()
```

### Python 实现

```python
import numpy as np

def kl_k1(log_p, log_q):
    """E_p[log p - log q]：无偏，高方差，样本少时可能为负"""
    return np.mean(log_p - log_q)

def kl_k3(log_p, log_q):
    """E_p[exp(log q - log p) - 1 - (log q - log p)]：无偏且恒非负"""
    log_ratio = log_q - log_p
    return np.mean(np.exp(log_ratio) - 1 - log_ratio)
```

### PyTorch 实现

```python
import torch

def kl_penalty(log_probs, ref_log_probs, mode="k3"):
    """
    log_probs:     [B, seq_len]  当前策略 p
    ref_log_probs: [B, seq_len]  参考策略 q
    """
    if mode == "k1":
        return (log_probs - ref_log_probs).mean()

    log_ratio = ref_log_probs - log_probs   # log(q/p)
    return (torch.exp(log_ratio) - 1 - log_ratio).mean()
```

### 两种估计的对比

样本来自 $p$，目标 $\text{KL}(p \| q)$：

| 估计器 | 公式                                               | 特点                         |
| ------ | -------------------------------------------------- | ---------------------------- |
| k1     | $\mathbb{E}_p[\log \frac{p}{q}]$                   | 无偏，简单，样本少时可能为负 |
| k3     | $\mathbb{E}_p[\frac{q}{p} - 1 - \log \frac{q}{p}]$ | 无偏，恒 $\geq 0$，GRPO 默认 |

::: warning 易错点
k3 中 ratio 必须是 $q/p$（ref/current）。由 $e^x - 1 - x \geq 0$ 对所有实数 $x$ 成立，保证非负；写反成 $p/q$ 后虽仍非负，但期望不再是 $\text{KL}(p \| q)$。
:::

---

## 易错点

| 易错                | 说明                                                  |
| ------------------- | ----------------------------------------------------- |
| shift 方向反了      | logits 砍**尾**，labels 砍**头**：位置 $t$ 预测 $t+1$ |
| 忘了 `ignore_index` | prompt 部分 token 标 `-100`，不计入 loss              |
| k3 ratio 方向反     | 必须是 $q/p$（ref/current）；写反期望偏离真值         |
| k1 样本太少         | 单批样本可能算出负数，是估计噪声，非 bug              |
| softmax 溢出        | 先减 `max(x)` 再 `exp`                                |
| `.contiguous()`     | PyTorch slice 后 `view` 可能报错，加 `.contiguous()`  |
