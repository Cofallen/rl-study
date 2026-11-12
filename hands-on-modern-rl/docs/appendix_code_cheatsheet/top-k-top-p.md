# B.7 采样方法

解码策略是 LLM 面试的常考题，和 RL 直接相关（RLHF 训练后的策略怎么采样、temperature 如何影响动作分布）。

三种方法都从 logits 出发，差别只在采样前怎么处理：

- **Temperature**：缩放所有 logits，调节整体随机性。
- **Top-k**：截断到固定数量的 token，砍掉尾部噪声。
- **Top-p**：截断到固定概率质量，自适应分布形状。

---

## Temperature

**核心问题**：softmax 后的分布往往太"尖"或太"平"，难以控制采样随机性。

**核心变量**：

- `logits`：模型原始输出
- `temperature`（$T$，$T>0$）：softmax **之前**的缩放因子
- `scaled_logits = logits / T`：喂给 softmax 的输入

### 一句话记忆

> **softmax 前先除以 T。T 大变平更随机，T 小变尖更确定。**

### 伪代码

```
scaled_logits = logits / T        # softmax 之前除
probs = softmax(scaled_logits)
sample from probs
```

### 数学

$$
p_i = \frac{\exp(x_i / T)}{\sum_j \exp(x_j / T)}
$$

- $T \to 0$：趋向 argmax（贪婪）
- $T = 1$：原始分布
- $T \to \infty$：趋向均匀分布

### PyTorch 实现

```python
def sample_with_temperature(logits, temperature=1.0):
    if temperature < 1e-8:
        return logits.argmax(dim=-1)  # T=0 退化为贪婪
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## Top-k Sampling

**核心问题**：纯按概率采样时长尾低概率 token 偶尔被采到，导致输出胡言乱语。

**核心变量**：

- `k`：固定保留的 token 数量（典型 50）
- `threshold`：第 k 大的 logit，低于它的位置置 `-inf`

### 一句话记忆

> **只留前 k 个 logit，其余设 $-\infty$，softmax 自动重归一化后采样。**

### 伪代码

```
threshold = 第 k 大的 logit
logits[logits < threshold] = -inf
probs = softmax(logits)            # -inf 自动归零并重新归一化
sample from probs
```

### Python 实现

```python
import numpy as np

def top_k_filtering(logits, k):
    """logits: [vocab_size] -> 非 top-k 位置置 -inf"""
    if k >= len(logits):
        return logits
    threshold = np.sort(logits)[-k]  # 第 k 大的值（升序的倒数第 k 个）
    return np.where(logits >= threshold, logits, -np.inf)
```

### PyTorch 实现

```python
import torch

def top_k_filtering(logits, k):
    """logits: [B, vocab_size] 或 [vocab_size]"""
    if k <= 0:
        return logits
    top_k = min(k, logits.size(-1))
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float('-inf'))

def top_k_sample(logits, k, temperature=1.0):
    logits = top_k_filtering(logits / temperature, k)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## Top-p (Nucleus) Sampling

**核心问题**：Top-k 固定保留 k 个，但分布尖锐程度随上下文变化——确定性强时 3 个就够，不确定时 50 个都不够。Top-p 改成保留**累积概率质量 $\geq p$ 的最小 token 集合**（称为"核"），自适应分布形状。

**核心变量**：

- `p`：累积概率阈值（典型 0.9）
- `sorted_logits` / `sorted_indices`：降序排序后的 logits 及原索引
- `cumulative_probs`：从高到低的累积概率
- `nucleus_mask`：累积值（减去当前 prob）超过 p 的位置

### 一句话记忆

> **按概率降序排，累加到 p 就停，剩下的设 $-\infty$。**

### 伪代码

```
sorted_logits, idx = sort_desc(logits)
sorted_probs = softmax(sorted_logits)
cumsum = cumsum(sorted_probs)
mask = cumsum - sorted_probs > p     # 减当前 prob 保证至少留一个
sorted_logits[mask] = -inf
logits = scatter_back(sorted_logits, idx)  # 还原原顺序
probs = softmax(logits); sample
```

### 对比

|          | Top-k         | Top-p                         |
| -------- | ------------- | ----------------------------- |
| 筛选依据 | 固定保留 k 个 | 保留累积概率达到 p 的最小集合 |
| 适应性   | 不随分布变化  | 自适应分布尖锐程度            |
| 极端情况 | k=1 → 贪婪    | p=0 → 贪婪，p=1 → 不限制      |

### Python 实现

```python
import numpy as np

def top_p_filtering(logits, p):
    """logits: [vocab_size] -> 核外位置置 -inf"""
    sorted_indices = np.argsort(logits)[::-1]           # 降序索引
    sorted_logits = logits[sorted_indices]
    sorted_probs = np.exp(sorted_logits - sorted_logits.max())
    sorted_probs /= sorted_probs.sum()
    cumulative_probs = np.cumsum(sorted_probs)

    # 核外位置：累积值减去当前 prob 超过 p（保证至少保留一个 token）
    cutoff = cumulative_probs - sorted_probs > p
    sorted_logits[cutoff] = -np.inf

    result = np.full_like(logits, -np.inf)
    result[sorted_indices] = sorted_logits              # 还原原顺序
    return result
```

### PyTorch 实现

```python
import torch

def top_p_filtering(logits, p):
    """logits: [B, vocab_size] -> 核外位置置 -inf"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = (cumulative_probs - sorted_probs) > p  # 核外掩码
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float('-inf'))

    return logits.scatter(1, sorted_indices, sorted_logits)  # 还原原顺序

def top_p_sample(logits, p, temperature=1.0):
    logits = top_p_filtering(logits / temperature, p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## 实际组合用法

工程中通常 Temperature → Top-k → Top-p 串联：

```python
def generate_sample(logits, temperature=1.0, top_k=50, top_p=0.9):
    logits = logits / max(temperature, 1e-8)   # 1. Temperature（softmax 前）
    logits = top_k_filtering(logits, top_k)    # 2. Top-k
    logits = top_p_filtering(logits, top_p)    # 3. Top-p
    probs = torch.softmax(logits, dim=-1)      # 4. 归一化并采样
    return torch.multinomial(probs, num_samples=1)
```

---

## 易错点

| 易错                 | 说明                                                                                   |
| -------------------- | -------------------------------------------------------------------------------------- |
| Temperature 顺序     | 必须在 softmax **之前**除 T，不是对概率除                                              |
| Top-p 的 cumsum 方向 | 必须**降序排列**后再 cumsum，升序无意义                                                |
| Top-p 保留至少一个   | 用 `cumsum - current_prob > p` 而非 `cumsum > p`，否则首个（最高概率）token 可能被误杀 |
| Top-k 阈值           | 用 `topk().values[..., -1]` 取第 k 大的值，不要用 sort 后取索引                        |
| Top-p 还原顺序       | 排序后必须 `scatter` 回原位，否则采样错乱                                              |
| 采样前重新归一化     | 置 `-inf` 后必须再过一次 softmax，让剩余 token 概率重新和为 1                          |
| `temperature=0`      | 特殊处理为 argmax，不要真的除以 0                                                      |
