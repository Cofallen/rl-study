---
title: B.6 Softmax and Cross-Entropy
---

# B.6 Softmax and Cross-Entropy

A common warm-up question. Before you write DPO or PPO on the whiteboard, an interviewer may ask you to handwrite a numerically stable softmax and cross-entropy.

---

## Numerically Stable Softmax

**Core problem**: map arbitrary real-valued logits to a probability distribution (summing to 1) while avoiding overflow such as $\exp(1000)=\text{inf}$.

**Core variables**:

- `x` / `logits`: the model's real-valued output vector
- `m = max(x)`: the shift constant; every element subtracts it
- `axis`: the normalization axis, the last dim (vocab) in LLMs

### One-Line Memory

> Subtract the max, then exp, then sum, then divide.

### Pseudocode

```
m = max(x)
exp_x = exp(x - m)
softmax = exp_x / sum(exp_x)
```

### Python Implementation

```python
import numpy as np


def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x_shifted)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
```

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F

# Use the built-in version in real code
probs = F.softmax(logits, dim=-1)


# Handwritten version (interview)
def manual_softmax(x, dim=-1):
    x_shifted = x - x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x_shifted)
    return e_x / e_x.sum(dim=dim, keepdim=True)
```

---

## Log-Sum-Exp and Log-Softmax

**Core problem**: LLM training needs log-probabilities, not probabilities. Computing softmax first and then taking log loses precision, and tiny probabilities can underflow to 0 before log turns them into `-inf`. Log-sum-exp merges the subtract-max and log into one step, yielding numerically stable log-probabilities.

**Core variables**:

- `m = max(x)`: same shift constant as softmax
- `lse = m + log(sum(exp(x - m)))`: the log of the logistic normalizer
- Output: `log_softmax(x)_i = x_i - m - log(sum(exp(x - m)))`

The identity is:

$$
\log\sum_j \exp(x_j) = m + \log\sum_j \exp(x_j - m), \quad m = \max(x)
$$

### One-Line Memory

> Don't softmax then log — $\log\text{softmax}_i = x_i - \text{LSE}(x)$, with max subtracted inside LSE to avoid overflow.

### Python Implementation

```python
def log_softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    return x_shifted - np.log(np.sum(np.exp(x_shifted), axis=axis, keepdims=True))
```

### PyTorch Implementation

```python
import torch.nn.functional as F

# Built-in and numerically stable
log_probs = F.log_softmax(logits, dim=-1)


def manual_log_softmax(x, dim=-1):
    max_val = x.max(dim=dim, keepdim=True).values
    return x - max_val - torch.log(torch.sum(torch.exp(x - max_val), dim=dim, keepdim=True))
```

---

## Cross-Entropy Loss

**Core problem**: classification / SFT tasks need a scalar loss that measures the gap between the predicted distribution and the true label. Cross-entropy reduces it to "the negative log-probability at the target position" — the better the prediction, the smaller the loss.

**Core variables**:

- `logits`: model output of shape `[N, C]`, N samples and C classes
- `targets`: true class indices of shape `[N]`
- `ignore_index`: positions to skip (e.g. padding / prompt), default `-100`
- `log_probs`: log-probabilities after log_softmax, used to pick the target position

When $p$ is one-hot (1 at the label position), cross-entropy collapses to:

$$
H(p, q) = -\sum_i p_i \log q_i \;=\; -\log q_{\text{label}}
$$

### One-Line Memory

> `-log_softmax(logits)[target].mean()` — one step.

### Pseudocode

```
log_probs = log_softmax(logits)
loss = -log_probs[target].mean()
```

### Python Implementation

```python
def cross_entropy(logits, targets, ignore_index=-100):
    """
    logits:  [N, C]
    targets: [N] integer class labels
    """
    log_probs = log_softmax(logits, axis=-1)
    total, count = 0.0, 0
    for i in range(len(targets)):
        if targets[i] == ignore_index:
            continue
        total += -log_probs[i, targets[i]]
        count += 1
    return total / max(count, 1)
```

### PyTorch Implementation

```python
def manual_cross_entropy(logits, targets, ignore_index=-100):
    """
    logits:  [B, C]
    targets: [B]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # gather selects log-prob at target index
    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    # mask out ignore_index
    mask = targets != ignore_index
    return -target_log_probs[mask].mean()
```

---

## Common Pitfalls

| Pitfall                         | Explanation                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------- |
| Forgot to subtract max          | The first thing interviewers look for.                                          |
| Softmax then log                | Numerically unstable. Use `log_softmax` directly.                               |
| Computing CE from probabilities | Do not do `softmax -> log -> CE`; use `F.cross_entropy(logits, targets)`.       |
| `ignore_index` handling         | In SFT loss questions, interviewers ask how you handle padding/prompt tokens.   |
| Temperature scaling             | Do `logits / temperature` before softmax. Larger $T$ flattens the distribution. |
