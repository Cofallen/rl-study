---
title: B.7 Sampling Methods
---

# B.7 Sampling Methods

Decoding strategies are a frequent LLM interview topic, and they connect directly to RL: how do you sample from a policy after RLHF? How does temperature change the action distribution?

All three methods start from logits and differ only in how they shape the sampling distribution:

- **Temperature**: scales all logits, controlling overall randomness.
- **Top-k**: truncates to a fixed number of tokens, removing the long tail.
- **Top-p**: truncates to a fixed probability mass, adapting to distribution shape.

---

## Temperature

**Core problem**: the softmax distribution is often too peaked or too flat, making sampling randomness hard to control.

**Core variables**:

- `logits`: raw model output
- `temperature` ($T$, $T>0$): scaling factor applied **before** softmax
- `scaled_logits = logits / T`: the input to softmax

### One-Line Memory

> Divide by T before softmax. Large T flattens (more random); small T sharpens (more confident).

### Pseudocode

```
scaled_logits = logits / T        # divide BEFORE softmax
probs = softmax(scaled_logits)
sample from probs
```

### Math

$$
p_i = \frac{\exp(x_i / T)}{\sum_j \exp(x_j / T)}
$$

- $T \to 0$: approaches argmax (greedy)
- $T = 1$: original distribution
- $T \to \infty$: approaches uniform

### PyTorch Implementation

```python
def sample_with_temperature(logits, temperature=1.0):
    if temperature < 1e-8:
        return logits.argmax(dim=-1)  # T=0 degenerates to greedy
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## Top-k Sampling

**Core problem**: pure probabilistic sampling occasionally draws a long-tail low-probability token, producing gibberish.

**Core variables**:

- `k`: the fixed number of tokens to keep (typical 50)
- `threshold`: the k-th largest logit; any value below it is set to `-inf`

### One-Line Memory

> Keep the top k logits, set the rest to $-\infty$, softmax re-normalizes automatically.

### Pseudocode

```
threshold = k-th largest logit
logits[logits < threshold] = -inf
probs = softmax(logits)            # -inf -> 0, others re-normalize
sample from probs
```

### Python (NumPy) Implementation

```python
import numpy as np

def top_k_filtering(logits, k):
    """logits: [vocab_size] -> non-top-k positions set to -inf"""
    if k >= len(logits):
        return logits
    threshold = np.sort(logits)[-k]  # k-th largest value (k-th from the end of ascending sort)
    return np.where(logits >= threshold, logits, -np.inf)
```

### PyTorch Implementation

```python
import torch

def top_k_filtering(logits, k):
    """logits: [B, vocab_size] or [vocab_size]"""
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

**Core problem**: Top-k keeps a fixed count, but distribution sharpness varies with context — 3 tokens may suffice when confident, while 50 may be too few when uncertain. Top-p keeps the **smallest token set whose cumulative mass $\geq p$** (the "nucleus"), adapting to the distribution shape.

**Core variables**:

- `p`: cumulative probability threshold (typical 0.9)
- `sorted_logits` / `sorted_indices`: logits sorted descending and the original indices
- `cumulative_probs`: cumulative probability from the top down
- `nucleus_mask`: positions where the running mass (minus current prob) exceeds p

### One-Line Memory

> Sort by probability descending, accumulate until the running total reaches p, mask the rest with $-\infty$.

### Pseudocode

```
sorted_logits, idx = sort_desc(logits)
sorted_probs = softmax(sorted_logits)
cumsum = cumsum(sorted_probs)
mask = cumsum - sorted_probs > p     # subtract current prob to keep at least one
sorted_logits[mask] = -inf
logits = scatter_back(sorted_logits, idx)  # restore original order
probs = softmax(logits); sample
```

### Comparison

|                | Top-k          | Top-p                                             |
| -------------- | -------------- | ------------------------------------------------- |
| Selection rule | keep exactly k | keep smallest set whose cumulative mass reaches p |
| Adaptivity     | does not adapt | adapts to distribution sharpness                  |
| Extremes       | k=1 -> greedy  | p=0 -> greedy, p=1 -> no restriction              |

### Python (NumPy) Implementation

```python
import numpy as np

def top_p_filtering(logits, p):
    """logits: [vocab_size] -> positions outside nucleus set to -inf"""
    sorted_indices = np.argsort(logits)[::-1]           # descending order
    sorted_logits = logits[sorted_indices]
    sorted_probs = np.exp(sorted_logits - sorted_logits.max())
    sorted_probs /= sorted_probs.sum()
    cumulative_probs = np.cumsum(sorted_probs)

    # Outside the nucleus: (cumsum - current prob) > p, ensuring at least one token kept
    cutoff = cumulative_probs - sorted_probs > p
    sorted_logits[cutoff] = -np.inf

    result = np.full_like(logits, -np.inf)
    result[sorted_indices] = sorted_logits              # restore original order
    return result
```

### PyTorch Implementation

```python
import torch

def top_p_filtering(logits, p):
    """logits: [B, vocab_size] -> positions outside nucleus set to -inf"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = (cumulative_probs - sorted_probs) > p  # outside-nucleus mask
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float('-inf'))

    return logits.scatter(1, sorted_indices, sorted_logits)  # restore original order

def top_p_sample(logits, p, temperature=1.0):
    logits = top_p_filtering(logits / temperature, p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## Typical Combined Usage

In practice, Temperature -> Top-k -> Top-p are usually chained:

```python
def generate_sample(logits, temperature=1.0, top_k=50, top_p=0.9):
    logits = logits / max(temperature, 1e-8)   # 1. Temperature (before softmax)
    logits = top_k_filtering(logits, top_k)    # 2. Top-k
    logits = top_p_filtering(logits, top_p)    # 3. Top-p
    probs = torch.softmax(logits, dim=-1)      # 4. normalize and sample
    return torch.multinomial(probs, num_samples=1)
```

---

## Common Pitfalls

| Pitfall                      | Explanation                                                                                   |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| Temperature order            | Must divide by T **before** softmax, not divide the probabilities                             |
| Top-p cumsum direction       | Sort **descending** before cumsum; ascending order is meaningless                             |
| Top-p keep at least one      | Use `cumsum - current_prob > p`, not `cumsum > p`, or the top (highest-prob) token may be cut |
| Top-k threshold              | Use `topk().values[..., -1]` for the k-th largest; do not index after sort                    |
| Top-p restore order          | After sorting, `scatter` back to original positions or sampling breaks                        |
| Re-normalize before sampling | After setting `-inf`, run softmax again so remaining tokens sum to 1                          |
| `temperature=0` edge case    | Treat as argmax; do not actually divide by zero                                               |
