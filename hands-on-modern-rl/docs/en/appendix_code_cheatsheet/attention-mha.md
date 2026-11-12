---
title: B.8 Attention Mechanism
---

# B.8 Attention Mechanism

Strictly speaking this is not an RL algorithm, but it is one of the top three most frequent "handwrite the code" questions in LLM interviews, and RL interviews often use it as prerequisite knowledge.

---

## Scaled Dot-Product Attention

**Core question**: let every position in a sequence aggregate context information weighted by relevance.

**Core variables**:

- `Q` (query): "what am I looking for" at the current position, shape $[seq_q, d_k]$
- `K` (key): "how can I be matched" at each position, shape $[seq_k, d_k]$
- `V` (value): "what information I carry" at each position, shape $[seq_k, d_v]$
- `d_k`: dimension of query/key; dot product is divided by $\sqrt{d_k}$ to prevent softmax saturation
- `mask`: a causal mask pushes "future" positions to $-\infty$, which softmax turns into zero

### One-Line Memory

> Score Q against K, divide by the square root, mask the future, softmax, then multiply by V.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Here $QK^T$ has shape $[seq_q, d_k]\times[d_k, seq_k]=[seq_q, seq_k]$, and softmax normalizes along the last dimension (the key dimension).

### Pseudocode

```
scores = Q @ K^T / sqrt(d_k)      # score + scale, [seq_q, seq_k]
if mask: scores = scores + mask    # push future positions to -inf
attn = softmax(scores, dim=-1)     # normalize over key dim
output = attn @ V                  # weighted aggregate, [seq_q, d_v]
```

### Python Implementation

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [seq_q, d_k], K: [seq_k, d_k], V: [seq_k, d_v]
    mask: [seq_q, seq_k], 0=keep, -inf=mask
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)        # [seq_q, seq_k]

    if mask is not None:
        scores = scores + mask

    # numerically stable softmax along the last dim (key dim)
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    return attn_weights @ V                # [seq_q, d_v]
```

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q/K/V: [B, heads, seq, d_k]
    mask: [B, 1, seq_q, seq_k], 1=keep, 0=mask
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


def causal_mask(seq_len):
    """Lower triangle = 1 (keep), upper triangle = 0 (mask future)."""
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
```

---

## Multi-Head Attention (MHA)

**Core question**: a single attention head can only learn one relation pattern. MHA splits $d_{model}$ into $h$ slices, each running attention independently, letting the model attend to multiple subspaces in parallel.

**Core variables**:

- `d_model`: model hidden dimension
- `n_heads` / $h$: number of heads, each with $d_k = d_{model}/h$
- `W_Q` / `W_K` / `W_V` / `W_O`: four linear projections

### One-Line Memory

> Slice the total dimension into h pieces; each runs attention on its own; concat, then project through $W_O$.

### Pseudocode

```
Q, K, V = x @ W_Q, x @ W_K, x @ W_V      # [B, seq, d_model]
Q, K, V = split_heads(Q, K, V)           # [B, h, seq, d_k]
attn = scaled_dot_product_attention(Q, K, V, mask)
attn = merge_heads(attn)                  # [B, seq, d_model]
output = attn @ W_O
```

Shape mnemonic: **view to split heads -> transpose to move head dim -> attention -> transpose back -> view to merge**.

### PyTorch Implementation

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, seq_len, d_model = x.shape

        # projections + split heads: [B, seq, d_model] -> [B, n_heads, seq, d_k]
        Q = self.W_Q(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        attn_out = scaled_dot_product_attention(Q, K, V, mask)

        # merge heads + output projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, d_model)
        return self.W_O(attn_out)
```

---

## MQA and GQA

**Core question**: in MHA the number of K/V heads equals the number of Q heads, so KV cache is large at inference. MQA makes all Q heads share one set of K/V (smallest cache); GQA is the middle ground — Q heads are grouped, each group shares one set of K/V, trading off cache and quality.

**Core variables**:

- `n_heads`: number of Q heads
- `n_kv_heads`: number of K/V heads (MQA=1, GQA=$g$, $1<g<h$)
- `n_groups`: $= n_{heads}/n_{kv\_heads}$, the number of Q heads sharing one set of K/V

### Quick Comparison

| Variant | # Q heads | # K/V heads     | KV cache | Example models     |
| ------- | --------- | --------------- | -------- | ------------------ |
| MHA     | $h$       | $h$             | largest  | GPT-2, BERT        |
| MQA     | $h$       | **1**           | smallest | PaLM, StarCoder    |
| GQA     | $h$       | **$g$** ($g<h$) | middle   | LLaMA 2/3, Mistral |

### One-Line Memory

> **MQA**: all Q heads share one set of K/V (smallest cache); **GQA**: split into $g$ groups sharing (middle ground); MQA is GQA with $g=1$.

### PyTorch Implementation (GQA)

```python
import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        """
        n_heads: number of query heads (e.g. 32)
        n_kv_heads: number of key/value heads (e.g. 8); must divide n_heads
        """
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, n_heads * self.d_k)
        self.W_K = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_V = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_O = nn.Linear(n_heads * self.d_k, d_model)

    def forward(self, x, mask=None):
        B, seq_len, _ = x.shape

        Q = self.W_Q(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # expand along head dim: [B, n_kv_heads, seq, d_k] -> [B, n_heads, seq, d_k]
        # adjacent n_groups Q heads share one set of K/V
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.W_O(attn_out)
```

---

## Follow-Up: Complexity

| Item           | Complexity         | Notes                                |
| -------------- | ------------------ | ------------------------------------ |
| self-attention | $O(n^2 \cdot d)$   | $n$ sequence length, $d$ hidden size |
| linear layers  | $O(n \cdot d^2)$   | per-token linear projections         |
| total (MHA)    | $O(n^2 d + n d^2)$ | when $n$ is large, $n^2$ dominates   |

---

## Common Pitfalls

| Pitfall                                        | Explanation                                                           |
| ---------------------------------------------- | --------------------------------------------------------------------- |
| Divide by $\sqrt{d_k}$, not $\sqrt{d_{model}}$ | Use per-head dimension $d_k = d_{model}/h$.                           |
| softmax along the last dimension               | Normalize over the key dim; each row (each query) sums to 1.          |
| Causal mask direction                          | `tril` lower triangle = keep, upper triangle = mask future.           |
| `view` after `transpose`                       | `transpose` makes tensors non-contiguous; call `.contiguous()` first. |
| Use `repeat_interleave` for GQA                | Not `repeat`; this makes adjacent $n_{groups}$ Q heads share one K/V. |
| MQA is a special case of GQA                   | When $n_{kv\_heads}=1$, GQA reduces to MQA.                           |
