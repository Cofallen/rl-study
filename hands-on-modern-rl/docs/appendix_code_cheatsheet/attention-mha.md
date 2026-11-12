# B.8 注意力机制

严格来说不属于 RL 算法，但它是大模型岗面试中**出现频率前三**的手写代码题，RL 岗面试也常作为前置知识考查。

---

## Scaled Dot-Product Attention

**核心问题**：让序列中每个位置按相关度加权聚合上下文信息。

**核心变量**：

- `Q`（query）：当前位置"我要查什么"，形状 $[seq_q, d_k]$
- `K`（key）：每个位置"我能被怎样匹配"，形状 $[seq_k, d_k]$
- `V`（value）：每个位置"我携带的信息"，形状 $[seq_k, d_v]$
- `d_k`：query/key 的维度，点积后除以 $\sqrt{d_k}$ 防止 softmax 饱和
- `mask`：causal mask 把"未来"位置置 $-\infty$，softmax 后归零

### 一句话记忆

> **Q 和 K 点积打分，除根号、遮未来、softmax、再乘 V。**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $QK^T$ 形状为 $[seq_q, d_k]\times[d_k, seq_k]=[seq_q, seq_k]$，softmax 沿最后一维（key 维度）归一化。

### 伪代码

```
scores = Q @ K^T / sqrt(d_k)      # 打分 + 缩放，[seq_q, seq_k]
if mask: scores = scores + mask    # 遮未来位置为 -inf
attn = softmax(scores, dim=-1)     # 沿 key 维归一化
output = attn @ V                  # 加权聚合，[seq_q, d_v]
```

### Python 实现

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [seq_q, d_k], K: [seq_k, d_k], V: [seq_k, d_v]
    mask: [seq_q, seq_k], 0=保留, -inf=遮掩
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)        # [seq_q, seq_k]

    if mask is not None:
        scores = scores + mask

    # 数值稳定的 softmax，沿最后一维（key 维）
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    return attn_weights @ V                # [seq_q, d_v]
```

### PyTorch 实现

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q/K/V: [B, heads, seq, d_k]
    mask: [B, 1, seq_q, seq_k], 1=保留, 0=遮掩
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


def causal_mask(seq_len):
    """下三角 = 1（保留），上三角 = 0（遮未来）。"""
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
```

---

## Multi-Head Attention (MHA)

**核心问题**：单头 attention 只能学到一种关系模式。MHA 把 $d_{model}$ 切成 $h$ 份，每个头独立做 attention，让模型在不同子空间同时建模多种关系。

**核心变量**：

- `d_model`：模型隐藏维度
- `n_heads` / $h$：头数，每头维度 $d_k = d_{model}/h$
- `W_Q` / `W_K` / `W_V` / `W_O`：四个线性投影

### 一句话记忆

> **总维度切 h 份，每份独立做 attention，concat 后过 $W_O$。**

### 伪代码

```
Q, K, V = x @ W_Q, x @ W_K, x @ W_V      # [B, seq, d_model]
Q, K, V = split_heads(Q, K, V)           # [B, h, seq, d_k]
attn = scaled_dot_product_attention(Q, K, V, mask)
attn = merge_heads(attn)                  # [B, seq, d_model]
output = attn @ W_O
```

维度变换口诀：**view 切头 → transpose 换位 → attention → transpose 回 → view 合头**。

### PyTorch 实现

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

        # 投影 + 切头: [B, seq, d_model] -> [B, n_heads, seq, d_k]
        Q = self.W_Q(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        attn_out = scaled_dot_product_attention(Q, K, V, mask)

        # 合头 + 输出投影
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, d_model)
        return self.W_O(attn_out)
```

---

## MQA 与 GQA

**核心问题**：MHA 中 K/V 的头数与 Q 相同，推理时 KV cache 占用大。MQA 让所有 Q 头共用一组 K/V，最省显存；GQA 折中——Q 分组、组内共用 K/V，平衡显存与质量。

**核心变量**：

- `n_heads`：Q 的头数
- `n_kv_heads`：K/V 的头数（MQA=1，GQA=$g$，$1<g<h$）
- `n_groups`：$= n_{heads}/n_{kv\_heads}$，每组共享同一组 K/V 的 Q 头数

### 对比速查

| 变体 | Q 头数 | K/V 头数        | KV cache | 代表模型           |
| ---- | ------ | --------------- | -------- | ------------------ |
| MHA  | $h$    | $h$             | 最大     | GPT-2、BERT        |
| MQA  | $h$    | **1**           | 最小     | PaLM、StarCoder    |
| GQA  | $h$    | **$g$** ($g<h$) | 折中     | LLaMA 2/3、Mistral |

### 一句话记忆

> **MQA：所有 Q 头共用一组 K/V（最省）；GQA：分 $g$ 组共用（折中）；MQA 是 $g=1$ 的 GQA。**

### PyTorch 实现（GQA）

```python
import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        """
        n_heads: Q 头数（如 32）
        n_kv_heads: K/V 头数（如 8），必须整除 n_heads
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

        # 沿 head 维扩展: [B, n_kv_heads, seq, d_k] -> [B, n_heads, seq, d_k]
        # 相邻 n_groups 个 Q 头共享同一组 K/V
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.W_O(attn_out)
```

---

## 计算复杂度

| 项             | 复杂度             | 说明                   |
| -------------- | ------------------ | ---------------------- |
| Self-Attention | $O(n^2 \cdot d)$   | $n$ 序列长度，$d$ 维度 |
| 线性投影       | $O(n \cdot d^2)$   | 每个 token 过线性层    |
| 总计（MHA）    | $O(n^2 d + n d^2)$ | 长序列时 $n^2$ 项主导  |

---

## 易错点

| 易错                                    | 说明                                                         |
| --------------------------------------- | ------------------------------------------------------------ |
| 除 $\sqrt{d_k}$ 不是 $\sqrt{d_{model}}$ | 用每个头的维度 $d_k = d_{model}/h$                           |
| softmax 沿最后一维                      | 沿 key 维归一化，每行（每个 query）权重和为 1                |
| causal mask 方向                        | `tril` 下三角 = 保留，上三角 = 遮未来                        |
| view 前 contiguous                      | `transpose` 后内存不连续，`.contiguous()` 再 `view`          |
| GQA 用 `repeat_interleave`              | 不是 `repeat`；前者让相邻 $n_{groups}$ 个 Q 头共享同一组 K/V |
| MQA 是 GQA 的特例                       | $n_{kv\_heads}=1$ 时 GQA 退化为 MQA                          |
