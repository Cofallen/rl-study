---
title: B.3 DPO Family
---

# B.3 DPO Methods

DPO loss is the single most frequently requested "handwritten code" question in post-training interviews — almost every interview covers it. This section gives one-page cheatsheets for DPO, IPO, KTO, and SimPO.

---

## DPO Loss

**Core problem**: Bypass the full RLHF pipeline (train a reward model + PPO) by directly fine-tuning the policy on preference pairs $(y_w, y_l)$. This is equivalent to maximum-likelihood under a Bradley-Terry model on implicit rewards, with no online sampling and no critic.

**Core variables**:

- `pi_chosen` / `pi_rejected`: log-probabilities of the chosen/rejected response under the current policy $\pi_\theta$
- `ref_chosen` / `ref_rejected`: log-probabilities under the reference policy $\pi_{ref}$ (must `detach`)
- `log_ratio_w` / `log_ratio_l`: $\log\frac{\pi_\theta}{\pi_{ref}}$, the implicit reward per response (divided by $\beta$)
- `beta` ($\beta$): temperature controlling how strongly the policy can drift from the reference; larger = more sensitive, typical 0.1–0.5

### One-Line Memory

> **4 log-probs (2 models × 2 responses); each takes "current − ref"; chosen minus rejected, scale by β, sigmoid, negative log.**

$$\mathcal{L}_{DPO} = -\mathbb{E}\Big[\log\sigma\Big(\beta\Big(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\Big)\Big]$$

### Pseudocode

```
log_ratio_w = log_pi_theta(y_w|x) - log_pi_ref(y_w|x)   # implicit reward, chosen
log_ratio_l = log_pi_theta(y_l|x) - log_pi_ref(y_l|x)   # implicit reward, rejected
logits      = beta * (log_ratio_w - log_ratio_l)        # advantage × temperature
loss        = -log_sigmoid(logits)                       # max advantage ↔ BT likelihood
```

### Python (NumPy) Implementation

```python
import numpy as np

def log_sigmoid(x):
    return -np.logaddexp(0, -x)  # numerically stable

def dpo_loss(logp_chosen, logp_rejected,
             logp_ref_chosen, logp_ref_rejected,
             beta=0.1):
    """
    All inputs: scalar or [B]. Returns a scalar loss.
    """
    log_ratio_w = logp_chosen - logp_ref_chosen
    log_ratio_l = logp_rejected - logp_ref_rejected
    logits = beta * (log_ratio_w - log_ratio_l)
    return -log_sigmoid(logits).mean()
```

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta=0.1):
    """All inputs: [B]"""
    log_ratio_w = policy_chosen_logps - ref_chosen_logps
    log_ratio_l = policy_rejected_logps - ref_rejected_logps
    logits = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(logits).mean()
```

---

## IPO

**Core problem**: DPO's $-\log\sigma$ loss saturates once the preference gap grows, and overfits noisy preferences. IPO replaces it with a squared loss that drives the chosen-vs-rejected advantage toward a fixed target $\frac{1}{2\beta}$, penalizing deviation in both directions and stabilizing training.

**Core variables**:

- `delta`: $\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$, the chosen-minus-rejected advantage (without β)
- Target margin $\frac{1}{2\beta}$, derived from the optimality condition

### One-Line Memory

> **Swap DPO's $-\log\sigma$ for $(\Delta - \frac{1}{2\beta})^2$ — no saturation, the target becomes a fixed point.**

$$\mathcal{L}_{IPO} = \Big(\Delta - \frac{1}{2\beta}\Big)^2, \quad \Delta = \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$$

### Pseudocode

```
delta = log_ratio_w - log_ratio_l     # advantage without β
loss  = (delta - 1 / (2 * beta)) ** 2 # regress to fixed target
```

### PyTorch Implementation

```python
def ipo_loss(log_ratio_w, log_ratio_l, beta=0.1):
    delta = log_ratio_w - log_ratio_l
    return ((delta - 1.0 / (2 * beta)) ** 2).mean()
```

---

## KTO

**Core problem**: DPO/IPO require chosen-rejected pairs, but real-world feedback is often single-sample binary ("user thumbs-up / thumbs-down"). KTO borrows from prospect theory to reformulate the alignment loss as a single-sample objective — no pairing required — and up-weights undesirable examples (loss aversion).

**Core variables**:

- `log_ratio`: $\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ for a single sample
- `z_ref`: baseline = EMA of the desirable logits, i.e. $z_{ref} \approx \beta \mathbb{E}[\log\frac{\pi_\theta}{\pi_{ref}}]$; must `detach`
- `lambda_D` / `lambda_U`: weights on desirable / undesirable samples; typically $\lambda_U > \lambda_D$ (loss aversion)

### One-Line Memory

> **No pairing needed: push desirable $\beta\log r$ above $z_{ref}$, push undesirable below; each through $-\log\sigma$.**

### Pseudocode

```
logit = beta * log_ratio                  # implicit reward × temperature for one sample
loss_desirable   = -log_sigmoid(logit - z_ref)        # desirable: logit above baseline
loss_undesirable = -log_sigmoid(z_ref - logit)        # undesirable: logit below baseline
loss = lambda_D * loss_desirable + lambda_U * loss_undesirable
```

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F

def kto_loss(log_ratio, is_desirable, z_ref=0.0,
             beta=0.1, lambda_D=1.0, lambda_U=1.33):
    """
    log_ratio: [B] = log_pi(y|x) - log_ref(y|x)
    is_desirable: [B] bool; z_ref already contains β (TRL convention)
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

**Core problem**: DPO keeps both the policy and the reference model resident in memory, which is expensive, and longer responses have inherently lower log-probabilities and are unfairly penalized. SimPO replaces the reference model with length normalization — removing ref entirely while eliminating length bias.

**Core variables**:

- `chosen_logps` / `rejected_logps`: log-probabilities of the chosen/rejected response under the current policy (sequence-summed)
- `chosen_lengths` / `rejected_lengths`: token counts of the two responses
- `beta` ($\beta$): preference-gap amplifier (SimPO typical 2.0, larger than DPO)
- `gamma` ($\gamma$): target margin, a learnable-style threshold for "how much higher the good response must beat the bad"

### One-Line Memory

> **DPO without a ref: log-prob divided by response length, chosen minus rejected, scale by β, subtract margin $\gamma$.**

### Pseudocode

```
logp_w = log_pi(chosen)  / len(chosen)   # length-normalized
logp_l = log_pi(rejected) / len(rejected)
logits = beta * (logp_w - logp_l) - gamma # subtract margin
loss   = -log_sigmoid(logits)
```

### PyTorch Implementation

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

## Quick Comparison: DPO Family

| Method | Needs ref? | Needs pairing?        | Key difference                                             |
| ------ | ---------- | --------------------- | ---------------------------------------------------------- |
| DPO    | yes        | yes (chosen/rejected) | $-\log\sigma(\beta\Delta)$, canonical                      |
| IPO    | yes        | yes                   | squared $(\Delta - \frac{1}{2\beta})^2$, avoids saturation |
| KTO    | yes        | no (good/bad labels)  | single-sample ± sigmoid + baseline $z_{ref}$, loss-averse  |
| SimPO  | **no**     | yes                   | length-normalized log-prob + margin $\gamma$               |

Note: $\Delta$ is the chosen-minus-rejected log-ratio difference.

---

## Common Pitfalls

| Pitfall                                  | Explanation                                                                                                      |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Mixing up the four log-probs             | Two models × two samples = four values; reference ones must be `.detach()`'d                                     |
| Wrong log-prob form                      | DPO/IPO/KTO use $\log\frac{\pi_\theta}{\pi_{ref}}$ (log-ratio), **not** raw log-prob                             |
| Numeric overflow in `log_sigmoid`        | PyTorch's `F.logsigmoid` handles it; in NumPy use `logaddexp`                                                    |
| $\beta$ meaning                          | $\beta$ is a temperature, larger = more sensitive to preference gaps; not a learning rate                        |
| Expecting sigmoid in IPO                 | IPO regresses to $\frac{1}{2\beta}$ with a squared loss, no sigmoid                                              |
| KTO's $z_{ref}$ convention               | In TRL $z_{ref}$ already includes $\beta$ (EMA of logits); write it as $\beta \cdot \text{log\_ratio} - z_{ref}$ |
| Forgetting length normalization in SimPO | Length normalization is the core of SimPO; long responses have smaller log-prob and must be divided out          |
| Swapped chosen vs rejected               | Check the dataset: chosen is the human-preferred response                                                        |
