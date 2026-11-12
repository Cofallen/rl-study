---
title: 6.1 The Policy Gradient Theorem and REINFORCE
---

# 6.1 Policy Gradient and REINFORCE

Chapter 3 followed one route: learn $Q(s,a)$, use it to score each action, and choose the action with the highest score. This route works well for CartPole and Atari because their action sets are finite. A robotic arm is different: its shoulder, elbow, and wrist can each apply a continuous torque, giving infinitely many possible action combinations. A language model also makes a sequence of decisions while preserving the ability to sample rather than always selecting the single most likely token.

These tasks suggest a second route: skip the intermediate $Q$ values and learn a **policy** $\pi_\theta(a\mid s)$ directly. This is the idea behind the [policy objective $J(\theta)$](../chapter03_mdp/policy-value) from Chapter 3 — define a score for the policy as a whole, then adjust its parameters $\theta$ to increase that score. The differences between the two routes can be summarized as:

| Property           | Value-based methods                     | Policy-based methods                   |
| ------------------ | --------------------------------------- | -------------------------------------- |
| Learned object     | $Q(s,a)$, a score per action            | $\pi_\theta(a\mid s)$, action probabilities |
| Action selection   | $\arg\max_a Q(s,a)$                    | sample from $\pi_\theta(\cdot\mid s)$  |
| Action space       | finite discrete actions in standard DQN | discrete or continuous                 |
| Exploration        | added by a separate rule                | represented by the policy distribution |
| Data reuse         | off-policy replay is common             | vanilla REINFORCE is on-policy         |

The two routes can be combined: Actor-Critic methods (Chapter 7) use a policy network to choose actions and a value network to reduce variance. Before that, we need the policy-gradient calculation that makes the actor trainable. A step-by-step comparison is available as supplementary reading: [Why Policy Gradients?](./pg-necessity).

::: tip Prerequisites
Three earlier ideas will be used throughout the chapter:

- [Policy $\pi_\theta$ and objective $J(\theta)$](../chapter03_mdp/policy-value): how a parameterized policy is represented and evaluated
- [Monte Carlo methods](../chapter03_mdp/dp-mc-td): finish an episode, then compute its return
- [State value $V(s)$](../chapter03_mdp/value-bellman): the expected return from a state and the source of a useful baseline
:::

This section answers two questions: what metric should we use to measure "how good" a policy is, and how do we optimize that metric?

## The Policy Objective

Chapter 3 introduced the [policy objective](../chapter03_mdp/policy-value) $J(\theta)$ — a measure of how well the policy performs overall. Across possible starting points, how much [discounted return](../chapter03_mdp/mdp) does policy $\pi_\theta$ accumulate on average?

$$J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]$$

| Symbol                    | Role              | Meaning                                                                                        |
| ------------------------- | ----------------- | ---------------------------------------------------------------------------------------------- |
| $\theta$                  | Policy parameters | Neural network weights: changing them changes the policy's behavior                            |
| $\pi_\theta$              | Policy            | Given a state, outputs a probability distribution over actions                                 |
| $J(\theta)$               | Objective         | The policy's "report card" — the average score achieved by the policy with parameters $\theta$ |
| $\mathbb{E}_{\pi_\theta}$ | Expectation       | Run the policy many times and average                                                          |
| $\gamma^t r_t$            | Discounted reward | Reward at time $t$; farther-future rewards are worth less                                      |

$J(\theta)$ is our north star. The goal is simple: find parameters $\theta$ that maximize $J(\theta)$.

### Numerical Example: $J(\theta)$ for a 3-Step Episode

Suppose an episode has only 3 steps, with rewards $r_0=1$, $r_1=2$, $r_2=3$ and discount factor $\gamma=0.9$. The discounted return of this trajectory is

$$
\begin{aligned}
\sum_{t=0}^{2} \gamma^t r_t &= \gamma^0 r_0 + \gamma^1 r_1 + \gamma^2 r_2 \\
&= 0.9^0 \times 1 + 0.9^1 \times 2 + 0.9^2 \times 3 \\
&= 1 \times 1 + 0.9 \times 2 + 0.81 \times 3 \\
&= 1 + 1.8 + 2.43 \\
&= 5.23.
\end{aligned}
$$

This is the return of **one** trajectory. $J(\theta)$ is the expectation over all possible trajectory returns — run the policy $\pi_\theta$ infinitely many times and take the average. Different policies $\pi_\theta$ produce different trajectory distributions, hence different $J(\theta)$. Suppose policy A tends to choose high-reward paths and policy B tends to choose low-reward paths:

| Policy | Average return of possible trajectories | $J(\theta)$ |
| ------ | --------------------------------------- | ----------- |
| A      | Around $5.23$                           | Larger      |
| B      | Around $2.10$                           | Smaller     |

The optimization objective is to find parameters $\theta$ that maximize $J(\theta)$ (i.e., the highest average return).

## Gradient Ascent

How do we make $J(\theta)$ larger? The classic move in deep learning: walk in the direction of the gradient.

$$\theta \leftarrow \theta + \alpha \, \nabla_\theta J(\theta)$$

| Symbol                    | Role          | Meaning                                                                 |
| ------------------------- | ------------- | ----------------------------------------------------------------------- |
| $\nabla_\theta J(\theta)$ | Gradient      | Which direction should we change parameters to improve the policy most? |
| $\alpha$                  | Learning rate | How big is one step? Too large oscillates; too small is slow            |
| $+$                       | Ascent        | Notice the plus sign: we maximize, not minimize                         |

### Numerical Example: One Parameter Update

Suppose parameters $\theta = [0.5,\ 0.3,\ -0.1]$, gradient $\nabla_\theta J(\theta) = [0.1,\ -0.2,\ 0.05]$, learning rate $\alpha = 0.01$. Writing out the update component by component:

$$
\begin{aligned}
\theta_0 &\leftarrow 0.5 + 0.01 \times 0.1 = 0.5 + 0.001 = 0.501, \\
\theta_1 &\leftarrow 0.3 + 0.01 \times (-0.2) = 0.3 - 0.002 = 0.298, \\
\theta_2 &\leftarrow -0.1 + 0.01 \times 0.05 = -0.1 + 0.0005 = -0.0995.
\end{aligned}
$$

After the update, $\theta = [0.501,\ 0.298,\ -0.0995]$. Each component has moved a small step in the direction indicated by the gradient. $\alpha$ controls the step size: with $\alpha=0.01$, each step moves only a few thousandths, but the effect accumulates over many rounds.

But how do we compute $\nabla_\theta J(\theta)$? The objective contains an expectation $\mathbb{E}$, which in principle requires averaging over all possible trajectories. The number of possible trajectories is astronomical; we cannot enumerate them all. It is like trying to compute the average height of every student in a school — you cannot measure everyone, but you can randomly sample 100 students and estimate. The sample mean computed from those 100 students is a **sampling estimate** of the true mean. Policy gradients use the same idea: run a few trajectories and use the average of their gradients to estimate the true $\nabla_\theta J(\theta)$.

## The Policy Gradient Theorem

This is exactly where the policy gradient theorem enters. In 1992, Williams showed in the REINFORCE paper that the seemingly intractable gradient $\nabla_\theta J(\theta)$ can be rewritten into a form that can be estimated by sampling. [^1] Sutton and colleagues later generalized and systematized this result in 2000. [^2]

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

Let's unpack it term by term:

| Symbol                                      | Role                 | Meaning                                                                                 |
| ------------------------------------------- | -------------------- | --------------------------------------------------------------------------------------- |
| $\nabla_\theta$                             | Take gradient        | Which direction should we change the parameters?                                        |
| $\log \pi_\theta(a_t \| s_t)$               | Log probability      | Under state $s_t$, the log-probability of choosing action $a_t$                         |
| $\nabla_\theta \log \pi_\theta(a_t \| s_t)$ | Gradient of log-prob | How do parameters change the probability of choosing this action?                       |
| $G_t$                                       | Return               | Total reward from time $t$ to the end — "how many points did we get after this action?" |
| Outer $\mathbb{E}$                          | Expectation          | "Run many episodes and average" — approximated by sampling                              |

In one sentence: **if an action leads to a good outcome (large $G_t$), increase the probability of taking it again; if it leads to a bad outcome (small $G_t$), decrease that probability.**

### Numerical Example: Gradient Computation in a Bandit Setting

Let's walk through a bandit scenario. Two-armed bandit: arm A wins with probability 30%, arm B with probability 70%. The policy has a single parameter $\theta$ (assume larger $\theta$ means higher probability of choosing B). Currently $\pi_\theta(B|s) = 0.7$, $\pi_\theta(A|s) = 0.3$.

**Trajectory 1: chose B, reward = 1.0**

First step, compute the log probability:

$$
\log \pi_\theta(B|s) = \log(0.7) = -0.357.
$$

Second step, compute the gradient of the log probability. Suppose that under the current parameters $\nabla_\theta \log \pi_\theta(B|s) = 0.5$ (this value depends on the specific parameterization; we pick a concrete number here).

Third step, multiply by the return. The bandit has only one step, so $G_0 = r_0 = 1.0$:

$$
\nabla_\theta \log \pi_\theta(B|s) \cdot G_0 = 0.5 \times 1.0 = 0.5.
$$

The gradient is positive, so the parameter updates as $\theta \leftarrow \theta + \alpha \times 0.5$. The parameter increases, and $\pi_\theta(B|s)$ increases accordingly — the policy learns "choosing B led to a good result, so choose B more often."

**Trajectory 2: chose A, reward = 0**

$$
\log \pi_\theta(A|s) = \log(0.3) = -1.204.
$$

Suppose $\nabla_\theta \log \pi_\theta(A|s) = -0.5$ (the gradient direction for choosing A is opposite to choosing B). Multiply by the return $G_0 = 0$:

$$
\nabla_\theta \log \pi_\theta(A|s) \cdot G_0 = (-0.5) \times 0 = 0.
$$

The gradient is zero — the parameters do not update. The policy chose A but got zero reward, so it neither encourages nor discourages this action.

**Trajectory 3: chose A, reward = 0.5**

$$
\nabla_\theta \log \pi_\theta(A|s) \cdot G_0 = (-0.5) \times 0.5 = -0.25.
$$

The gradient is negative, so the parameter updates as $\theta \leftarrow \theta + \alpha \times (-0.25)$. The parameter decreases, which means $\pi_\theta(B|s)$ decreases and $\pi_\theta(A|s)$ increases — the policy learns "A got some reward this time, so we can choose A slightly more often." But since A's average reward is much lower than B's, over many samples the positive gradient from B will accumulate and dominate, and the policy will eventually converge to preferring B.

### Numerical Example: Gradient Computation for a 3-Step Episode

Consider a CartPole-like scenario: 3-step episode, $\gamma=0.9$.

| Step | State | Action        | Reward    |
| ---- | ----- | ------------- | --------- |
| 0    | $s_0$ | $a_0$ = right | $r_0 = 1$ |
| 1    | $s_1$ | $a_1$ = left  | $r_1 = 2$ |
| 2    | $s_2$ | $a_2$ = right | $r_2 = 3$ |

First, compute the return $G_t$ at each step. $G_2$ includes only the last step:

$$
G_2 = r_2 = 3.
$$

$G_1$ accumulates from step 1 to the end:

$$
G_1 = r_1 + \gamma r_2 = 2 + 0.9 \times 3 = 2 + 2.7 = 4.7.
$$

$G_0$ accumulates from step 0 to the end:

$$
G_0 = r_0 + \gamma r_1 + \gamma^2 r_2 = 1 + 0.9 \times 2 + 0.81 \times 3 = 1 + 1.8 + 2.43 = 5.23.
$$

Suppose the log-probability gradient values at each step are:

| Step | $\log \pi_\theta(a_t \| s_t)$                  | $\nabla_\theta \log \pi_\theta(a_t \| s_t)$ | $G_t$  |
| ---- | ---------------------------------------------- | ------------------------------------------- | ------ |
| 0    | $\log \pi_\theta(\text{right}\mid s_0) = -0.4$ | $[0.3, -0.1]$                               | $5.23$ |
| 1    | $\log \pi_\theta(\text{left}\mid s_1) = -0.7$  | $[-0.2, 0.4]$                               | $4.7$  |
| 2    | $\log \pi_\theta(\text{right}\mid s_2) = -0.3$ | $[0.1, 0.2]$                                | $3$    |

The gradient contribution from each step:

$$
\begin{aligned}
\text{Step 0:} \quad \nabla_\theta \log \pi_\theta(a_0|s_0) \cdot G_0 &= [0.3, -0.1] \times 5.23 = [1.569, -0.523], \\
\text{Step 1:} \quad \nabla_\theta \log \pi_\theta(a_1|s_1) \cdot G_1 &= [-0.2, 0.4] \times 4.7 = [-0.94, 1.88], \\
\text{Step 2:} \quad \nabla_\theta \log \pi_\theta(a_2|s_2) \cdot G_2 &= [0.1, 0.2] \times 3 = [0.3, 0.6].
\end{aligned}
$$

The gradient estimate from this single trajectory is the sum of the three:

$$
\begin{aligned}
\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t &= [1.569, -0.523] + [-0.94, 1.88] + [0.3, 0.6] \\
&= [1.569 - 0.94 + 0.3,\ -0.523 + 1.88 + 0.6] \\
&= [0.929,\ 1.957].
\end{aligned}
$$

If $\alpha = 0.01$, the parameter update is $\theta \leftarrow \theta + 0.01 \times [0.929,\ 1.957]$. Step 1 contributes the most (with $G_1=4.7$ and a relatively large gradient component of $0.4$), indicating that the decision "choose left in $s_1$" has a prominent contribution to the final return, so the parameters will move in the direction of increasing the probability of choosing left in $s_1$.

### The Log-Derivative Trick

Why don't we write something like $\nabla_\theta \pi_\theta(a_t|s_t) \cdot G_t$, and instead introduce a $\log$?

This is a mathematical technique known as the **log-derivative trick**. By the chain rule:

$$\nabla_\theta \log \pi = \frac{\nabla_\theta \pi}{\pi}$$

The division by $\pi$ cancels the $\pi$ factor hidden inside the expectation, making the entire expression clean and computable. From an engineering point of view, since probabilities $\pi$ lie in $(0, 1)$, directly computing gradients of probabilities can produce extremely small values, destabilizing training. The $\log$ maps $(0, 1)$ to $(-\infty, 0)$, yielding more numerically stable gradients.

### Numerical Example: The Effect of the Log-Derivative Trick

Take a concrete example: $\pi_\theta(a|s) = 0.7$. Suppose a small perturbation of the parameter changes $\pi_\theta(a|s)$ to $0.71$. Then

$$
\nabla_\theta \pi_\theta(a|s) \approx 0.71 - 0.7 = 0.01.
$$

Using the raw probability gradient:

$$
\nabla_\theta \pi_\theta(a|s) = 0.01.
$$

Using the log-derivative trick:

$$
\nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} = \frac{0.01}{0.7} = 0.0143.
$$

Now consider another action with $\pi_\theta(a'|s) = 0.05$, which becomes $0.06$ after the same perturbation:

$$
\nabla_\theta \pi_\theta(a'|s) = 0.01, \quad \nabla_\theta \log \pi_\theta(a'|s) = \frac{0.01}{0.05} = 0.2.
$$

The two actions have the same probability gradient ($0.01$), but the low-probability action's log-derivative gradient is $0.2/0.0143 \approx 14$ times that of the high-probability action. This makes intuitive sense: raising a $5\%$ probability to $6\%$ is a $20\%$ relative increase; raising a $70\%$ probability to $71\%$ is only a $1.4\%$ relative increase. Dividing by $\pi$ converts absolute changes into relative changes, giving actions at different probability scales comparable magnitudes in gradient space.

<details>
<summary>Math derivation: from the objective to the policy gradient theorem</summary>

The gradient of the objective differentiates through the trajectory distribution:

$$\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau} P(\tau; \theta) \sum_t r_t(\tau)$$

Here $\tau = (s_0, a_0, s_1, a_1, \ldots)$ denotes a trajectory, and $P(\tau; \theta)$ is the probability that policy $\pi_\theta$ generates trajectory $\tau$. The gradient can only act on $P(\tau; \theta)$, since the reward does not depend on $\theta$:

$$\nabla_\theta J(\theta) = \sum_{\tau} \nabla_\theta P(\tau; \theta) \sum_t r_t(\tau)$$

The key step is the identity $\nabla_\theta P = P \cdot \nabla_\theta \log P$:

$$\nabla_\theta J(\theta) = \sum_{\tau} P(\tau; \theta) \left( \nabla_\theta \log P(\tau; \theta) \right) \sum_t r_t(\tau)$$

The trajectory probability factorizes as $P(\tau; \theta) = \prod_t \pi_\theta(a_t|s_t) \cdot P(s_{t+1}|s_t, a_t)$. Taking logs and differentiating with respect to $\theta$, the environment transition probability $P(s'|s,a)$ disappears because it does not depend on $\theta$, leaving only the policy terms:

$$\nabla_\theta \log P(\tau; \theta) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

Substituting back into the expectation yields the policy gradient theorem. The most elegant aspect of this derivation is that the environment dynamics — the state transition probabilities — cancel out during differentiation. This means policy gradients **do not need a model of the environment**, which is the fundamental reason they are much more flexible than dynamic programming approaches.

</details>

## The REINFORCE Algorithm

The policy gradient theorem gives us the form of the gradient. **REINFORCE** is the most straightforward implementation of the theorem: it uses [Monte Carlo sampling](../chapter03_mdp/dp-mc-td) to estimate the expectation. The algorithm flow is:

1. Run one full episode with the current policy $\pi_\theta$, recording state, action, and reward at each step.
2. For each time step, compute the return from that step until the end of the episode: $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$.
3. Estimate the gradient with samples: $\nabla_\theta J \approx \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$.
4. Update parameters along the gradient: $\theta \leftarrow \theta + \alpha \nabla_\theta J$.

In PyTorch, the core update can be written in one line:

```python
loss = -log_prob * G_t  # minus sign because PyTorch does gradient descent (minimize) by default, while we want ascent (maximize)
```

A full multi-step version:

```python
# REINFORCE core (multi-step version)
for t in range(len(rewards)):
    G_t = sum(gamma ** k * rewards[t + k] for k in range(len(rewards) - t))
    loss += -log_probs[t] * G_t

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Numerical Example: A Complete Episode's REINFORCE Update

Continuing with the 3-step episode from earlier, $\gamma=0.9$:

| Step | State | Action | Reward  | $\log \pi_\theta(a_t \| s_t)$ |
| ---- | ----- | ------ | ------- | ----------------------------- |
| 0    | $s_0$ | right  | $r_0=1$ | $-0.4$                        |
| 1    | $s_1$ | left   | $r_1=2$ | $-0.7$                        |
| 2    | $s_2$ | right  | $r_2=3$ | $-0.3$                        |

**Step 1: Compute the return $G_t$ at each step.**

$G_2$ from step 2 to the end (only 1 step):

$$
G_2 = r_2 = 3.
$$

$G_1$ from step 1 to the end (2 steps):

$$
G_1 = r_1 + \gamma r_2 = 2 + 0.9 \times 3 = 2 + 2.7 = 4.7.
$$

$G_0$ from step 0 to the end (3 steps):

$$
G_0 = r_0 + \gamma r_1 + \gamma^2 r_2 = 1 + 0.9 \times 2 + 0.81 \times 3 = 1 + 1.8 + 2.43 = 5.23.
$$

Summary:

| Step | $r_t$ | $G_t$ |
| ---- | ----- | ----- |
| 0    | 1     | 5.23  |
| 1    | 2     | 4.7   |
| 2    | 3     | 3     |

**Step 2: Compute the gradient estimate.**

Each step's gradient term is $\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$. In PyTorch, `log_prob` is $\log \pi_\theta(a_t|s_t)$, and autograd handles the gradient. Here we only look at the scalar loss construction:

$$
\begin{aligned}
\text{loss} &= -\sum_{t=0}^{2} \log \pi_\theta(a_t|s_t) \cdot G_t \\
&= -(\log \pi_\theta(\text{right}|s_0) \cdot G_0 + \log \pi_\theta(\text{left}|s_1) \cdot G_1 + \log \pi_\theta(\text{right}|s_2) \cdot G_2) \\
&= -((-0.4) \times 5.23 + (-0.7) \times 4.7 + (-0.3) \times 3) \\
&= -(−2.092 + (−3.29) + (−0.9)) \\
&= -(-6.282) \\
&= 6.282.
\end{aligned}
$$

**Step 3: Backpropagation and parameter update.**

`loss.backward()` computes $\nabla_\theta \text{loss}$. Since $\text{loss} = -\sum_t \log \pi_\theta(a_t|s_t) \cdot G_t$, we have

$$
\nabla_\theta \text{loss} = -\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t.
$$

`optimizer.step()` executes $\theta \leftarrow \theta - \alpha \cdot \nabla_\theta \text{loss}$ (PyTorch defaults to gradient descent). The two negatives cancel:

$$
\theta \leftarrow \theta - \alpha \cdot \left(-\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right) = \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t.
$$

This is exactly gradient ascent. This round of updates moves the parameters in the direction of increasing the probability of high-return actions.

### A Minimal Example: Multi-Armed Bandits

Before we dive into CartPole, let's use a minimal setting to understand what `loss = -log_prob * reward` is really doing.

Imagine a bandit with two arms: arm A wins with probability 30%, arm B wins with probability 70%. The policy network is just a Softmax layer that outputs the probability of choosing A and B. The core training code:

```python
probs = policy(state)
dist = torch.distributions.Categorical(probs)
action = dist.sample()          # sample an action according to probability
log_prob = dist.log_prob(action)  # log π(a|s)

reward = pull_arm(action.item())  # take the action

loss = -log_prob * reward         # REINFORCE core
```

After 300 episodes, the probability of choosing B will typically climb from around 0.5 to 0.85–0.95: the policy learns to prefer the arm with the higher win rate. But the curve will not be smooth — it will be jagged and noisy. If you increase the learning rate from 0.01 to 0.1, the policy may swing violently between A and B.

This is the core pain point of REINFORCE: **high variance**.

## The Variance Problem in REINFORCE

$G_t$ is the cumulative return from time $t$ until the episode ends, which includes all randomness along that future trajectory. For the same action, different sampled trajectories can produce very different $G_t$ values:

| Case      | What actually happened                    | $G_t$ |
| --------- | ----------------------------------------- | ----- |
| Good luck | Subsequent steps all yielded high rewards | Large |
| Bad luck  | Subsequent steps all yielded low rewards  | Small |

Policy gradients use $G_t$ to decide whether "this action was good." But when $G_t$ fluctuates heavily, **the same good action can be penalized due to bad luck, and the same bad action can be rewarded due to good luck**. It is like judging a student's ability by a single exam — a bad score does not necessarily mean poor mastery; it may just be an off day.

### Numerical Example: Comparing Gradient Signals from Two Trajectories

Back to the bandit scenario: $\pi_\theta(B|s) = 0.7$, $\pi_\theta(A|s) = 0.3$. Both times we chose B (the same action), but the subsequent luck differed.

**Episode 1: chose B, reward 1.0 (good outcome)**

$$
\text{gradient term} = \nabla_\theta \log \pi_\theta(B|s) \times 1.0.
$$

The gradient pushes $\pi_\theta(B|s)$ up.

**Episode 2: chose B, reward 0.0 (bad outcome, bad luck)**

$$
\text{gradient term} = \nabla_\theta \log \pi_\theta(B|s) \times 0.0 = 0.
$$

The gradient signal is zero — the parameters do not change.

Now consider a multi-step example. Same 3-step episode structure, $\gamma=0.9$, where the policy chose the same action "right" at $s_0$:

| Episode | $r_0$ | $r_1$ | $r_2$ | $G_0$                   | Gradient signal direction |
| ------- | ----- | ----- | ----- | ----------------------- | ------------------------- |
| 1       | 1     | 2     | 3     | $1 + 1.8 + 2.43 = 5.23$ | Strongly positive         |
| 2       | 1     | 0     | 0     | $1 + 0 + 0 = 1$         | Weakly positive           |

Both episodes took the same action "right" at $s_0$, yet $G_0$ differs by more than a factor of 4. Episode 1 would push $\pi(\text{right}|s_0)$ up significantly, while Episode 2 would only push it up weakly. The problem is that $G_0$ does not only reflect whether "choosing right at $s_0$ is good" — it also includes the randomness of $r_1$ and $r_2$. The rewards from the subsequent two steps are not controlled by the current action, yet they are all counted into $G_0$.

In the bandit experiment, this manifests as jagged and oscillating training curves. In more complex environments (such as CartPole), high variance makes training even more unstable — sometimes the policy improves nicely, then suddenly gets derailed by a run of bad luck.

## Discrete vs. Continuous Action Spaces

In this chapter's experiments we use discrete action spaces (choose A vs. B, CartPole left vs. right), but the policy gradient theorem applies equally to continuous action spaces:

|                    | Discrete action space                 | Continuous action space                                         |
| ------------------ | ------------------------------------- | --------------------------------------------------------------- |
| Example            | CartPole left/right, LLM token choice | Robot joint angle, steering wheel angle                         |
| Output layer       | Softmax, probability for each action  | Gaussian parameters, mean $\mu$ and standard deviation $\sigma$ |
| Sampling           | Sample according to Softmax           | Sample from $\mathcal{N}(\mu, \sigma^2)$                        |
| Compute $\log \pi$ | `log_softmax`                         | Log-density formula of a Gaussian distribution                  |

With the same policy gradient formula, changing only the output layer lets us switch from "left/right" to "continuous torque." This is where policy gradients are more flexible than value-based methods: DQN's $\arg\max$ is not computable in continuous spaces, while policy gradients differentiate through a probability density, which is naturally compatible with continuous actions.

<details>
<summary>Thinking question: what is the essential difference between REINFORCE and Q-Learning updates?</summary>

Q-Learning updates the value function $Q(s,a)$ — "how many points is this action worth" — and the policy is obtained implicitly via $\arg\max Q$. REINFORCE updates the policy parameters $\theta$ directly, skipping the intermediate step of learning $Q$ values.

This difference leads to two key consequences: Q-Learning is off-policy (it can reuse old data for training), while REINFORCE is on-policy (it must use fresh data generated by the current policy); Q-Learning can only handle discrete actions (it needs to enumerate all actions to take the max), while REINFORCE can handle continuous actions (it directly differentiates the log probability density).

</details>

REINFORCE can work, but its high variance makes it nearly unusable in practice. Fortunately, the policy gradient theorem has a remarkable property: we can subtract a baseline that does not depend on the action from the gradient estimator, without changing the expected direction of the gradient, while significantly reducing variance. We will develop this in [6.2 Policy Gradient Improvements](./pg-improvements). For hands-on practice, first run the [Dice-Game Bandit](./dice-game), then apply REINFORCE to [CartPole](./cartpole).

---

[^1]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8(3-4), 229-256. [DOI](https://doi.org/10.1007/BF00992696)

[^2]: Sutton, R. S., et al. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.
