---
title: '7.3 Critic Training Details'
---

# 7.3 Critic Training Details

In the previous section, we defined the advantage function $A(s,a) \approx \delta = r + \gamma V(s') - V(s)$ and introduced the Critic network as an estimator of $V(s)$. This section expands on the three classic value-estimation methods from Chapter 4 -- [DP, MC, and TD](../chapter03_mdp/dp-mc-td) -- and shows how each one trains the Critic in practice.

::: tip Prerequisites

- [DP/MC/TD value-estimation methods](../chapter03_mdp/dp-mc-td) -- principles and comparisons
- [Bellman expectation equation](../chapter03_mdp/value-bellman) -- theoretical basis for DP updates
- [TD Error $\delta$](../chapter03_mdp/dp-mc-td) -- the core signal of TD methods
  :::

We continue with the three-cell corridor from Chapter 4, using a fixed policy $\pi$: at both $S$ and $M$, move right with probability 0.8 and left with probability 0.2. The transitions and rewards are:

| Current state | Action | Policy prob | Next state | Reward |
| ------------- | ------ | ----------- | ---------- | ------ |
| $S$           | left   | 0.2         | $S$        | $-2$   |
| $S$           | right  | 0.8         | $M$        | $-1$   |
| $M$           | left   | 0.2         | $S$        | $-2$   |
| $M$           | right  | 0.8         | $G$        | $-1$   |
| $G$           | end    | 1.0         | --         | $0$    |

We set $\gamma=1$. All three methods estimate the same value table; they differ only in where the update targets come from.

## DP: A Theoretical Baseline

If we knew the full transition probabilities $P$ and reward function $R$ (recall the [MDP 5-tuple](../chapter03_mdp/mdp)), we could iterate the Critic directly using the [Bellman expectation equation](../chapter03_mdp/value-bellman):

$$V_\phi(s) \leftarrow \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_\phi(s') \right]$$

Each symbol in this equation:

| Symbol           | Meaning                                                               |
| ---------------- | --------------------------------------------------------------------- |
| $V_\phi(s)$      | Critic's current value estimate for state $s$, with parameters $\phi$ |
| $a$              | An action available at state $s$ (e.g., left, right)                  |
| $\pi(a \mid s)$  | Probability that the current policy selects action $a$ at state $s$   |
| $R(s,a)$         | Immediate reward received after taking action $a$ in state $s$        |
| $s'$             | A possible next state after taking action $a$                         |
| $P(s' \mid s,a)$ | Probability of transitioning to $s'$ from state $s$ and action $a$    |
| $V_\phi(s')$     | Critic's current value estimate for next state $s'$                   |
| $\gamma$         | Discount factor                                                       |

Expanding for state $S$ in the corridor. The outer sum weights over actions according to the policy; the inner sum weights over next states according to the transition probabilities. Since transitions are deterministic (moving right always goes right; moving left either hits the wall or goes back), the inner $\sum_{s'}$ has probability 1 only for the actual destination:

$$
\begin{aligned}
V_\phi(S) &\leftarrow \pi(\text{right} \mid S)\left[R(S,\text{right}) + V_\phi(M)\right] + \pi(\text{left} \mid S)\left[R(S,\text{left}) + V_\phi(S)\right] \\
          &= 0.8\left[-1 + V_\phi(M)\right] + 0.2\left[-2 + V_\phi(S)\right]
\end{aligned}
$$

Similarly for $M$, where moving right reaches the terminal state $G$ and moving left returns to $S$:

$$
\begin{aligned}
V_\phi(M) &\leftarrow \pi(\text{right} \mid M)\left[R(M,\text{right}) + V_\phi(G)\right] + \pi(\text{left} \mid M)\left[R(M,\text{left}) + V_\phi(S)\right] \\
          &= 0.8\left[-1 + V_\phi(G)\right] + 0.2\left[-2 + V_\phi(S)\right]
\end{aligned}
$$

By repeatedly applying this update to all states, $V_\phi$ converges to the exact $V^\pi$. Starting from an all-zero table, we substitute numbers round by round.

**Round 1** -- the old table is all zeros, so the target reduces to the average immediate cost of each action:

$$
\begin{aligned}
V_1(S) &= 0.8[-1 + V_0(M)] + 0.2[-2 + V_0(S)] \\
       &= 0.8(-1 + 0) + 0.2(-2 + 0) = -0.8 - 0.4 = -1.2
\end{aligned}
$$

$$
\begin{aligned}
V_1(M) &= 0.8[-1 + V_0(G)] + 0.2[-2 + V_0(S)] \\
       &= 0.8(-1 + 0) + 0.2(-2 + 0) = -0.8 - 0.4 = -1.2
\end{aligned}
$$

**Round 2** -- using the round-1 results as the old table:

$$
\begin{aligned}
V_2(S) &= 0.8[-1 + V_1(M)] + 0.2[-2 + V_1(S)] \\
       &= 0.8[-1 + (-1.2)] + 0.2[-2 + (-1.2)] \\
       &= 0.8 \times (-2.2) + 0.2 \times (-3.2) = -1.76 - 0.64 = -2.4
\end{aligned}
$$

$$
\begin{aligned}
V_2(M) &= 0.8[-1 + V_1(G)] + 0.2[-2 + V_1(S)] \\
       &= 0.8[-1 + 0] + 0.2[-2 + (-1.2)] \\
       &= 0.8 \times (-1) + 0.2 \times (-3.2) = -0.8 - 0.64 = -1.44
\end{aligned}
$$

**Round 3** -- using the round-2 results as the old table:

$$
\begin{aligned}
V_3(S) &= 0.8[-1 + V_2(M)] + 0.2[-2 + V_2(S)] \\
       &= 0.8[-1 + (-1.44)] + 0.2[-2 + (-2.4)] \\
       &= 0.8 \times (-2.44) + 0.2 \times (-4.4) = -1.952 - 0.88 = -2.832
\end{aligned}
$$

$$
\begin{aligned}
V_3(M) &= 0.8[-1 + V_2(G)] + 0.2[-2 + V_2(S)] \\
       &= 0.8(-1 + 0) + 0.2[-2 + (-2.4)] \\
       &= -0.8 + 0.2 \times (-4.4) = -0.8 - 0.88 = -1.68
\end{aligned}
$$

Summary of each round:

| Round     | $V(S)$ | $V(M)$ | $V(G)$ |
| --------- | ------ | ------ | ------ |
| 0         | 0      | 0      | 0      |
| 1         | -1.2   | -1.2   | 0      |
| 2         | -2.4   | -1.44  | 0      |
| 3         | -2.832 | -1.68  | 0      |
| converged | -3.375 | -1.875 | 0      |

In each round, the values for $S$ and $M$ encode the "average consequence of acting under the current policy" -- moving right is generally better, but the policy occasionally moves left, and the costs of detours and wall bumps must also enter the value table.

On this basis, we can also perform **policy improvement** -- at state $s$, choose the action that maximizes $Q(s,a)$ (recall the [greedy optimal policy](../chapter03_mdp/value-q)). The loop "evaluate the policy $\rightarrow$ improve the policy $\rightarrow$ evaluate again" is exactly **Policy Iteration**, which is guaranteed to converge to an optimal policy.

In real-world problems, however, it is almost never feasible to know the complete $P$ and $R$. DP's role in Actor-Critic is primarily a theoretical baseline -- it tells you the Critic's optimal answer when everything is known.

## MC: Update the Critic Using Complete Trajectories

Monte Carlo (MC) updates wait until a complete episode finishes, then use the [actual return $G_t$](../chapter03_mdp/mdp) to train the Critic. The Critic loss is a mean squared error:

$$L_{\text{Critic}} = \left( G_t - V_\phi(s) \right)^2 \tag{6.3}$$

Each symbol in this equation:

| Symbol              | Meaning                                                                               |
| ------------------- | ------------------------------------------------------------------------------------- |
| $L_{\text{Critic}}$ | Critic loss, measuring the prediction error                                           |
| $G_t$               | Actual discounted return from time step $t$ to the end of the episode (the MC target) |
| $V_\phi(s)$         | Critic's current value prediction for state $s$                                       |

$G_t - V_\phi(s)$ is the Critic's prediction error -- the episode actually returned $G_t$, but the Critic previously predicted $V_\phi(s)$. The loss is the square of this error.

### Numerical Example

Suppose we sample the following trajectory:

$$
S \xrightarrow{-2} S \xrightarrow{-1} M \xrightarrow{-2} S \xrightarrow{-1} M \xrightarrow{-1} G
$$

With $\gamma=1$, we compute $G_t$ by summing the rewards from each visit position to the end:

| Visit | State | Remaining rewards | $G_t$ computation                | MC target $G_t$ |
| ----- | ----- | ----------------- | -------------------------------- | --------------- |
| 1     | $S$   | $-2,-1,-2,-1,-1$  | $-2 + (-1) + (-2) + (-1) + (-1)$ | $-7$            |
| 2     | $S$   | $-1,-2,-1,-1$     | $-1 + (-2) + (-1) + (-1)$        | $-5$            |
| 3     | $M$   | $-2,-1,-1$        | $-2 + (-1) + (-1)$               | $-4$            |
| 4     | $S$   | $-1,-1$           | $-1 + (-1)$                      | $-2$            |
| 5     | $M$   | $-1$              | $-1$                             | $-1$            |

### Loss Computation and Gradient Update

Assume the Critic is a simple value table with $V(S) = 0$ and $V(M) = 0$. Using the first visit to $S$ as an example, the MC target is $G_t = -7$:

$$
L = (G_t - V(S))^2 = (-7 - 0)^2 = 49
$$

The gradient-descent update (learning rate $\alpha = 0.5$):

$$
V(S) \leftarrow V(S) - \alpha \cdot \frac{\partial L}{\partial V(S)} = V(S) - \alpha \cdot 2(V(S) - G_t)
$$

Here $\frac{\partial L}{\partial V(S)} = 2(V(S) - G_t) = 2(0 - (-7)) = 14$, but it is more common to absorb the $\frac{1}{2}$ into the learning rate and write directly:

$$
V(S) \leftarrow V(S) + \alpha (G_t - V(S)) = 0 + 0.5 \times (-7 - 0) = -3.5
$$

The complete update process across all visits:

| Updated state | MC target $G_t$ | Old value | Update computation                                                     | New value |
| ------------- | --------------- | --------- | ---------------------------------------------------------------------- | --------- |
| 1st $S$       | $-7$            | 0         | $0 + 0.5 \times (-7 - 0) = -3.5$                                       | $-3.5$    |
| 2nd $S$       | $-5$            | $-3.5$    | $-3.5 + 0.5 \times [-5 - (-3.5)] = -3.5 + 0.5 \times (-1.5) = -4.25$   | $-4.25$   |
| 1st $M$       | $-4$            | 0         | $0 + 0.5 \times (-4 - 0) = -2$                                         | $-2$      |
| 3rd $S$       | $-2$            | $-4.25$   | $-4.25 + 0.5 \times [-2 - (-4.25)] = -4.25 + 0.5 \times 2.25 = -3.125$ | $-3.125$  |
| 2nd $M$       | $-1$            | $-2$      | $-2 + 0.5 \times [-1 - (-2)] = -2 + 0.5 \times 1 = -1.5$               | $-1.5$    |

MC methods (recall the [MC value update](../chapter03_mdp/dp-mc-td): $V(s) \leftarrow V(s) + \alpha[G_t - V(s)]$) provide an **unbiased estimate** because they use the true return, but they have two limitations:

1. **You must wait until the episode ends** to compute $G_t$; you cannot learn online step by step.
2. **High variance** -- $G_t$ can fluctuate drastically across different episodes.

In a neural-network implementation, the MC method is equivalent to: run one full episode, collect all $(s_t, G_t)$ pairs, then perform a gradient-descent update on the Critic parameters $\phi$ using this batch.

## TD: One-Step Updates

Temporal Difference (TD) learning updates the Critic using the [TD Error](../chapter03_mdp/dp-mc-td). The Critic loss is:

$$L_{\text{Critic}} = \left( r + \gamma V_\phi(s') - V_\phi(s) \right)^2 = \delta^2 \tag{6.4}$$

Each symbol in this equation:

| Symbol              | Meaning                                                 |
| ------------------- | ------------------------------------------------------- |
| $L_{\text{Critic}}$ | Critic loss, measuring the magnitude of the TD Error    |
| $r$                 | Immediate reward received at the current step           |
| $\gamma$            | Discount factor                                         |
| $V_\phi(s')$        | Critic's current value prediction for next state $s'$   |
| $V_\phi(s)$         | Critic's current value prediction for current state $s$ |
| $\delta$            | TD Error, i.e., $r + \gamma V_\phi(s') - V_\phi(s)$     |

Minimizing $\delta^2$ makes the Critic's predictions progressively more accurate. The meaning of $\delta$: after taking one step, the difference between "the reward actually received plus the next-step prediction" and "the current prediction." $\delta > 0$ means this step was better than expected; $\delta < 0$ means it was worse.

### Numerical Example

Using the same trajectory as MC:

$$
S \xrightarrow{-2} S \xrightarrow{-1} M \xrightarrow{-2} S \xrightarrow{-1} M \xrightarrow{-1} G
$$

The initial value table is all zeros, with learning rate $\alpha = 0.5$. TD updates after every step, reading from the **current latest table**.

**Step 1**: $S \xrightarrow{-2} S$. Current $V(S) = 0$, $V(S') = V(S) = 0$.

$$
\delta = r + \gamma V(s') - V(s) = -2 + 1 \times 0 - 0 = -2
$$

$$
V(S) \leftarrow V(S) + \alpha \cdot \delta = 0 + 0.5 \times (-2) = -1
$$

**Step 2**: $S \xrightarrow{-1} M$. Current $V(S) = -1$ (updated in the previous step), $V(M) = 0$.

$$
\delta = -1 + 1 \times 0 - (-1) = -1 + 0 + 1 = 0
$$

$$
V(S) \leftarrow -1 + 0.5 \times 0 = -1
$$

$\delta = 0$ means "received $-1$ and arrived at $V(M) = 0$" exactly equals the previous estimate of $V(S) = -1$ -- the prediction had no error.

**Step 3**: $M \xrightarrow{-2} S$. Current $V(M) = 0$, $V(S) = -1$.

$$
\delta = -2 + 1 \times (-1) - 0 = -3
$$

$$
V(M) \leftarrow 0 + 0.5 \times (-3) = -1.5
$$

Note that $V(S) = -1$ here is the value just updated in step 1 -- TD immediately uses freshly learned information.

**Step 4**: $S \xrightarrow{-1} M$. Current $V(S) = -1$, $V(M) = -1.5$.

$$
\delta = -1 + 1 \times (-1.5) - (-1) = -1.5
$$

$$
V(S) \leftarrow -1 + 0.5 \times (-1.5) = -1 + (-0.75) = -1.75
$$

**Step 5**: $M \xrightarrow{-1} G$. Current $V(M) = -1.5$, $V(G) = 0$.

$$
\delta = -1 + 1 \times 0 - (-1.5) = 0.5
$$

$$
V(M) \leftarrow -1.5 + 0.5 \times 0.5 = -1.5 + 0.25 = -1.25
$$

$\delta = 0.5 > 0$ indicates that moving right from $M$ to the terminal was better than the current estimate of $V(M)$, so $V(M)$ is adjusted upward.

### Step-by-step Summary

| Step | Transition             | Updated state | Old $V(s)$ | $r$  | $V(s')$ | TD target $r + \gamma V(s')$ | $\delta$ | New $V(s)$ |
| ---- | ---------------------- | ------------- | ---------- | ---- | ------- | ---------------------------- | -------- | ---------- |
| 1    | $S \xrightarrow{-2} S$ | $S$           | 0          | $-2$ | 0       | $-2 + 0 = -2$                | $-2$     | $-1$       |
| 2    | $S \xrightarrow{-1} M$ | $S$           | $-1$       | $-1$ | 0       | $-1 + 0 = -1$                | $0$      | $-1$       |
| 3    | $M \xrightarrow{-2} S$ | $M$           | 0          | $-2$ | $-1$    | $-2 + (-1) = -3$             | $-3$     | $-1.5$     |
| 4    | $S \xrightarrow{-1} M$ | $S$           | $-1$       | $-1$ | $-1.5$  | $-1 + (-1.5) = -2.5$         | $-1.5$   | $-1.75$    |
| 5    | $M \xrightarrow{-1} G$ | $M$           | $-1.5$     | $-1$ | 0       | $-1 + 0 = -1$                | $0.5$    | $-1.25$    |

### TD Loss Computation

Using step 3 as an example, $\delta = -3$:

$$
L = \delta^2 = (-3)^2 = 9
$$

The gradient-descent direction:

$$
\frac{\partial L}{\partial V(M)} = -2\delta = -2 \times (-3) = 6
$$

The parameter moves in the direction of $-\frac{\partial L}{\partial V(M)}$, i.e., $V(M)$ decreases. In practice, the update is equivalently $V(M) \leftarrow V(M) + \alpha \cdot \delta$, consistent with the table above.

The advantages of TD methods (recall the [TD(0) update](../chapter03_mdp/dp-mc-td): $V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$):

1. **No need to wait for the episode to end** -- you can update at every step.
2. **Lower variance** -- $V_\phi(s')$ acts as an "anchor" that stabilizes the estimate.
3. **Matches the Actor's update cadence** -- both update once per environment step.

The price is introducing **bias**: $V_\phi(s')$ is itself an estimate, not the true value. This is called [bootstrapping](../chapter03_mdp/dp-mc-td) -- using your own estimates to update your own estimates. In practice, however, this bias is far smaller than the benefit gained from reducing variance.

## Comparing the Three Methods

|                           | **DP**               | **MC** | **TD**                         |
| ------------------------- | -------------------- | ------ | ------------------------------ |
| **Used to train Critic?** | Theoretical baseline | Usable | **Practical default**          |
| **Need episode to end?**  | No                   | Yes    | No                             |
| **Unbiased?**             | Yes                  | Yes    | No (biased but lower variance) |
| **Variance**              | Low                  | High   | Medium                         |
| **Bootstrapping**         | Yes                  | No     | Yes                            |

### MC vs. TD: A Numerical Comparison

Same trajectory $S \xrightarrow{-2} S \xrightarrow{-1} M \xrightarrow{-2} S \xrightarrow{-1} M \xrightarrow{-1} G$, initial table all zeros, $\alpha = 0.5$, $\gamma = 1$.

**MC** -- updates only after the entire episode ends. At the first visit to $S$, the target is the complete return over the whole trajectory:

$$
G_0 = (-2) + (-1) + (-2) + (-1) + (-1) = -7
$$

$$
V(S) \leftarrow 0 + 0.5 \times (-7 - 0) = -3.5
$$

MC uses all information from start to finish in a single update.

**TD** -- updates immediately after the first step. Step 1 uses only one-step information:

$$
\delta = -2 + V(S) - V(S) = -2 + 0 - 0 = -2
$$

$$
V(S) \leftarrow 0 + 0.5 \times (-2) = -1
$$

The TD target ($-2$) is much smaller in magnitude than the MC target ($-7$), but TD does not need to wait for the episode to end. As more trajectories accumulate, TD's $V(S)$ also gradually approaches the true value of $-3.375$.

Both methods eventually converge to the same $V^\pi$, but their update paths differ: MC makes large single updates ($-3.5$) with high variance; TD makes small updates ($-1$) but more frequently, with lower variance.

In practice, Actor-Critic methods almost always use TD to train the Critic. In more advanced implementations (e.g., [GAE in Chapter 8](../chapter10_ppo/gae-reward-model)), MC and TD are combined -- a parameter $\lambda$ interpolates between them to achieve an optimal bias-variance tradeoff.

## The Full Critic-Training Workflow

Putting the pieces together, a one-step Actor-Critic training loop looks like this:

1. **Interact**: At state $s$, the Actor selects action $a$; the environment returns $r$ and $s'$.
2. **Forward pass**: The Critic computes the current prediction $V_\phi(s)$ and the next-step prediction $V_\phi(s')$.
3. **Compute TD Error**: $\delta = r + \gamma V_\phi(s') - V_\phi(s)$.
4. **Update Critic**: Update the Critic parameters $\phi$ using $\delta^2$ as the loss.
5. **Update Actor**: Update the Actor parameters $\theta$ using $\delta$ as the advantage estimate.

### Numerical Walkthrough

Assume the current Critic value table is $V(S) = -1$, $V(M) = -0.5$, $V(G) = 0$, with $\gamma = 0.9$, Critic learning rate $\alpha_\phi = 0.1$, and Actor learning rate $\alpha_\theta = 0.01$.

**Step 1: Interact**

At state $S$, the Actor chooses right with probability 0.8 and left with probability 0.2. Suppose this sample picks right; the environment returns $r = -1$, $s' = M$.

**Step 2: Forward Pass**

$$
V_\phi(S) = -1, \quad V_\phi(M) = -0.5
$$

**Step 3: Compute TD Error**

$$
\delta = r + \gamma V_\phi(s') - V_\phi(s) = -1 + 0.9 \times (-0.5) - (-1) = -1 + (-0.45) + 1 = -0.45
$$

$\delta = -0.45 < 0$ indicates that moving right from $S$ to $M$ was worse than the current prediction -- actually receiving $-1$ plus $M$'s estimate of $-0.45$ totals $-1.45$, which is lower than $S$'s estimate of $-1$.

**Step 4: Update Critic**

$$
L_{\text{Critic}} = \delta^2 = (-0.45)^2 = 0.2025
$$

Parameter update (using a value table as an example):

$$
V(S) \leftarrow V(S) + \alpha_\phi \cdot \delta = -1 + 0.1 \times (-0.45) = -1 + (-0.045) = -1.045
$$

The Critic lowered $V(S)$ -- this experience suggests $S$'s value is lower than previously estimated.

**Step 5: Update Actor**

$\delta = -0.45$ indicates that this action (moving right) performed worse than expected. The Actor's update direction is to decrease the probability of this action. Using the policy gradient as an example:

$$
\theta \leftarrow \theta + \alpha_\theta \cdot \delta \cdot \nabla_\theta \log \pi(\text{right} \mid S)
$$

Since $\delta < 0$, the parameters move opposite to $\nabla_\theta \log \pi(\text{right} \mid S)$, reducing the probability $\pi(\text{right} \mid S)$.

If $\delta > 0$, the action was better than expected, and the Actor increases its probability.

The Critic parameters $\phi$ update in the direction that makes $\delta^2$ smaller -- predictions become more accurate. The Actor parameters $\theta$ update in the direction that assigns higher probability to actions with positive $\delta$ -- choices become better. This creates a virtuous cycle: the more accurate the Critic's evaluation, the faster the Actor improves; the more diverse actions the Actor tries, the richer the data the Critic sees, and the more accurate its evaluation becomes.

## References

[^1]: Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. _Machine Learning_, 3(1), 9-44.

[^2]: Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. _ICML_. [arXiv:1602.01783](https://arxiv.org/abs/1602.01783)
