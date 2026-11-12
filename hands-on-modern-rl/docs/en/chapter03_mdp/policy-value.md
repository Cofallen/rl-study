# 2.3 Policies, Values, and Returns

The previous section used the MDP tuple to describe a sequential decision problem. $\mathcal{S}$ and $\mathcal{A}$ specify what the agent can encounter and do; $P$ and $R$ specify how the environment changes and what immediate reward it provides; $\gamma$ specifies the weight of future rewards.

These objects are enough to record one CartPole interaction. At each step, the agent observes the current state, pushes left or right, receives a reward, and enters the next state. Repeating this process produces a trajectory. Once the pole falls, we can calculate the return $G_t$ from every time step along that trajectory.

Decision making requires an estimate before the trajectory ends. From an intermediate state, the agent needs to know the average future return from continuing, and how that outcome changes if it chooses one action first. A single return $G_t$ describes one realized trajectory, while repeated visits to the same state can produce different actions, transitions, and returns.

This section therefore follows the notation from Section 2.2: first a policy produces a trajectory, then return measures that trajectory, and finally expected return defines the state value $V^\pi(s)$ and action value $Q^\pi(s,a)$. This moves us from recording an interaction to evaluating states and actions before acting.

## Policies and Decision Rules

Consider one decision in CartPole. At each step, the environment returns four numbers: cart position, cart velocity, pole angle, and pole angular velocity. Suppose we observe

$$
s_1=[0.00, 0.00, 0.08, 0.15].
$$

This four-dimensional vector is one concrete **state**. The cart is near the center and almost stationary. The pole is tilted by $0.08$ radians and is still rotating at $0.15$ radians per second.

Another run might produce

$$
s_2=[0.12, -0.04, -0.05, -0.10].
$$

This is a different state because all four measurements may differ. Collecting every state that CartPole can produce gives the **state space** $\mathcal{S}$:

$$
\mathcal{S}=\left\{[x,\dot{x},\theta,\dot{\theta}]\mid
\text{the values form a possible CartPole state}\right\}.
$$

The calligraphic capital $\mathcal{S}$ denotes the whole set, while lowercase $s$ denotes one member of that set. Thus, $s_1\in\mathcal{S}$ reads “$s_1$ belongs to the state space $\mathcal{S}$.” Because the CartPole state changes continuously, $\mathcal{S}$ contains many four-dimensional vectors, not just the two above.

CartPole has only two available actions. Its **action space** is

$$
\mathcal{A}=\{\text{push left}, \text{push right}\}.
$$

The calligraphic capital $\mathcal{A}$ denotes all available actions, while lowercase $a$ denotes the action selected at one step. For example, $a=\text{push right}$ and $a\in\mathcal{A}$.

The agent receives a state $s$ and must select an action $a$ from $\mathcal{A}$. The rule that turns a state into a choice is a **policy**, written as the Greek letter $\pi$, pronounced “pi.” Policies have two common forms.

A **deterministic policy** directly returns one action for each state:

$$
\pi:\mathcal{S}\to\mathcal{A},
\qquad
a=\pi(s).
$$

The arrow $\to$ means “maps from the expression on the left to the expression on the right.” Therefore, $\pi:\mathcal{S}\to\mathcal{A}$ says that the policy accepts a state from the state space and returns an action from the action space.

To make the mapping concrete, consider three states and a simple teaching policy. This is not claimed to be an optimal CartPole controller.

| Input state $s$ | Pole motion | Policy output $\pi(s)$ |
| --- | --- | --- |
| $s_1=[0.00,0.00,0.08,0.15]$ | tilted in the positive direction and still rotating that way | push right |
| $s_2=[0.12,-0.04,-0.05,-0.10]$ | tilted in the negative direction and still rotating that way | push left |
| $s_3=[-0.08,0.03,0.01,-0.02]$ | nearly upright | push right |

The first row can be written as

$$
\pi(s_1)=\text{push right}.
$$

$s_1$ is the input, and “push right” is the output. If the same $s_1$ is passed to this deterministic policy again, the output remains “push right.” A complete deterministic policy specifies one action for every state.

A **stochastic policy** returns action probabilities instead of immediately fixing one action:

$$
\pi:\mathcal{S}\to\Delta(\mathcal{A}),
\qquad
a\sim\pi(\cdot\mid s).
$$

$\Delta(\mathcal{A})$ means “the set of all probability distributions over the action space $\mathcal{A}$.” Because CartPole has two actions, one such distribution can be written as

$$
[p_{\text{left}},p_{\text{right}}],
\qquad
p_{\text{left}}+p_{\text{right}}=1.
$$

For example, $[0.1,0.9]$ assigns probability $0.1$ to pushing left and $0.9$ to pushing right. The vectors $[0.5,0.5]$, $[0.8,0.2]$, and $[0,1]$ are also valid distributions in $\Delta(\mathcal{A})$.

| Input state $s$ | Probability of left | Probability of right | Policy output $\pi(\cdot\mid s)$ |
| --- | ---: | ---: | --- |
| $s_1$ | $0.1$ | $0.9$ | $[0.1,0.9]$ |
| $s_2$ | $0.85$ | $0.15$ | $[0.85,0.15]$ |
| $s_3$ | $0.45$ | $0.55$ | $[0.45,0.55]$ |

For $s_1$, the policy returns $[0.1,0.9]$. If the agent encounters $s_1$ many times, it will push left about $10\%$ of the time and right about $90\%$ of the time.

In $\pi(\cdot\mid s)$, the vertical bar $\mid$ reads “given,” so the expression refers to action probabilities given the current state $s$. The dot $\cdot$ stands for every candidate action. The symbol $\sim$ means “sample according to the distribution on the right.” Thus, $a\sim\pi(\cdot\mid s)$ says to sample the actual action $a$ from the probabilities produced by the policy.

The full path is now explicit: $s_1$ is a concrete member of $\mathcal{S}$; the stochastic policy maps it to the concrete distribution $[0.1,0.9]$ in $\Delta(\mathcal{A})$; sampling that distribution produces a concrete action in $\mathcal{A}$.

A deterministic policy is a special case. The distribution $[0,1]$ assigns probability $1$ to pushing right, so every sample returns the same action. Policy-gradient methods commonly learn stochastic policies directly, whereas value-based methods such as DQN often derive a deterministic greedy policy from action values.

```python
# A simple stochastic policy for CartPole
import torch
import torch.nn as nn

class CartPolePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.Tanh(),
            nn.Linear(32, 2)  # Logits for the two actions
        )

    def forward(self, state):
        logits = self.net(state)
        return torch.distributions.Categorical(logits=logits)

    def act(self, state):
        dist = self.forward(state)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
```

If we pass `torch.tensor([0.00, 0.00, 0.08, 0.15])` to this network, the first layer receives the four state values. The final layer outputs two `logits`, one for pushing left and one for pushing right. `Categorical` converts those scores into action probabilities, and `sample()` draws action $0$ or $1$. `log_prob(action)` records the log-probability of the sampled action, which policy-gradient methods later use to adjust the network parameters.

### The Optimal Policy

Once policies have been defined, we need a way to compare them. Suppose policy A controls CartPole for $100$ episodes and survives for $180$ steps on average, while policy B survives for $420$ steps on average. Because CartPole gives $+1$ for each surviving step, policy B has the larger average long-term return.

The policy with the largest expected long-term return among all candidates is called an **optimal policy**, written $\pi^*$:

$$\pi^* = \arg\max_\pi \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right]$$

The star $*$ marks an optimal quantity. $\arg\max_\pi$ means “find the policy $\pi$ that makes the expression on the right as large as possible,” and $\mathbb{E}_\pi$ means averaging over the different outcomes that the policy may produce. DQN, PPO, and SAC represent and update policies differently, but all aim to improve this expected return.

## Returns and Trajectory Evaluation

Consider a trajectory with only three steps. The agent starts in $s_0$, selects $a_0$, receives $r_1=1$, and reaches $s_1$. It then selects $a_1$ and receives $r_2=1$. Finally, it selects $a_2$, receives $r_3=1$, and the task ends. Written in order, this experience is a **trajectory**:

$$
\tau=(s_0,a_0,r_1,s_1,a_1,r_2,s_2,a_2,r_3,s_3).
$$

$\tau$ is the Greek letter tau and denotes the whole trajectory. The subscripts $0,1,2$ on states and actions mark decision times; $r_1$ is the feedback received after the action at time $0$. One complete run from the initial state to a terminal state is usually called an episode.

A trajectory contains several one-step rewards. To evaluate the entire future from a particular time, we combine those rewards into a **return**:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

$G_t$ is the return from time $t$. $R_{t+1}$ is the reward received after the current action, $k$ counts how many steps into the future a reward lies, and $\sum$ means to add all of these terms.

### The Role of the Discount Factor $\gamma$

$\gamma \in [0, 1]$ is the **discount factor**. It determines how much weight future rewards retain. Smaller values reduce the weight quickly; values close to $1$ preserve more of the distant reward.

For the three-step trajectory above, let $\gamma=0.9$. The return from the beginning is

$$
G_0=1+0.9\times1+0.9^2\times1=2.71.
$$

All three rewards equal $1$, but the second has weight $0.9$ and the third has weight $0.9^2=0.81$. If we change $\gamma$ to $0.5$, the same trajectory has return $1+0.5+0.25=1.75$. The more distant rewards now matter less.

For infinite-horizon tasks with bounded rewards, choosing $\gamma<1$ also keeps the discounted sum finite. For a finite-horizon task with a clear terminal time, $\gamma=1$ may be appropriate when all rewards should have equal weight. The choice of $\gamma$ is part of the task objective.

| γ value | Meaning                          | Applications                      |
| ------- | -------------------------------- | --------------------------------- |
| 0       | Immediate rewards only (greedy)  | Rarely used                       |
| 0.9     | Short horizon (about 10 steps)   | Board games, recommender systems  |
| 0.99    | Medium horizon (about 100 steps) | Atari, CartPole                   |
| 0.999   | Long horizon (about 1,000 steps) | Long-term tasks, robot navigation |
| 1.0     | No discounting                   | Finite-horizon tasks              |

### Returns in CartPole

CartPole gives a reward of 1 at every step while the pole remains upright. An episode ends when the pole falls or the cart leaves the permitted range. The return is

$$G_0 = 1 + \gamma + \gamma^2 + \cdots + \gamma^{T-1} = \frac{1 - \gamma^T}{1 - \gamma}$$

where $T$ is the number of steps in the episode. When $\gamma=0.99$ and $T=500$, $G_0\approx99.34$. If the pole survives for only $100$ steps, then $G_0\approx63.40$. Under the same reward rule, surviving longer produces a larger return.

## Value Functions and Long-Term Benefit

The same state can lead to different outcomes. Suppose we restart CartPole from state $s$ three times, follow the same policy, and obtain returns $80$, $100$, and $120$. Their average is

$$
\frac{80+100+120}{3}=100.
$$

With enough repetitions, this average approaches the expected return from the state under the policy. A **value function** estimates this average future return before the future trajectory has actually happened.

### State Value $V(s)$

$$V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid s_t = s\right] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid s_t = s\right]$$

$V$ stands for value, and the superscript $\pi$ reminds us that the value depends on the policy followed afterward. The vertical bar $\mid$ means “given.” Thus, $V^\pi(s)$ is the average of $G_t$ given that the current state is $s_t=s$ and future actions follow policy $\pi$.

### Action Value $Q(s, a)$

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid s_t = s, a_t = a\right]$$

$Q^\pi(s,a)$ additionally fixes the first action. Suppose repeated trials from the same state have average return $70$ when the first action is push left and $110$ when it is push right. Then

$$
Q^\pi(s,\text{push left})=70,\qquad
Q^\pi(s,\text{push right})=110.
$$

Both numbers assume that after this first action, the agent continues with policy $\pi$. They refine the question “how good is this state?” into “how good is a particular first action in this state?”

### The Relationship Between V and Q

$$V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$$

This equation says that $V^\pi(s)$ is the probability-weighted average of the action values. If the policy pushes left with probability $25\%$ and right with probability $75\%$, then

$$
V^\pi(s)=0.25\times70+0.75\times110=100.
$$

$\sum_a$ means to visit every action and add the resulting terms. $\pi(a\mid s)$ is the probability of action $a$, and $Q^\pi(s,a)$ is its long-term value when taken first.

::: details How the V–Q formula follows from the definitions

The two definitions already distinguish what is known before the first action. The state value conditions only on the current state:

$$V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]$$

The action value conditions on the current state and on a chosen first action:

$$Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,\,a_t=a]$$

Under policy $\pi$, that first action is still random: $a_t$ is drawn from $\pi(\cdot\mid s)$. The expectation of $G_t$ can therefore be grouped by which first action occurred. This is the law of total expectation from Section 2.1: the overall mean equals each group's probability times the mean inside that group.

$$
\begin{aligned}
V^\pi(s)
&=\mathbb{E}_\pi[G_t\mid s_t=s] \\
&=\sum_a \pi(a\mid s)\,\mathbb{E}_\pi[G_t\mid s_t=s,\,a_t=a] \\
&=\sum_a \pi(a\mid s)\,Q^\pi(s,a).
\end{aligned}
$$

The first line is the definition of $V^\pi(s)$. The second line expands the same expectation over the possible first actions. The third line replaces each inner conditional expectation with $Q^\pi(s,a)$.

With two CartPole actions this is the arithmetic above:

$$V^\pi(s)=0.25\times 70+0.75\times 110=100.$$

Here $0.25$ and $0.75$ are $\pi(\text{push left}\mid s)$ and $\pi(\text{push right}\mid s)$, while $70$ and $110$ are the two action values. If the policy is deterministic, one probability is $1$ and the others are $0$, so $V^\pi(s)$ equals the $Q$ of that single action.
:::

### A Numerical GridWorld Example

Consider a corridor with three nonterminal states. The policy always moves right, entering the terminal state gives reward $+1$, every other transition gives reward $0$, and $\gamma=0.9$:

```
S0 ──→ S1 ──→ S2 ──→ terminal
0.81   0.90   1.00      0
```

Starting from $S_2$, the next move gives $+1$, so $V^\pi(S_2)=1$. Starting from $S_1$, that reward arrives one step later, so $V^\pi(S_1)=0.9$. Similarly, $V^\pi(S_0)=0.9^2=0.81$. States closer to the terminal reward have larger values because the same reward is discounted fewer times.

## The Advantage Function and Action Evaluation

The **advantage function** measures how much better action $a$ is than the average action:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

- $A > 0$: action $a$ is better than average
- $A < 0$: action $a$ is worse than average
- $A = 0$: action $a$ is average

Continue with $V^\pi(s)=100$. Pushing right has action value $110$, so

$$
A^\pi(s,\text{push right})=110-100=10.
$$

The positive value says that pushing right is $10$ return units better than the policy's average choice. Pushing left has advantage $70-100=-30$, so it is $30$ below average. Advantage measures relative quality, which is why positive and negative values can appear in the same state.

The advantage function is central to policy-gradient methods ([Chapter 6](../chapter08_policy_gradient/reinforce)) and Actor-Critic methods ([Chapter 7](../chapter09_actor_critic/actor-critic)).

## A Preview of the Bellman Equation

Value functions satisfy the **Bellman equation**, a recursive relationship that expresses $V(s)$ in terms of $V(s')$:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right]$$

First consider one term inside the brackets. Suppose action $a$ in state $s$ gives immediate reward $2$ and reaches $s'$. If $V^\pi(s')=10$ and $\gamma=0.9$, the immediate reward plus discounted future value is

$$
R(s,a,s')+\gamma V^\pi(s')=2+0.9\times10=11.
$$

The full equation also accounts for every action the policy may choose and every next state the environment may produce. The outer sum $\sum_a\pi(a\mid s)$ averages over action probabilities, while the inner sum $\sum_{s'}P(s'\mid s,a)$ averages over transition probabilities. [Chapter 3: Value Functions and Bellman Equations](./value-bellman) develops this equation from concrete examples.

## Section Summary

Policies, returns, and value functions are three core concepts in an MDP:

1. **Policy $\pi$**: the agent's decision rule; the stochastic policy $a \sim \pi(\cdot \mid s)$ is the most general form
2. **Return $G_t$**: the discounted cumulative reward from time step $t$ onward, $\sum \gamma^k r$
3. **Value functions**: $V^\pi(s)$ is the state value and $Q^\pi(s, a)$ is the action value; the advantage $A = Q - V$ measures relative quality

These objects can describe what happens along a trajectory, but return $G_t$ is available only after the trajectory unfolds. When an agent must act from an intermediate state, it needs to estimate the long-term outcome in advance. Chapter 3 begins with this problem and develops value functions and Bellman equations as the solution.
