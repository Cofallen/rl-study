# 9.1 Deterministic Policy Gradients and DDPG

The previous chapter introduced PPO. By clipping its objective, PPO makes Actor-Critic training more stable, and the BipedalWalker experiment showed that it can learn continuous robot control. PPO still has one clear limitation: **it is on-policy, so its sample efficiency is low**.

On-policy training can update only from data collected by the current policy. Once the parameters change, earlier data can no longer be reused as if it came from the new policy, so another round of interaction is required. In a simulator this may be acceptable: two million BipedalWalker steps can take only tens of minutes. Real robots make every interaction costly:

- motors wear out and batteries have limited capacity;
- road testing for autonomous vehicles introduces safety risks;
- one incorrect industrial-robot action can damage equipment.

Millions of real samples may therefore require days or weeks of uninterrupted operation. This chapter follows a single thread—**improving sample efficiency**—through three levels:

| Level | Method                        | Source of sample efficiency                                                        | Representative algorithms    |
| ----- | ----------------------------- | ---------------------------------------------------------------------------------- | ---------------------------- |
| 1     | Off-policy continuous control | Reuse historical interactions instead of requiring samples from the current policy | DDPG → TD3 → SAC             |
| 2     | Model-based RL                | Learn an environment model and train on imagined data, reducing real interaction   | Dyna → PETS → MBPO           |
| 3     | Search and world models       | Use the learned model directly for planning at decision time                       | AlphaZero → MuZero → Dreamer |

::: tip Prerequisites
This chapter repeatedly uses the following ideas:

- [experience replay and target networks in DQN](../chapter07_dqn/from-q-to-dqn), the basic components of off-policy deep RL;
- the [Actor-Critic architecture](../chapter09_actor_critic/actor-critic), which DDPG and SAC extend;
- [PPO's stability problem](../chapter10_ppo/trust-region-clipping), which motivates the move from on-policy to off-policy learning.
  :::

The chapter is organized as follows:

- **9.1 Deterministic Policy Gradients and DDPG** (this section) asks how DQN's off-policy approach can be extended to continuous actions. DDPG provides an answer, although its training is unstable.
- [9.2 TD3 and SAC](./td3-sac) presents two remedies. TD3 adds three stabilization techniques to DDPG, while SAC rebuilds the objective around maximum-entropy RL.
- [9.3 Model-Based Reinforcement Learning](./model-based) goes beyond replaying historical data and learns a model that can generate imagined data, often reducing real interaction by one or two orders of magnitude.
- [9.4 Search and World Models](./search-world-models) places the model inside the decision loop. AlphaZero, MuZero, and Dreamer combine planning or imagined rollouts with neural value estimation.

---

### Section Preview

This section develops four ideas:

- why discrete-action Q-learning methods such as DQN cannot directly handle continuous actions—the required argmax cannot be computed by enumeration;
- how the Deterministic Policy Gradient (DPG) theorem extends policy gradients from stochastic to deterministic policies and permits off-policy training;
- how DDPG combines DQN's experience replay and target networks with Actor-Critic;
- why Q-value overestimation, hyperparameter sensitivity, and unstable feedback make DDPG difficult to use in practice.

We begin with the first question: **why can DQN not be applied directly to continuous actions?**

## The Fundamental Difficulty of Continuous Actions

CartPole offers two actions: push left or push right. Atari games likewise expose a small set of buttons. Because these action spaces are finite, an algorithm can enumerate them.

Robot control is different. A seven-joint arm may choose any torque vector in $[-1,1]^7$. A steering angle may take any value in $[-\pi,\pi]$, and a throttle command may take any value in $[0,1]$. These are continuous action spaces and contain infinitely many candidates.

Recall DQN's central decision rule:

$$
a^* = \arg\max_a Q(s,a).
$$

For a discrete action space, the computation is straightforward. Evaluate $Q(s,\text{left})$ and $Q(s,\text{right})$, then choose the larger value. For $a\in\mathbb{R}^n$, evaluating every possible action is impossible.

This is the first difficulty: **the Q-function cannot be maximized by direct enumeration**.

PPO avoids this argmax. It learns a stochastic policy $\pi_\theta(a\mid s)$, usually a Gaussian distribution whose network outputs a mean and standard deviation, and samples an action from that distribution. The policy-gradient theorem provides an update rule, but the samples must come from the current policy. That restriction makes PPO on-policy and limits its sample efficiency.

The natural next step is to combine the two approaches: reuse data off-policy as DQN does, while still producing continuous actions as PPO does. Silver et al. (2014) provided the key result: the **Deterministic Policy Gradient (DPG)** theorem.

## The Deterministic Policy Gradient Theorem

First recall the stochastic policy-gradient theorem from Chapter 6:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{s\sim d^\pi,\,a\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(a\mid s)\,Q^\pi(s,a)
\right].
$$

Here $\pi_\theta$ is a probability distribution. The update increases the probability of actions with high $Q$ values. The expression contains an expectation over $a\sim\pi_\theta$, so in a continuous action space many sampled actions may be needed for an accurate estimate.

Silver et al. showed that a related theorem still holds for a deterministic policy $a=\mu_\theta(s)$, which outputs one action directly:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{s\sim d^\mu}
\left[
\nabla_\theta\mu_\theta(s)
\cdot
\nabla_a Q^\mu(s,a)\bigg|_{a=\mu_\theta(s)}
\right].
$$

The chain rule separates this expression into two parts:

- $\nabla_\theta\mu_\theta(s)$ asks how a small parameter change moves the action produced by the policy;
- $\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}$ asks how a small action change moves the critic's value estimate.

Their product moves the policy parameters in the direction that changes the action toward a higher $Q$ value.

The deterministic form has two practical advantages:

- **No expectation over actions is required.** The stochastic theorem averages over possible actions; the deterministic theorem needs an expectation only over states, which reduces estimation variance.
- **It supports off-policy learning.** Training can use states collected by a different behavior policy, including older noisy versions of the actor, instead of requiring every sample to come from the current target policy.

A deterministic policy does not explore by itself. If $\mu_\theta(s)$ always returns the same action in state $s$, no alternative is ever tested. DDPG separates exploration from the learned target policy by adding noise to the actor's output while collecting training data.

## DDPG: Deep Deterministic Policy Gradient

Lillicrap et al. (2015) combined DPG with the deep-network techniques used by DQN to create DDPG. Its components are familiar:

- **Actor:** $\mu_\theta(s)$ maps a state directly to a continuous action vector.
- **Critic:** $Q_\phi(s,a)$ receives both the state and the action and returns one scalar Q-value. DQN instead receives only the state and returns one Q-value per discrete action.
- **Target networks:** both the actor and critic have slowly updated target copies that stabilize their learning targets.
- **Experience replay:** a replay buffer stores interactions, and random mini-batches permit off-policy reuse.

The complete update can be seen in a compact implementation:

```python
class DDPG:
    def __init__(self, state_dim, action_dim, action_max):
        # Online networks
        self.actor = Actor(state_dim, action_dim, action_max)
        self.critic = Critic(state_dim, action_dim)
        # Target networks
        self.actor_target = copy(self.actor)
        self.critic_target = copy(self.critic)
        self.replay_buffer = ReplayBuffer(capacity=1_000_000)
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state, explore=True):
        with torch.no_grad():
            action = self.actor(state)
        if explore:
            # Add Gaussian exploration noise during data collection.
            action += np.random.normal(0, 0.1, size=action.shape)
        return np.clip(action, -self.action_max, self.action_max)

    def update(self, batch_size=256):
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optim.zero_grad(); critic_loss.backward()
        self.critic_optim.step()

        # Actor update: maximize Q(s, mu(s)).
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad(); actor_loss.backward()
        self.actor_optim.step()

        # Soft-update the target networks.
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
```

The `update` method performs three steps.

**First, update the critic.** The target actor chooses the next action,

$$
a'=\mu_{\theta'}(s'),
$$

and the target critic builds a TD target,

$$
y=r+\gamma Q_{\phi'}(s',a').
$$

The critic parameters are then updated so that $Q_\phi(s,a)$ approaches $y$.

**Second, update the actor.** The actor maximizes the critic's score $Q(s,\mu_\theta(s))$. Since optimizers perform gradient descent, the loss is its negative:

$$
L_{\text{actor}}=-\mathbb{E}_s[Q_\phi(s,\mu_\theta(s))].
$$

The gradient first flows from the critic's output to its action input and then through $a=\mu_\theta(s)$ to the actor parameters. This is exactly the product $\nabla_aQ\cdot\nabla_\theta\mu$ in the DPG theorem.

**Third, update the target networks.** DDPG usually avoids DQN's periodic hard copy and instead moves each target a small amount after every update:

$$
\theta'\leftarrow\tau\theta+(1-\tau)\theta'.
$$

With $\tau=0.005$, the target moves only 0.5% of the distance toward the online network at each step. Consequently, the TD target changes slowly.

During data collection, `select_action` adds Gaussian noise $\mathcal{N}(0,0.1^2)$ to the deterministic action. The replay buffer therefore contains data from a noisy behavior policy, while the actor being optimized remains deterministic. This separation of behavior and target policies is the practical benefit of off-policy learning.

DDPG demonstrated that deep off-policy RL could solve continuous-control tasks such as HalfCheetah, Hopper, and Walker2d. Its basic mechanism is important, but the algorithm is difficult to tune and can fail abruptly.

## Three Weaknesses of DDPG

DDPG introduced deep off-policy continuous control, but three problems limit its practical reliability.

**1. Q-value overestimation.** DQN can overestimate values because a maximum tends to select actions whose estimates contain positive noise. In DDPG, the actor itself searches for actions that the critic scores highly. If the critic overestimates one action, the actor moves toward it; the changed actor can then reinforce the critic's error, creating a positive feedback loop.

**2. Hyperparameter sensitivity.** Small changes to the learning rate, exploration-noise scale, network architecture, or soft-update coefficient $\tau$ can change a successful run into a divergent one. Settings often have to be retuned for each environment.

**3. Training instability.** A slightly inaccurate critic gives the actor an incorrect gradient. The resulting actor collects poorer data, which further degrades the critic, and the coupled system can collapse. Recovery after this feedback loop begins is difficult.

These are the problems addressed in the next section. TD3 adds targeted stabilization techniques to DDPG, while SAC changes the objective through maximum-entropy RL.

::: details Additional note: Ornstein-Uhlenbeck noise and Gaussian noise
The original DDPG paper used an Ornstein-Uhlenbeck (OU) process rather than independent Gaussian noise:

$$
x_{t+1}=x_t+\vartheta(\mu-x_t)+\sigma\epsilon_t.
$$

OU noise is temporally correlated, so an exploratory movement has inertia from one step to the next. This property was intended to suit physical-control tasks. Later work, including TD3, found that independent Gaussian noise works at least as well in common benchmarks. Modern implementations therefore usually prefer the simpler Gaussian choice.
:::

## Section Summary

The Deterministic Policy Gradient theorem extends policy gradients from stochastic policies to a direct action mapping:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{s\sim d^\mu}
\left[
\nabla_\theta\mu_\theta(s)
\cdot
\nabla_aQ^\mu(s,a)\bigg|_{a=\mu_\theta(s)}
\right].
$$

DDPG combines this theorem with experience replay, target networks, and soft updates. The actor outputs a deterministic continuous action, the critic evaluates that action, and a noisy behavior policy supplies exploration data.

DDPG remains vulnerable to Q-value overestimation, hyperparameter sensitivity, and unstable actor-critic feedback. [Section 9.2](./td3-sac) presents two complementary remedies: TD3 stabilizes the DDPG update, and SAC rebuilds it around a maximum-entropy objective.
