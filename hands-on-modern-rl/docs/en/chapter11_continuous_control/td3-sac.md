# 9.2 TD3 and SAC

> [9.1](./ddpg) explained Deterministic Policy Gradients (DPG) and DDPG, which transfer DQN's off-policy approach to continuous actions. DDPG, however, has three widely criticized weaknesses: Q-value overestimation, hyperparameter sensitivity, and unstable training. This section presents two complementary remedies: **TD3** stabilizes DDPG with engineering techniques, while **SAC** reformulates the objective through maximum-entropy RL.

## Stability Improvements for DDPG

Twin Delayed Deep Deterministic Policy Gradient (Fujimoto et al., 2018) addresses DDPG's three weaknesses with three modifications.

### 1. Twin Q-Networks

Following the idea behind Double DQN, TD3 trains **two independent critics**, $Q_{\phi_1}, Q_{\phi_2}$, and uses the smaller value as the target:

$$y = r + \gamma \cdot \min(Q_{\phi_1'}, Q_{\phi_2'})(s', \mu_{\theta'}(s'))$$

This structure suppresses Q-value overestimation because two networks are much less likely to overestimate the same value simultaneously than one network is.

```python
class TD3Critic:
    def __init__(self, state_dim, action_dim):
        self.Q1 = QNetwork(state_dim, action_dim)
        self.Q2 = QNetwork(state_dim, action_dim)  # Initialized independently

    def forward(self, s, a):
        return self.Q1(s, a), self.Q2(s, a)

    def target_min(self, s, a):
        return torch.min(self.Q1(s, a), self.Q2(s, a))
```

### 2. Delayed Policy Updates

The critic has a more difficult learning problem than the actor: the critic must fit the two-argument function $Q(s,a)$, while the actor needs to learn only the single-argument function $\mu(s)$. TD3 updates the actor only once every $d$ steps, with $d=2$, giving the critic more updates before it supplies a signal to the actor:

```python
for step in range(total_steps):
    # Update the critic at every step
    update_critic()

    # Update the actor and target networks only every d=2 steps
    if step_count % policy_delay == 0:
        update_actor()
        soft_update_targets()
```

The intuition is that while the critic is still inaccurate, its gradients are noisy. Delayed updates keep the actor from following these inaccurate gradients.

### 3. Target Policy Smoothing

DDPG's target action $a' = \mu_{\theta'}(s')$ is deterministic, but the function approximator's values may vary sharply around $s'$. TD3 adds a small amount of smoothing noise to the target action:

$$a' = \text{clip}(\mu_{\theta'}(s') + \epsilon, a_{\text{low}}, a_{\text{high}}), \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)$$

This performs a local average in the action space, making the Q-function smoother along the action dimension and **reducing the critic's sensitivity to small perturbations**. Common settings are $\sigma = 0.2, c = 0.5$.

### The Combined Effect of the Three Modifications

TD3 makes DDPG **substantially more stable** on MuJoCo and outperformed contemporary early versions of SAC. TD3 remains a strong baseline for continuous control.

```python
class TD3:
    def update(self, batch_size=256):
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        # === Critic update (twin Q-networks) ===
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            # Target policy smoothing
            noise = (torch.randn_like(next_actions) * 0.2).clamp(-0.5, 0.5)
            next_actions = (next_actions + noise).clamp(-self.action_max, self.action_max)
            # Take the minimum of the twin Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optim.zero_grad(); critic_loss.backward()
        self.critic_optim.step()

        # === Actor update (delayed) ===
        if self.step_count % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            self.actor_optim.zero_grad(); actor_loss.backward()
            self.actor_optim.step()

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
```

## Maximum-Entropy RL

Soft Actor-Critic (Haarnoja et al., 2018) takes a different approach: the policy maximizes **return plus entropy**, rather than expected return alone.

### The Maximum-Entropy RL Objective

$$J(\pi) = \mathbb{E}_{(s_t, a_t) \sim \pi}\left[\sum_t \gamma^t \big(r_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t))\big)\right]$$

Here, $\mathcal{H}(\pi) = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)]$ is the policy entropy, and the temperature coefficient $\alpha$ controls its weight.

**Why add entropy?**

- **Encouraging exploration**: a high-entropy policy does not converge prematurely to a single action
- **Robustness**: a multimodal policy that assigns probability to several good actions is more robust to environmental perturbations
- **Training stability**: entropy regularization makes the Q-function smoother and reduces overestimation

### The Soft Bellman Equation

The modified Bellman backup is

$$Q^\pi(s, a) = \mathbb{E}_{s'}\left[r + \gamma \cdot V^\pi(s')\right], \quad V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)] + \alpha \mathcal{H}(\pi(\cdot \mid s))$$

The key change is that $V$ is no longer $\max_a Q$. It is a **soft maximum**, expressed in log-sum-exp form and evaluated as an expectation for continuous actions:

$$V^\pi(s) = \alpha \log \int \exp\left(\frac{Q^\pi(s, a)}{\alpha}\right) da$$

### Reparameterizing the Stochastic Policy

SAC's policy $\pi_\theta(a \mid s)$ is Gaussian. It uses the reparameterization trick to compute actor gradients:

$$a = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This makes the actor loss differentiable:

$$\mathcal{L}_{\text{actor}} = \mathbb{E}_{s \sim \mathcal{D}, \epsilon}\left[\alpha \log \pi_\theta(a \mid s) - Q_\phi(s, a)\right]$$

### Automatic Temperature Tuning

The most difficult hyperparameter is $\alpha$. An engineering innovation in SAC is **automatic temperature tuning**:

$$\alpha^* = \arg\max_\alpha \mathbb{E}\left[-\alpha \log \pi(a \mid s) - \alpha \mathcal{H}_0\right]$$

Here, $\mathcal{H}_0$ is the target entropy, usually set to $-|\mathcal{A}|$. This allows $\alpha$ to adjust automatically during training: when entropy is too high, $\alpha$ decreases; when entropy is too low, $\alpha$ increases.

```python
# Optimize alpha for automatic temperature tuning
def update_alpha(self, states, actions):
    # Learn alpha so that policy entropy approaches target_entropy
    log_pi = -self.actor.log_prob(states, actions)  # Current policy's negative log-likelihood
    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
    self.alpha_optim.zero_grad()
    alpha_loss.backward()
    self.alpha_optim.step()
    self.alpha = self.log_alpha.exp()
```

### Advantages of SAC

SAC has remained a leading method on MuJoCo because it provides

1. **High off-policy sample efficiency**, inherited from DDPG
2. **Automatic exploration through maximum entropy**, without a manually tuned noise scale
3. **Stable training** through twin Q-networks and soft targets
4. **Performance beyond the human level**, reaching scores above 15,000 on HalfCheetah

### Comparing the Three Algorithms

| Dimension                  | DDPG          | TD3           | SAC                       |
| -------------------------- | ------------- | ------------- | ------------------------- |
| Policy type                | Deterministic | Deterministic | Stochastic (Gaussian)     |
| Number of Q-networks       | 1             | 2 (twin)      | 2 (twin)                  |
| Exploration method         | Added noise   | Added noise   | Entropy reward (built in) |
| Stability                  | Poor          | Moderate      | Strong                    |
| Hyperparameter sensitivity | High          | Moderate      | Low                       |
| Recommended first choice   | ❌            | ⚠️            | ✅                        |

**Practical recommendation**: use SAC as the first choice for continuous control. If a deterministic policy is required, for example to eliminate randomness at deployment time, use TD3.

## Training Curves on HalfCheetah

The following diagram compares training for one million steps in the MuJoCo HalfCheetah-v3 environment:

```
return
12000 │                    ╭─────── SAC (stable convergence)
10000 │                  ╭─╯
 8000 │                ╭─╯  ╭─────── TD3 (stable but slightly slower)
 6000 │              ╭─╯   ╱
 4000 │            ╭─╯    ╱
 2000 │          ╭─╯     ╱  ╭───── DDPG (occasional recovery after divergence)
     0 │─────────╯──────╱──╯
       └───────────────────────────────
        0    200K  400K  600K  800K  1M steps
```

Three observations follow:

- **SAC** converges fastest and most steadily because maximum-entropy exploration accelerates early learning
- **TD3** is slightly slower than SAC but reaches similar final performance because its stability modifications make DDPG practical
- **DDPG** diverges much of the time and trains successfully only under some random seeds

## Section Summary

DDPG → TD3 → SAC forms a three-stage development in continuous control:

1. **DDPG** extends DQN's ideas to continuous actions, but is unstable
2. **TD3** stabilizes DDPG with twin Q-networks, delayed updates, and target smoothing
3. **SAC** reformulates the objective through maximum-entropy RL, incorporating exploration and automatic temperature tuning

In practice, SAC is the first choice, TD3 is the alternative when a deterministic policy is required, and DDPG is no longer recommended.

The next section, [9.3 Model-Based RL](./model-based), turns to another direction: when sampling the real environment is expensive, learning an environment model can generate "synthetic" data and improve sample efficiency by a factor of 10–100.
