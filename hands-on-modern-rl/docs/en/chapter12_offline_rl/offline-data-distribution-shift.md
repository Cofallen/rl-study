# 10.1 Offline Data and Distribution Shift

Part II relied on agents continually interacting with an environment to collect experience. Many real systems can use only existing logs because new trial and error may be expensive, slow, or unsafe. Part III begins with offline reinforcement learning, then extends fixed-data learning to imitation learning, inverse reinforcement learning, meta-reinforcement learning, exploration, multi-agent learning, and hierarchical decision-making.

DDPG, TD3, and SAC in [Chapter 9](../chapter11_continuous_control/ddpg) can reuse historical data through a replay buffer, and model-based reinforcement learning can reduce real interaction with an environment model. These methods still allow a new policy to collect samples and correct old experience. Offline reinforcement learning removes this feedback channel: training can use only one fixed dataset.

This section follows three questions: why fixed data cause distribution shift, how BCQ, CQL, and IQL constrain estimation errors, and how AWAC and TD3+BC incorporate behavior cloning into policy updates. With this foundation, [Section 10.2](./sequence-modeling) will turn to Decision Transformer's sequence-modeling approach.

## 1. Why Fixed Data Cause Distribution Shift

Both [Chapter 5: DQN](../chapter07_dqn/from-q-to-dqn) and [Chapter 9: SAC](../chapter11_continuous_control/ddpg) update the value of the current state using the estimated value of the next state. First write the one-step target:

$$y = r + \gamma \cdot \mathbb{E}_{s' \sim P(\cdot \mid s, a)}\left[V(s')\right]$$

Read this expression from left to right: $y$ is the target fitted in the current update, $r$ is the reward already obtained from the current action, $V(s')$ is the long-term value after the next state, and $\gamma$ controls the weight of future value in the target. Even if online training temporarily overestimates a new state, the policy can visit it later and correct the estimate with observed rewards.

In online RL, the $V(s')$ in the target is supported by future exploration. Even if a new policy reaches an unseen state, the agent continues interacting with the environment and collects new data to correct the estimate. **Offline RL has no such safeguard.** Dataset $\mathcal{D} = \{(s, a, r, s')\}$ is collected by a behavior policy $\pi_\beta$ and remains **completely fixed** during training:

$$\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^{N}, \quad (s, a) \sim d^{\pi_\beta}(s) \pi_\beta(a \mid s)$$

The new policy $\pi_\theta$ is deployed after training, but its action distribution $\pi_\theta(a \mid s)$ differs from $\pi_\beta(a \mid s)$. This creates **distribution shift**.

### 1.1 Where Extrapolation Error Comes From

Fujimoto et al. 2019 precisely characterized the source of offline-RL failure in the BCQ paper. Let the dataset's action support be $\mathcal{D}_\mathcal{A}(s) = \{a : (s, a) \in \text{support}(\pi_\beta(\cdot \mid s))\}$. The Bellman operator receives no supervision for $a' \notin \mathcal{D}_\mathcal{A}(s')$. A neural network **extrapolates** at these OOD (out-of-distribution) points, and its output may be arbitrary.

To identify the source of the problem, we can write an illustrative decomposition of value-estimation error:

$$\underbrace{Q_\phi(s, a) - Q^\pi(s, a)}_{\text{total error}} = \underbrace{\epsilon_{\text{stat}}}_{\substack{\text{statistical error}\\\text{(finite samples)}}} + \underbrace{\epsilon_{\text{approx}}}_{\substack{\text{approximation error}\\\text{(network capacity)}}} + \underbrace{\max_{a'} Q_\phi(s', a') - Q^\pi(s', \pi(s'))}_{\text{extrapolation error}}$$

The first two terms appear in both online and offline training. The third appears when maximization selects an action unsupported by the data: the network may happen to assign this action a large Q-value, and the max operator preferentially selects it, inserting the unverified estimate into the next target.

The accumulation of extrapolation error can be expanded recursively. Let $Q_0$ be the initial estimate. After $T$ Bellman iterations, the error satisfies

$$\|Q_T - Q^\pi\|_\infty \leq \gamma^T \|Q_0 - Q^\pi\|_\infty + \sum_{k=0}^{T-1} \gamma^k \|\mathcal{T} Q_k - \mathcal{T}^\pi Q_k\|_\infty$$

Here, $\mathcal{T}$ is the Bellman update with action maximization, and $\mathcal{T}^\pi$ is the update computed under the true policy. The first term on the right is the initial error, which gradually decays after multiplication by $\gamma^T$. The sum contains the new error introduced at each iteration. If each iteration introduces error of approximately $\epsilon_{\text{ood}}$, its total effect is amplified by a geometric series to about $\epsilon_{\text{ood}}/(1-\gamma)$. For example, when $\gamma=0.99$, the amplification factor approaches 100. The error is **added repeatedly**; its magnitude does not itself grow exponentially.

::: warning Why More Data Alone Is Insufficient
Broader data coverage can reduce the number of OOD actions, but covering every possible $a$ in a continuous action space is difficult. As long as the update maximizes over unsupported regions, extrapolation error can occur. Data coverage and conservative updates must therefore be addressed together.
:::

### 1.2 What Offline RL Must Optimize Simultaneously

The preceding diagnosis yields a formal offline-RL objective: learn a policy $\pi_\theta$ within the support of the dataset that maximizes expected return, while ensuring that $\pi_\theta$ **does not depart too far from $\pi_\beta$**, which would move it into OOD regions. Modern offline-RL algorithms balance these two objectives:

$$\max_\theta \; \mathbb{E}_{s \sim \mathcal{D}}\left[Q^\pi(s, \pi_\theta(s))\right] \quad \text{subject to} \quad D(\pi_\theta \| \pi_\beta) \leq \epsilon$$

We next examine how this constraint can be imposed in the action space or value function, then how behavior cloning can be included directly in the policy loss.

## 2. Constraining Out-of-Dataset Actions with Conservative Value Estimates

The most direct idea is to **make the Q-function pessimistic for OOD actions**. If $Q(s,a)$ assigns low values to unseen $a$, then $\max_a Q(s,a)$ will not select imagined actions. Three classic algorithms—BCQ, CQL, and IQL—implement this principle in different ways.

### 2.1 BCQ: Constraining Actions to Remain Near the Data Distribution

Batch-Constrained Q-Learning (Fujimoto et al. 2019) was the first deep algorithm shown to be stable on offline continuous-action data. Its central constraint is that **target action $a'$ must lie within the support of $\pi_\beta$**.

BCQ trains a conditional VAE $\pi_\beta(a \mid s)$ to approximate the behavior policy, samples candidate actions $\{a_i\} \sim \pi_\beta$, and maximizes only over those candidates:

$$a' = \arg\max_{a \in \{a_i + \xi \Phi(s, a_i)\}} Q_\phi(s', a)$$

Here, $\Phi(s,a)$ is a perturbation network that makes a small adjustment to a sampled action to approach a local optimum, and $\xi$ is the perturbation magnitude. This constrains the continuous-action argmax to high-density regions of the behavior policy.

### 2.2 CQL: Lowering the Value of Actions Outside the Dataset

Conservative Q-Learning (Kumar et al. 2020) approaches the problem differently. It does not constrain actions; it **directly penalizes Q-values for OOD actions**. A regularizer is added to the standard Bellman error:

$$\mathcal{L}_{\text{CQL}}(Q) = \alpha \left(\mathbb{E}_{s \sim \mathcal{D}}\left[\log \sum_a \exp(Q(s, a))\right] - \mathbb{E}_{(s, a) \sim \mathcal{D}}[Q(s, a)]\right) + \mathcal{L}_{\text{Bellman}}(Q)$$

The first term, $\log\sum_a\exp(Q(s,a))$, is logsumexp, a soft maximum over Q-values for **all actions**, including OOD actions. Reducing it requires lowering Q-values across actions. The second term restores the Q-values of state-action pairs actually observed in the dataset to their normal range. Their difference creates a penalty gap that systematically underestimates OOD actions.

CQL provides a theoretical guarantee that learned $\hat Q$ is a **lower bound** on the true $Q^\pi$: $\hat Q(s,a)\leq Q^\pi(s,a)$ for all $(s,a)$. It can further be shown that $\hat Q$ values for OOD actions are lower than those for in-distribution actions by an $\mathcal O(\alpha)$ gap. A policy derived from $\hat Q$ therefore does not overestimate any action's return. In practice, $\alpha$ is adjusted automatically through a Lagrangian so that conservatism reaches an appropriate level:

$$\mathcal{L}(\alpha) = -\alpha \cdot \left(\mathbb{E}_s\left[\log\sum_a \exp(\hat{Q}(s, a))\right] - \mathbb{E}_{(s, a) \sim \mathcal{D}}[\hat{Q}(s, a)] - \xi\right)$$

Here, $\xi$ is the target gap, such as 5.0. When the actual gap is below $\xi$, $\alpha$ increases; otherwise it decreases, stabilizing the gap near the target.

```python
class CQL(SAC):
    def critic_loss(self, batch):
        s, a, r, s_next, done = batch
        # Standard Bellman error inherited from SAC.
        with torch.no_grad():
            a_next = self.actor(s_next)
            q_target = torch.min(self.critic_target1(s_next, a_next),
                                  self.critic_target2(s_next, a_next))
            y = r + self.gamma * (1 - done) * q_target
        bellman_loss = F.mse_loss(self.critic1(s, a), y) + \
                       F.mse_loss(self.critic2(s, a), y)

        # CQL conservative regularizer.
        # First term: apply logsumexp to random (OOD) actions.
        rand_a = torch.rand_like(a) * 2 - 1
        q_rand1 = self.critic1(s, rand_a).flatten()
        q_curr1 = self.critic1(s, a).flatten()  # in-distribution
        q_next1 = self.critic1(s, a_next).flatten()
        cat_q1 = torch.cat([q_rand1, q_curr1, q_next1], dim=1)
        logsumexp_q1 = torch.logsumexp(cat_q1, dim=1).mean()

        conservative_loss = \
            self.alpha * (logsumexp_q1 - q_curr1.mean()) \
            + self.alpha * (logsumexp_q2 - q_curr2.mean())

        return bellman_loss + conservative_loss
```

### 2.3 IQL: Avoiding Explicit Evaluation of Out-of-Dataset Actions

Implicit Q-Learning (Kostrikov et al. 2022) avoids maximizing over actions outside the dataset. It learns $V(s)$ through expectile regression, biasing $V$ toward higher-value actions in the data:

$$\mathcal{L}_V = \mathbb{E}_{(s, a) \sim \mathcal{D}}\left[L_2^\tau(Q_{\bar{\theta}}(s, a) - V_\psi(s))\right]$$

First compute residual $x=Q_{\bar\theta}(s,a)-V_\psi(s)$, then apply

$$L_2^\tau(x) = |\tau - \mathbb{1}(x < 0)| \cdot x^2$$

which assigns different weights to positive and negative residuals. This is the **expectile loss**. When $\tau=0.7$, $V(s)$ lies closer to the higher $Q(s,a)$ values in the data, while training still uses only actions that appear in the dataset. After obtaining $V$, define advantage $A(s,a)=Q_{\bar\theta}(s,a)-V_\psi(s)$ and train the policy:

$$\mathcal{L}_\pi = -\mathbb{E}_{(s, a) \sim \mathcal{D}}\left[\exp(\beta \cdot A(s, a)) \cdot \log \pi_\theta(a \mid s)\right]$$

If $A(s,a)>0$, the action is better than the baseline value for that state in the data, so its exponential weight exceeds 1. If $A(s,a)<0$, its imitation weight decreases. $\beta$ controls how strongly this difference is amplified. IQL never applies $\max$ to an out-of-dataset action, avoiding this route to extrapolation error. CQL actively lowers the value of actions outside the dataset; IQL learns its Q-function, value function, and policy only from actions in the data.

### 2.4 Comparing BCQ, CQL, and IQL

| Dimension                      | BCQ                           | CQL                         | IQL                        |
| ------------------------------ | ----------------------------- | --------------------------- | -------------------------- |
| Constraint location            | Action space                  | Value function              | Implicit (expectile + AWR) |
| Evaluates OOD actions          | No (sampling constraint)      | Yes (logsumexp)             | No (avoids explicit query) |
| Additional network             | VAE $\pi_\beta$               | None                        | $V$ network                |
| Hyperparameter sensitivity     | High (perturbation magnitude) | Medium (automatic $\alpha$) | Low ($\tau,\beta$)         |
| Performance on medium datasets | Medium                        | Strong                      | Strong                     |
| Stability on sparse datasets   | Medium                        | Occasionally unstable       | Strong                     |
| Implementation complexity      | High                          | Medium                      | Low                        |

For a first implementation, IQL provides a useful baseline because its updates depend only on in-dataset actions. CQL can then be compared when explicit control over conservatism is required. BCQ is useful for understanding the approach of constraining candidate actions.

## 3. Constraining Policy Updates with Behavior Cloning

Another approach is more direct in engineering terms: **retain the on-policy or off-policy actor-critic loop and add behavior-cloning regularization directly to the policy loss**. These methods are compatible with the PPO and SAC frameworks from Chapters 8 and 9 and require only small implementation changes.

### 3.1 TD3+BC: Adding Behavior Cloning to the Policy Loss

TD3+BC, proposed by Fujimoto and Gu 2021, uses a direct implementation: add a behavior-cloning term to the TD3 actor loss and adjust weight $\lambda$ adaptively:

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}_{s \sim \mathcal{D}}\left[Q(s, \mu_\theta(s))\right] + \lambda \cdot \mathbb{E}_{(s, a) \sim \mathcal{D}}\left[(\mu_\theta(s) - a)^2\right]$$

Here, $\lambda = \frac{\alpha}{\frac{1}{N}\sum_i |Q(s_i, \mu_{\theta_{\text{old}}}(s_i))|}$. The denominator is the scale of current Q-values, so $\lambda$ adapts automatically to the reward scale of different environments without further tuning. The paper uses the same setting, $\alpha=2.5$, for every D4RL MuJoCo task.

TD3+BC's simplicity makes it a strong offline-RL baseline. Its performance highlights a counterintuitive fact: **on many offline-RL benchmarks, simple BC regularization can approach the performance of CQL and IQL**.

### 3.2 AWAC: Increasing the Imitation Weight of High-Quality Actions

Advantage-Weighted Actor-Critic (Nair et al. 2020) and IQL's policy loss share the same source—advantage-weighted regression—but AWAC uses an explicit Q-function rather than an expectile value function:

$$\mathcal{L}_\pi^{\text{AWAC}} = -\mathbb{E}_{(s, a) \sim \mathcal{D}}\left[\underbrace{\exp\left(\frac{A(s, a)}{\beta}\right)}_{\text{advantage weight}} \cdot \log \pi_\theta(a \mid s)\right]$$

Here, $A(s,a)=Q(s,a)-V(s)$, and $\beta$ is a temperature. Actions in the data that perform above average receive greater weight, while below-average actions receive less. AWAC generalizes BC into weighted BC by imitating the better parts of the dataset more strongly.

AWAC's main engineering advantage is its **smooth transition from offline to online training**: it can be pretrained entirely offline and then fine-tuned with a small amount of online interaction. This is useful in applications such as physical robotics and recommender systems.

### 3.3 How AWAC and IQL Differ

Compare the two objectives:

$$\mathcal{L}_\pi^{\text{AWAC}} = -\mathbb{E}\left[\exp\left(\frac{A(s, a)}{\beta}\right) \log \pi(a \mid s)\right], \quad \mathcal{L}_\pi^{\text{IQL}} = -\mathbb{E}\left[\exp\left(\beta \cdot A(s, a)\right) \log \pi(a \mid s)\right]$$

They are nearly identical in form. Although $\beta$ appears in a different position, it acts as a temperature in both. The difference lies in estimating $A(s,a)$:

- **AWAC**: $A=Q_\phi(s,a)-V_\psi(s)$, where $Q$ still uses a standard Bellman backup whose target retains maximization through $\pi$.
- **IQL**: $A=Q_\phi(s,a)-V_\psi(s)$, but $Q$ is backed up through $V$—the target uses $V(s')$ instead of $\max_a Q(s',a)$—and $V$ uses expectile regression to favor better actions in the data.

By changing the Bellman target to $V(s')$ and removing the maximum, IQL eliminates a source of extrapolation error. AWAC retains the standard Bellman target and constrains the policy through weighted BC. This constraint is weaker than IQL's implicit constraint, so AWAC is more likely to enter OOD regions when Q-values in the dataset are noisy.

### 3.4 Comparing AWAC, TD3+BC, and IQL

| Method | Policy-loss form                    | Requires $V$ | Supports online fine-tuning |
| ------ | ----------------------------------- | ------------ | --------------------------- |
| TD3+BC | $-\!Q + \lambda \|\mu-a\|^2$        | No           | Medium                      |
| AWAC   | $-\!w(A)\log\pi$, $w=\exp(A/\beta)$ | Yes          | Strong                      |
| IQL    | $-\!\exp(\beta A)\log\pi$ (AWR)     | Yes          | Medium                      |

AWAC and IQL have very similar policy-loss structures; the distinction is the source of $A$. AWAC uses an explicit Q–V difference, while IQL estimates it implicitly through expectile regression. This small difference can substantially affect stability on sparse data.

## Section Summary

Starting from distribution shift and extrapolation error, this section compared three approaches. BCQ restricts candidate actions to remain near the data, CQL lowers the estimated value of out-of-dataset actions, and IQL avoids explicitly maximizing over those actions. All three still use Bellman updates; they differ in how they prevent unreliable estimates from entering policy improvement.

The next section, [10.2 Offline Reinforcement Learning through Sequence Modeling](./sequence-modeling), follows a different route: it abandons Bellman updates and formulates RL as conditional sequence generation.
