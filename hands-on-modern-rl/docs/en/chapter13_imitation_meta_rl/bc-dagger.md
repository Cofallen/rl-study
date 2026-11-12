# 11.1 Behavior Cloning and Interactive Imitation Learning

[Chapter 10: Offline Reinforcement Learning](../chapter12_offline_rl/offline-data-distribution-shift) improves a policy using fixed historical data, but the data still retain rewards. Imitation learning operates with less information: the data only tell us which action an expert took in a given state, without a ready-made reward function explaining why that action was good.

Chapter 11 proceeds in three steps. We first learn expert actions directly through behavior cloning and DAgger, then infer rewards from demonstrations with inverse reinforcement learning and GAIL, and finally study how policies adapt rapidly to new tasks through MAML, RL², PEARL, and in-context reinforcement learning. This section begins with four questions: how behavior cloning is trained, why its errors compound, how DAgger collects error states, and what data each approach requires.

## 1. Formulating Expert Demonstrations as Supervised Learning

[Chapter 6: Policy Gradients](../chapter08_policy_gradient/reinforce) assumes that the environment provides rewards. In many real tasks, however, we have only **expert demonstrations**—trajectories from human drivers, operation logs from skilled workers, or high-quality question-answer pairs. **Imitation learning** learns a policy directly from demonstrations, bypassing reward-function design.

### 1.1 The Behavior-Cloning Objective

The most direct method treats expert data as supervised examples: the state $s$ is the input, and the expert action $a$ is the label. The higher the probability that the policy assigns to the expert action, the lower the loss:

$$\mathcal{L}_{BC}(\theta) = -\mathbb{E}_{(s, a) \sim \mathcal{D}_{\text{expert}}}\left[\log \pi_\theta(a \mid s)\right]$$

Here, $\mathcal{D}_{\text{expert}}=\{(s_i,a_i)\}_{i=1}^N$ is the expert-demonstration dataset, and $\pi_\theta(a\mid s)$ is the probability that the policy selects expert action $a$ in state $s$. The negative sign turns “increasing the probability of the expert action” into a minimization problem. Discrete actions usually use cross-entropy, while continuous actions can use mean squared error or the negative log-likelihood of a probability distribution. Supervised fine-tuning of an LLM uses the same conditional-likelihood objective, except that the action is the next token.

Consider one numerical example. An expert demonstration contains a state in which a car is centered in its lane and the expert action is a slight turn to the left. Two policies assign different probabilities to that action:

- Policy 1 gives $\pi_1(a\mid s)=0.7$, so its loss is $-\log 0.7\approx0.36$.
- Policy 2 gives $\pi_2(a\mid s)=0.95$, so its loss is $-\log 0.95\approx0.05$.

Policy 2 agrees more strongly with the expert and therefore has the smaller loss. Behavior cloning minimizes this quantity over every demonstration, using the same cross-entropy mechanism as handwritten-digit classification.

```python
def behavior_cloning_step(policy_net, expert_batch):
    states, actions = expert_batch
    log_probs = policy_net.log_prob(states, actions)
    loss = -log_probs.mean()  # Negative log-likelihood
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

## 2. Why Behavior-Cloning Errors Compound

During training, BC sees the state distribution visited by the expert, $d_{\text{expert}}(s)$. During deployment, it visits the distribution induced by the current policy, $d_{\pi_\theta}(s)$. A small error can take the agent to a state absent from the training set, making subsequent errors more likely.

Suppose the policy has an error rate of only $\epsilon=0.01$ on states visited by the expert. It agrees with the expert 99% of the time, but the probability of completing an entire $T$-step task without an error is $(1-\epsilon)^T$:

| Task length $T$         | 10 steps | 50 steps | 100 steps | 1,000 steps |
| ----------------------- | -------- | -------- | --------- | ----------- |
| Probability of no error | 0.90     | 0.61     | 0.37      | 0.00004     |

In a 100-step task, two runs out of three contain at least one error. This is still optimistic because it assumes the error rate remains 1% after the policy leaves the expert trajectory. Ross et al. (2011) express the resulting cumulative task cost as order $O(T^2\epsilon)$:

$$\mathbb{E}\left[\sum_{t=0}^T \mathbb{1}[\pi_\theta(s_t) \neq \pi^*(s_t)]\right] \leq O(T^2 \epsilon)$$

Here, $T$ is the task horizon and $\epsilon$ is the supervised-learning error. The factor $T^2$ indicates that an early error affects both the current step and the states encountered during many later steps. The longer the task, the greater the cost of training only on expert states.

## 3. Using DAgger to Collect States the Policy Actually Visits

Dataset Aggregation directly supplements the states that the policy will visit but the expert dataset does not cover. The current policy first performs the task, and the expert then supplies the correct actions for those states.

```python
def dagger(env, expert, policy_net, n_iterations=20, n_traj_per_iter=50):
    dataset = []
    for it in range(n_iterations):
        # 1. Roll out the current policy (not the expert).
        trajectories = []
        for _ in range(n_traj_per_iter):
            s = env.reset()
            traj = []
            done = False
            while not done:
                # beta mixture: favor the expert early for safety, then the policy later
                beta = max(0.0, 1.0 - it / 10)
                if np.random.rand() < beta:
                    a = expert(s)
                else:
                    a = policy_net.act(s)
                s_next, r, done, _ = env.step(a)
                traj.append((s, a))
                s = s_next
            trajectories.append(traj)

        # 2. Crucially, ask the expert to relabel all policy-visited states,
        #    including failure states.
        for traj in trajectories:
            for s, _ in traj:
                a_expert = expert(s)
                dataset.append((s, a_expert))

        # 3. Retrain the policy on the expanded dataset.
        train_bc(policy_net, dataset)
```

Following the iterations makes the distribution change concrete. During the first rounds, $\beta$ is close to 1, so the expert performs most actions and the data mainly contain ordinary states such as a centered car at a suitable speed. Later, $\beta$ falls toward one half and the imperfect policy begins producing states such as crossing a lane marking or driving too fast. The expert labels these previously unseen states, filling precisely the gaps in which the policy is likely to fail. Once $\beta$ reaches zero, every new failure state produced by independent policy execution can be labeled and learned in the next round. The growing dataset therefore approaches the policy's own state distribution $d_{\pi_\theta}$.

Under conditions such as no-regret online learning, the cumulative cost can improve from BC's $O(T^2\epsilon)$ to order $O(T\epsilon)$. The cost is that the expert must be queried repeatedly during training.

## 4. Comparing BC, DAgger, and GAIL

| Method | Source of training data             | Addresses distribution shift | Requires online expert labels |
| ------ | ----------------------------------- | ---------------------------- | ----------------------------- |
| BC     | Offline expert data only            | ❌                           | ❌                            |
| DAgger | Expert data + policy-visited states | ✅                           | ✅ (key limitation)           |
| GAIL   | Expert data + policy rollouts       | ✅ (implicitly)              | ❌ (only state-action pairs)  |

DAgger's engineering bottleneck is its **requirement for online expert interaction**. A human driver, for example, cannot easily label the correct action in real time for every unusual state visited by a policy. This limitation motivates the inverse-RL approach in the next section, which infers rewards from demonstrations.

## Section Summary

Behavior cloning (BC) is the simplest form of imitation learning: it treats expert trajectories as supervised data for policy training. Its central difficulty is **distribution shift**: training covers only the expert state distribution, and once the deployed policy deviates, it may not recover. DAgger addresses this problem by having an expert correct the agent's actual trajectories.

The next section, [11.2 Inverse Reinforcement Learning and GAIL](./irl-gail), no longer imitates actions directly. Instead, it **infers the reward function from expert behavior**—the defining idea of inverse reinforcement learning (IRL).
