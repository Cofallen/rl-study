# 9.3 Model-Based RL

## What This Section Covers

[9.2](./td3-sac) brought model-free continuous-control algorithms to a stable, practical level. SAC and TD3 can learn good MuJoCo policies after roughly one million training steps. In a physical robot, each sample consumes time, battery capacity, and mechanical lifetime, while a collision can damage the system. A million real steps may require weeks or months.

**Model-based RL** addresses this cost by learning an environment model,

$$
\hat P(s'\mid s,a),\qquad \hat R(s,a),
$$

and then planning or training largely inside that model. This section develops three routes: Dyna uses the model as a data generator, PETS quantifies uncertainty with probabilistic ensembles, and MBPO limits error with short rollouts.

## The Fundamental Difference Between Model-Based and Model-Free RL

All preceding algorithms—DDPG, TD3, and SAC—are **model-free**: the agent does not attempt to understand the environment and learns a policy only from the rewards supplied by the environment. **Model-based RL** takes the opposite approach. It first learns an environment model $\hat{P}(s' \mid s, a), \hat{R}(s, a)$ and then uses the model to plan or generate data.

### Why Use a Model?

The main reason is **sample efficiency**. If one real interaction takes one second, one million steps require about 11.5 days of uninterrupted operation, whereas 100,000 steps require about one day. Model-based methods can reuse a learned model to generate inexpensive imagined experience, reducing real interaction by roughly 10–100 times in the representative results discussed below.

### Overview of Three Major Paradigms

| Paradigm | Central idea                                    | Representative algorithm | Applicable setting                                     |
| -------- | ----------------------------------------------- | ------------------------ | ------------------------------------------------------ |
| **Dyna** | Use the model for data augmentation             | Dyna-Q                   | Discrete actions and rapid training                    |
| **PETS** | Probabilistic ensembles and trajectory sampling | PETS                     | High-precision control where model uncertainty matters |
| **MBPO** | Short-horizon rollouts                          | MBPO                     | General continuous control                             |

We now examine each paradigm in turn.

## The Model as Data Augmentation

Dyna is Sutton's classic 1990 approach. It divides each real interaction into four steps: the third trains the model, and the fourth uses the model to generate "synthetic" data that accelerates model-free training.

```python
for step in range(total_steps):
    # 1. Interact with the real environment
    a = policy.select(s)
    s_prime, r = env.step(a)
    replay_buffer.add(s, a, r, s_prime)

    # 2. Update a model-free algorithm such as Q-Learning with real data
    q_learning_update(replay_buffer.sample())

    # 3. Train the environment model with real data
    model.train(s, a, r, s_prime)

    # 4. Generate synthetic data with the model and perform N more Q-Learning updates
    for _ in range(N):  # N = 10–100
        s_sim, a_sim = replay_buffer.sample_state_action()
        s_sim_next, r_sim = model.predict(s_sim, a_sim)
        q_learning_update(s_sim, a_sim, r_sim, s_sim_next)
```

Dyna treats the model as an additional data generator. After every real interaction, it performs $N$ simulated updates, improving **sample efficiency by approximately a factor of $N$**.

### A Key Limitation of Dyna

Dyna works well in a small discrete world, but repeatedly feeding predictions back into a learned model accumulates error in continuous dynamics. Suppose the model's one-step error is at most $\epsilon$, and the true dynamics are Lipschitz in the state with constant $L$. If

$$
e_t=\lVert s_t^{\text{predicted}}-s_t^{\text{true}}\rVert,
$$

then

$$
e_{t+1}\leq L e_t+\epsilon,
\qquad
e_T\leq \epsilon\sum_{i=0}^{T-1}L^i.
$$

The three regimes are different. With $L=0.9$, the bound converges to $10\epsilon$. With $L=1$, it grows linearly to $T\epsilon$. With $L=1.1$ and $T=50$, the geometric sum is about $1174\epsilon$; a one-step error of 0.01 can therefore grow beyond 11. This is why long imagined trajectories can become unrelated to the real system.

PETS and MBPO respond differently. PETS explicitly represents uncertainty and plans conservatively across plausible models. MBPO avoids long predictions and resets each short rollout to a state taken from real data.

## Probabilistic Ensembles with Trajectory Sampling

The key observation behind Probabilistic Ensembles with Trajectory Sampling (Chua et al., 2018) is that the model itself has **two kinds of uncertainty**:

- **Epistemic uncertainty**: uncertainty in the model caused by limited training data, represented by an **ensemble** $M_1, \ldots, M_K$
- **Aleatoric uncertainty**: randomness inherent in the environment, such as a die roll, represented by a **probabilistic output** $p(s' \mid s, a)$

Epistemic uncertainty can shrink when the data set covers the relevant state-action region. Aleatoric uncertainty remains even with unlimited data because it belongs to the environment itself. PETS commonly uses an ensemble of five probabilistic networks: disagreement across networks represents the first kind, while each network's predicted variance represents the second.

### Model Architecture

PETS uses an ensemble of $K$ probabilistic neural networks:

```python
class PEtsModel:
    def __init__(self, n_models=5):
        self.models = [ProbabilisticNN() for _ in range(n_models)]

    def predict(self, s, a):
        # Each model outputs (mean, variance)
        means, vars = [], []
        for m in self.models:
            mu, sigma = m(s, a)
            means.append(mu); vars.append(sigma)
        return means, vars  # Ensemble disagreement = epistemic uncertainty
```

Planning uses samples from the ensemble rather than a single model, making the policy robust to the possibility that the model is inaccurate.

### Trajectory Sampling Strategy

PETS plans with the **Cross-Entropy Method (CEM)**. At every step, it samples and selects among candidate action sequences $\{a_1, \ldots, a_H\}$:

```python
def cem_planning(model, s, horizon=10, n_samples=500, n_iters=5):
    # Initialize the action distribution
    action_mean = zeros(horizon, action_dim)
    action_var = ones(horizon, action_dim)

    for it in range(n_iters):
        # 1. Sample N action sequences
        action_seqs = sample_normal(action_mean, action_var, n_samples)

        # 2. Roll out each sequence with a randomly selected ensemble model
        rewards = []
        for seq in action_seqs:
            model_id = random_int(0, K)
            s_pred = s
            total_r = 0
            for a in seq:
                s_pred, r = model[model_id].predict(s_pred, a)
                total_r += r
            rewards.append(total_r)

        # 3. Select the top 20% of sequences and update the distribution
        elite = top_k_indices(rewards, k=0.2 * n_samples)
        action_mean = action_seqs[elite].mean(0)
        action_var = action_seqs[elite].var(0)

    return action_mean[0]  # Execute only the first action, following MPC
```

Each sampled trajectory keeps one randomly selected ensemble member for the whole rollout. This TS1 choice preserves distinct plausible dynamics; changing models at every step would instead approximate an artificial average environment.

CEM begins with a broad Gaussian over action sequences, evaluates 500 candidates, fits the next distribution to the best 20%, and repeats for five iterations. It then executes only the first action and replans from the next observed state, following model predictive control.

### Experimental Results for PETS

PETS was the first model-based method to match model-free performance on MuJoCo while using **10–50 times fewer samples**. Its cost is expensive planning. With five CEM iterations, 500 candidates, and a horizon of ten, selecting one action requires about $5\times500\times10=25{,}000$ model predictions.

## Model-Based Policy Optimization

The central innovation of Model-Based Policy Optimization (Janner et al., 2019) is to **generate finite-length rollouts with the model**, such as five steps, before returning to the real environment. This prevents model error from growing without bound as rollout length increases.

### Short-Horizon Rollouts

The key MBPO parameter is the rollout length $k$. The practical observation is that most of the sample-efficiency benefit appears with one to five model steps, before compounded model bias dominates.

For intuition, take $L=1.01$ and $\epsilon=0.01$. The error bound is about 0.01 after one step, 0.05 after five, 0.22 after twenty, and 2.7 after one hundred. MBPO uses the early, relatively reliable part of the curve. Every rollout starts from a state sampled from real experience, advances only a few model steps, and then stops.

```python
# Short-horizon rollouts keep model error under control
for rollout_step in range(K_short):  # K_short = 5
    a = policy(s_sim)
    s_sim, r = model.predict(s_sim, a)
    replay_buffer.add(s_sim, a, r, s_sim)
    # Reset to a real state every five steps
    if rollout_step % K_short == 0:
        s_sim = real_env.state
```

### MBPO Training Process

```
┌──────────────────────────────────────────────────────┐
│ 1. Train model M with real data                      │
│    M.predict(s, a) → s', r                           │
├──────────────────────────────────────────────────────┤
│ 2. Generate short rollouts (5 steps) with M          │
│    Start: a state s from the real data               │
│    Each step: a = policy(s), s' = M(s, a)            │
│    Add the five (s, a, r, s') tuples to replay       │
├──────────────────────────────────────────────────────┤
│ 3. Update SAC on a replay buffer mixing real and     │
│    synthetic data                                    │
└──────────────────────────────────────────────────────┘
```

MBPO matches the performance of model-free SAC on MuJoCo while using **10–100 times fewer samples**.

The rollout length can grow with training. Early in training, when the learned dynamics are inaccurate, $k$ may remain at one. As prediction improves, it can increase toward five. The final policy is still an ordinary SAC policy, so deployment does not require PETS-style CEM planning.

### Comparing Three Model-Based RL Algorithms

| Algorithm | Model type             | Planning method         | Sample efficiency | Computational cost |
| --------- | ---------------------- | ----------------------- | ----------------- | ------------------ |
| Dyna      | Deterministic          | One-step synthetic data | ~10×              | Low                |
| PETS      | Probabilistic ensemble | CEM MPC                 | ~50×              | High               |
| MBPO      | Deterministic          | Short rollouts          | ~100×             | Moderate           |

Practical choices:

- **Rapid experiments**: Dyna, which is simple and stable
- **High-precision control**: PETS, for robotic manipulation and precision manufacturing
- **General continuous control**: MBPO, across the full MuJoCo suite

## Section Summary

Model-based RL improves sample efficiency by **learning an environment model**:

1. **Dyna** uses the model for data augmentation and performs N simulated updates after every real interaction
2. **PETS** represents model uncertainty with probabilistic ensembles and maintains robustness through CEM planning
3. **MBPO** uses short-horizon rollouts to limit error accumulation, matching SAC's performance with 100 times fewer samples

The next section, [9.4 Search and World Models](./search-world-models), turns to another branch of model-based methods: explicit search with neural-network evaluation, tracing its development from AlphaGo to Dreamer V3.
