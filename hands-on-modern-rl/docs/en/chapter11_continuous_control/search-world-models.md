# 9.4 Search and World Models

> [9.3](./model-based) presented the data-augmentation branch of model-based RL: Dyna, PETS, and MBPO use models to generate data that accelerates model-free training. This section presents another major branch of model-based RL: **explicit search with neural-network evaluation**. From AlphaGo (2016) and AlphaZero (2017), through MuZero (2019), to Dreamer V3 (2023), this line of work represents the frontier of model-based RL and directly inspired Process Reward Model search in the era of LLMs.

## AlphaZero and the Full Realization of Search with Learning

The progression AlphaGo (2016) → AlphaGo Zero (2017) → AlphaZero (2017) → MuZero (2019) represents another philosophy of model-based RL: **explicit search with neural-network evaluation**.

### The Core AlphaZero Loop

```python
def alphazero_search(state, neural_net, n_simulations=800):
    root = MCTSNode(state)
    for _ in range(n_simulations):
        # 1. Selection: select the best child using PUCT
        node = root
        while not node.is_leaf():
            node = node.select_child()

        # 2. Expansion: evaluate the leaf with the neural network
        policy, value = neural_net(node.state)
        node.expand(policy)

        # 3. Backup: propagate value back to the root
        node.backup(value)

    # Return root visit counts as an action-probability distribution
    return root.compute_action_distribution()
```

AlphaZero combines Monte Carlo Tree Search (MCTS) with neural networks:

- **Policy network** $p_\theta(a \mid s)$: reduces the search width by focusing on promising actions
- **Value network** $v_\theta(s)$: reduces the search depth by evaluating leaves directly instead of searching to terminal states

### The PUCT Formula

AlphaZero selects child nodes with PUCT (Predictor + UCB):

$$\text{PUCT}(a) = Q(s, a) + c_{\text{prior}} \cdot p_\theta(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

- $Q(s, a)$: the current value estimate for action $a$
- $p_\theta(a \mid s)$: the policy network prior
- $\sqrt{N(s)} / (1 + N(s, a))$: an exploration bonus derived from UCB

The first term exploits current knowledge, the prior in the second term narrows the search, and the exploration factor ensures that every action is tried.

### Self-Play Training

The two networks are trained through **self-play**:

1. Play one game against itself using the current network and MCTS
2. Use the search result as a better policy target: the action distribution returned by MCTS is an improved policy
3. Use the outcome as a better value target: a win gives +1 and a loss gives -1
4. Train the network under supervision from these targets

```python
def self_play_training(network, n_games=10000):
    for game in range(n_games):
        # 1. Self-play
        trajectory = []
        state = initial_state()
        while not state.is_terminal():
            policy = alphazero_search(state, network)
            action = sample_from(policy)
            trajectory.append((state, policy, action))
            state = state.next(action)

        # 2. Label outcomes
        winner = state.winner()
        for s, p, a in trajectory:
            value = +1 if winner == s.current_player else -1
            train_network(s, p, value)
```

**No human game records are required**—starting from scratch, AlphaZero defeated Stockfish after four hours and surpassed every human Go program after 72 hours.

## MuZero and Implicit Model Learning

AlphaZero requires the game rules, including state transitions and legal actions. MuZero's key innovation (Schrittwieser et al., 2019) is to **learn an implicit model** that maps state $s$ to a hidden representation $h(s)$ and performs planning and value estimation in the hidden space.

$$s \xrightarrow{h} x_0 \xrightarrow{g} x_1 \xrightarrow{g} x_2 \to \ldots$$

### MuZero's Three Networks

- **Representation network** $h(s) \to x$: encodes the real state into a hidden space
- **Dynamics network** $g(x, a) \to x', r$: predicts the next hidden state and reward
- **Prediction network** $f(x) \to p, v$: predicts a policy and value from the hidden state

```python
class MuZero:
    def plan(self, state, n_simulations):
        # 1. Encode the real state into the hidden space
        root_hidden = self.representation(state)
        root_policy, root_value = self.prediction(root_hidden)

        # 2. Run MCTS in the hidden space
        for _ in range(n_simulations):
            self._mcts_iteration(root_hidden)

        # 3. Return the action distribution at the root
        return root.action_distribution()

    def _mcts_iteration(self, root):
        # Select, expand, and back up in the hidden space
        path = self._select_path(root)
        next_hidden, reward = self.dynamics(path[-1].hidden, path[-1].action)
        policy, value = self.prediction(next_hidden)
        path[-1].expand(policy, reward)
        for node in path:
            node.update(value, reward)
```

### The Significance of MuZero

MuZero can learn without knowing the game rules—it **learns the rules itself**. This allows it to extend to

- **Atari**, learning directly from pixels without a simulator
- **Board games**, including Go, chess, and shogi
- **Poker**, with partial observability
- **Any MDP**

MuZero is a unified model-based RL architecture: the same algorithm and network structure span visual and vector inputs as well as discrete and continuous actions.

## Dreamer V3 and a New Generation of World Models

The Dreamer series (Hafner et al., 2020–2023) is a modern flagship of model-based RL. Its central idea is to **learn a recurrent latent-variable world model** and train an actor-critic by "dreaming" within that model.

### The Recurrent State-Space Model

Dreamer uses a **Recurrent State-Space Model (RSSM)** to represent both

- **Deterministic trajectories**, through the RNN hidden state $h_t$
- **A stochastic posterior**, in which an encoder infers $z_t$ from observations
- **A stochastic prior**, which predicts $\hat{z}_t$ from $h_t$

During training, $\hat{z}_t$ is made to match $z_t$, allowing the model to imagine trajectories consistent with the real environment.

```python
class RSSM:
    def forward(self, obs_seq, action_seq):
        h = zeros(batch, hidden_dim)
        posterior_zs = []
        prior_zs = []

        for t in range(T):
            # Prior: predict z_t from h_t
            prior_mean, prior_std = self.prior(h)
            prior_zs.append((prior_mean, prior_std))

            # Posterior: infer z_t from h_t and obs_t
            posterior_mean, posterior_std = self.posterior(h, encoder(obs_seq[t]))
            z = reparameterize(posterior_mean, posterior_std)
            posterior_zs.append((posterior_mean, posterior_std))

            # Update the RNN hidden state
            h = self.rnn(h, z, action_seq[t])

        return prior_zs, posterior_zs
```

### Actor-Critic in Imagination

The actor is trained with model rollouts rather than real data:

```python
# "Dream" within the world model
h = world_model.encode(real_observation_sequence)
for t in range(H):  # H = 15-step imagination horizon
    a = actor(h)
    h, r = world_model.predict(h, a)
    imagined_trajectory.append((h, a, r))

# Train the actor-critic on imagined trajectories
for (h, a, r) in imagined_trajectory:
    critic_loss = ...
    actor_loss = ...
```

### The Generality of Dreamer V3

The key contribution of Dreamer V3 (Hafner et al., 2023) is a **single hyperparameter configuration** that works across more than 150 tasks, including

- Atari, with discrete actions and visual input
- MuJoCo, with continuous actions and vector input
- Crafter, an open-world survival environment
- DMLab, for first-person 3D navigation
- BSuite, for cognitive tasks

Without task-specific tuning, Dreamer V3 **outperforms model-free state-of-the-art methods** on most benchmarks. This was the first time that model-based RL surpassed methods such as SAC and PPO in generality.

### Three Key Engineering Innovations

1. **Discretized latent variables**: replacing the Gaussian distribution for $z$ with a categorical distribution stabilizes training
2. **Symlog loss**: $\text{symlog}(x) = \text{sign}(x) \log(|x| + 1)$ compresses the value-function range and adapts to different reward scales
3. **No KL annealing**: the method directly maximizes the ELBO, making the posterior match the prior

These three changes allow Dreamer V3 to work across more than 150 tasks with the same configuration.

## Model-Based vs. Model-Free: When to Use Each

| Dimension                      | Model-free               | Model-based                                 |
| ------------------------------ | ------------------------ | ------------------------------------------- |
| **Sample efficiency**          | Low (millions of steps)  | High (tens of thousands of steps)           |
| **Asymptotic performance**     | High                     | Limited by model error                      |
| **Computational cost**         | Low (uses data directly) | High (model training plus search/planning)  |
| **Interpretability**           | Black box                | The model can be analyzed                   |
| **Transfer capability**        | Weak                     | The model can transfer to downstream tasks  |
| **Hyperparameter sensitivity** | Moderate                 | High (model quality determines performance) |

**Choose model-free methods when:**

- The simulator is inexpensive, as in Atari, MuJoCo, or StarCraft
- Final performance matters and the number of samples is unrestricted
- Deployment should avoid the inference cost of a model

**Choose model-based methods when:**

- Sampling the real environment is expensive, as in robotics, autonomous driving, or chemical reactions
- Rapid adaptation is required, as in meta-RL or online learning
- Interpretability is required in safety-critical settings

## Connections to RL for LLMs

In LLM training:

- **Model-free**: RLHF and GRPO train directly from reward-model scores
- **Model-based**: Process Reward Models and verifier models act as a form of environment model; PRM-guided search ([Chapter 17: PRMs and Search](../chapter20_prm_search/inference-time-search)) is analogous to AlphaZero
- **World model**: a Code World Model ([Chapter 20: SWE-Agent](../chapter23_rl_based_swe/world-model-and-deep-swe)) predicts the outcomes of code execution and serves as the LLM-era analogue of MuZero

The tradeoff between model-based and model-free methods explains why Tongyi DeepResearch uses PRM-guided search and why SWE-Agent uses a Code World Model to improve sample efficiency.

## Chapter Summary

Continuous control and model-based RL are two major advanced directions in classical deep RL:

1. **DDPG → TD3 → SAC** traces the development of deterministic policy gradients, from exploration through added noise, to stabilization with twin Q-networks and delayed updates, and finally to automatic exploration through maximum entropy
2. **Dyna → PETS → MBPO** traces the model-based data-augmentation approach, in which the model acts as a data generator
3. **AlphaZero → MuZero → Dreamer V3** traces the leading approach based on explicit search and learned models, representing the frontier of model-based RL

The next chapter, [Chapter 10: Offline Reinforcement Learning](../chapter12_offline_rl/offline-data-distribution-shift), turns to another question: **What can an agent do when it cannot interact with the environment and has access only to historical data?** This is a central problem in practical settings such as LLM post-training and recommender systems.

## Further Reading

- [Silver et al. 2018 "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)
- [Schrittwieser et al. 2020 "Mastering Atari, Go, chess and shogi by planning with a learned model" (MuZero)](https://arxiv.org/abs/1911.08265)
- [Hafner et al. 2023 "Mastering Diverse Domains through World Models" (Dreamer V3)](https://arxiv.org/abs/2301.04104)
- [Janner et al. 2019 "When to Trust Your Model: Model-Based Policy Optimization" (MBPO)](https://arxiv.org/abs/1906.08253)
