# 10.2 Offline Reinforcement Learning through Sequence Modeling

[Section 10.1](./offline-data-distribution-shift) remains within the Bellman framework and estimates value while constraining the policy or value function to stay near the fixed dataset. Decision Transformer takes a different approach: it arranges states, actions, and returns into a sequence and directly learns which action to take for a specified target return.

This section proceeds in four steps. We first construct Decision Transformer using return-to-go, then explain how Trajectory Transformer searches complete trajectories, introduce Diffuser's conditional generation, and finally compare the tasks suited to each method.

## 1. Training Decision Transformer with a Target Return

The methods in [Section 10.1](./offline-data-distribution-shift) stabilize Bellman updates by constraining actions, lowering Q-values outside the dataset, or adding behavior-cloning regularization. Decision Transformer (Chen et al. 2021) does not learn a Q-function. Instead, it reformulates offline trajectories as a conditional sequence-generation problem.

### 1.1 Using Return-to-Go as a Condition

Write a trajectory as $\tau=(s_1,a_1,r_1,\ldots,s_T,a_T,r_T)$. At step $t$, the model must know both the current state and the return it should obtain from that point onward. Decision Transformer represents this target as the sum of rewards from the current time through the end of the trajectory:

$$\hat{R}_t = \sum_{t'=t}^{T} r_{t'}$$

This quantity is called the **return-to-go (RTG)**. For example, if the rewards over the next three steps are $2,1,3$, the RTGs at those positions are $6,4,3$. As time advances, rewards already obtained are subtracted from the target. Given $\hat R_t$ and $s_t$, the training task is to predict the action $a_t$ that actually appears in the data.

DT reorganizes a trajectory into a sequence of triplets:

$$\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \ldots, \hat{R}_T, s_T, a_T$$

Each time step contributes an RTG, a state, and an action in that order. Causal attention ensures that the model can access only current and preceding information when predicting $a_t$:

$$\pi_\theta(a_t \mid \hat{R}_t, s_t, a_{t-1}, \ldots) = \text{Transformer}(\hat{R}_{1:t}, s_{1:t}, a_{1:t-1})$$

The left side is the probability of selecting action $a_t$ in the current context; the right side indicates that the Transformer reads past target returns, states, and actions. Training uses no Bellman update. It only requires the predicted action to match the action in the dataset.

```python
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim, n_heads, n_layers,
                 max_ep_len=4096):
        super().__init__()
        # Three embedding layers map RTG, state, and action to hidden_dim.
        self.embed_rtg  = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(act_dim, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)
        # Positional encoding: timestep embeddings.
        self.pos_emb = nn.Embedding(max_ep_len, hidden_dim)
        # GPT backbone.
        self.transformer = GPT(
            d_model=hidden_dim, n_heads=n_heads, n_layers=n_layers,
            # Each timestep occupies three tokens; the attention mask must match.
            attn_pdrop=0.1, resid_pdrop=0.1
        )
        # Action-prediction head for continuous-action regression.
        self.action_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, rtg, states, actions, timesteps):
        B, T, _ = states.shape
        # Embed and interleave: (R1, s1, a1, R2, s2, a2, ...).
        rtg_emb   = self.embed_rtg(rtg)
        state_emb = self.embed_state(states) + self.pos_emb(timesteps)
        action_emb = self.embed_action(actions)

        # Stack into (B, 3T, H) in RTG, state, action order.
        stacked = torch.stack([rtg_emb, state_emb, action_emb], dim=1)
        stacked = stacked.permute(0, 2, 1, 3).reshape(B, 3 * T, -1)
        stacked = self.embed_ln(stacked)

        # Causal attention: each token can attend only to the past.
        h = self.transformer(stacked)
        # Use outputs at state positions to predict the corresponding actions.
        h_states = h[:, 1::3, :]  # indices 1, 4, 7, ...
        return self.action_head(h_states)  # Regress continuous actions.

    @torch.no_grad()
    def act(self, state, target_rtg, history, t):
        # At inference, use the target RTG as a prompt and generate autoregressively.
        rtg_seq = torch.cat([history.rtg, target_rtg[None]], dim=0)[-self.K:]
        s_seq   = torch.cat([history.states, state[None]], dim=0)[-self.K:]
        a_seq   = history.actions[-self.K - 1:-1]  # Shifted by one position.
        t_seq   = torch.arange(len(s_seq))
        pred_a = self.forward(rtg_seq, s_seq, a_seq, t_seq)
        return pred_a[-1]  # Prediction for the final timestep.
```

### 1.2 Predicting Actions with Supervised Learning

DT's training loss is simply MSE for continuous-action regression, or cross-entropy for discrete actions:

$$\mathcal{L} = \mathbb{E}_{\tau \sim \mathcal{D}}\left[\sum_t \|\hat{a}_t - a_t\|^2\right]$$

Continuous actions use mean squared error, while discrete actions use cross-entropy. Training follows the same procedure as an ordinary sequence model: read a segment of history and predict the next action. Data loaders, optimizers, and distributed-training components can therefore reuse the Transformer training stack.

### 1.3 Using the Target Return at Inference Time

DT deployment does not require an argmax over Q. We simply **specify a target RTG**, such as an expert score for the environment, and DT generates actions autoregressively so that cumulative return approaches the target:

```python
target_return = 9000  # HalfCheetah expert level
state = env.reset()
history = TrajectoryBuffer()
for t in range(max_steps):
    action = model.act(state, target_return, history, t)
    next_state, reward, done, _ = env.step(action)
    history.append(state, action, reward)
    state = next_state
    # Crucially, subtract the observed reward from RTG to obtain the remaining target.
    target_return -= reward
```

RTG serves as a control condition during inference. After each step, the observed reward is subtracted from the remaining target, so the next input represents the return still required. A target set too high may place the model in a conditioning range not covered by the training data, so the dataset's return range should guide the choice.

### 1.4 Why Decision Transformer Can Work

Decision Transformer works when the data contain a stable relationship among states, actions, and final returns. If the dataset contains trajectories of varying quality, RTG helps the model distinguish actions that frequently occur in high-return trajectories.

- The dataset contains expert trajectories with high RTG, medium trajectories with intermediate RTG, and random trajectories with low RTG.
- Given a high target RTG, the learned conditional distribution $p(a \mid \hat{R}_{\text{high}}, s)$ naturally favors high-return actions.
- The model can be understood as imitating trajectories that achieved a similar RTG from similar states.

Formally, the policy learned by DT can be written as

$$\pi_\theta(a \mid s, \hat{R}) \propto \exp\left(-\frac{1}{2\sigma^2}\|a - f_\theta(s, \hat{R})\|^2\right)$$

where $f_\theta$ is the Transformer's regression output. As $\sigma \to 0$, this reduces to the deterministic policy $a = f_\theta(s, \hat{R})$. Its relationship to $\pi_\beta$ is

$$\pi_\theta(a \mid s, \hat{R}) \approx \pi_\beta(a \mid s, \text{return} \approx \hat{R})$$

Thus, DT learns the behavior policy's conditional distribution at a specified return. State-action combinations absent from the data remain difficult to predict reliably, and the ability to combine trajectory segments varies across tasks and datasets.

This observation motivated extensive subsequent work, including RL via supervised learning in online RL, in-context RL through Algorithm Distillation, Star-Vector, and Eyre et al.'s “language modeling is all you need for RL.”

### 1.5 Limitations of Decision Transformer

1. **It can learn only optimal behavior present in the data**—if the dataset contains no expert trajectories, even a very high target RTG cannot produce expert behavior.
2. **Weak stitching ability**—traditional offline RL can combine useful segments of two suboptimal trajectories into a better policy, a process called subtrajectory stitching. As a purely supervised method, DT has limited compositional generalization of this kind.
3. **Sensitivity to RTG selection**—a target set too high can produce incoherent actions, while one set too low yields conservative behavior.

## 2. Searching Trajectories with Trajectory Transformer

After DT, the “RL as sequence modeling” approach quickly developed into several variants. Two representative methods are Trajectory Transformer, which models a complete trajectory as a token sequence and performs beam-search inference, and Diffuser, which generates a complete trajectory directly with a diffusion model.

### 2.1 Discretizing Trajectories and Using Beam Search

Janner et al. 2021 discretize RTG, state, action, and reward into tokens, then train a standard Transformer to predict the next token:

$$p_\theta(\tau) = \prod_{t=1}^{T} p_\theta(s_t, a_t, r_t \mid s_{<t}, a_{<t}, r_{<t})$$

At inference, beam search maximizes trajectory probability, optionally subject to a reward constraint. TT has three main properties:

- Discretizing continuous quantities avoids regression, but the number of tokens grows rapidly because every state dimension must be discretized.
- Beam-search inference is slow because it expands multiple candidate trajectories.
- It supports **planning**: future reward constraints can be injected explicitly during search, producing a form of implicit model-based RL.

## 3. Generating Complete Trajectories with Diffuser

Janner et al. 2022 treat a complete trajectory as the object generated by a diffusion model. If the state dimension is $d_s$, the action dimension is $d_a$, and the trajectory length is $T$, the trajectory is a matrix of shape $T\times(d_s+d_a)$. During training, noise is added to real trajectories, and the network predicts that noise:

$$\min_\theta \; \mathbb{E}_{\tau, t, \epsilon}\left[\|\epsilon - \epsilon_\theta(\tau_t, t)\|^2\right]$$

Here, $\tau_t$ is the noisy trajectory, $t$ is the diffusion step, $\epsilon$ is the noise actually added, and $\epsilon_\theta$ is the network's prediction. Their mean squared error decreases as the prediction improves. At inference, the process starts from random noise and repeatedly subtracts predicted noise until it produces a complete trajectory.

Diffuser uses classifier-free guidance to control generation. During training, it randomly drops state or reward conditions so that the model learns both conditional and unconditional distributions:

$$\tilde{\epsilon}_\theta = (1 + w) \cdot \epsilon_\theta(\tau_t, t, c) - w \cdot \epsilon_\theta(\tau_t, t)$$

Here, $c$ is a condition such as maximizing future reward, and $w$ controls conditioning strength. At inference, the reward condition changes the denoising direction so that high-reward trajectories receive greater generation probability. Optimization therefore changes from explicitly selecting the maximum-value action to sampling from a reward-guided trajectory distribution.

## 4. Comparing the Three Sequence-Modeling Methods

| Dimension                              | Decision Transformer         | Trajectory Transformer                        | Diffuser                                 |
| -------------------------------------- | ---------------------------- | --------------------------------------------- | ---------------------------------------- |
| Modeling target                        | Conditional policy given RTG | Joint distribution over complete trajectories | Diffusion model of complete trajectories |
| Discretization                         | No                           | Yes (each state dimension)                    | No                                       |
| Inference                              | Autoregressive sampling      | Beam search                                   | Iterative denoising                      |
| Planning ability                       | Weak (implicit)              | Strong (explicit)                             | Strong (conditional generation)          |
| Stitching ability                      | Weak                         | Medium                                        | Strong                                   |
| Inference speed                        | Fast                         | Slow                                          | Medium (dozens of denoising steps)       |
| Compatibility with LLM training stacks | Strong (most GPT-like)       | Strong                                        | Weak (different architecture)            |

## Section Summary

Decision Transformer generates actions autoregressively for a specified return-to-go and can therefore reuse supervised Transformer training directly. Trajectory Transformer searches complete trajectories with beam search, while Diffuser generates trajectories through iterative denoising. All three depend on behavioral coverage already present in offline data; they differ in how they represent trajectories and select actions.

The next section, [10.3 Offline Reinforcement Learning and Preference Data](./experiments), applies the fixed-data perspective to preference optimization and compares DPO, preference data, and sequence modeling.
