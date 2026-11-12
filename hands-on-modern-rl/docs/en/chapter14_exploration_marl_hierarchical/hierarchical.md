# 12.3 Hierarchical Reinforcement Learning and World Models

[Section 12.2](./marl) coordinates multiple agents through centralized training. This section returns to a single agent but extends the task horizon: a robot may need thousands of actions to clean an entire house, making it difficult for the final reward to guide each earlier action.

We first explain how hierarchy shortens the decision horizon, then compare Options, FeUdal Networks, and HIRO, treat a generative world model as a learnable environment, and finally discuss how exploration, multi-agent, and hierarchical methods can be recombined in new environments.

## 1. Shortening the Horizon of Long Tasks with Hierarchy

Long-horizon tasks separate final rewards from early actions. In _Atari Montezuma's Revenge_, for example, the agent must first obtain a key, then open a door, and finally enter the next room. Under direct PPO training, the final reward is difficult to assign accurately to earlier actions. **Hierarchical RL** decomposes decisions into two or more levels:

- **High-level policy**: acts occasionally and outputs a subgoal or option.
- **Low-level policy**: executes primitive actions under the high-level subgoal until that subgoal is completed.

The high level therefore considers only a sparse sequence of subgoals. It divides a long horizon into shorter horizons, within which gradient signals can propagate.

## 2. Comparing Three Hierarchical Reinforcement-Learning Methods

### 2.1 Options: Encapsulating a Policy Segment as a Temporally Extended Action

The **options** framework of Sutton, Precup, and Singh 1999 is the formal foundation of hierarchical RL. An option $\omega = (\mathcal{I}_\omega, \pi_\omega, \beta_\omega)$ consists of three components:

- **initiation set** $\mathcal{I}_\omega$: the set of states in which the option may start.
- **intra-option policy** $\pi_\omega$: the policy followed while the option executes.
- **termination function** $\beta_\omega(s)$: the probability that the option terminates upon reaching $s$.

An option may execute for $T$ consecutive steps, so a high-level update cannot use only a one-step reward. It must sum the rewards obtained while the option executes and then include the value of the termination state:

$$Q^\mu(s, \omega) = \mathbb{E}\left[\sum_{t=0}^{T-1}\gamma^t r_t + \gamma^T \max_{\omega'} Q^\mu(s_T, \omega')\right]$$

Here, $s$ is the state in which the option starts, $\omega$ is the current option, $T$ is its duration, and $s_T$ is the termination state. The sum records the rewards already obtained during the option, while $\gamma^T Q(s_T,\omega')$ represents the value of selecting the next option after termination. The high level selects options in an SMDP, and the low level executes each option's internal policy.

### 2.2 FeUdal Networks: The Manager Sets a Direction and the Worker Acts

FeUdal Networks (Vezhnevets et al. 2017) make options learnable end to end through two networks:

- **Manager** $M_\theta$: every $c$ steps, outputs a direction vector $g_t \in \mathbb{R}^k$ in a latent space rather than a literal subgoal.
- **Worker** $W_\phi$: at every primitive step in the $c$-step window, outputs an action from $\pi_\phi(a \mid s; g_t)$, whose target distribution is modulated by $g_t$.

After the Manager outputs direction $g_t$, training must determine whether the Worker actually moved in that direction over the next $c$ steps. FeUdal represents the actual movement as the difference between two latent states:

$$\mathcal{L}_M = -\langle g_t,\ \hat{z}_{t+c} - \hat{z}_t\rangle$$

Here, $\hat z_t$ is the representation produced by the shared encoder at step $t$, and angle brackets denote an inner product. If actual change $\hat z_{t+c}-\hat z_t$ is aligned with $g_t$, their inner product is large and the leading negative sign makes the loss small. This auxiliary objective trains the Manager to specify directions the Worker can execute; task return still determines which directions are useful.

For a two-dimensional example, let the Manager output $g_t=(1,0)$, pointing right in latent space. If the representation moves by $(0.9,0.2)$ over the next $c$ steps, the inner product is $0.9$ and the loss is $-0.9$: the directions largely agree. If it moves by $(-0.8,0)$, the inner product is $-0.8$ and the loss is $0.8$, strongly penalizing a direction that the Worker followed in reverse.

FeUdal demonstrated the potential of this hierarchy on long-horizon tasks such as _Montezuma's Revenge_, while also revealing sensitivity to hyperparameters when the Manager and Worker are trained jointly.

### 2.3 HIRO: Learning Continuous Subgoals from Off-Policy Data

Data-Efficient Hierarchical Reinforcement Learning (HIRO, Nachum et al. 2018) modernizes FeUdal through **off-policy training and goal relabeling**:

- The high level outputs a continuous subgoal $g_t \in \mathbb{R}^d$, directly representing a displacement in state space, and changes it every $c$ steps.
- The low-level reward is $r^l_t=-\|s_{t+1}-(s_t+g_t)\|$: the target point is the current state displaced by $g_t$, and the reward increases as the next state approaches it.
- The high level is trained with an off-policy algorithm such as TD3.

For example, suppose the state is a robot's ground position and the high level outputs $g_t=(10,0)$, meaning “move ten meters east.” From current position $(2,0)$, the target is $(12,0)$. After one step to $(3,0)$, the remaining distance is 9 and the intrinsic reward is $-9$. At $(11,0)$, the remaining distance is 1 and the reward is $-1$. The low level does not need to understand the external task; it only needs to move closer to the target, while the external reward is assigned to the high level.

The main technical difficulty is **off-policy bias**. An old subgoal $g$ sampled by the high level from the replay buffer was executed by an earlier low-level policy, but the low-level policy has since changed. HIRO addresses this with **goal relabeling**: it remaps old subgoal $g$ to a new subgoal $g'$ that would account for the observed trajectory under the current low-level policy, keeping the high-level training data consistent.

```python
# Skeleton of the HIRO training loop.
for step in range(total_steps):
    if step % c == 0:
        # The high level samples a subgoal every c steps.
        goal = high_level_policy(state)
    # Conditional low-level policy.
    a = low_level_policy(state, goal)
    s_next, r_ext, done = env.step(a)
    # Intrinsic reward for the low level.
    r_int = -np.linalg.norm(s_next - (state + goal))
    low_buffer.add(state, a, r_int, s_next)
    if step % c == 0:
        # The high-level reward is cumulative external reward over c steps.
        high_buffer.add(state, goal, ext_reward_sum, s_next_c)
    update(low_level_policy, low_buffer)
    update(high_level_policy, high_buffer, goal_transition=transition_fn)
```

### 2.4 Comparing Options, FeUdal, and HIRO

| Algorithm | High-level output    | Low-level objective        | Training method       | Main issue                 |
| --------- | -------------------- | -------------------------- | --------------------- | -------------------------- |
| Options   | Option ID            | Fixed subpolicy            | SMDP-Q                | Options must be predefined |
| FeUdal    | Latent direction $g$ | Worker intrinsic objective | On-policy, end to end | Unstable training          |
| HIRO      | State displacement   | State matching             | Off-policy            | Goal-relabeling design     |

### 2.5 Practical Difficulties of Hierarchical RL

A hierarchy introduces additional assumptions: the high-level subgoal must be feasible, and the low level must actually use it. With an unsuitable decomposition, the Manager may output meaningless directions while the Worker learns to ignore the high level. LLM agents often express hierarchy explicitly as a “plan, then execute” process partly because plans and execution traces are easier to inspect. Whether hierarchy is internal to the network or explicit in the interaction process, one must verify that subgoals shorten the decision horizon of the original task.

## 3. Using a Generative World Model as the Training Environment

The hierarchical methods above assume an existing environment and train only the policy. A generative world model also makes the environment learnable: the model predicts future states from the current state and action, and the policy can train or plan within the resulting generated trajectories.

### 3.1 From Dreamer to Genie

[Chapter 9: Dreamer V3](../chapter11_continuous_control/search-world-models#dreamer-v3-and-a-new-generation-of-world-models) demonstrated the feasibility of training an actor-critic inside a world model: first train an RSSM world model on real data, then optimize the policy over imagined trajectories. Dreamer's world model remains task-specific, trained on a particular Atari game or MuJoCo environment.

Genie (Bruce et al. 2024) advances world models to a **generative, cross-task** setting. Given a video segment or a single image, Genie can learn an interactive “game engine”: an action is supplied, and the model generates the next frame. This has three implications:

- **Environment data come from internet video**, without dependence on a game engine or physics simulator.
- **One model can generate multiple environments**, enabling cross-task generalization.
- **RL can train in generated environments**, without a real physics engine.

Genie 3 further introduces **latent actions**: the model automatically discovers latent control variables that cause changes between video frames, without action labels. Formally,

$$z_t = \text{LatentAction}(x_t, x_{t+1}),\quad x_{t+1} = \text{Decoder}(x_t, z_t)$$

$z_t$ represents the latent factor responsible for the transition from $x_t$ to $x_{t+1}$. If these factors can be controlled consistently, $z_t$ can serve as an action and the decoder can predict the frame that follows its execution. An important limitation remains: whether latent actions correspond to executable controls must be verified in each environment.

Consider a Super Mario video. Frame 100 shows Mario in front of a pipe and frame 101 shows him jumping over it. A latent action that consistently explains this transition will eventually correspond roughly to pressing Jump. Clouds may move between the same frames, but no control action explains their motion, so the latent-action model learns to ignore it. In the reverse direction, the decoder receives “Mario in front of the pipe” together with the Jump latent action and predicts the next frame in which Mario clears the pipe.

## 4. Using Generated Environments for Exploration, Cooperation, and Hierarchy

Once a world model serves as a generative environment, the three method classes in this chapter act at different parts of it:

1. **Exploration**: intrinsic rewards can operate in the generated environment's latent space. ICM's forward-prediction error has a similar structure to a world model's state-prediction loss.
2. **Multi-agent learning**: Genie-like models can generate environments containing NPCs, in which multiple agents perform self-play ([Chapter 26: Self-Play](../chapter32_selfplay/self-play-outlook/)).
3. **Hierarchy**: a high-level policy can output a latent subgoal that the world model decodes into an environmental state change, producing an implicit form of option learning.

Together, these directions advance the world model from “predicting the next state” to “providing interactive training trajectories.” The trainer must still check whether the generated environment obeys task rules and whether the policy exploits model errors. Later chapters on agents continue this discussion by replacing pure text simulation with tool environments and executable feedback.

## Chapter Summary

This chapter studied three classes of problems beyond the classical single-agent, dense-reward setting:

1. **Sparse rewards → intrinsic rewards**: ICM uses forward-prediction error, while RND uses random network distillation. NGU combines short-term episodic novelty with long-term RND, and Agent57 switches adaptively between exploration and exploitation, becoming the first algorithm to exceed human performance on all 57 Atari games.
2. **Multi-agent nonstationarity → CTDE**: MADDPG gives each agent a centralized critic, while MAPPO uses a shared critic and on-policy clipping. MAPPO is a common strong baseline for cooperative tasks such as SMAC and Hanabi.
3. **Long horizons → hierarchy**: the Options framework, FeUdal Networks' end-to-end Manager-Worker architecture, and HIRO's off-policy goal relabeling allow the high level to focus only on a sequence of subgoals.

These approaches also provide tools for later work on LLM reinforcement learning: tool calls can be analyzed as temporally extended actions, multi-agent cooperation requires joint training and decentralized execution, and world models can generate interactive training environments. The next chapter, [Chapter 13: The RLHF Training Pipeline](../chapter15_rlhf/base-model-to-assistant), begins the main thread on aligning large language models. Exploration, cooperation, and hierarchical decision-making will recur in later chapters on agents.

## Further Reading

- [Pathak et al. 2017 "Curiosity-driven Exploration by Self-Supervised Prediction" (ICM)](https://arxiv.org/abs/1705.05363)
- [Burda et al. 2018 "Exploration by Random Network Distillation" (RND)](https://arxiv.org/abs/1810.12894)
- [Badia et al. 2020 "Agent57: Outperforming the Atari Human Benchmark"](https://arxiv.org/abs/2003.13350)
- [Lowe et al. 2017 "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (MADDPG)](https://arxiv.org/abs/1706.02275)
- [Yu et al. 2022 "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (MAPPO)](https://arxiv.org/abs/2103.01955)
- [Vezhnevets et al. 2017 "FeUdal Networks for Hierarchical Reinforcement Learning"](https://arxiv.org/abs/1703.01161)
- [Nachum et al. 2018 "Data-Efficient Hierarchical Reinforcement Learning" (HIRO)](https://arxiv.org/abs/1805.08296)
- [Bruce et al. 2024 "Genie: Generative Interactive Environments"](https://arxiv.org/abs/2402.15391)
