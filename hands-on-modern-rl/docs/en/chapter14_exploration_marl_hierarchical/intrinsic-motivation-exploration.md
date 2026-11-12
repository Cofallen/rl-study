# 12.1 Intrinsic Motivation and Exploration

[Chapter 9: Continuous Control](../chapter11_continuous_control/ddpg) studies how a single agent can learn stably, while [Chapter 10: Offline RL](../chapter12_offline_rl/offline-data-distribution-shift) studies how to use fixed data when further interaction is impossible. Chapter 12 relaxes the classical setting further: rewards may take a long time to appear, other agents may change the environment simultaneously, and a task may be too long for a single-level policy to receive useful training signals.

This section begins with sparse rewards. We first explain why random exploration fails, then construct intrinsic rewards from ICM prediction error, use RND to estimate state novelty, and finally examine how NGU and Agent57 manage short- and long-term exploration together.

<OnlineTraining studios="minigrid" compact />

## 1. Why Sparse Rewards Require Intrinsic Motivation

[Chapter 2's exploration-exploitation problem](../chapter03_mdp/bandit) introduced this trade-off in a stateless setting: because the expected return of each arm is unknown, the agent must divide its budget between pulling the currently best arm (exploitation) and uncertain arms (exploration). UCB encodes uncertainty directly in action values with the upper confidence bound $U_t(a) = \hat{\mu}_t(a) + c\sqrt{\ln t / N_t(a)}$.

Deep RL magnifies this problem. In Atari games such as _Montezuma's Revenge_ and _Pitfall_, reaching the first reward from the initial state requires dozens of meaningful actions—avoiding traps, collecting a key, and opening a door. The probability that random $\epsilon$-greedy exploration reaches the first reward is approximately $10^{-18}$. DQN consequently scores zero for long periods on these **hard-exploration** games.

The difficulty is that the reward signal is too sparse. Suppose the agent starts at $s_0$ and needs at least $H^\star$ steps to reach reward state $s^\star$. Any update based only on environment reward $r_t$ must wait for the first successful trajectory, whose density falls rapidly as $H^\star$ increases. An **intrinsic reward** provides an auxiliary signal before an external reward is reached, encouraging the agent to visit novel or unpredictable states.

Training adds the environment and exploration rewards:

$$r^{\text{total}}_t = r^{\text{ext}}_t + \beta \cdot r^{\text{int}}_t$$

Here, $r_t^{\text{ext}}$ is the environment reward at step $t$, $r_t^{\text{int}}$ is the novelty reward computed by the agent, and $\beta$ controls its contribution. If the external reward is 0, the intrinsic reward is 0.4, and $\beta=0.1$, the total reward is 0.04. An intrinsic reward must satisfy two requirements:

1. **Computable**: it depends only on observed data and requires no external supervision.
2. **Exhaustible**: after a state has been visited often enough, its intrinsic reward should decay to zero so the agent cannot remain there collecting reward.

The next two sections present two major approaches: prediction-error-based ICM and random network distillation (RND).

## 2. Driving Exploration with ICM Prediction Error

### 2.1 Constructing Intrinsic Reward from State-Prediction Error

The central idea of the Intrinsic Curiosity Module (Pathak et al. 2017) is that a region is unfamiliar, and therefore worth exploring, when the agent cannot predict its next state. Larger prediction error yields greater intrinsic reward.

Direct prediction in pixel space fails because the next frame contains too many details and irrelevant high-frequency noise dominates the error. ICM first uses an **inverse model** $g_\phi$ to learn a feature space $\Phi(s)$: given $(s_t,s_{t+1})$, it predicts action $a_t$. These features retain components affected by actions while filtering irrelevant changes such as background flicker and camera shake.

The filtering mechanism is easier to see as a prediction problem. Give the inverse model two consecutive Mario frames and ask which key was pressed. Mario moving half a tile to the right directly reveals the Right key. Leaves changing angle and a score indicator flickering at the same moment reveal nothing about the action. Encoding those details would only make action prediction harder, so the learned $\Phi(s)$ retains controllable information such as Mario's position, terrain, and enemies while suppressing unrelated animation.

Given features $\Phi(s_t)$, forward model $f_\psi$ predicts the next-state features from the current features and action:

$$\hat{\Phi}(s_{t+1}) = f_\psi(\Phi(s_t), a_t)$$

The distance between actual and predicted features becomes the intrinsic reward:

$$r^{\text{int}}_t = \tfrac{1}{2}\|\Phi(s_{t+1}) - \hat{\Phi}(s_{t+1})\|^2$$

The distance is large upon entering an unfamiliar region because the model predicts poorly. As similar states recur, the forward model learns to predict them and the reward decreases. The factor $1/2$ only simplifies differentiation and does not change the reward ordering.

Training combines the policy, inverse-model, and forward-model losses:

$$\mathcal{L} = \mathcal{L}_{\text{policy}}(\theta) + \lambda_{\text{inv}}\,\mathcal{L}_{\text{inv}}(\phi) + \lambda_{\text{fwd}}\,\mathcal{L}_{\text{fwd}}(\psi)$$

$\lambda_{\text{inv}}$ and $\lambda_{\text{fwd}}$ weight the two auxiliary tasks. The inverse model makes the features retain action-relevant information, the forward model provides novelty, and the policy learns from the total reward including intrinsic reward.

```python
class ICM(nn.Module):
    def __init__(self, feat_dim=256, action_dim=6):
        self.encoder = CNNtoMLP(out=feat_dim)              # Phi(s)
        self.inverse = nn.Linear(feat_dim * 2, action_dim) # g_phi
        self.forward_net = MLP(feat_dim + action_dim, feat_dim)

    def intrinsic_reward(self, s, a, s_next):
        phi, phi_next = self.encoder(s), self.encoder(s_next)
        phi_pred = self.forward_net(torch.cat([phi, a], -1))
        return 0.5 * (phi_next - phi_pred).pow(2).sum(-1)

    def forward_loss(self, s, a, s_next):
        phi, phi_next = self.encoder(s), self.encoder(s_next)
        phi_pred = self.forward_net(torch.cat([phi, a], -1))
        return F.mse_loss(phi_pred, phi_next.detach()) + \
               F.cross_entropy(self.inverse(torch.cat([phi, phi_next], -1)), a)
```

In visual-control tasks such as _Super Mario Bros_, ICM enables an agent to traverse an entire map without external rewards. Its weakness is the **noisy-TV problem**: if the environment contains an unpredictable random source, such as television static in a corner, the forward model never learns to predict it. The intrinsic reward remains high, and the agent stays in front of the television.

## 3. Using RND to Recognize Unvisited States

Random Network Distillation (Burda et al. 2018) does not predict the next frame. It fixes a randomly initialized target network $\hat f(s)$ that is never updated, then trains predictor network $f_\psi(s)$ to match its output:

$$\mathcal{L}_{\text{RND}}(\psi) = \mathbb{E}_s\bigl[\|f_\psi(s) - \hat{f}(s)\|^2\bigr]$$

$$r^{\text{int}}(s) = \|f_\psi(s) - \hat{f}(s)\|^2$$

The first line is the predictor's training loss; the second uses the same error as exploration reward. Frequently visited states have appeared in many updates and usually have small error. New states have not been fitted and usually have large error. The random target network contains no task semantics; it simply provides a fixed target for every input.

RND has three advantages:

- **No inverse model**, reducing computation.
- **No action dependence**, so it can be added to any model-free algorithm, including PPO and A2C.
- **Natural robustness to noisy television**: the random target has finite complexity, so prediction error is bounded rather than increasing without limit.

```python
class RND(nn.Module):
    def __init__(self, obs_shape, feat_dim=512):
        # Target network: frozen and never updated.
        self.target = CNN(obs_shape, feat_dim)
        for p in self.target.parameters():
            p.requires_grad = False
        # Predictor network: trained.
        self.predictor = CNN(obs_shape, feat_dim)

    def intrinsic_reward(self, s):
        with torch.no_grad():
            target = self.target(s)
        pred = self.predictor(s)
        return (pred - target).pow(2).sum(-1)  # One scalar per state.
```

Burda et al. found in large-scale experiments that PPO agents using only RND intrinsic rewards, without any external reward, discovered complex behavior in several Atari games. On sparse-reward _Montezuma's Revenge_, RND was the first method to surpass a score of zero.

### 3.1 Comparing ICM and RND

| Dimension                      | ICM                               | RND                    |
| ------------------------------ | --------------------------------- | ---------------------- |
| Action-dependent               | Yes (the forward model needs $a$) | No                     |
| Trainable modules              | Encoder + inverse + forward       | Predictor only         |
| Robustness to noisy television | Weak                              | Strong                 |
| Computational cost             | High                              | Medium                 |
| Representative applications    | Visual exploration (Mario, DMLab) | Atari hard exploration |

## 4. Managing Short- and Long-Term Exploration Together

ICM and RND solve parts of the problem but share a blind spot: they lack **episodic memory**. A state may be novel within one episode but have been visited millions of times across episodes. Novelty based only on neural-network prediction error cannot distinguish these time scales. Never Give Up (Badia et al. 2020) and its successor Agent57 (Badia et al. 2020) model both and became the **first algorithm to exceed human performance across all 57 Atari games**.

### 4.1 NGU's Two-Timescale Intrinsic Reward

NGU combines two components multiplicatively:

$$r^{\text{int}}_t(s) = r^{\text{episodic}}_t(s) \cdot r^{\text{life-long}}_t(s)$$

The **episodic component** $r^{\text{episodic}}$ maintains a fixed-capacity table of controllable states visited during the current episode. A state far from every stored feature under k-nearest-neighbor distance has high novelty; novelty decays for frequently visited states:

$$r^{\text{episodic}}_t = \frac{1}{\sqrt{k} + c \sum_{i=1}^{k} \frac{1}{\sqrt{d(s_t,s_i)}}}$$

Here, $s_i$ denotes one of the $k$ stored states nearest to $s_t$, $d(s_t,s_i)$ is the distance to that neighbor, and $c$ controls the decay caused by revisiting nearby states. A close neighbor makes $1/\sqrt d$ large, increasing the denominator and reducing within-episode novelty.

For a numerical example, take $k=1$ and $c=1$. If the nearest stored state has distance $d=4$, novelty is $1/(1+1/\sqrt4)=1/1.5\approx0.67$. If the agent has already visited almost the same state and $d=0.01$, novelty falls to $1/(1+10)\approx0.09$. Within one episode, the second visit to a location therefore receives about one seventh of the earlier intrinsic reward.

The **life-long component** $r^{\text{life-long}}$ uses RND across episodes to identify states that are new in the current episode but have been visited repeatedly in earlier episodes. After multiplication, only states rare both within the current episode and over long-term training receive high intrinsic reward.

### 4.2 Propagating Rewards with Retrace and Distributed Actors

NGU uses R2D2's distributed architecture—parallel actors plus an LSTM for partial observability—and estimates off-policy Q-values with Retrace($\lambda$), propagating intrinsic rewards stably across long horizons. The system is extremely expensive to train, requiring billions of frames and hundreds of TPUs, but it first demonstrated that end-to-end RL could solve Atari hard-exploration games.

### 4.3 How Agent57 Selects Exploration Strength Adaptively

NGU leaves one problem unresolved: intrinsic-reward weight $\beta$ is fixed. In simple games such as _Pong_ and _Space Invaders_, a large $\beta$ causes excessive exploration instead of exploiting the known optimal policy. In hard-exploration games, a small $\beta$ provides insufficient exploration. **Agent57** introduces an **adaptive policy scheduler**:

- It maintains a family of policies $\pi_i$ with different exploration parameters $(\beta_i,\gamma_i,c_i)$ spanning pure exploitation to pure exploration.
- A meta-controller estimates each policy's relative return online and preferentially samples high-performing policies.
- The policies share a replay buffer and Q-network during training.

This removes the need to fix one $\beta$ for each game. Agent57 was the first algorithm to surpass human benchmark scores on every game in the Atari 57 suite, showing that one system can cover tasks emphasizing either exploration or exploitation.

## Section Summary

Intrinsic motivation supplies a sustained training signal for sparse-reward tasks. ICM uses state-prediction error, RND uses prediction error against a random target network, NGU combines within-episode and across-episode novelty, and Agent57 selects among different exploration strengths according to task performance. This line of work substantially improves results on hard-exploration games such as _Montezuma's Revenge_.

The next section, [12.2 Multi-Agent RL: CTDE, MADDPG, and MAPPO](./marl), turns to another challenge: when multiple agents learn simultaneously, nonstationarity violates the MDP assumption.
