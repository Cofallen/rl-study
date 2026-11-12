# 9.2 TD3 与 SAC

## 本节导读

**核心内容**

- 理解 TD3 如何用三个针对性的工程技巧（双 Q 网络、延迟策略更新、目标策略平滑）修复 DDPG 的过估计和不稳定问题。
- 理解最大熵 RL 的核心思想：不只是最大化奖励，还要最大化策略熵——鼓励探索、提高鲁棒性、平滑 Q 函数。
- 掌握 SAC 如何把最大熵目标转化为实际算法：随机策略 + reparameterization trick + 自动温度调节。
- 对比 DDPG、TD3、SAC 三者的优劣，知道在实践中应该首选哪个算法。

上一节我们看到 DDPG 虽然开了连续控制 off-policy 的先河，但它有三个致命缺陷：Q 值过估计、超参数敏感、训练容易崩溃。这一节我们来看两套修补方案：一套是"打补丁"——**TD3** 用三个简单但有效的工程技巧，针对性解决 DDPG 的问题；另一套是"重新设计"——**SAC** 从根本上改变目标函数，用最大熵 RL 让训练自然稳定。

先从简单的开始：TD3 是怎么给 DDPG 打补丁的？

## TD3：给 DDPG 打三个稳定性补丁

Twin Delayed Deep Deterministic Policy Gradient（Fujimoto et al. 2018）这个名字有点长，但它其实就是三个技巧的组合，每个技巧解决 DDPG 的一个具体问题。我们一个一个来看。

### 补丁一：双 Q 网络（Twin Q）——解决过估计

DDPG 的第一个问题是 Q 值过估计。回忆 Double DQN 是怎么解决 DQN 过估计的？它用两个网络，一个选动作，一个评估 Q 值，避免"自己选自己评"导致的高估。

TD3 用了更直接的方法：**训练两个独立的 Critic 网络 $Q_{\phi_1}$ 和 $Q_{\phi_2}$，计算 target 的时候取两者的较小值**：

$$
y = r + \gamma \cdot \min(Q_{\phi_1'}, Q_{\phi_2'})(s', \mu_{\theta'}(s'))
$$

这为什么有效？直觉很简单：如果一个 Q 网络因为噪声偶然给某个动作高估了值，另一个独立训练的网络不太可能也在同一个动作上高估。取较小值，就能系统性地压低高估。

举个例子：假设某个 $(s', a')$ 的真实 Q 值是 10。由于网络估计有噪声：

- $Q_1$ 可能估计成 12（高估了 2）
- $Q_2$ 可能估计成 9（低估了 1）

取 min 就是 9，虽然略低于真实值，但比用 12 好得多——**低估是保守的，不会导致策略被错误的高估值带偏**。过估计会导致正反馈崩溃，低估虽然可能让学习稍慢，但至少不会发散。

```python
class TD3Critic:
    def __init__(self, state_dim, action_dim):
        self.Q1 = QNetwork(state_dim, action_dim)
        self.Q2 = QNetwork(state_dim, action_dim)  # 独立初始化

    def forward(self, s, a):
        return self.Q1(s, a), self.Q2(s, a)

    def target_min(self, s, a):
        return torch.min(self.Q1(s, a), self.Q2(s, a))
```

两个 Critic 网络结构完全一样，只是参数独立初始化。训练的时候，两个网络都用同一个 target y 来更新，分别计算自己的 MSE loss 加在一起。

### 补丁二：延迟策略更新（Delayed Policy Updates）——让 Critic 先学好

DDPG 的第二个问题是训练不稳定：Critic 还没学好的时候，它给 Actor 的梯度是噪声，Actor 被这些噪声梯度更新后，又生成更差的数据给 Critic，形成恶性循环。

TD3 的观察是：**Critic 比 Actor 难学得多**。Critic 要拟合一个二元函数 $Q(s,a)$，需要准确估计每个状态-动作对的长期价值；而 Actor 只需要学一个一元函数 $\mu(s)$——在每个状态选一个能让 Q 最大的动作。既然 Critic 学得慢，那就让 Actor 等一等，等 Critic 多更新几轮、更准确了，再更新 Actor。

具体做法：**Critic 每一步都更新，Actor 每 $d$ 步才更新一次**（论文里取 $d=2$）。目标网络也和 Actor 同步更新。

```python
for step in range(total_steps):
    # 每步都更新 critic
    update_critic()

    # 每 d=2 步才更新 actor + 目标网络
    if step_count % policy_delay == 0:
        update_actor()
        soft_update_targets()
```

这个技巧看起来简单，但效果非常显著。它避免了 Actor 被还不准确的 Critic 梯度"带歪"，等 Critic 更稳定了再给 Actor 方向。

### 补丁三：目标策略平滑（Target Policy Smoothing）——防止 Q 函数过拟合尖峰

DDPG 的第三个问题比较微妙：确定性策略 $\mu_{\theta'}(s')$ 在计算 target 时是一个确定的点，但 Q 函数可能在某些点上有尖锐的高峰——这些高峰可能只是函数逼近的噪声，不是真实的高价值区域。如果 Actor 总是瞄准这些尖峰，策略就会不稳定。

TD3 的解决方案是：**给 target 动作加一点截断的噪声，相当于在动作空间做局部平滑**：

$$
a' = \text{clip}\left(\mu_{\theta'}(s') + \epsilon,\ a_{\text{low}},\ a_{\text{high}}\right), \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)
$$

这个式子的意思是：在目标动作 $\mu_{\theta'}(s')$ 上加一个高斯噪声，然后把噪声截断到 $[-c, c]$ 范围内，最后再把整个动作裁剪到合法范围内。常用配置是 $\sigma=0.2$，$c=0.5$。

这相当于什么？相当于我们不是在一个精确的点 $a'$ 上评估 Q 值，而是在 $a'$ 附近的一个小区域里取平均。如果 Q 函数在 $a'$ 处有一个很窄的尖峰，这个平均会把尖峰抹平，避免策略追逐虚假的峰值。**Q 函数变得更平滑，策略更新也就更稳定**。

### 三招合起来的效果

三个补丁都很简单，没有复杂的数学，但组合在一起效果惊人。TD3 在 MuJoCo 上**显著稳定了 DDPG**，性能甚至超过了同期 SAC 的早期版本。直到今天，TD3 仍然是连续控制的强基线算法。

让我们看完整的 TD3 更新代码：

```python
class TD3:
    def update(self, batch_size=256):
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        # === Critic 更新（双 Q + 目标平滑） ===
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            # 目标策略平滑：给 target 动作加截断噪声
            noise = (torch.randn_like(next_actions) * 0.2).clamp(-0.5, 0.5)
            next_actions = (next_actions + noise).clamp(-self.action_max, self.action_max)
            # 双 Q 取 min
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optim.zero_grad(); critic_loss.backward()
        self.critic_optim.step()

        # === Actor 更新（延迟） ===
        if self.step_count % self.policy_delay == 0:
            # 注意：用 Q1 来更新 Actor 就够了
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            self.actor_optim.zero_grad(); actor_loss.backward()
            self.actor_optim.step()

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
```

注意一个细节：更新 Actor 的时候，只用了 $Q_1$，没有用 $Q_2$。为什么？因为 Actor 的目标是最大化 Q 值，用哪个 Q 都行——只要 Q 已经是保守估计了。用一个就够了，两个反而可能互相干扰。

TD3 很好，但它本质上还是"在 DDPG 框架上打补丁"——策略仍然是确定性的，探索仍然靠外加噪声，超参数虽然比 DDPG 少了，但还是要调噪声大小。有没有办法从根本上改变目标函数，让探索和稳定性内生于算法，而不是靠外加补丁？

这就是 SAC 要做的事。

## SAC：最大熵强化学习

Soft Actor-Critic（Haarnoja et al. 2018）从一个完全不同的角度切入。它说：我们之前一直把目标定义为"最大化期望累积奖励"，这个目标太"贪心"了——它只想要最高的分数，不管策略是不是在一棵树上吊死。如果我们把目标改一改，不仅要奖励高，还要**策略熵大**——也就是在每个状态都保持一定的随机性，不要过早收敛到一个确定的动作——会怎么样？

### 最大熵目标

SAC 的目标函数是：

$$
J(\pi) = \mathbb{E}_{(s_t, a_t) \sim \pi}\left[\sum_t \gamma^t \left(r_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t))\right)\right]
$$

让我们拆开看。原来的目标是 $\sum \gamma^t r_t$——只加奖励。现在多了一项：$\alpha \mathcal{H}(\pi(\cdot \mid s))$。

- $\mathcal{H}(\pi(\cdot \mid s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)]$ 是策略在状态 $s$ 的**熵**。熵是什么？简单说就是"随机性"。如果策略在一个状态确定选一个动作（概率 1），熵是 0；如果策略在多个动作上分散概率，熵就大。
- $\alpha$ 是**温度系数**，控制熵奖励的权重。$\alpha$ 越大，越鼓励探索；$\alpha = 0$ 就退化成普通的 RL。

你可能会问："为什么要加熵奖励？鼓励随机性有什么好处？"

**第一个好处：自动探索**。DDPG 和 TD3 都需要我们手动加高斯噪声来探索，噪声大小还要调。SAC 的策略本身就是随机的——它输出一个分布，从分布里采样动作——熵奖励会阻止策略过早变成 delta 函数，探索自然就有了，不用手动加噪声。

**第二个好处：鲁棒性**。如果一个状态有多个动作都差不多好，最大熵策略会给这些动作都分配一定的概率，而不是只选一个。这样当环境有噪声或者模型有误差时，策略不会因为一个小扰动就崩溃。

**第三个好处：Q 函数更平滑**。熵奖励相当于鼓励策略在 Q 值差不多的动作之间"分散"，这会让 Q 函数的学习信号更平滑，减少过拟合尖峰的问题——和 TD3 的目标平滑有异曲同工之妙，但这是从目标函数层面自然实现的，不是外加的补丁。

我们用一个简单例子感受一下。假设在某个状态，有三个动作：

- 动作 A：Q = 10（最好）
- 动作 B：Q = 9.9（几乎一样好）
- 动作 C：Q = 1（很差）

普通 RL（DDPG/TD3）会收敛到选 A 的概率 100%——熵为 0。
最大熵 RL 会怎么做？它要在"选 Q 高的动作"和"保持熵"之间权衡。如果 $\alpha$ 合适，它可能给 A 分配 60% 概率，B 分配 39% 概率，C 分配 1% 概率——熵比较大，同时期望 Q 也很高。如果以后发现 A 其实有问题（比如环境变了），它有很大概率已经在尝试 B 了，能快速切换。

### Soft Bellman 方程

目标函数变了，Bellman 方程也要跟着变。原来的 Bellman 方程是：

$$
Q(s,a) = \mathbb{E}_{s'}\left[r + \gamma \max_{a'} Q(s', a')\right]
$$

最大熵下，我们不再取 $\max_{a'}$，而是取一个"软最大值"（soft max）——它考虑了每个动作的 Q 值，但同时也考虑了熵：

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi}\left[Q^\pi(s, a)\right] + \alpha \mathcal{H}(\pi(\cdot \mid s))
$$

$$
Q^\pi(s, a) = \mathbb{E}_{s'}\left[r + \gamma V^\pi(s')\right]
$$

这里 $V(s)$ 不再是 $\max_a Q(s,a)$，而是 Q 的期望加上熵奖励。对于连续动作，soft max 有一个解析形式：

$$
V^\pi(s) = \alpha \log \int \exp\left(\frac{Q^\pi(s, a)}{\alpha}\right) da
$$

这个积分看起来有点吓人，但你不需要记住它——SAC 在实现时用了一个巧妙的技巧（reparameterization trick）来避免直接计算这个积分。

### 随机策略与 reparameterization trick

DDPG/TD3 的策略是确定性的 $a = \mu_\theta(s)$。SAC 的策略是随机的——它输出一个高斯分布的均值和标准差：

$$
a \sim \mathcal{N}\left(\mu_\theta(s),\ \sigma_\theta^2(s)\right)
$$

但是有个问题：如果我们直接从高斯分布采样 $a$，然后计算 Q 值，梯度怎么传回 $\mu$ 和 $\sigma$？采样操作是不可微的——你不知道采样得到的 $a$ 变化一点点，是因为 $\mu$ 变了还是因为采样的噪声变了。

解决方案是 **reparameterization trick**（重参数化技巧）。我们把随机性和确定性分开：先从一个固定的标准正态分布采样噪声 $\epsilon$，然后用 $\mu$ 和 $\sigma$ 把噪声"变换"成动作：

$$
a = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

这里 $\odot$ 是逐元素相乘。这样一来，$a$ 还是高斯分布的采样，但随机性只来自 $\epsilon$——$\epsilon$ 是固定的（没有梯度），所有的梯度都可以通过 $\mu_\theta(s)$ 和 $\sigma_\theta(s)$ 流回策略网络。

有了 reparameterization，Actor 的损失函数就可以写成：

$$
\mathcal{L}_{\text{actor}} = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}}\left[\alpha \log \pi_\theta(a \mid s) - Q_\phi(s, a)\right]
$$

这里 $a = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon$ 是重参数化后的动作。最小化这个 loss 等价于最大化 $Q(s,a)$ 同时减去 $\alpha \log \pi(a|s)$——后者就是熵项的贡献。

```python
def actor_loss(self, states):
    # 重参数化采样
    mu, sigma = self.actor(states)
    dist = Normal(mu, sigma)
    # 用 rsample() 而不是 sample()——rsample 支持重参数化梯度
    actions = dist.rsample()
    log_probs = dist.log_prob(actions).sum(-1, keepdim=True)

    # 最小化：α * log_prob - Q(s,a)
    q1, q2 = self.critic(states, actions)
    min_q = torch.min(q1, q2)
    return (self.alpha * log_probs - min_q).mean()
```

注意两个细节：

1. 用的是 `rsample()` 而不是 `sample()`——前者保留了梯度路径（重参数化），后者不保留。
2. 和 TD3 一样，SAC 也用了双 Q 网络，取 min 来抑制过估计。

### 自动温度调节

SAC 里有一个关键超参数 $\alpha$——温度系数，它控制"探索 vs 利用"的平衡。$\alpha$ 太大，策略会太随机，光探索不收敛；$\alpha$ 太小，策略太快确定，又回到 DDPG 的问题。手动调 $\alpha$ 很麻烦。

SAC 的一个非常实用的工程创新是**自动调温**：把 $\alpha$ 也当成一个可学习的参数，在训练中自动调整。

目标是什么呢？我们希望策略的熵维持在一个目标值 $\mathcal{H}_0$ 附近——如果实际熵比目标大（太随机），就减小 $\alpha$，让奖励项更重要；如果实际熵比目标小（太确定），就增大 $\alpha$，多鼓励探索。

形式化地，$\alpha$ 的优化目标是：

$$
\alpha^* = \arg\max_\alpha \mathbb{E}\left[-\alpha \log \pi(a \mid s) - \alpha \mathcal{H}_0\right]
$$

实践中目标熵 $\mathcal{H}_0$ 通常设为 $-|\mathcal{A}|$——也就是动作维度的负数。比如动作是 6 维连续值，目标熵就设为 -6。这是一个启发式，但在大多数任务上都工作得很好，不需要手动调。

```python
def update_alpha(self, states):
    mu, sigma = self.actor(states)
    dist = Normal(mu, sigma)
    actions = dist.rsample()
    log_pi = dist.log_prob(actions).sum(-1)

    # alpha loss：让熵接近 target_entropy
    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
    self.alpha_optim.zero_grad()
    alpha_loss.backward()
    self.alpha_optim.step()
    self.alpha = self.log_alpha.exp().item()
```

注意代码里用的是 `log_alpha` 而不是直接优化 $\alpha$——这是为了保证 $\alpha$ 始终是正数（因为 $\exp(\log \alpha) > 0$）。

### SAC 的优势

SAC 把这些东西合在一起：

- **随机策略 + 熵奖励**：自动探索，不用手动加噪声
- **双 Q 网络**：抑制过估计（和 TD3 一样）
- **重参数化技巧**：让随机策略的梯度可计算
- **自动温度调节**：不用手动调探索强度

效果如何？在 MuJoCo 连续控制基准上，SAC **长期霸榜**，原因：

1. **样本效率高**：off-policy，继承了 DDPG 的优点
2. **训练稳定**：最大熵 + 双 Q，超参数不敏感
3. **开箱即用**：自动调温，一套超参数能在大部分环境工作
4. **最终性能好**：在 HalfCheetah 等环境上能达到 15000+ 分，超过 TD3 和 DDPG

### DDPG、TD3、SAC 三者对比

我们把三个算法放在一起比：

| 维度     | DDPG     | TD3          | SAC            |
| -------- | -------- | ------------ | -------------- |
| 策略类型 | 确定性   | 确定性       | 随机（高斯）   |
| Q 网络   | 1 个     | 2 个（Twin） | 2 个（Twin）   |
| 探索方式 | 外加噪声 | 外加噪声     | 熵奖励（内置） |
| 稳定性   | 差       | 中           | 强             |
| 超参敏感 | 高       | 中           | 低（自动调温） |
| 推荐首选 | ❌       | ⚠️           | ✅             |

**实战建议**：连续控制任务**首选 SAC**。只有在你明确需要确定性策略（比如部署时不想要任何随机性）的时候，才选 TD3。DDPG 现在主要是教学价值，实践中基本不用了。

## 训练曲线对比

在 MuJoCo HalfCheetah-v3 环境上训练 100 万步，三个算法的典型表现是这样的：

```
回报
12000 │                    ╭─────── SAC (稳定收敛)
10000 │                  ╭─╯
 8000 │                ╭─╯  ╭─────── TD3 (稳定但稍慢)
 6000 │              ╭─╯   ╱
 4000 │            ╭─╯    ╱
 2000 │          ╭─╯     ╱  ╭───── DDPG (发散后偶尔恢复)
     0 │─────────╯──────╱──╯
       └───────────────────────────────
        0    200K  400K  600K  800K  1M steps
```

可以看到三个明显的特点：

- **SAC** 收敛最快、最稳定——最大熵自带的探索让前期学得快，自动调温让后期稳定收敛
- **TD3** 略慢于 SAC 但最终性能接近——三个稳定性补丁让 DDPG 终于可用了
- **DDPG** 大部分时间在发散——只在某些随机种子下偶尔训练成功，非常不可靠

::: details 加餐：SAC 的 Soft Q 更新细节
对于想深入理解 SAC 数学的读者，这里补充一下 Critic 更新的完整形式。因为策略是随机的，target y 不是 $r + \gamma Q(s', a')$，而是要对动作 $a' \sim \pi(\cdot | s')$ 求期望，并加上熵项：

$$
y = r + \gamma \left( \min(Q_{\phi_1'}, Q_{\phi_2'})(s', \tilde{a}') - \alpha \log \pi_\theta(\tilde{a}' \mid s') \right)
$$

这里 $\tilde{a}'$ 是从目标策略重参数化采样的动作。注意减去 $\alpha \log \pi$ 就是加上熵项（因为 $\mathcal{H} = -\mathbb{E}[\log \pi]$）。

用双 Q 取 min 和 TD3 一样是为了抑制过估计。实际代码里，这个 target 的计算和 TD3 结构类似，只是多了 log_prob 项和用采样动作代替确定性动作。
:::

## 本节总结

DDPG → TD3 → SAC 是连续控制 off-policy 算法的三步演进：

1. **DDPG** 把 DQN 思想扩展到连续动作——确定性策略 + Actor-Critic + 经验回放 + 目标网络，但不稳定，容易崩溃。
2. **TD3** 用三个工程技巧稳定 DDPG：双 Q 取 min 抑制过估计、延迟策略更新让 Critic 先学好、目标策略平滑避免追逐 Q 函数尖峰。
3. **SAC** 从根本上重构目标函数——最大熵 RL 在奖励之外加上熵奖励，让探索内置、训练稳定；配合双 Q、重参数化技巧、自动温度调节，成为连续控制的首选算法。

实战中 SAC 是首选，TD3 是确定性策略场景的备选，DDPG 已不推荐。

下一节 [9.3 Model-Based RL](./model-based) 转向另一个方向——当真实环境采样非常昂贵时（比如真实机器人），我们不满足于只复用历史数据，而是学一个环境模型来生成"想象"的数据，把样本效率再提升 10-100 倍。
