# CartPole PPO 训练思维导图

对应 [1.3.py](../01-cartpole入门/1.3.py)。下面用 `N` 表示本轮采样步数，默认 2048，用 `B` 表示当前 mini-batch 的实际样本数，默认 64。

## 1. 循环与更新次数

```text
train() → 默认 40 个 iteration
└─ 每个 iteration
   ├─ collect_rollout → 顺序执行 N 次 env.step，期间可能经历多个 episode
   ├─ compute_gae → 倒序遍历 N 步，再生成按原时间顺序排列的两个向量
   ├─ 设置 Adam 的学习率 → 3e-4 × (1 - iteration / total_iterations)
   └─ ppo_update → 用这批 rollout 训练 10 个 epoch
      └─ 每个 epoch → 重新打乱全部 N 个样本下标，切成 32 个 mini-batch
         └─ 每个 mini-batch → 前向 → 算 loss → 反向 → 梯度裁剪 → Adam 更新一次
```

默认每轮采样 2048 步，更新参数 `10 × 32 = 320` 次。采样循环、GAE 循环和 epoch 循环依次执行。一个 episode 可以跨过 rollout 的边界，下一轮从返回的当前 `obs` 继续。手写图的训练顺序和默认更新次数都对。参数更新公式写成「Adam 更新」更准确，`θ -= lr * grad` 是普通 SGD 的写法。

## 2. Rollout 与 GAE 的形状

```text
collect_rollout 返回一个 6 元组
├─ transitions → list[dict]，长度 N
│  └─ transitions[t] → 第 t 步的 dict
│     ├─ obs → (4,)：位置、速度、杆角度、杆角速度
│     └─ action、log_prob、value、reward、terminated、truncated、next_value
│        → 各一个标量
└─ 其余返回项 → 当前 obs、已结束 episode 的奖励和长度列表、
                以及尚未结束 episode 的累计奖励和长度

compute_gae(transitions) → 接收整个列表，逐步读取每条记录的价值、奖励和结束标志
├─ delta 和 gae → 每个时间步各一个标量
├─ raw_advantages.insert(0, gae) → 倒序算，头插恢复原时间顺序
└─ 把长度 N 的 raw_advantages 列表转成张量
   ├─ returns = raw_advantages + 采样时的 values → (N,)，Critic 的回归目标
   └─ advantages = 标准化(raw_advantages) → (N,)，Actor loss 的权重
```

采样时，`model.get_action(obs_tensor)` 已经把 `obs` 转成动作分布与 `value`。采样函数也用下一状态得到 `next_value`。GAE 直接复用这些已存的估值，递推 `delta = reward + gamma * next_value - value`。真正终止时 `next_value = 0`，时间上限截断时用 reset 前的下一状态估值。两种 episode 结束都会切断 GAE 跨 reset 的递推。走到 rollout 末尾时，递推初值为 0，最后一步仍可通过 `next_value` 引入下一状态的估值。

## 3. Actor 和 Critic 的训练输入

```text
[t["obs"] for t in transitions]
└─ 堆叠并转为 obs 张量 (N, 4)
   └─ batch_obs = obs[idx] → (B, 4)
      └─ logits, values = model(batch_obs)
         ├─ Actor  → self.actor(batch_obs) → logits (B, 2)
         │  └─ 对采样时的 batch_actions (B,) 求新 log_prob (B,)
         │     └─ 与旧 log_prob (B,) 算概率比，再和 batch_advantages (B,)
         │        算 PPO 裁剪目标，取均值得到标量 policy_loss
         └─ Critic → self.critic(batch_obs) → (B, 1) → squeeze(-1) → values (B,)
            └─ 与 batch_returns (B,) 逐样本算平方误差，取均值得到标量 value_loss
```

同一个 `ActorCritic` 对象里有两个各自独立的 MLP。它们的前向输入都是 `batch_obs`。`advantages` 和 `returns` 从 GAE 走到 loss，分别在每行提供一个权重和一个目标值。更新时，当前模型重新计算同一批状态上的 logits 和 values，并用当初采样的动作计算新 log probability。旧 log probability、advantage 和 return 在这 10 个 epoch 内保持固定。源码计算了熵，但其权重为 0，实际优化的是 `policy_loss + 0.5 * value_loss`。

## 4. 打乱下标，再完整分批

```text
假设 N = 7，batch_size = 3
└─ 一个 epoch → indices = [4, 0, 6, 1, 3, 5, 2]
   ├─ start = 0 → indices[0:3] → [4, 0, 6]
   ├─ start = 3 → indices[3:6] → [1, 3, 5]
   └─ start = 6 → indices[6:9] → [2]
```

`np.random.permutation(N)` 把 `0…N-1` 各放一次。你说的「整体分割、每个都用到」正是这段循环做的事。`start` 是在打乱后的数组里的切片起点，`idx` 中的数才是原 rollout 的行号。`obs[idx]`、`actions[idx]`、`old_log_probs[idx]`、`advantages[idx]` 和 `returns[idx]` 都用相同的行号，所以打乱后仍然一一配对。下一 epoch 换一个排列。每条样本每个 epoch 用一次，整轮共用 10 次。若不能整除，最后一个小批次照常更新。

读代码时可以始终追踪一条 `transitions[t]`：先按时间顺序为它算出 `advantages[t]` 和 `returns[t]`，再随同一行的 `obs[t]` 和 `action[t]` 一起进入某个 mini-batch。GAE 需要时间顺序，PPO 更新只需要保持同行数据对齐。
