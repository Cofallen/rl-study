# 9.4 搜索与世界模型

## 本节导读

**核心内容**

- 理解 AlphaZero 如何把蒙特卡洛树搜索（MCTS）和神经网络结合——策略网络缩小搜索宽度，价值网络缩短搜索深度——通过自我对弈达到超人类水平。
- 拆解 PUCT 公式，理解"利用 + 先验 + 探索"三项如何平衡，为什么这比单纯的 UCB 更有效。
- 理解 MuZero 的核心创新：隐式模型学习——不需要知道游戏规则，在隐藏空间里学动力学、做规划，真正实现"通用强化学习"。
- 掌握 Dreamer V3 的循环状态空间模型（RSSM）：如何同时建模确定性轨迹和随机性，在"想象"里训练 actor-critic。
- 对比 Model-Based 与 Model-Free 的优劣，理解何时该用哪个，以及这些思想如何启发 LLM 时代的 PRM 搜索和世界模型。

上一节 [9.3](./model-based) 我们讲了 model-based RL 的"数据增强"路线——Dyna/PETS/MBPO 都是用模型生成额外数据，最终还是用 model-free 的方式训练策略。但 model-based 还有另一条更激进、也更强大的旗舰路线：**显式搜索 + 神经网络估值**。

让我们停下来想一下。人在下棋的时候，是怎么思考的？你不会直接输出一个动作——你会往前看几步："如果我走这里，对方大概率会应那里，然后我再走这里……"你在脑子里推演未来的可能性，比较不同走法的结果，然后选看起来最好的那一步。这种"思考"过程，就是**搜索**。

从 AlphaGo（2016）击败李世石，到 AlphaZero（2017）从零开始超越所有人类围棋程序，到 MuZero（2019）不需要知道规则就能学，再到 Dreamer V3（2023）用世界模型在 150+ 任务上"开箱即用"——这条线代表了 model-based RL 的理论天花板，也直接启发了今天 LLM 时代的 Process Reward Model 推理搜索。这一节我们就来走一遍这条进化路线。

## AlphaZero 与 搜索 + 学习的极致

AlphaGo（2016）→ AlphaGo Zero（2017）→ AlphaZero（2017）→ MuZero（2019），这条技术线代表了 model-based RL 的另一种哲学：**不只是学一个模型来生成数据，而是把搜索直接嵌入到决策过程中——用神经网络快速估值引导搜索，用搜索结果反过来训练神经网络**。

这种"神经网络 + 搜索"的结合有多强大？AlphaZero 从零开始，不需要任何人类棋谱，只通过自我对弈，4 小时打败国际象棋世界冠军程序 Stockfish，24 小时打败将棋世界冠军 Elmo，72 小时超越所有人类围棋程序。而且——最惊人的是——**同一个算法、同一套超参数，在围棋、国际象棋、将棋三个完全不同的游戏上都达到了超人类水平**。

### AlphaZero 的核心循环

AlphaZero 的核心是把 MCTS（蒙特卡洛树搜索）和神经网络完美结合起来。我们先看搜索函数的伪代码，再拆解每个部分：

```python
def alphazero_search(state, neural_net, n_simulations=800):
    root = MCTSNode(state)
    for _ in range(n_simulations):
        # 1. Selection: 按 PUCT 选最优子节点
        node = root
        while not node.is_leaf():
            node = node.select_child()

        # 2. Expansion: 神经网络评估叶子
        policy, value = neural_net(node.state)
        node.expand(policy)

        # 3. Backup: 把 value 反向传播到根
        node.backup(value)

    # 返回根节点的访问次数分布作为动作概率
    return root.compute_action_distribution()
```

这个搜索循环做 800 次模拟（simulation）——也就是在真正走一步棋之前，先在脑子里"下"800 次。每次模拟分三步：选择（Selection）、扩张/评估（Expansion/Evaluation）、回溯（Backup）。

但等等，你可能会问："传统的 MCTS 不是这样的啊？传统 MCTS 要 rollout 到游戏结束才能拿到价值，为什么这里直接用神经网络输出 value 就行了？"

这就是 AlphaZero 相对于传统 MCTS 的两大创新，用两个网络分别解决了搜索的两个痛点：

- **策略网络** $p_\theta(a \mid s)$：输出"在这个状态下，每个动作看起来有多好"的先验概率。它的作用是**缩小搜索宽度**——传统 MCTS 在每个节点要平等考虑所有合法动作，但有了策略网络，我们只需要重点搜索那些先验概率高的动作。
- **价值网络** $v_\theta(s)$：输出"这个状态最终能赢的概率"的估计。它的作用是**缩短搜索深度**——传统 MCTS 必须 rollout 到游戏结束（终局）才能拿到奖励，但有了价值网络，到了叶子节点直接用 $v_\theta$ 估值，不需要继续往下搜了。

就这么两个改变，让搜索效率提升了几个数量级。传统 MCTS 在围棋上需要百万次模拟才能下出好棋，AlphaZero 只需要 800 次。

### PUCT 公式

搜索的核心是：在树的每个节点，我们应该优先扩展哪个子节点？AlphaZero 用的是 PUCT（Predictor + Upper Confidence Bound）公式来给每个候选动作打分：

$$
\text{PUCT}(a) = Q(s, a) + c_{\text{prior}} \cdot p_\theta(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}
$$

这个公式看起来有点复杂，我们把它拆成三项，每一项都有明确的含义：

1. **第一项：$Q(s, a)$**——这是**利用项**。动作 $a$ 的当前价值估计，也就是到目前为止，这条路径在模拟中平均拿到了多少价值。$Q$ 越高，说明这个动作历史表现越好，我们越应该继续考虑它。

2. **第二项：$c_{\text{prior}} \cdot p_\theta(a \mid s)$**——这是**先验项**。$p_\theta(a|s)$ 是策略网络给出的"这个动作应该被考虑的概率"。策略网络认为好的动作，即使还没怎么被探索过，我们也会给它初始的优先级。$c_{\text{prior}}$ 是一个常数，控制先验的权重。

3. **第三项：$\sqrt{N(s)} / (1 + N(s, a))$**——这是**探索项**，来自经典的 UCB（Upper Confidence Bound）算法。$N(s)$ 是当前节点被访问的总次数，$N(s, a)$ 是动作 $a$ 被选中的次数。这个项的意思是：
   - 总访问次数越多，探索奖励越大（因为我们已经有足够多数据利用了，可以多探索）；
   - 某个动作被选得越少，它的探索奖励越大（鼓励探索那些还没怎么试过的动作）。

让我们用具体数字感受一下。假设根节点已经被访问了 100 次（$N(s)=100$），有两个动作：

- 动作 A：$Q=0.8$（看起来很好），$p=0.3$（策略网络也觉得不错），$N(s,a)=60$（已经被选了 60 次）
- 动作 B：$Q=0.2$（目前看起来一般），$p=0.4$（策略网络觉得其实很有希望），$N(s,a)=5$（只试过 5 次）

我们取 $c_{\text{prior}}=1$，算一下 PUCT 分数：

- PUCT(A) = 0.8 + 1 × 0.3 × √100 / (1 + 60) = 0.8 + 0.3 × 10 / 61 ≈ 0.8 + 0.049 = 0.849
- PUCT(B) = 0.2 + 1 × 0.4 × √100 / (1 + 5) = 0.2 + 0.4 × 10 / 6 ≈ 0.2 + 0.667 = 0.867

有意思吧？虽然动作 A 目前的 Q 值高很多，但因为动作 B 还没怎么被探索过，而且策略网络认为它其实不错，它的 PUCT 分数反而更高——下一步模拟就会选 B 来探索。这就是"利用-探索"平衡的精妙之处：既要选目前看起来好的，也要给那些"有潜力但还没试过"的机会。

### 自我对弈训练

你可能会问："策略网络和价值网络是怎么训练出来的？一开始它们是随机的啊，搜索不也是乱搜吗？"

答案是**自我对弈（self-play）**——AI 自己和自己下棋，用搜索的结果作为标签来训练自己，然后用更强的网络再做搜索，再对弈，循环往复，越来越强。这是一个"左右互搏"的过程。

具体来说，每一局自我对弈是这样的：

```python
def self_play_training(network, n_games=10000):
    for game in range(n_games):
        # 1. 自我对弈
        trajectory = []
        state = initial_state()
        while not state.is_terminal():
            policy = alphazero_search(state, network)
            action = sample_from(policy)
            trajectory.append((state, policy, action))
            state = state.next(action)

        # 2. 标注胜负
        winner = state.winner()
        for s, p, a in trajectory:
            value = +1 if winner == s.current_player else -1
            train_network(s, p, value)
```

我们一步步看：

1. **自我对弈阶段**：从初始状态开始，每一步都用当前网络做 MCTS 搜索，得到一个"更优的动作分布"（MCTS 算出来的访问次数），然后从这个分布里采样一个动作（加点探索），把 $(状态, MCTS策略分布, 动作)$ 存下来。
2. **标注阶段**：一局棋下完，有了赢家。对于每一步，如果走棋的玩家最终赢了，这一步的价值标签就是 +1；如果输了，就是 -1。
3. **训练阶段**：用这些数据监督训练两个网络：
   - 策略网络：让它的输出 $p_\theta(a|s)$ 尽量接近 MCTS 搜索出来的动作分布——因为 MCTS 搜索比直接用网络选动作更强，这相当于"用搜索结果作为更好的策略标签"。
   - 价值网络：让它的输出 $v_\theta(s)$ 尽量接近最终的胜负结果（+1/-1）。

这个过程的美妙之处在于：**标签不需要人给，是搜索自己生成的**。一开始网络是随机的，搜索也很弱；但哪怕是弱搜索，也比纯随机网络强一点；用弱搜索的结果训练网络，网络变强一点；变强的网络又能让搜索变强一点……这样正反馈循环下去，网络和搜索一起进化，最终达到超人类水平。

最惊人的是：**整个过程不需要任何人类棋谱**。AlphaZero 从零开始，不知道任何定式、任何开局、任何人类总结的围棋知识——它只知道规则，然后自己和自己下，72 小时后就超越了所有人类围棋程序。

::: details 加餐：AlphaGo → AlphaGo Zero → AlphaZero 的演进
可能你会好奇这几个名字的区别，这里简单梳理一下：

- **AlphaGo Fan（2015）**：第一个击败人类职业棋手的版本，用了大量人类棋谱做监督学习预训练，然后用策略梯度强化学习。
- **AlphaGo Lee（2016）**：击败李世石的版本，仍然用人类棋谱监督预训练，但价值网络更强，搜索用了 40 个 TPU。
- **AlphaGo Master（2017）**：在线上对战平台以 60-0 击败所有人类顶尖棋手，用了单个神经网络同时输出策略和价值，仍然从人类棋谱起步。
- **AlphaGo Zero（2017）**：**完全抛弃人类棋谱**，从零开始自我对弈，只用了 40 天就超过 AlphaGo Lee，用单个残差网络，更简洁高效。
- **AlphaZero（2017）**：把 AlphaGo Zero 的算法推广到国际象棋、将棋等所有完美信息博弈，同一套超参数通吃三个游戏，不需要任何领域知识。
  :::

AlphaZero 非常强大，但它有一个前提条件：你必须知道**游戏规则**——也就是知道状态转移是什么、合法动作有哪些、终局条件是什么。因为 MCTS 需要知道"做了动作 $a$ 之后，下一个状态是什么"。但如果我们不知道规则呢？如果环境的动力学是未知的呢？比如 Atari 游戏从像素输入，你根本不知道游戏代码；或者真实世界，物理规则太复杂没法写出来。

这时候——你应该能猜到——我们需要学一个模型。但这个模型不是用来生成数据的，而是用来在隐藏空间里做规划的。这就是 MuZero 要做的事。

## MuZero 与 隐式模型学习

AlphaZero 需要知道游戏规则（状态转移、合法动作）才能做搜索。但现实世界里，大多数时候我们根本没有精确的环境模型——你怎么把"如果机器人手臂往前推 5 厘米，物体会怎么动"写成解析公式？MuZero（Schrittwieser et al. 2019）的关键创新就是：**不需要知道真实规则，我学一个隐式模型——把状态编码到隐藏空间，在隐藏空间里预测状态转移、奖励、策略和价值——然后直接在隐藏空间里做 MCTS 搜索**。

你可能会问："什么叫'在隐藏空间里搜索'？" 思路是这样的：

- 我不需要预测下一状态的像素或者物理状态——那些细节太多了，很多和决策无关；
- 我只需要学一个**表示网络**，把真实状态 $s$（不管是棋盘还是像素）编码成一个隐藏表示 $x = h(s)$，这个隐藏表示只要包含"做决策需要的所有信息"就行；
- 然后学一个**动力学网络**，在隐藏空间里做预测：给定当前隐藏状态 $x$ 和动作 $a$，预测下一隐藏状态 $x'$ 和即时奖励 $r$；
- 最后学一个**预测网络**，从隐藏状态直接输出策略和价值。

整个过程就是一个链式预测：

$$
s \xrightarrow{h} x_0 \xrightarrow{g} x_1 \xrightarrow{g} x_2 \to \ldots
$$

从真实状态 $s$ 编码到 $x_0$，然后在隐藏空间里一步一步用动力学网络 $g$ 推演，每一步都可以用预测网络 $f$ 看策略和价值。最妙的是：**整个隐藏空间的动力学不需要和真实世界的像素/物理状态一一对应，它只要能正确预测对决策重要的东西（奖励、价值、策略）就行**。

### MuZero 的三大网络

具体来说，MuZero 有三个核心网络，分工明确：

- **表示网络（Representation Network）** $h(s) \to x$：把真实状态 $s$（比如围棋棋盘、Atari 像素帧）编码到隐藏表示 $x$。这一步只在搜索根节点做一次——把当前真实状态"翻译"成模型能理解的隐藏语言。

- **动力学网络（Dynamics Network）** $g(x, a) \to x', r$：这是"隐式世界模型"本身。给定当前隐藏状态 $x$ 和动作 $a$，它预测两个东西：下一个隐藏状态 $x'$，以及**做这个动作会得到多少即时奖励** $r$。有了它，我们就可以在隐藏空间里"往前看"了。

- **预测网络（Prediction Network）** $f(x) \to p, v$：这和 AlphaZero 的策略+价值网络作用一样——给定隐藏状态 $x$，输出策略 $p$（动作先验）和价值 $v$（这个状态有多好）。搜索的时候，叶子节点用它评估，扩张子节点用它给先验。

我们来看代码理解这三个网络怎么配合做规划：

```python
class MuZero:
    def plan(self, state, n_simulations):
        # 1. 编码真实状态到隐藏空间（只做一次）
        root_hidden = self.representation(state)
        root_policy, root_value = self.prediction(root_hidden)

        # 2. MCTS 在隐藏空间搜索
        for _ in range(n_simulations):
            self._mcts_iteration(root_hidden)

        # 3. 返回根的动作分布（和 AlphaZero 一样）
        return root.action_distribution()

    def _mcts_iteration(self, root):
        # 在隐藏空间选择、扩张、回溯
        path = self._select_path(root)
        # 用动力学网络在隐藏空间前进一步
        next_hidden, reward = self.dynamics(path[-1].hidden, path[-1].action)
        # 用预测网络评估叶子
        policy, value = self.prediction(next_hidden)
        path[-1].expand(policy, reward)
        for node in path:
            node.update(value, reward)
```

对比 AlphaZero 的搜索代码，你发现区别了吗？整个 MCTS 搜索流程**几乎完全一样**——PUCT 选择、扩张、回溯——唯一的区别是：

- AlphaZero 从真实状态出发，用真实游戏规则做状态转移；
- MuZero 从真实状态出发，但**只在根节点用真实状态编码一次**，之后所有的状态转移都在隐藏空间里用动力学网络 $g$ 做预测。

而且训练过程也和 AlphaZero 类似——自我对弈，用 MCTS 结果作为策略标签，用最终胜负作为价值标签——只不过多了几个损失项：让动力学网络预测的奖励准确，让整个预测链在 rollout 时一致。

### MuZero 的意义

MuZero 的真正意义是什么？它**不需要知道环境规则也能学习规划**。它自己学规则——虽然学的是隐藏空间里的规则，但只要这个规则能正确预测奖励、价值、策略，就能用来做高质量的搜索。

这让 model-based RL 第一次能推广到：

- **Atari 游戏**：不需要访问模拟器代码，直接从像素输入学，在 57 个 Atari 游戏上达到了 model-free SOTA 性能。
- **棋盘游戏**：围棋、国际象棋、将棋——和 AlphaZero 一样强，但不需要预先写规则。
- **部分可观察环境**：比如扑克这类不完美信息游戏（虽然 MuZero 本身是为完美信息设计的，但思想可以扩展）。
- **任何 MDP**：理论上，只要你能把状态编码成向量，MuZero 就能在上面学模型、做搜索。

MuZero 可以说是 model-based RL 的"统一架构"探索——同一个算法思路、同一套网络结构，可以跨越视觉输入（Atari 像素）、矢量输入（棋盘）、离散动作，甚至（后来的扩展）连续动作。但是——MuZero 还是用了 MCTS 搜索，搜索时每一步还是要做很多次网络前向，计算成本不低。而且 MuZero 主要在游戏上验证，还没有到"通用开箱即用"的程度。

真正把世界模型推向通用、开箱即用的，是 Dreamer 系列，尤其是 Dreamer V3。

## Dreamer V3 与 世界模型的新世代

Dreamer 系列（Hafner et al. 2020-2023）是 model-based RL 的现代旗舰。如果说 AlphaZero/MuZero 是"搜索 + 模型"路线——规划的时候做大量搜索——那 Dreamer 就是"纯模型"路线：**学一个足够准确的循环隐变量世界模型，然后完全在这个模型里面"做梦"训练 actor-critic，训练时和真实环境零交互，只在收集数据的时候用真实环境**。

你可能会问："这和 MBPO 有什么区别？" MBPO 也用模型生成数据，但它是在模型 rollout 5 步后就回到真实数据混合训练；而且 MBPO 的模型是简单的前馈网络，没有考虑环境的部分可观察性和长期记忆。Dreamer 用的是**循环状态空间模型（Recurrent State-Space Model, RSSM）**——这是一个更强大的世界模型架构，能同时建模环境的确定性和随机性，甚至能处理部分可观察环境（POMDP）。

### 循环状态空间模型

为什么需要循环？为什么需要隐变量？让我们停下来想一下。真实世界是部分可观察的：你当前看到的一帧画面，不足以告诉你所有信息——比如你看不到物体的速度，看不到门后面有什么。要做准确的多步预测，你需要**记忆**——把过去的观察信息整合起来。RNN 的隐藏状态 $h_t$ 正好可以做这个。

但只有确定性的 RNN hidden state 还不够——环境本身有随机性（还记得 PETS 里讲的偶然不确定性吗？）。所以 RSSM 同时建模两部分：

- **确定性部分**：RNN 的隐藏状态 $h_t$，它随着时间确定性地更新，整合所有历史信息，形成"记忆"。
- **随机性部分**：隐变量 $z_t$，代表环境中无法从历史完全预测的随机因素。RSSM 学两个分布：
  - **后验分布** $q(z_t \mid h_t, o_t)$：看了当前观察 $o_t$ 之后，推断当前的随机隐变量是什么——这是"看了之后才知道"。
  - **先验分布** $p(\hat{z}_t \mid h_t)$：不看当前观察，只靠历史记忆预测 $z_t$ 应该是什么——这是"没看之前的预测"。

训练世界模型的时候，我们要让先验预测 $\hat{z}_t$ 尽量接近后验推断的 $z_t$——如果两者能很好地对齐，说明模型已经能准确预测接下来会发生什么，不需要看真实观察也能"想象"出合理的未来。

我们来看 RSSM 的代码结构：

```python
class RSSM:
    def forward(self, obs_seq, action_seq):
        h = zeros(batch, hidden_dim)
        posterior_zs = []
        prior_zs = []

        for t in range(T):
            # 先验：只看 h_t（历史记忆），预测 z_t
            prior_mean, prior_std = self.prior(h)
            prior_zs.append((prior_mean, prior_std))

            # 后验：看了 h_t 和当前观察 obs_t，推断 z_t
            posterior_mean, posterior_std = self.posterior(h, encoder(obs_seq[t]))
            z = reparameterize(posterior_mean, posterior_std)
            posterior_zs.append((posterior_mean, posterior_std))

            # 用 z 和动作 a_t 更新 RNN 确定性隐藏状态 h_t → h_{t+1}
            h = self.rnn(h, z, action_seq[t])

        return prior_zs, posterior_zs
```

这个流程要仔细看：

1. 每一步我们有两个 $z$：先验（只靠记忆预测）和后验（看了观察修正）。训练时让两者靠近（用 KL 散度损失）。
2. RNN 状态 $h$ 的更新用的是**后验的 $z$**——因为训练时我们有真实观察，所以用更准确的后验来刷新记忆。
3. 但做想象 rollout 的时候——也就是在世界模型里"做梦"的时候——我们**没有真实观察**，这时候就只能用先验来采样 $z$，然后用它更新 RNN，继续往前预测。

如果先验足够准确——不看观察也能预测出合理的 $z$——那么想象 rollout 出来的轨迹就会和真实环境的轨迹很像，我们完全可以在上面训练策略。

### Actor-Critic in Imagination

有了训练好的世界模型，怎么训练策略？Dreamer 的做法非常直接：**完全在世界模型里训练 actor-critic，不需要和真实环境有任何交互**。

流程是这样的：

1. 先用之前的真实交互数据（存在 dataset 里），用 RSSM 编码出初始的隐藏状态（包括确定性 $h$ 和随机 $z$）；
2. 从这个初始状态出发，用当前 actor 选动作，用世界模型 rollout H 步（通常 H=15）——完全在想象空间里，不需要真实环境；
3. 把想象出来的轨迹存下来，在上面像常规 actor-critic 一样计算价值损失、策略损失，更新 actor 和 critic。

我们来看伪代码：

```python
# 在世界模型里"做梦"训练
h = world_model.encode(real_observation_sequence)
for t in range(H):  # H = 15 想象 horizon
    a = actor(h)
    h, r = world_model.predict(h, a)
    imagined_trajectory.append((h, a, r))

# 在想象轨迹上训练 actor-critic
for (h, a, r) in imagined_trajectory:
    critic_loss = ...
    actor_loss = ...
```

等等——你可能会问——这和 Dyna 有什么区别？Dyna 也是在模型里生成数据然后训练啊。区别在于 Dreamer 的世界模型强太多了：

- RSSM 有 RNN 记忆，能处理部分可观察环境，做长 horizon 预测更准确；
- 隐变量建模了环境随机性，不是简单的确定性预测；
- Dreamer 直接在**隐空间**里训练 actor-critic——actor 的输入不是原始观察，而是世界模型的隐藏状态 $h$，这让策略学习更容易（隐藏状态已经是高维特征了）；
- 想象 rollout 可以做 15 步甚至更长——因为世界模型足够准确，误差不会像简单前馈模型那样爆炸。

整个训练循环是：**用真实环境收集少量数据 → 用数据训练世界模型 → 在世界模型里做大量想象训练 → 用训好的策略再去真实环境收集更多数据 → 重复**。收集真实数据是为了让世界模型见过更多场景、更准确；但策略梯度更新——那些需要大量样本的更新——完全在想象里做，几乎零成本。

### Dreamer V3 的统一性

Dreamer V1（2020）已经在很多连续控制任务上工作得很好了，但还是要针对不同领域调超参数。Dreamer V2（2021）引入了离散隐变量，在 Atari 上达到了 model-free SOTA。而 **Dreamer V3（Hafner et al. 2023）** 真正做到了里程碑式的突破：**单一超参数设置，在跨越 150+ 个不同领域的任务上，都达到了或者超过了 model-free SOTA**。

这些任务包括：

- **Atari**：离散动作 + 视觉像素输入（26 个游戏）
- **MuJoCo**：连续动作 + 矢量状态输入（标准连续控制基准）
- **Crafter**：开放世界生存游戏（视觉输入、稀疏奖励、复杂任务链）
- **DMLab**：第一人称 3D 导航（视觉输入、第一人称视角）
- **BSuite**：一套认知任务，专门测试 RL 算法的各种核心能力

这是什么概念？在此之前，没有任何一个 RL 算法——不管是 model-free 还是 model-based——能做到"一套超参数通吃所有这些领域"。你要在 Atari 上做得好，得用 Atari 特有的超参数；要在 MuJoCo 上做得好，得用 MuJoCo 特有的超参数。Dreamer V3 第一次证明了：世界模型路线可以做到**通用**。

那 Dreamer V3 到底做对了什么，突然就这么通用了？论文里总结了三个关键的工程创新——说穿了也都是工程细节，但组合在一起效果惊人。

### 三个关键工程创新

1. **离散化隐变量（Discrete Latents）**。Dreamer V1/V2 已经用了离散隐变量，但 Dreamer V3 把这个做到位：把连续的高斯隐变量 $z$ 改成 categorical（类别型）分布——类似 VQ-VAE 的思想。为什么这么做？因为离散隐变量的训练更稳定，不会出现高斯分布方差坍缩的问题，而且更容易建模多模态的未来（同一状态下可能有多种合理的未来）。

2. **Symlog 损失**。不同任务的奖励尺度差别太大了：MuJoCo 里奖励可能是 0-10，Atari 里得分可能是 0-1000，Crafter 里奖励非常稀疏。价值函数如果直接拟合原始尺度，大奖励任务的梯度会淹没小奖励任务。Dreamer V3 用了一个简单但极其有效的变换——symlog：

$$
\text{symlog}(x) = \text{sign}(x) \log(|x| + 1)
$$

这个函数长什么样？它对正数是 $\log(x+1)$，对负数是 $-\log(-x+1)$，在 0 附近近似线性，在大绝对值处是对数增长——相当于自动压缩了大值的范围，同时保持符号。价值网络在 symlog 空间里预测，输出再用 symexp（symlog 的逆变换）映射回来，这样不管奖励尺度多大，都能稳定学习。

3. **不使用 KL 退火（No KL Annealing）**。训练隐变量模型的时候，通常会用 KL 退火——一开始 KL 项权重小，让模型先学会重建观察，然后慢慢增大 KL 权重，让先验后验对齐。Dreamer V3 发现这完全没必要——直接固定 KL 权重为 1，直接最大化 ELBO（证据下界），让后验自然匹配先验。去掉 KL 退火这个超参数，大大提升了鲁棒性。

就这三招——离散隐变量、symlog 损失、固定 KL 权重——让 Dreamer V3 能在 150+ 任务上"开箱即用"，不需要针对每个任务调参。这在 RL 历史上是第一次。

::: details 加餐：Dreamer V3 的世界模型损失函数
Dreamer V3 训练世界模型时，是在最大化 ELBO，它的损失函数可以拆成三项：

$$
\mathcal{L}_{\text{world model}} = \mathcal{L}_{\text{rec}} + \beta_{\text{KL}} \mathcal{L}_{\text{KL}} + \mathcal{L}_{\text{reward}}
$$

- $\mathcal{L}_{\text{rec}}$：重建损失——从隐变量解码回观察（图像或状态），让隐变量包含足够信息重建真实输入。对于视觉输入用 MSE 或离散熵，对于矢量输入用 MSE。
- $\mathcal{L}_{\text{KL}}$：KL 散度——让后验 $q(z_t|h_t,o_t)$ 靠近先验 $p(z_t|h_t)$，这样想象时用先验采样的 $z$ 才合理。Dreamer V3 里 $\beta_{\text{KL}}=1$，固定不调。
- $\mathcal{L}_{\text{reward}}$：奖励预测损失——让模型预测的即时奖励和真实奖励对齐。

三项加起来端到端训练，整个世界模型就学会了"看了之后知道怎么回事，不看的时候也能预测未来"。
:::

Dreamer V3 的成功让很多人相信：**世界模型是通往通用强化学习的一条可行路径**。你学一个足够好的世界模型，然后在模型里训练策略——就像人一样，先在脑子里建立对世界的理解，然后在脑子里推演、练习，再去真实世界行动。

讲完了 AlphaZero → MuZero → Dreamer V3 这三条技术线，我们现在可以回过头来，从更宏观的角度回答一个问题：**Model-Based 和 Model-Free 到底各有什么优劣？实战中我该选哪个？**

## Model-Based vs Model-Free 与 何时用哪个

我们把 model-free 和 model-based 放在一张表里对比，从多个维度看它们的权衡：

| 维度         | Model-Free       | Model-Based                |
| ------------ | ---------------- | -------------------------- |
| **样本效率** | 低（百万步）     | 高（万步）                 |
| **渐进性能** | 高               | 受模型误差限制             |
| **计算成本** | 低（直接用数据） | 高（训练模型 + 搜索/规划） |
| **可解释性** | 黑盒             | 模型可分析                 |
| **迁移能力** | 弱               | 模型可迁移到下游任务       |
| **超参敏感** | 中               | 高（模型质量决定一切）     |

让我们逐个解释这几个维度：

- **样本效率**：这是 model-based 最大的优势。正如我们前面算过，真实样本昂贵的场景，model-based 能把交互次数降低 10-100 倍。
- **渐进性能**：当你有无限多样本的时候，model-free 通常最终性能更高——因为它没有模型偏差。如果模型不是完美的（现实中永远不是），model-based 最终会被模型误差卡住天花板；而 model-free 只要给足够多样本，能一直优化到最优策略。
- **计算成本**：这是 model-based 的代价——你不仅要训练策略，还要训练模型；如果是 AlphaZero/MuZero 那种搜索路线，部署时每一步还要做大量计算。Model-free 训练完之后，部署就是一次网络前向，非常快。
- **可解释性**：模型本身是对环境动力学的理解，你可以查看模型预测了什么、哪里预测不准、模型对哪里不确定。Model-free 的策略网络和 Q 网络基本上是黑盒，很难解释为什么它选这个动作。
- **迁移能力**：如果你已经在一个环境上学好了世界模型，换个新任务（换个奖励函数），你不需要重新学动力学——直接用现成的模型在新奖励下训策略就行。Model-free 每次换任务都要从头开始重新训。
- **超参数敏感性**：Model-based 对超参数更敏感——尤其是模型容量、模型训练频率、rollout 长度这些和模型相关的超参数。模型学坏了，一切都白搭。Model-free 虽然也有超参数，但相对更鲁棒一些。

那实战中到底怎么选？

**何时选 Model-Free：**

- **仿真器很便宜**：比如 Atari、MuJoCo、StarCraft 这类，一秒钟能跑几千步，样本几乎是免费的——这时候不需要省样本，直接上 SAC/PPO 简单粗暴有效。
- **只关心最终性能**：不限制训练时间、不限制样本数，只要最终策略尽可能强——model-free 的渐进性能通常更高。
- **部署时计算资源有限**：机器人、手机、边缘设备上部署，没有额外算力跑模型和搜索——model-free 策略网络一次前向就出动作，延迟最低。

**何时选 Model-Based：**

- **真实环境采样昂贵**：这是最典型的场景——真实机器人、自动驾驶、化学反应控制、工业流程优化——这些场景每一次真实交互都有时间、金钱、安全成本，model-based 的 10-100 倍样本效率是"能用"和"不能用"的区别。
- **需要快速适应新任务**：Meta-RL、在线学习、持续学习——你已经有一个预训练好的世界模型，新任务直接在模型里规划/训练，几秒钟就能适应，不需要重新和环境交互百万步。
- **需要可解释性和安全性**：安全关键场景（比如自动驾驶、医疗），你想知道智能体"为什么"做这个决策，想预测它接下来会做什么，想在部署前验证策略不会出问题——模型提供了可分析的内部表示。

这不是非此即彼的选择。实战中你也可以混合：比如先在世界模型里预训练策略，再用少量真实样本做 model-free 微调；或者用 model-based 做高层规划，用 model-free 做底层控制——就像人一样，大脑里有世界模型做高层思考，但很多低级运动控制是反射式的（model-free）。

## 与 LLM RL 的连接

如果你一直在做 LLM 相关的工作，你可能会觉得这些东西有点熟悉——没错，今天 LLM 训练和推理里的很多核心思想，都能在 model-based RL 里找到源头。理解了这一节讲的搜索、世界模型、模型偏差这些概念，你就能理解为什么 LLM 领域最近的一些方向看起来那么自然。

LLM 训练和推理中：

- **Model-Free 路线**：就是大家熟悉的 RLHF、GRPO——直接用奖励模型（RM）给整个回答打分，然后用 PPO/GRPO 这类策略梯度算法优化语言模型。这是纯 model-free 的——语言模型本身就是策略，奖励模型就是奖励信号，中间没有"世界模型"。

- **Model-Based / 搜索路线**：**Process Reward Model（PRM）** 和推理时搜索（比如 MCTS、beam search、best-of-n）就是 model-based 思想的直接体现——PRM 就像 AlphaZero 的价值/策略网络，给推理链的**每一步**打分；然后用搜索算法（比如 MCTS）在"可能的推理步骤空间"里搜索更好的推理链，而不是直接一步输出答案。这和 AlphaZero 里"MCTS 用网络引导搜索，搜索结果比网络本身更强"是一模一样的逻辑——我们会在 [第 17 章 PRM 与搜索](../chapter20_prm_search/inference-time-search) 深入讲这个。

- **World Model 路线**：最近的 **Code World Model**、**SWE-Agent 里的世界模型组件**（[第 20 章](../chapter23_rl_based_swe/world-model-and-deep-swe)）就是在 LLM 领域重建 MuZero/Dreamer 的思路——学一个"代码世界模型"，预测"如果我做这个编辑、执行这段代码，会发生什么、测试会不会过"，然后在这个世界模型里搜索/规划补丁方案，提升解决代码问题的样本效率。

你看，Tongyi DeepResearch、OpenAI o1/o3 这类推理模型用 PRM 引导搜索、SWE-Agent 这类代码 Agent 用世界模型提升效率——这些都不是凭空冒出来的想法，它们都是 model-based RL 核心思想在 LLM 时代的自然延伸。理解了 model-based 和 model-free 的根本权衡——样本效率 vs 渐进性能、计算成本 vs 决策质量、模型偏差 vs 无偏但低效——你就能更本质地理解 LLM 后训练和推理的各种技术路线为什么是现在这个样子。

## 本章总结

连续控制和 model-based RL 是经典深度强化学习的两大进阶方向。这一章我们走完了三条主要的技术演进路线：

1. **DDPG → TD3 → SAC**：确定性策略梯度的演进，解决连续动作 off-policy 学习问题。从 DDPG 的开创性工作，到 TD3 用三个工程补丁（双 Q、延迟更新、目标平滑）稳定训练，再到 SAC 用最大熵 RL 从根本上重构目标函数——随机策略内置探索、自动温度调节、开箱即用，SAC 成为今天连续控制的首选算法。

2. **Dyna → PETS → MBPO**：model-based 数据增强路线的演进。从 Dyna 最简单的"模型当额外数据生成器"，到 PETS 用概率集成建模两种不确定性 + CEM MPC 规划，再到 MBPO 用短 horizon rollout 巧妙绕开误差累积——这条线把模型作为"数据放大器"，最终还是用 model-free 方式训练策略，但样本效率提升 10-100 倍。

3. **AlphaZero → MuZero → Dreamer V3**：显式搜索 + 学习模型的旗舰路线。AlphaZero 把 MCTS 和神经网络完美结合，通过自我对弈零人类知识达到超人类棋力；MuZero 把这个推进到隐空间，不需要知道环境规则也能学规划；Dreamer V3 用循环隐变量世界模型，第一次做到单一超参数设置在 150+ 任务上通用——这代表了 model-based RL 的天花板，也直接启发了 LLM 时代的 PRM 搜索和世界模型工作。

走完这一章，你应该对"如何做连续控制"、"如何用模型提升样本效率"有了完整的图景。但我们一直有一个隐含假设：**智能体可以和环境自由交互，想采多少数据就采多少数据**。

现实中是这样吗？不是。很多场景——推荐系统、医疗、LLM 后训练——你根本没法在线和环境"交互探索"。你有一堆历史数据，但不能随便上线试错；或者试错成本极高。这时候怎么办？这就是下一章要讲的核心问题：**离线强化学习（Offline RL）**——当智能体不能交互，只能用预先收集好的历史数据学习时，怎么学到好策略？这是 LLM 后训练（DPO、RLHF 里的离线阶段）、推荐系统、医疗 AI 等真实场景的核心问题。

下一章 [第 10 章 离线强化学习](../chapter12_offline_rl/offline-data-distribution-shift)，我们来回答这个问题。

## 延伸阅读

- [Silver et al. 2018 "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)
- [Schrittwieser et al. 2020 "Mastering Atari, Go, chess and shogi by planning with a learned model" (MuZero)](https://arxiv.org/abs/1911.08265)
- [Hafner et al. 2023 "Mastering Diverse Domains through World Models" (Dreamer V3)](https://arxiv.org/abs/2301.04104)
- [Janner et al. 2019 "When to Trust Your Model: Model-Based Policy Optimization" (MBPO)](https://arxiv.org/abs/1906.08253)
