# 18.4 分布式 RL 训练

> **本节目标**：理解训推不一致的来源与修正方法，掌握 veRL 怎样用 ResourcePool、Worker 和 Driver 安排多个模型与 GPU，了解生成吞吐与训练显存的常用技术，并判断什么时候需要改用异步调度。

上一节排查训练失稳时，最后一步是核对生成端与训练端的模型版本和概率。那套排查依赖一个前提：生成、打分、更新都发生在同一个脚本里，三个先后调用的函数，版本不会错乱。这一节把训练搬上集群，这个前提不再成立。

训练一旦分到多组机器上，立刻会遇到一个具体的问题。比如训练一个 70B 的数学模型，加上 Reference 和奖励模型，参数有上千亿，一组 GPU 根本放不下：生成只能放在一组卡上，更新放在另一组卡上。

训练一分开，一批回答就要在两组 GPU 之间转成一个循环。

第一步是生成。rollout GPU 根据当前策略生成一批回答，生成一条通常要花几秒钟。

第二步是打分。奖励进程读走回答，调用验证器或奖励模型算出分数。

第三步是更新。训练 GPU 拿到分数计算梯度、更新 Actor，这一步只要几百毫秒。

最后是同步。更新结束，新的参数必须立刻送回 rollout GPU。

在单机代码里，这只是几个先后调用的函数；到了多机系统，每一步都变成不同进程之间的数据传输和等待。

参数送回得慢，下一批回答还会从旧策略生成，上一节刚建立的一致性又被破坏。这一节要解决的就是怎样安排这个循环，依次回答四个问题：生成端和训练端的概率差异从哪里来，多个模型怎样安排到 GPU 上，生成速度和训练显存两个瓶颈先解决哪个，以及什么时候用异步调度换取同步等待。模型换成 MoE 以后，这些安排还要再加上专家路由和流水线两道题。

::: tip 第一次阅读到这里即可
记住数据顺序：**生成 → 奖励 → 更新 → 同步参数**。veRL、slime 和 OpenRLHF 的实现不同，都在安排这四步使用哪些 GPU、何时交换数据。
:::

---

## 训推不一致从哪里来

参数同步解决了版本问题：rollout GPU 和训练 GPU 拿到的是同一组权重。但同一组权重，不等于同一组概率。

设想 rollout 侧的 vLLM 用 FP8 算出某个 token 的概率是 0.30；训练端拿同一组权重重算这个 token，用的是 BF16 精度和另一套算子，得到 0.29。两次计算都"没错"，结果却差了一截。原因在于两边走的是完全不同的计算路径：推理侧通常使用 vLLM 或 SGLang，配合 KV Cache 和低精度计算，一切为生成速度服务；训练侧常用 FSDP 或 Megatron，还要保留反向传播所需的计算图。

把 rollout 引擎实际执行的策略记为 $\pi_{\text{rollout}}$，把训练端记录的旧策略记为 $\pi_{\text{old}}$。二者本该相同；浮点精度、算子实现和 MoE 路由的误差，都会让它们产生偏差。这就是**训推不一致**（Training-Inference Mismatch）。

**生成策略与训练策略的偏差来源**：

- rollout 侧通常使用 vLLM 或 SGLang，以 FP8/BF16 生成回答，并启用 KV Cache 优化。
- 训练侧通常使用 FSDP 或 Megatron，以 BF16/FP32 计算 log-probability 和梯度，并可能启用激活重计算。

知道了偏差从哪里来，再看它进到训练里会去哪里。高概率 token 的微小误差通常影响有限；低概率 token 数量多，误差累积后可能明显改变梯度估计。更直接的去处，是 PPO 损失函数里的重要性采样比率。

### 为什么 PPO Clipping 无法修正训推偏差

PPO 使用下面的重要性采样比率限制一次更新的幅度：

$$
\mathcal{L}^{\text{CLIP}} = \mathbb{E}\left[\min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)\right],
$$

其中

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}.
$$

$\hat A_t$ 表示动作 $a_t$ 比当前平均水平好多少，$\epsilon$ 决定允许比率偏离 1 的范围。假设旧策略生成某个 token 的概率是 $0.20$，新策略把它提高到 $0.24$，那么 $r_t=0.24/0.20=1.2$。当 $\epsilon=0.2$ 时，这次变化刚好到达 clipping 的上边界。

这个计算有一个前提：分母 $\pi_{\text{old}}$ 必须是生成动作时真正使用的策略。如果回答来自 $\pi_{\text{rollout}}$，训练端却用另一条计算路径重算 $\pi_{\text{old}}$，那么 $r_t$ 在更新开始以前就已经不准确。clipping 只能限制参数更新，不能修正两个引擎算出的概率差异。

**训推偏差的排查顺序**：

1. 核对 rollout 使用的模型版本，确认参数同步已经完成。
2. 用同一批 token 分别记录 rollout 侧和训练侧的 log-probability，观察误差集中在哪些位置。
3. 对齐两侧的浮点精度和算子实现，再比较误差是否缩小。
4. 对 MoE 模型额外记录专家路由，确认训练端是否复现了生成时的路由。

### 训推偏差的修正方法

- **统一计算精度**：先用 FP16/BF16 替代 rollout 侧的 FP8，判断低精度计算是否是主要误差来源。需要继续使用 FP8 时，应同时加入偏差监控和重要性采样修正。
- **记录真实行为策略**：直接保存 rollout 时的 log-probability，避免训练端把重新计算的结果当作真实行为概率。
- **重新计算并校验**：训练前用训练引擎重算 log-probability，并与 rollout 记录逐 token 对比。重算本身不能恢复真实行为策略，但能暴露差异的位置和大小。
- **限制极端比率**：Truncated IS（TIS）等方法会截断过大的重要性采样比率，降低少量异常 token 对梯度的影响。
- **处理长尾 token**：动态词表剪枝等方法会过滤偏差最大的低概率区域，减少误差在长序列中的累积。
- **回放 MoE 路由**：R3（Rollout Routing Replay）在训练时复现 rollout 的专家选择，减少路由变化造成的概率偏差。

工程中的 On-policy 程度取决于 $\pi_{\text{rollout}}$ 与当前训练策略之间的距离。参数同步控制模型版本，精度对齐、概率记录和重要性采样修正继续控制计算路径带来的偏差。

---

## 模型与 GPU 的资源安排

概率对上以后，下一个问题落在资源上：Actor、Critic、Reference Model、Reward Model 和 rollout 引擎，五个模型要放在哪些 GPU 上，彼此怎样传递数据。veRL（Volcano Engine Reinforcement Learning）把这个问题拆成算法主循环、模型计算和资源分配三个层级。

HybridFlow 的核心设计是 single-controller 多模型编排：一个 Driver 作为单控制器运行算法主循环和资源调度，指挥分设在各个 GPU 上的 Worker 完成具体计算。Worker 按角色分为 Actor（FSDP 训练）、Critic（FSDP 训练）、Reference（冻结模型）、Reward Model 和 Rollout Engine（vLLM/SGLang 推理）五类，共享同一个 ResourcePool（GPU 集合）。

### 三个核心抽象

**ResourcePool：GPU 资源分组**。把 GPU 分组，每组可以放一个或多个模型。不同模型可以共享 GPU（colocate）或独占 GPU（disaggregated）。例如 Actor 和 Rollout 可以共享同一组 GPU，也可以分开设池。

**Worker：模型实例封装**。每个 Worker 是一个独立的模型实例，封装了具体的训练/推理逻辑。ActorWorker 负责 loss 计算、反向传播和优化器更新；RolloutWorker 负责批量生成和权重同步。

**Driver：单控制器编排**。Driver 是 RL 算法的主循环，顺序执行：同步权重给 Rollout → 采样 responses → 用 Reward Model 算分 → 用 Critic 算 value → 算 advantage 更新 Actor → 更新 Critic。

**HybridFlow 的混合并行策略**：这里的 Hybrid 指统一的混合并行策略，即在同一个框架内可以组合 3D Parallelism（TP×PP×DP）、Colocate vs Disaggregated、多种训练后端（FSDP、Megatron、DeepSpeed ZeRO）和多种推理后端（vLLM、SGLang、HuggingFace generate）。

### 主流框架架构对比

| 框架                                                   | 编排模式                            | 训练后端                                                                                                       | 推理后端                                                                                          | 典型规模   | 代表使用者               | 适用场景                         |
| ------------------------------------------------------ | ----------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------- | ------------------------ | -------------------------------- |
| [veRL (HybridFlow)](https://arxiv.org/abs/2409.19256)  | Single-controller                   | FSDP、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)、[DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) | [vLLM](https://arxiv.org/abs/2309.06180)、[SGLang](https://arxiv.org/abs/2312.07104)、HF generate | 8–1024 GPU | Qwen、DeepSeek、字节跳动 | 大规模生产训练、需要灵活资源组合 |
| [OpenRLHF](https://arxiv.org/abs/2405.11143)           | Single-controller（Ray Actor 隔离） | FSDP、DeepSpeed                                                                                                | vLLM                                                                                              | 8–256 GPU  | 社区、研究团队           | 研究实验、中等规模训练           |
| [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner) | Multi-controller                    | [Megatron](https://github.com/NVIDIA/Megatron-LM)                                                              | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)                                            | 8–512 GPU  | NVIDIA 生态、企业集群    | 已采用 NVIDIA NeMo 栈的生产环境  |
| [TRL](https://github.com/huggingface/trl)              | Single-process                      | [HuggingFace Accelerate](https://huggingface.co/docs/accelerate)                                               | HF generate                                                                                       | 1–8 GPU    | 入门学习、快速原型       | 学习算法、小规模实验验证         |

选型可以从当前规模和已有技术栈出发：学习和原型用 TRL；研究和中等规模用 OpenRLHF 或 veRL；大规模生产用 veRL 或 NeMo-Aligner（看硬件栈）。

---

## 生成速度与训练显存

模型和 GPU 安排妥当以后，两个最容易被卡住的环节浮出来。还是看那个 70B 的训练：一批回答的生成时间往往占掉整轮训练的大半，训练端还要为权重、梯度、优化器状态和激活腾出空间。这一节先解决前者，看 rollout 引擎怎样提高生成吞吐；再解决后者，看训练显存怎样分摊到多张卡上。

### vLLM 的核心优化技术

| 技术                                               | 解决的问题                                                     | 核心原理                                                                                | 典型收益                                                    | GRPO 中的重要性                  |
| -------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------- | -------------------------------- |
| [PagedAttention](https://arxiv.org/abs/2309.06180) | KV cache 连续分配导致显存碎片、利用率低（50-70%）              | 借鉴操作系统虚拟内存分页，把 KV cache 分成固定大小 block，按需分配与回收                | 显存利用率提升至 95%+，有效 batch size 提升 2–4 倍          | 基础优化，所有场景必需           |
| Continuous Batching                                | 传统 static batching 要等整批序列全部生成完才换，导致 GPU 空等 | 某条序列生成 EOS 后立刻在同一 iteration 填入新序列，实现迭代级动态调度                  | 整体生成吞吐提升 5–10 倍                                    | 长回答场景收益最大               |
| Speculative Decoding                               | 自回归生成逐 token 解码，计算密度低                            | 用小模型（draft model）先预测多个 token，大模型并行验证，accept 匹配的、reject 后重采样 | 典型 LLM 推理吞吐提升 2–3 倍                                | 短回答、对延迟敏感场景           |
| Prefix Caching                                     | 同一 prompt 重复生成时，前缀 KV 重复计算浪费                   | 对 prompt 部分的 KV cache 做哈希复用，相同前缀直接命中缓存                              | GRPO 中同一 prompt 生成 $G=8$ 条回答时，前缀计算节省 70–80% | **GRPO 核心优化**，veRL 默认启用 |

### SGLang 的生成与调度优化

[SGLang](https://arxiv.org/abs/2312.07104) 由 LMSYS 团队开发，在 agentic 场景下比 vLLM 更快：RadixAttention 用基数树管理 KV cache 支持跨请求复用，Programmatic Frontend 支持复杂的控制流（多轮调用、分支、循环），Constrained Decoding 内置 JSON、regex 约束生成。

工业实践中推理引擎选择：

| 引擎                                                   | 核心优势                                     | 最适合场景                                               |
| ------------------------------------------------------ | -------------------------------------------- | -------------------------------------------------------- |
| [vLLM](https://arxiv.org/abs/2309.06180)               | PagedAttention、Continuous Batching 生态成熟 | 通用 rollout、单轮生成、GRPO 数学/代码训练               |
| [SGLang](https://arxiv.org/abs/2312.07104)             | RadixAttention、多轮控制流、结构化输出       | Agentic rollout、多轮工具调用、需要 constrained decoding |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA GPU 深度优化、FP8 支持最好            | NVIDIA 硬件栈上的最高吞吐生产部署                        |

### 多 GPU 显存分摊技术

生成速度解决以后，训练端仍要容纳权重、梯度、优化器状态和激活。一个 70B 模型的 BF16 全参数训练远超单张 80GB H100 的容量。

**训练显存的构成**：以常见的 BF16 权重与梯度、FP32 主权重和 Adam 一阶、二阶动量为例，每个参数大约需要 16 字节：

| 组件                          | 数据类型  | 单参数字节数   | 70B 模型占用 | 说明                          |
| ----------------------------- | --------- | -------------- | ------------ | ----------------------------- |
| 模型权重（Weight）            | BF16/FP16 | 2 B            | 140 GB       | 训练时需要保留当前参数        |
| 梯度（Gradient）              | BF16/FP16 | 2 B            | 140 GB       | 反向传播后累积                |
| 优化器主权重（Master Weight） | FP32      | 4 B            | 280 GB       | Adam 需要 FP32 副本做稳定更新 |
| Adam 一阶动量（m）            | FP32      | 4 B            | 280 GB       | 梯度的指数移动平均            |
| Adam 二阶动量（v）            | FP32      | 4 B            | 280 GB       | 梯度平方的指数移动平均        |
| 激活值（Activation）          | BF16/FP16 | 动态           | ~100 GB      | 与 batch size、序列长度正相关 |
| **合计（全参数训练）**        | -         | **16 B/param** | **~1.22 TB** | 远超单张 80GB H100 容量       |

**ZeRO：零冗余优化器级别对比**。[DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) 按分片内容分为三个级别：

| ZeRO 级别 | 分片 Optimizer State | 分片 Gradient | 分片 Weight | 单卡显存节省倍数  | 通信开销 | 适用场景                              |
| --------- | -------------------- | ------------- | ----------- | ----------------- | -------- | ------------------------------------- |
| ZeRO-1    | ✅                   | ❌            | ❌          | ~4×               | 低       | 中小模型、通信受限场景                |
| ZeRO-2    | ✅                   | ✅            | ❌          | ~8×               | 中       | 大多数训练场景的默认选择              |
| ZeRO-3    | ✅                   | ✅            | ✅          | $N$×（$N$=GPU数） | 高       | 大模型全参数训练，需要额外 all-gather |

ZeRO-3 把权重也切分到各 GPU，每个 GPU 只存 $1/N$ 的权重，但前向反向传播时需要通过 all-gather 临时聚合所需参数。

[FSDP（Fully Sharded Data Parallel）](https://pytorch.org/docs/stable/fsdp.html) 是 PyTorch 原生实现，等价于 ZeRO-3，与 PyTorch 生态兼容性更好，是 veRL 默认的训练后端。

**Gradient Checkpointing：梯度检查点**。这种做法用计算换显存：前向时不保存全部中间激活，反向传播时重新计算所需激活。激活显存从 $O(L)$ 降到 $O(\sqrt{L})$（$L$ 是 Transformer 层数），代价是训练速度降低 20–30%。

**显存优化方案组合**（以 70B 模型、单张 80GB H100 为例）：

| 方案组合                               | 单卡显存需求 | 训练速度                    | 可行性                  |
| -------------------------------------- | ------------ | --------------------------- | ----------------------- |
| 全参数 + Adam（不做任何分片）          | ~940 GB      | 基准（最快）                | ❌ 不可行，远超单卡容量 |
| ZeRO-3（仅分片训练状态）               | ~118 GB      | 比基准慢 10–15%（通信开销） | ❌ 单卡仍然 OOM         |
| ZeRO-3 + Gradient Checkpointing        | ~30 GB       | 比基准慢 30–40%             | ✅ 可行                 |
| ZeRO-3 + Gradient Checkpointing + LoRA | ~8 GB        | 比基准慢 ~40%（但参数量少） | ✅ 最快的工业方案       |

工业级 70B RL 训练通常使用 LoRA + FSDP 的组合，在显存、速度和训练效果之间取得平衡。

---

## 任务耗时差别变大以后：异步调度

前面的安排都隐含了一个假设：一批 rollout 里各条轨迹的耗时接近。数学题和代码题大致如此，每道题解起来花的时间差不多，等最慢的一条结束再更新，损失不大。任务换成工具调用和浏览器操作以后，有的轨迹还在等环境返回，有的早已结束，耗时相差几十倍，同步训练会让训练 GPU 一直等最慢的那条。

异步训练让生成和更新各自推进，通过队列交换轨迹。代价随之而来：轨迹生成期间，策略可能已经更新过多次，拿到手里的轨迹已经陈旧，需要修正。下面三个框架走了三条不同的路线。

### 三种异步框架对比

| 框架                                        | 发布方                | 核心设计                                                                                                                      | 陈旧度处理                                                                                                 | 典型规模   | 公开加速比                          | 代表场景                                       |
| ------------------------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------- | ----------------------------------- | ---------------------------------------------- |
| [LlamaRL](https://arxiv.org/abs/2505.24034) | Meta（2025）          | 去中心化，无主节点；Rollout worker 持续取 prompt 生成，Train worker 持续取 batch 更新，权重异步广播                           | 不做显式重要性修正，靠持续版本更新覆盖                                                                     | 4096+ GPU  | Llama-3-70B GRPO 比同步快 **10.4×** | 超大规模同步瓶颈明显的推理任务                 |
| [AReaL](https://arxiv.org/abs/2505.24298)   | 蚂蚁集团+清华（2025） | 完全异步 rollout，每条轨迹记录生成时的策略版本和 logprob                                                                      | 显式计算 token-level 重要性权重 $\exp(\text{current\_logprob} - \text{gen\_logprob})$，截断到 $[0.8, 1.2]$ | 1024 GPU   | 671B MoE GRPO 比同步快 **2.77×**    | MoE 模型、需要显式控制偏差的任务               |
| [AgentRL](https://arxiv.org/abs/2510.04206) | THUDM/智谱（2025）    | 异步生成训练流水线 + 统一环境接口；训练侧分 rollout/Actor/Reference worker 池，环境侧通过 Controller/Task Worker 管理异构任务 | 异步队列 + 任务隔离，多轮环境会话单独维护                                                                  | 多机多环境 | 支撑 AutoGLM 训练                   | 多轮 Agent（SWE、Computer Use、Deep Research） |

三个框架代表处理陈旧轨迹的三条路线。

[LlamaRL](https://arxiv.org/abs/2505.24034) 不做显式修正：去中心化架构没有主节点，rollout worker 持续从队列取 prompt 生成回答，train worker 持续取 batch 更新，权重异步广播，靠持续的版本更新覆盖陈旧样本。没有单点故障，容易横向扩展到上万张卡。

[AReaL](https://arxiv.org/abs/2505.24298) 选择显式修正：每条轨迹记录生成时的策略版本和 logprob，训练端计算重要性权重 $\exp(\text{current\_logprob} - \text{gen\_logprob})$ 并截断到 $[0.8, 1.2]$，避免旧样本造成过大梯度。偏差被控制在可计算的范围里，代价是每条轨迹都要多存一份元数据。

[AgentRL](https://arxiv.org/abs/2510.04206) 面对的是第三种情况：轨迹里夹着多轮环境交互，等待环境返回的时间可能比模型生成还长。它把异步流水线和统一环境接口放在一起：训练侧维护 rollout、Actor 和 Reference 三类 worker 池，环境侧用函数调用接口、容器、Controller 和 Task Worker 管理异构任务，支撑了 AutoGLM 的训练。

---

## MoE 与流水线空闲

前面的安排还默认了两件事：模型是 Dense 架构，多卡流水线已经排满。模型换成 MoE、轨迹长短差距拉大以后，这两点都不再成立：MoE 带来专家路由与跨卡通信问题，多卡流水线也不会自动保持忙碌。

### MoE 带来的额外系统复杂度

DeepSeek V3、Qwen3 和 GLM-4.5 都采用 MoE 架构。每个 token 只激活少量专家，参数能够分散到更多 GPU；与此同时，RL 的样本分布会改变专家负载，训练系统还要记录路由和通信状态。

以 DeepSeek V3 为例：Dense 部分（attention 等）约 20B 参数，MoE 部分有 256 个 expert × 5B 参数 = 1.28T，每条样本激活 8 个 expert（实际激活 40B），总参数 1.3T，激活参数 60B。

**MoE RL 的三项系统挑战**：

| 挑战                    | 现象                                                                            | 解决方案                                                                              | 代表工作                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Expert 负载不均         | 某些 expert 被频繁激活（hot expert），其他 expert 闲置；部分 expert 训练不充分  | Expert Balancing Loss，鼓励激活频率接近均匀分布 $1/\text{num\_experts}$；动态路由调整 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[GShard](https://arxiv.org/abs/2006.16668) |
| 跨卡通信开销            | Expert 分布在多 GPU（Expert Parallelism），每条样本都需要 all-to-all 路由 token | 优化 all-to-all 通信 kernel，计算与通信重叠                                           | [DeepEP](https://github.com/deepseek-ai/DeepEP)                                             |
| Token-level IS 方差过大 | MoE 路由差异导致 token 级重要性采样比率波动剧烈，梯度方差高                     | 将 IS 比率从 token 级改为序列级（整个序列共享一个 ratio）                             | [GSPO](https://arxiv.org/abs/2507.18071)（Qwen3 全系采用）                                  |

### 减少流水线空闲时间

模型切到多张 GPU 后，每张卡并不会自动保持忙碌。

| 技术                                         | 解决的问题                                              | 核心原理                                                               | 收益                                                                                        |
| -------------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [DualPipe](https://arxiv.org/abs/2412.19437) | 传统流水线并行存在大量气泡（bubble），前向/反向无法重叠 | 双向流水线调度，让前向 stage N 和反向 stage N-1 在同一 GPU 上重叠执行  | 气泡比例从传统 $\frac{P-1}{M}$ 降到 $\frac{P-1}{2M}$（$P$=PP stage 数，$M$=micro-batch 数） |
| Best-Fit Packing                             | Micro-batch 大小不均导致部分 GPU 提前完成后空等         | 用装箱算法（bin packing）把不同大小的 micro-batch 分配到 GPU，平衡负载 | DeepSeek V3 中 GPU 利用率从 70% 提升到 95%                                                  |

### 性能瓶颈定位方法

前面的每项技术都可能把瓶颈推到下一处：生成加快以后，权重同步可能变慢；模型切分以后，跨卡通信可能占据主要时间；样本装箱改善以后，数据读取又可能跟不上。

**常用性能分析工具**：

| 工具                                                                 | 用途                      | 能看到什么                                                          | 适用阶段                                |
| -------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------- | --------------------------------------- |
| [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)    | 通用 PyTorch 训练性能分析 | CPU/CUDA 活动时间线、显存占用、kernel 执行时间、top 耗时操作        | 训练阶段优化                            |
| [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) | 系统级 GPU 性能可视化     | 每个 CUDA kernel 执行时间、CPU-GPU 同步点、NCCL 通信开销、多流重叠  | 通信/调度瓶颈定位                       |
| veRL 内置 Profiler                                                   | RL 训练流程时间分解       | Rollout 生成、Actor 更新、Critic 更新、权重同步、通信各占总时间比例 | **RL 训练首选**，直接定位 pipeline 瓶颈 |

**常见瓶颈与优化方向**：

| 瓶颈现象              | 判断标准（时间占比）             | 优化方向                                                                       |
| --------------------- | -------------------------------- | ------------------------------------------------------------------------------ |
| Rollout 生成慢        | rollout 占总时间 80%+            | 增加 rollout GPU 数量、启用 vLLM Prefix Caching、增大 batch size、考虑分离部署 |
| 权重同步慢            | weight sync 占 5%+               | 使用 LoRA（只同步 adapter 权重，而非完整模型）、NCCL 打包传输、减少同步频率    |
| 跨卡通信开销大        | all-reduce/all-gather 占 10%+    | 增大 micro-batch size、使用 gradient accumulation、优化并行策略切分            |
| 激活显存爆炸（OOM）   | 训练中途 CUDA out of memory      | 启用 Gradient Checkpointing、降低 max sequence length、减小 batch size         |
| Expert 负载不均       | 部分 GPU 利用率 90%+、部分 30%   | 开启 Expert Balancing Loss、调整 MoE 路由策略、使用 EP 负载均衡                |
| 慢人问题（Straggler） | batch 内最长序列决定整批完成时间 | 长度分桶（length bucketing）、Seer divided rollout（按预测生成长度分组）       |

**MFU（Model FLOPs Utilization）**：用实际执行的浮点运算量除以硬件在同一时间内能够提供的峰值运算量：

$$\text{MFU} = \frac{\text{实际 FLOPs}}{\text{峰值 FLOPs} \times \text{时间}}$$

不同训练配置的典型 MFU 参考：

| 训练配置                                           | 典型 MFU 范围                    | 瓶颈来源                         |
| -------------------------------------------------- | -------------------------------- | -------------------------------- |
| Dense 模型 + FSDP + Gradient Checkpointing（同步） | 35–45%                           | 激活重计算、跨卡 all-reduce      |
| MoE 模型 + Expert Parallelism + DualPipe           | 50–60%                           | Expert all-to-all 通信、负载不均 |
| 异步 RL（生成/训练分离部署，rollout 用 vLLM）      | 训练端 40–50%，rollout 端 70–80% | 权重同步、队列等待               |

MFU 低于 30% 时，应结合时间分解继续检查通信、数据加载和 rollout 等待。

---

## 本节小结

多机训练仍然沿着**生成 → 奖励 → 更新 → 同步参数**的循环运转，本节的所有技术都在解决这个循环里的某一个卡点。

- **训推不一致**：同一组权重不保证同一组概率。推理引擎和训练引擎的计算路径差异仍会改变 log-probability，PPO 的 clipping 只能限制更新幅度，修正不了这个偏差；对应的办法是精度对齐、记录真实行为策略、截断极端重要性采样比率。
- **资源安排**：veRL 用 ResourcePool 给 GPU 分组，Worker 封装模型实例，Driver 作为单控制器运行算法主循环。
- **两个瓶颈**：生成吞吐靠 vLLM 的 PagedAttention、Continuous Batching 和 Prefix Caching；训练显存靠 FSDP/ZeRO 分片，再用梯度检查点换计算。70B 模型约 1.2TB 的全参数训练状态，用 ZeRO-3 加梯度检查点能压到单卡 30GB 左右。
- **同步与异步**：同步训练保证数据较新；异步训练减少等待，但要处理轨迹陈旧。LlamaRL 不修正，AReaL 显式修正，AgentRL 隔离多轮环境任务。
- **MoE 与流水线**：MoE 在普通并行之外多了专家负载不均和 all-to-all 通信；DualPipe 和 Best-Fit Packing 减少流水线空闲。

多机系统解决了算力怎样协同，训练仍然需要持续供应可执行、可验证的数据。[18.5 大规模 RL 数据工程](./data-engineering) 将沿着一条轨迹的生命周期，说明任务、环境、奖励和失败样本怎样进入下一轮训练。

## 延伸阅读

- [HybridFlow: A Flexible and Efficient RLHF Framework (veRL)](https://arxiv.org/abs/2409.19256)
- [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [SGLang](https://arxiv.org/abs/2312.07104)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [LlamaRL: A Distributed Asynchronous Reinforcement Learning Framework](https://arxiv.org/abs/2505.24034)
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298)
- [AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework](https://arxiv.org/abs/2510.04206)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [GSPO: Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
