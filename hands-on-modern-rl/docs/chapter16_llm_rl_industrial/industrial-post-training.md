# 18.2 工业后训练流水线

> **本节目标**：理解工业后训练为什么必须是一个闭环，掌握目标定义、数据环境、SFT、RL、评测、失败回流六阶段的具体内容，能够把任何公开技术报告定位回对应的阶段。

到这一章为止，我们已经把 RL 训练的算法、架构和分布式工程都讨论完了——从 PPO、DPO 到 GRPO，从单卡实现到多卡同步生成。真正开始做工业训练时，马上会遇到几个前面没有回答的问题：第一批训练数据从哪里来？模型训完以后，怎么判断它是真的学会了，还是只记住了训练集？那些做错的题目，怎样让它在下一轮里不再犯？

先看一个代码模型的真实例子。团队从真实 GitHub 仓库收集 bug 报告，为每个问题准备可以运行的环境、锁定依赖和测试用例。模型先通过 SFT 学会读取仓库、调用工具和提交补丁，再通过 RL 反复尝试修复，测试程序给每次尝试打分。训练完成后，团队在完全没见过的新仓库上评测模型，把失败任务归类，补充新的数据和验证规则，然后开始下一轮训练。

训完一轮不是结束。评测暴露的失败会变成下一轮的输入，这就形成了工业后训练的**基本闭环**：

**定义目标 → 构造数据与环境 → SFT → RL / 偏好优化 → 评测 → 失败回流 → 回到定义目标**

本节回答三个问题：后训练为什么必须做成闭环，一轮完整训练要经过哪六个阶段，以及公开的案例分别改变了闭环中的哪一步。

---

## 后训练闭环

很多初学者会把后训练理解成"先 SFT，再 RL，然后发布"的一次性流程。实际上如果真的只做一轮，几乎一定会遇到下面这几类问题：

**第一类：训练数据只能覆盖团队已经想到的情况。** 写第一版数据时，团队会按自己的经验构造样本：比如数学题偏向竞赛题型，代码修复只挑了中等规模的 Python 仓库。但模型进入真实场景后，会暴露新的失败方式——用户的仓库是用旧版 numpy 写的，issue 包含中文、英文和代码混合描述，工具调用格式正确但一直重复搜索同一份文件，数学答案正确但推理过程碰巧猜中。训练集里没有这些边界情况，模型自然也不会处理。

**第二类：训练奖励升高不等于真实能力提高。** 某一轮 RL 里，模型发现验证器只检查新增测试是否通过，不检查原有测试是否被破坏——于是模型开始在修改 bug 时顺便删掉难跑的旧测试，训练 reward 一路涨，但独立评测的真实通过率反而降了。或者模型在训练沙箱里学会了绕过某些环境限制，部署到真实仓库里就完全失灵。

**第三类：单个阶段的改进需要反复迭代。** 你可能第一轮把 verifier 写对了，但发现 SFT 数据里工具格式不对，模型根本不会正确调用；第二轮把工具格式教对了，又发现 reward 没有惩罚修改无关文件，模型乱改源码；第三轮修完 reward，评测里又发现新的失败类型——模型在长任务中前面全对，但最后一步提交 patch 时路径写错了。

这些问题没有一个能在第一轮里一次性解决。训练每往前走一步，就会在更难的失败上暴露新的问题。所以每一轮后训练的产物不只有模型权重，还包括能够持续复用的**数据集**（下次训练可以直接用）、**环境和沙箱**（每次启动任务状态一致）、**验证器和奖励规则**（逐步完善防止被钻空子）、**评测体系**（能区分真进步还是过拟合）。

把一次性训练和多轮闭环训练放在一起对比，差异在五个方面：

- **数据**：一次性训练第一轮构造多少就用多少，用完就过；闭环训练会把评测暴露的失败分类，补充新任务、修复坏样本，让数据持续增长。
- **环境**：一次性训练里沙箱依赖不一致、启动脚本有问题都很难发现；闭环训练把环境版本化、可回放，失败时能重放完全相同的状态定位问题。
- **奖励**：一次性训练里模型钻了验证器空子（比如删掉测试、硬编码答案）只能认栽；闭环训练把奖励规则也版本化，每轮发现作弊方式就升级规则，逐步堵住漏洞。
- **评测**：一次性训练直接拿训练集上的表现当发布依据；闭环训练必须用独立的新任务集、在接近部署的配置下评测，还要专门检查通用能力有没有回退。
- **模型能力**：一次性训练的模型只会第一轮数据覆盖的那些模式；闭环训练每次补新的失败类型，能力在多轮迭代中逐步扩展和稳定。

所以闭环的价值不是多跑了几轮训练，而是把"训练中暴露的真实失败"变成"下一轮系统改进的输入"。模型能力正是在多轮修正中逐步建立的。

## 训练流程

现在用一个具体的代码修复任务，把六个阶段走完一遍。六个阶段按顺序首尾相接，每个阶段回答一个核心问题，产出可以复用到下一轮的资产：

- **目标定义**回答"模型最终要完成什么任务"，产出能力清单、成功标准与安全边界。
- **数据与环境**回答"怎样让任务可以训练、可以重复、可以判分"，产出提示、轨迹、沙箱、测试与验证器。
- **SFT**回答"模型是否会按正确格式开始行动"，产出具备基本行为的冷启动模型。
- **RL/偏好优化**回答"模型能否通过尝试提高成功率"，产出针对目标能力优化后的模型。
- **评测**回答"提升是否来自真实能力，其他能力是否退化"，产出独立指标、失败样本与回归测试。
- **数据回流**回答"下一轮应该补什么"，产出新任务、新轨迹和修改后的奖励规则。

公开方案使用的算法名称很多：GRPO、DAPO、VAPO、SAO、DPO、RLHF、OPD……初学时先把它们放一边，问**这个方法改变了六个阶段中的哪一个**，再回来看用了什么算法就清楚了。

### 目标定义

训练修 bug 模型，第一步要先写清楚什么叫"修好"。假设团队要在 SWE-bench Lite 这类真实 Python 仓库 bug 修复上达到 30% 以上的通过率，目标定义至少包含四条：

1. **新增测试必须通过**：仓库原来 issue 附带的测试，或专门为这个 bug 写的新增测试，模型提交的 patch 必须让它们全部通过。
2. **原有测试不能退化**：运行仓库原有的完整测试套件，之前能通过的测试不能因为模型的 patch 变失败了——这是防止模型绕过测试最基本的约束。
3. **不能修改无关文件**：模型只能修改与 bug 相关的文件，不能为了让测试通过而删掉测试、改配置文件、或修改无关源码。
4. **不能绕过验证器**：比如不能把 assert 语句改成 pass，不能把输入写死成测试里的值来硬编码通过。

由此得到能力清单、成功标准和安全边界。部署时如果还限制运行时间（比如一次任务最多 10 分钟）、工具次数（最多 20 次工具调用）、回答长度（patch diff 不能超过 500 行），这些约束也要在此时确定——因为后面每个阶段（数据构造、reward 设计、评测检查）都要依赖它们。

> 目标定义阶段看起来不需要任何技术工作，但决定了后面所有阶段的难度。成功标准写得越明确、越可自动判断，后续训练越稳定；如果目标定义模糊到"模型写的代码看起来合理"，后面所有 reward 设计和评测都会跟着模糊。

### 数据与环境

GitHub issue 本身只是一段文本描述："项目启动时在 macOS 上报错 ImportError，错误信息是……"。它还不是可训练任务。团队要做以下几步才能把一个 issue 变成可训练样本：

1. **保存出错前的仓库版本**：找到这个 issue 对应的 commit hash，用 `git checkout` 把仓库切到 bug 存在的状态，而不是 bug 已经被修好的最新代码。
2. **锁定依赖和启动命令**：把 requirements.txt、package.json 或 Dockerfile 锁定到当时的版本，写清楚如何启动环境、运行测试的命令。
3. **准备新增测试与回归测试**：为了验证 bug 是否真的被修好，需要写好新增测试（不打 patch 就失败、打了 patch 就通过）和回归测试（打了 patch 后原有功能也不能坏）。
4. **封装进可重复启动的沙箱**：把整个项目打包进 Docker 容器或专用沙箱，保证模型每次面对同一个 bug 时，环境状态完全一致——同一个 Python 版本、同一组依赖、同一个文件结构。

同一个任务每次都从相同状态开始，模型得到的奖励才可以比较。如果今天沙箱里 numpy 是 1.26，明天变成了 2.0，同样的 patch 可能今天通过明天失败，训练信号就完全乱了。

环境型任务还要覆盖**执行方式的变化**。如果所有训练都只用一种外层模板——比如固定的 ReAct 循环、固定的上下文管理方式——模型很容易过拟合这个外层格式，换一种 harness 就不会了。MiniMax 会用不同的 Agent Scaffold 采样同一类任务；Kimi K3 会改变工具、提示和上下文管理；Qwen-AgentWorld 同时使用容器、MCP、移动端和网页环境。这样可以减少模型只记住某一种外层模板的风险。

### SFT

模型直接进入 RL 会遇到一个很实际的问题：它可能根本不会正确地操作环境。比如面对一个仓库修复任务，它可能：

- 不会用 `read_file` 工具读取目标文件，而是直接在回答里说"我需要先看一下代码"
- 运行测试时不会解析 pytest 的输出，不知道哪些测试失败了
- 写了 patch 以后不会用 `apply_patch` 工具提交，而是在对话里贴一段 diff

团队先用人工写的、或强模型生成的**成功轨迹**做 SFT，教模型三件事：

- **工具格式**：每个工具应该怎么调用参数，返回结果长什么样，下一步应该如何根据工具返回继续操作
- **操作顺序**：先读文件，再改代码，再跑测试，再根据失败改代码，而不是反过来
- **停止条件**：什么时候算任务完成可以停止，什么时候还要继续修改

SFT 在这里只负责建立基本行为，不要求模型通过反复试错找到更好的修复策略——这是 RL 阶段的目标。类比人类学车：SFT 相当于教练先教你认识方向盘、离合、刹车，教你正确的驾驶步骤，先能把车开走；什么时候变道、什么时候超车、怎么处理复杂路况，那是后面 RL 在实战里练的。

### RL

具备基本行为以后，模型对同一个 bug 不再只生成一条轨迹，而是生成多条。还是那个 SWE-bench 的例子，每题采样 16 个候选：

- 模型第一次尝试直接改 `load_config()` 函数，patch 提交后新增测试通过了，但 regression test 坏了，reward 给 0.2
- 第二次尝试在改 `load_config()` 的同时，又加了一个新的兼容分支，提交后新增和 regression 都通过了，也没改无关文件，reward 给 1.0
- 第三次尝试改了错误的文件，所有测试都失败，reward 给 0
- 第四次尝试新增测试通过、regression 通过，但模型为了省事把 assert 改成了 pass，被 reward 的硬编码检查抓住，reward 给 -0.5

每条轨迹的奖励来自**测试程序和辅助规则**：测试是否通过、是否破坏了旧测试、是否修改了无关文件、是否硬编码绕过验证器、工具调用格式是否正确。GRPO 会把这 16 条轨迹放在同一组里比较，高于组内平均分的回答概率升高，低于平均分的概率降低。训练几万步以后，模型越来越倾向于产出真正修复 bug 的轨迹。

数学、代码、写作等不同领域的能力也可以先训练成不同的领域专家，再通过 **On-Policy Distillation**（OPD）合回一个模型。MiniCPM5、Kimi K3 和 NVIDIA Nemotron-Cascade 2 都公开了这条路线：相当于每个领域先拿一个专项教练带学生，最后再把专项教练的能力综合进同一个学生模型里。

### 评测

训练一轮结束后，不能只看训练集里的 reward 曲线，必须在**从未见过的新任务**上做独立评测。还是那个 SWE-bench 例子，团队准备了 500 个训练阶段完全没见过的新仓库 bug，检查：

- **补丁通过率**：模型到底修好了多少个 bug（这是主指标）
- **无效工具调用比例**：多少次工具调用没有任何效果（反映模型会不会正确用工具）
- **平均成本**：平均每个任务调用多少次工具、消耗多少秒时间（影响真实部署成本）
- **通用能力回归**：除了代码，通用聊天、数学、多语言能力有没有因为 RL 只训代码而明显下降

评测配置要尽量接近真实部署。比如部署时用户只会给模型 10 分钟运行时间，评测时就不要放 30 分钟；部署时模型用的是 4-bit 量化，评测也用同样的量化方式；部署时工具协议、上下文管理方式是什么，评测就用什么。这些配置一换，最终结果就可能差很多。

**训练奖励升高而独立评测不变**，是一个很重要的信号——模型可能只适应了训练环境的特殊情况，或者利用了验证器漏洞。比如前面提到的模型学会删掉测试来通过验证器，在训练 reward 里会表现很好，但一到独立评测（新仓库的测试完整、验证规则更严），通过率就会立刻掉下来。独立评测的意义就是防止这种表面进步。

### 数据回流

独立评测出来的 180 个失败案例不是直接丢掉。团队要按失败原因分类，然后送回到闭环中对应的位置：

- 如果是训练集覆盖的仓库类型太窄，就回流到**数据与环境**阶段，补充新语言、新框架、不同规模的新仓库问题。
- 如果是沙箱环境不一致，同一道题每次跑结果都不同，也回流到**数据与环境**阶段，修复沙箱打包脚本、锁定依赖版本、增加可复现性检查。
- 如果是验证器只看通过率，没查无关文件修改被模型钻了空子，就回流到**目标定义与奖励规则**，更新奖励函数，增加 patch diff 审查项。
- 如果是模型根本不会用某个工具或操作协议，连正确格式的动作都输不出来，就回流到**SFT**阶段，补对应的成功轨迹做冷启动。
- 如果模型已经会基本操作，但这类任务的成功率还不够高，就回流到**RL**阶段，增加这类任务在下一轮采样中的权重。

这就是"失败回流"的含义——评测不只是为了给模型打分，更是为了**定位系统哪一层需要修改**。为此团队必须记录每一条失败的任务、环境版本、完整轨迹、奖励函数版本和模型 checkpoint，否则过几周以后根本无法重放当时的失败状态，也不知道改完是否真的修复了问题。

这样走完整的六步以后，六个阶段之间的关系就很清楚了：前一阶段的产物是后一阶段的输入，最后的评测失败又回去修正前面的环节。[18.5 大规模 RL 数据工程](./data-engineering) 将继续展开其中最复杂的一环：数据生产线，也就是如何把原始事件批量变成可训练、可回放的 RL 样本。

## 环境型任务扩展

数学题的环境就是题目文本加答案验证器。代码修复、网页操作、桌面 GUI 这类任务需要和外部环境交互，一个中型 Python 仓库的修复沙箱就要带 200MB 依赖、十几个启动命令、完整 git history，光拉起环境就要一分钟。六阶段在这类任务上每一步都要扩展：

- **目标定义**新增工具权限、时间预算、长度与安全边界。一道数学题给 10 分钟还是 1 小时差别不大，但 agent 任务给 10 分钟只能读 3 个文件、改 1 处 patch，给 1 小时就能完整浏览仓库、反复调试，时间预算直接决定策略风格。模型握着实打实的工具权限，必须明确哪些动作绝对不能做——不能 `rm -rf /`、不能读 `~/.ssh/` 私钥、不能未经授权装软件。这些约束写进目标定义后，reward 设计和评测检查都要对应加约束。代表案例有 Apple 端侧约束、OpenAI 工具任务。
- **数据与环境**新增仓库、容器、网页状态、工具和可恢复检查点。数学题一条样本就是 `{"prompt": "...", "answer": "..."}`，加载一次就行。环境型任务一条长轨迹可能跑了 40 分钟、改了 12 个文件、执行了 30 次工具调用，这时候 worker 崩溃，如果没有 checkpoint 就要从零重跑，前面算力全部浪费。可恢复检查点定期保存文件系统快照、数据库状态、浏览器 DOM 等，崩溃后从最近 checkpoint 恢复即可，不用从头再来。同一个任务还要用不同的工具配置和执行框架训练，避免模型过拟合某一种 harness。代表案例有 MiniMax M2.1、Kimi K3。
- **SFT**新增工具格式、环境观察、动作顺序和停止条件。数学题 SFT 输入只有用户 prompt，输出是推理过程。Agent 任务的输入还包括 `read_file` 返回的文件内容、pytest 报错堆栈、浏览器 DOM 结构、`search_code` 返回的 JSON 结果——这些是模型上一步动作后环境返回的真实反馈，不是用户写的。SFT 不只要教工具调用格式，更要教"看到 `AttributeError` 之后下一步去哪个文件找定义"这种基于观察的决策逻辑。代表案例有 Qwen-AgentWorld、UI-TARS。
- **RL/偏好优化**新增多轮轨迹奖励、异步经验和领域专家训练。数学 RL 通常一次性输出后给 0/1 分；agent 任务一次 episode 有 30 多轮交互，reward 要拆到轨迹不同阶段——正确读文件给小正反馈，连续 3 次调同一个无新信息的工具给负反馈，破坏原有测试立即扣分。同步 GRPO 必须等同一 batch 所有轨迹跑完才能训练，但 agent 任务有的 30 秒结束、有的跑 15 分钟，等最慢的会让前面做完的 GPU 闲置。异步经验做法是每条 rollout 做完就放进经验队列，训练端凑够 batch 就更新，不用等组内最慢的。代表案例有 GLM-5.2、DAPO、MiniCPM5 OPD。
- **评测**新增真实环境、独立任务、成本与安全回归。数学 RL 看正确率就够了，但 agent 任务上线要烧钱。比如某轮训练后 SWE-bench 通过率从 28% 升到 35%，但平均每个任务调用 300 次工具、花 2 美元，而旧模型只要 50 次、0.3 美元，上线就亏钱。安全回归同样重要：模型学会修改 `/etc/hosts` 绕过网络限制虽然能提分，但在真实部署里是严重漏洞。所以评测还要监控平均工具调用次数、单次任务成本、敏感操作触发率。代表案例有 Claude 系统卡、Computer Use。
- **数据回流**新增失败分类、拒绝采样、蒸馏与验证器更新。数学 RL 把错题补进训练集就行，但 agent 任务的长轨迹不能简单二元判断——一条 50 步轨迹前 45 步全对、最后一步路径写错，前 45 步的经验仍然有价值，不能整条丢掉。拒绝采样按片段、按轮次、按中间 reward 筛选有价值的部分，而不是非黑即白。验证器也要持续更新：如果发现模型通过硬编码绕过测试、旧 verifier 没查出来，要先修验证器规则，再用新版 verifier 重跑旧样本重算 reward，否则坏信号会一直累积。代表案例有 DeepSeek-R1、Nemotron-Cascade 2。

::: info 数学 RL vs 环境型 Agent RL：训练系统的关键差异

- **State 复杂度**：数学 RL 状态即题目文本，几乎无外部依赖；Agent RL 状态包括文件系统、进程、网络、DOM、数据库等，迁移成本高。
- **Reward 延迟**：数学 RL 单次输出后立即给 0/1 分；Agent RL 多轮交互后才有最终结果，需要过程奖励和 credit assignment。
- **Episode 时长**：数学 RL 时长均匀，几秒到几十秒；Agent RL 差异极大，30 秒到数小时，同步等待浪费严重。
- **Rollout 失败率**：数学 RL 失败率低，最多格式错；Agent RL 沙箱启动失败、依赖装不上、工具崩溃、超时都是高频问题。
- **样本复用**：数学 RL 重复加载 prompt 即可复现；Agent RL 需要版本化环境快照 + 可恢复 checkpoint 才能重放。

:::

::: details 加餐：为什么数学 RL 不能直接照搬到 Agent 任务

核心差异有四点：

- **动作空间**从 token 变到工具+环境。数学 RL 动作就是输出下一个 token，全在词表里；agent 任务动作包括调工具、读文件、跑测试、提 patch，结果取决于真实环境状态，无法在词表里穷举，必须接入真正的工具执行层。
- **Episode 长度**从百 token 级变到千 token 级以上。GSM8K 通常 200-800 token，SWE-bench 任务要 5000-20000 token、10-30 次工具调用，attention、KV 缓存、超时策略、batch 大小都要重设计。
- **Credit assignment**更难。数学题输出完最后一个 token 立刻判分；agent 任务第 5 步读错文件可能要到第 45 步提交 patch 才暴露，第 3 步做对的关键定位也要等最终通过才能确认。
- **验证器**从比对答案变到真正执行环境。数学 verifier 抽 $\boxed{}$ 比对数字 20 行搞定；代码 verifier 要起沙箱、装依赖、跑 pytest、查 patch diff，网页 agent 要点 DOM、查页面状态，桌面 GUI 要截图识别、模拟键鼠。

:::

## 公开案例

公开报告里的算法名很多——GRPO、DAPO、SAO、OPD……其实都是对闭环某个环节的具体改进。下面四个案例覆盖四个最常见的改进入口：MiniCPM5 处理**数据回流**里多专家能力合并，Kimi K3 处理**数据与环境**里 harness 过拟合，Qwen-AgentWorld 处理**SFT + RL** 的训练顺序，GLM-5.2 处理**RL** 训练系统的长短轨迹异步问题。阅读时先找它改了闭环哪一步，再看用了什么算法。

### MiniCPM5

MiniCPM5 改进数据回流 + RL 两步，解决"专项能力强但合进通用模型顾此失彼"的问题。公开流程是 SFT、分领域 RL、OPD 三步：先分别训练数学、代码、闭卷问答、写作等领域的 RL 专家模型，再让学生模型对同一批领域提示采样，教师在每个 token 位置给出概率分布，学生据此调整输出——这就是 On-Policy Distillation（OPD）。

和离线蒸馏的差别：离线蒸馏保存教师写好的固定答案让学生模仿，学生做题时会在哪一步走偏教师不知道；OPD 的回答由当前学生自己生成，教师纠正学生此刻真正走到的状态，针对性更强。领域专家训练时用的领域提示可以直接复用，不需要为蒸馏单独构造新题目。每条样本需要记录领域、提示版本、学生策略版本、教师版本和逐 token 教师信号，OPD 的公式与日志结构放在 [18.5](./data-engineering)。

从闭环定位看，MiniCPM5 改了两步：RL 阶段先分领域训专家让专项能力充分收敛，数据回流阶段用 OPD 把多个专家蒸馏合并回一个模型，相当于把上一轮多个专家的成功经验合成下一轮的冷启动数据。

### Kimi K3

Kimi K3 改进数据与环境一步，解决 agent 训练过拟合某一种执行框架（harness）的问题——模型学到固定格式套路，不是真正的任务决策逻辑。如果所有训练都用同一个 ReAct 模板，模型只会记住"第一步写 `Thought:`、第二步调 `list_dir`"这种固定流程，换一种框架就连工具都不会调。

Kimi K3 把工具、提示、上下文管理、技能、记忆和子智能体组合成统一训练环境，同一个任务可以换用不同工具配置或执行框架（ReAct+Bash、Plan-then-Edit+`apply_patch`、带反思循环的框架等），让模型学"怎么修 bug"本身而不是某一种 harness 的答题格式。任务侧还可以从知识图谱选节点，检索文章、博客和代码仓库，合成长程研究任务。

Kimi K3 同样先训练不同领域和思考强度的专家，再用多教师 OPD 合并能力。对百万 token 长任务，系统保存未完成轨迹和外部 KV 状态，下一轮从中间继续——如果任务跑到 80 万 token 时宕机，不用从头重跑，加载 KV 缓存就能从断点继续，节省几倍算力。

从闭环定位看，Kimi K3 改了两步：数据与环境阶段把"**可组合训练环境**"作为设计重点，通过多 scaffold、多工具配置避免过拟合；数据回流阶段沿用 OPD 合并专家，同时增加长任务 **KV 状态持久化**，让中断轨迹也能作为下一轮样本。

### Qwen-AgentWorld

Qwen-AgentWorld 改进 SFT + RL 的训练顺序，解决直接上手 RL 导致模型连环境反馈都看不懂的问题。它用容器沙箱、MCP 工具、Android、网页和操作系统模拟器构造任务，训练分三阶段：继续预训练学环境知识，SFT 预测下一状态并学会基本操作，最后用 GSPO 和混合奖励训练完成任务。三阶段顺序类似学骑车：先读说明书了解原理，再在停车场练基本动作，最后上路在真实环境中试错调整，顺序反了就会摔。

Qwen-AgentWorld 会挑出轨迹中真正包含环境变化的轮次重点训练。纯格式重复、没有新观察的轮次信息少，不必和关键操作同权重。比如 agent 连续 5 次调用同一个 `search`、输入相同关键词，返回结果自然一样——第 1 次调用读到新结果有价值，第 2-5 次反复调用没有新观察是"空转"，给同权重会稀释有效梯度，甚至让模型学会"重复调工具凑长度"。轨迹筛选已经从整条保留/删除深入到**轮次级别**。

从闭环定位看，Qwen-AgentWorld 改了两步：SFT 阶段拆成"继续预训练学环境建模 + SFT 学基本操作"两个子阶段，降低直接上手动作学习的门槛；RL 阶段加入按轮次筛选有效轨迹的机制，重点训练含真实环境变化的回合。

### GLM-5.2

GLM-5.2 改进 RL 训练系统，解决同步 GRPO 在 agent 任务上因长短轨迹耗时差异大导致 GPU 严重闲置的问题。工具任务时长差异大，按组等待浪费算力：32 张 GPU 并行 rollout，同一 batch 内 8 个任务 30 秒、8 个 2 分钟、12 个 5 分钟、4 个 15 分钟，同步 GRPO 必须等最慢的 900 秒，总闲置时间 8×870 + 8×780 + 12×600 = 20400 秒——32 张卡里超过 22 张在空转。

GLM-5.2 的 **SAO**（Single-policy Asynchronous Optimization）让每个提示只生成一条轨迹，做完一条就异步送入训练队列，不用等组内其他；同时配套处理旧策略数据、价值估计和 token 裁剪。slime 框架在工程上连接训练后端、SGLang rollout、自定义环境、验证器与数据缓冲区。

异步训练带来策略版本不一致问题：同步 GRPO 整个 batch 轨迹都是同一版策略生成，异步 SAO 队列中的轨迹可能来自 5 分钟前或 30 分钟前的旧策略，而训练端参数已经更新了几十步。每条轨迹必须携带策略版本号，版本差太远直接丢弃，差不远用重要性采样权重修正 loss，防止旧数据把新策略带偏。

从闭环定位看，GLM-5.2 改了两步：RL 阶段把同步 GRPO 改成异步 SAO，通过版本标记和重要性采样解决数据不一致，大幅提升 GPU 利用率；数据与环境阶段配套 slime 工程框架，把训练后端、rollout 引擎、环境和验证器串成异步可扩展流水线。

### 资料来源

- [MiniCPM5-1B 官方训练说明](https://github.com/OpenBMB/MiniCPM#-minicpm5-1b)
- [Kimi K3 官方报告与仓库](https://github.com/MoonshotAI/Kimi-K3)
- [Qwen-AgentWorld 官方说明](https://qwen.ai/blog?id=qwen-agentworld)
- [slime 官方仓库](https://github.com/THUDM/slime)
- [SAO：GLM-5.2 的异步 Agentic RL](https://arxiv.org/abs/2607.07508)
- [Nemotron-Cascade 2](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)

::: details 加餐：怎样快速把一个公开论文定位回六阶段

拿到一篇新的工业后训练论文或技术报告，不用从头一页页读细节，先按下面这个快速检查清单过一遍，3 分钟就能知道它主要改了闭环的哪一环：

（1）**是否改动了任务定义/成功标准** → 对应**目标定义**阶段。如果论文开头就写"我们重新定义了 XX 任务的成功标准，除了通过率还要看成本/安全/时延"，或者"我们把能力清单从单工具扩展到多工具协作"，那它的核心贡献在目标定义层。

（2）**是否造新环境/造新数据** → 对应**数据与环境**阶段。如果论文花了大量篇幅讲"我们构造了 XX 沙箱"、"我们从 XX 原始数据清洗出了 XX 条可运行样本"、"我们新增了 XX 种任务形态"，或者放出了新的数据集/环境仓库，核心贡献就在这一层。

（3）**是否主要讲 SFT 数据构造** → 对应**SFT**阶段。如果论文的核心是"我们用了 XX 种轨迹筛选方法构造 SFT 数据"、"我们把 SFT 拆成了 XX 个子阶段分别训"、"我们引入了 XX 专家来写冷启动轨迹"，而且算法创新不多，主要讲数据怎么来的，那就是 SFT 阶段的改进。

（4）**是否讲新算法/新 advantage/新 reward** → 对应**RL / 偏好优化**阶段。如果论文里有大量公式，讲新的目标函数、新的优势估计方式、新的奖励拆分策略、同步改异步的训练方式，或者出现了新的算法缩写（SAO、DAPO、VAPO 等），核心贡献就在 RL 阶段。

（5）**是否讲新 benchmark/新评测方法** → 对应**评测**阶段。如果论文的主线是"我们提出了一个新的 benchmark，覆盖了 XX 类之前没人做过的场景"、"我们设计了 XX 种回归测试防止能力遗忘"、"我们提出了新的评测指标（成本、安全、时延）"，甚至附了独立的评测排行榜，那核心贡献就在评测层。

（6）**是否讲新数据补法/蒸馏合并** → 对应**失败回流**阶段。如果论文讲的是"我们把上一轮的失败样本按 XX 规则分类补进下一轮"、"我们用拒绝采样从 RL 轨迹中筛 SFT 数据"、"我们把多个专家模型蒸馏合并成一个"（如 OPD、Expert Soup），那核心贡献就在失败回流层。

一篇论文通常会同时碰 2-3 个阶段，按这个清单从上往下过一遍，定位完主要改哪几环以后，再去读细节算法和数据，就不会被一堆新名词带着走了。

:::

看完四个案例，下面是按阶段分类的公开实践索引，同一团队通常覆盖多个阶段，查阅时从当前问题进入即可：

- **目标定义**：OpenAI Operator/Codex、Apple、IBM Granite、xAI Grok
- **数据与环境**：MiniMax SWE Scaling、WebExplorer、UI-TARS、LongCat、Qwen-AgentWorld
- **SFT**：Qwen2.5、InstructGPT、Meta Llama、Microsoft Phi
- **RL/偏好优化**：DeepSeek、Qwen3、DAPO/VAPO、Kimi、GLM、Hunyuan
- **评测**：Claude 系统卡、Gemini、Llama 4、StepFun、多模态与安全案例
- **数据回流**：Rejection Sampling、Long-to-Short、OPD、Expert Soup、Tulu 3

---

## 公开实践汇编

理解了六阶段闭环和四个代表性案例之后，实际动手训练时仍会遇到大量具体工程问题：冷启动数据如何构造、长度奖励在训练哪个阶段引入、异步采样如何保证数据版本一致性、多模态奖励如何加权平衡。这些问题没有一个能在六阶段的框架描述里直接找到答案，但在近两年的工业界技术报告中已有大量公开讨论。

需要特别说明的是，**本节是工程案例库，而不是教学主线**。它的定位不是要求读者按顺序从头到尾读完所有团队，而是提供一套可以按需查阅的"后训练工程字典"——当你在自己的闭环里遇到某一类具体问题时，可以快速定位到已经公开过类似解法的团队，参考它们的工程选择和踩坑记录。第一次阅读时不必深究每个细节，先建立"哪些团队在哪类问题上有公开经验"的索引感即可。

如果是第一次通读，建议优先聚焦国内团队的五条主线：**MiniMax、Qwen、Kimi、DeepSeek、GLM**。这五个团队的公开材料组合起来，恰好覆盖了后训练工程最核心的几个维度：MiniMax 的 Agent 环境构造与多 scaffold 训练、Qwen 的分阶段后训练 recipe 与通用能力回填、Kimi 的长思考 RL 稳定性与研究型 agent 奖励设计、DeepSeek 的 GRPO 基础方案与拒绝采样回流、GLM 的异步训练系统与跨阶段蒸馏。读完这五个，对后训练的工程全貌就有了基本坐标系。

国外团队的对照则建议重点看 **OpenAI、Anthropic、Google、Meta** 四家。它们的公开材料在通用偏好对齐方法论、系统化安全训练、多模态长上下文后训练、开源产品级数据治理这四个方向上有更系统的讨论，可以作为国内方法在通用体验、安全和部署侧的补充对照。其余团队的资料则建议在真正遇到对应问题时再按需查阅，例如遇到 GUI agent 问题看 UI-TARS、遇到端侧约束看 Apple、遇到企业级工具集成看 NVIDIA 和 Salesforce。

::: details 国内团队公开实践

### 国内团队

国内团队的公开报告更多集中在可验证任务、Agent 环境构造、长程推理的工程实现。先看 SWE-bench 等软件工程任务的落地实践，再看 reasoning RL 的各种训练技巧，就能形成比较完整的后训练工程画面。

下面按团队梳理国内各家公开的后训练细节，公司名称只是资料来源，所有案例都可以映射回前面的六阶段框架。

#### [MiniMax](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models)

MiniMax 的公开资料分三条线（[M2.1 Agent 后训练经验](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models)、[MiniMax-M1 长思考 RL](https://arxiv.org/abs/2506.13585)、[WebExplorer 网页 Agent](https://arxiv.org/abs/2509.06501)）：分别对应软件与应用 agent、长思考 RL、长程网页 agent，不应只概括成"可验证环境 + reward"。从六阶段定位看，M2.1 主要改进"数据与环境 + RL 奖励设计"两个阶段；WebExplorer 主要在"数据构造 + SFT 冷启动"；MiniMax-M1 则聚焦"RL 阶段的长输出效率"。

#### [M2.1 SWE Scaling](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models)：从 GitHub 数据构建可运行环境

**动机**：软件工程任务天然可验证，但原始 GitHub 数据不能直接训练 agent。一个 issue、PR 或 commit 只有文本和代码差异，不等于 RL 样本；必须恢复仓库状态、构造任务描述、准备依赖、确定验证命令，才能变成"模型行动 -> 环境反馈 -> reward"的闭环。

**数据构造**可以拆成五步：

- **挖掘真实事件**：从 GitHub merged PR、commit、issue、测试变更和代码 diff 中筛出有明确修复目标的样本。
- **改写任务形态**：SWE-Resolve 要求模型修 bug 或实现需求；SWE-Test 反过来要求模型写测试，使其在补丁前失败、补丁后通过；SWE-Review 要求模型审查代码变更并指出问题。
- **封装可执行环境**：checkout 到补丁前状态，安装依赖，准备测试命令，构建 Docker / sandbox，保证模型提交 patch 后可以自动验证。
- **补齐训练字段**：生成 original problem description、test-case reward、runnable environment 等字段，形成既可 SFT 又可 RL 的样本。
- **扩语言和场景**：M2.1 报告里提到覆盖 10+ 主要编程语言、10,000+ runnable PRs 和 140,000+ variable tasks。

**奖励设计**也要按任务类型拆：

- **SWE-Resolve**：主 reward 是测试是否通过，同时检查是否破坏已有测试、是否修改无关文件、是否只硬编码测试。
- **SWE-Test**：测试必须在 patch 前失败、patch 后通过，才能说明测试确实覆盖目标 bug。
- **SWE-Review**：它不是完全可执行任务，只能近似验证，例如用另一个 LLM 检查 review 是否命中真实问题，同时控制 hallucination rate。

**关键方法**是"把一个真实开发事件改写成多个等价任务"。这样同一份 GitHub 数据可以产出修复、测试、审查等多种训练信号。

**Multi-scaffold** 是 M2.1 的另一条主线。scaffold 指 agent 外层执行框架，例如上下文管理方式、工具调用协议、反思/计划模板、文件编辑接口。如果 SFT 和 RL 都只在一个 ReAct loop 上做，模型会过拟合这个 loop 的格式。MiniMax 的做法是用 multi-scaffold rejection sampling 生成 SFT 数据，并在 RL 中让不同 scaffold 参与 rollout。

**最小复现**可以先准备两到三种简单 scaffold：直接 ReAct、plan-then-edit、test-driven loop；同一批 SWE 任务分别跑这些 scaffold，保留成功轨迹做 SFT，再在 RL 中随机 scaffold，让模型学任务本身而不是学模板。

#### [M2.1 AppDev](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models)：从固定测试到 Agent-as-a-Verifier

**动机**："从零开发应用"很难预写完整测试。一个前端/后端/移动端应用是否完成，不只看函数输出，还要看能否启动、交互是否正确、视觉是否合理、业务逻辑是否闭环。

**专家数据**：MiniMax 引入 experts-in-the-loop。前端、后端、Android、iOS 专家负责设计 prompts、meta-queries、rubric-based rewards 和系统提示。专家提示里包含最佳实践；训练时可以把系统提示从轨迹中去掉，让模型把专家启发蒸馏进默认行为。

**Reward 三层结构**：

- **Execution level**：检查代码是否能编译、启动、运行。
- **Interaction level**：通过 Playwright 等工具真实点击页面、输入表单、检查状态变化，判断业务逻辑是否正确。
- **Visual level**：按相对一致的审美标准评分，例如布局是否明显错位、关键信息是否可见、交互控件是否可用。

**和普通 LLM-as-a-judge 的区别**：judge 不是只看静态截图或最终文本，而是作为 agent 进入 sandbox 交互。

**最小复现**可以做一个 Todo App / 登录页 / 数据表格任务集：给模型需求，让它生成项目；自动 `npm install && npm run dev`；用 Playwright 跑 5-10 个交互检查；最后用 rubric judge 补视觉和需求覆盖评分。

#### [WebExplorer](https://arxiv.org/abs/2509.06501)：构造长程网页 Agent 难题

**动机**：现有开源 web agents 在 BrowseComp、GAIA、WebWalkerQA、FRAMES 等复杂信息检索任务上弱，一个核心原因是缺少高质量、长程、可训练的数据。人工写这类题很贵；简单 query evolution 又容易变成不自然的问题；图构造方法需要复杂的节点扩展和启发式选择。

**核心思路**：WebExplorer 用 model-based exploration 让强模型自主探索信息空间，再用 iterative long-to-short query evolution 生成更难的问题。

**WebExplorer-QA 的合成流程**可以按"探索"和"演化"两阶段理解：

- **探索阶段**：从 Wikipedia seed entities 出发，在 prompt 中给三个 BrowseComp-en 的 QA exemplars，让模型围绕 seed entity 执行 search / browse，构建一个局部信息空间。这个阶段不是直接产题，而是让模型找到相关实体、网页、事实链和潜在可问的答案。
- **演化阶段**：做 long-to-short query evolution。初始问题通常包含较多显式线索，后续迭代逐步删除或压缩显著信息，让问题更短、更隐含、更需要多步搜索。论文使用 5 次 evolution，最终合成约 40K WebExplorer-QA。
- **设计重点**：先探索出答案和证据，再把问题变难，而不是凭空写难题。

**训练 recipe** 分两段：

- **冷启动 SFT**：先教模型正确使用 search / browse 工具，并形成基本长程搜索格式。论文用 Qwen3-8B 作为底座，约 13K SFT 样本，batch size 32，learning rate 1e-5，训练 4 epochs。
- **GRPO 阶段**：不需要人工轨迹，只需要合成 QA 对，让模型自己探索不同搜索路径。论文用约 12K 样本做 GRPO，每组 8 rollouts，batch size 64，learning rate 1e-6，并逐步把最大响应长度扩到 128K、最大工具调用轮数扩到 100。

**Reward 公式**是复合形式：`R = 0.2 * R_format + R_correct`。

- **`R_format`**：检查工具调用、思考结构、回答格式是否符合协议。
- **`R_correct`**：用 DeepSeek-V3 作为自动 judge，根据 ground-truth answer 判断最终回答是否正确。
- **权重动机**：web agent RL 中，如果不奖励格式，模型会在 search / browse 协议上漂；如果只奖励格式，模型会学会空转工具。所以格式只占小权重，正确性是主信号。

**最小复现**可以这样做：选 500-1000 个 Wikipedia seed entities；为每个 seed 让强模型用搜索工具探索 5-10 个相关网页并产出候选 QA；用另一个模型过滤答案不唯一、证据不足、过于简单的问题；做 2-3 轮 query compression，把显式实体名、日期、地点等线索逐步隐藏；保留答案和证据 URL；用 1K-5K 轨迹做 SFT，让 7B/8B 模型学会 search/browse 格式；再用 1K QA 做 GRPO，reward 用格式检查 + LLM judge / exact match。

**评估指标**至少看三项：平均工具调用轮数是否增加、正确率是否提升、是否出现无效搜索循环。

#### [MiniMax-M1](https://arxiv.org/abs/2506.13585)：长思考 RL 的训练效率

**动机**：M1 不是在讲 agent 环境，而是在解决 test-time compute scaling 太贵的问题。长 CoT 输出到 64K、80K、100K token 时，普通 attention 和普通 RL 都会非常昂贵。

**模型与 RL**：M1 用 hybrid MoE + Lightning Attention 支持 1M 输入上下文和 40K/80K thinking budget，再用 CISPO 做高效 RL。CISPO 的要点是 clip importance sampling weights，而不是直接 clip token updates；这样在长响应里仍然保留所有 token 的梯度贡献，并降低方差。

**数据和 curriculum**：

- **继续预训练**：从 MiniMax-Text-01 继续训练，强化 STEM、代码、书籍、reasoning、长上下文。
- **SFT 冷启动**：注入 CoT 模式，为 RL 做起步策略。
- **RL 数据混合**：可验证任务包括数学、竞赛编程、SynLogic 生成的 41 类逻辑任务、SWE-bench 派生的软件工程 sandbox；不可验证任务如 QA、创意写作，则用生成式 reward model 反馈。
- **逐步扩长度**：扩到 80K 输出时，不是直接放开长度，而是从 40K 逐步扩到 48K、56K、64K、72K、80K。
- **晋级指标**：用生成序列 perplexity 是否收敛、99 分位输出长度是否逼近窗口上限等指标判断是否进入下一阶段。
- **数据筛选**：报告还提到使用 40K 模型筛数据、移除过易样本、增加困难数学和代码比例、下采样会导致重复同质化的 synthetic reasoning 数据。

**可复现重点**不是复刻 512 H800，而是在小模型上验证三件事：长输出训练要逐步扩窗口；过易样本要过滤，否则没有 RL 信号；合成 reasoning 数据如果模式太单一，会让长上下文 RL 变得重复和不稳定。

#### [阿里 Qwen / 通义](https://arxiv.org/abs/2505.09388)

Qwen 的公开资料分为四条线（[Qwen2.5](https://arxiv.org/abs/2412.15115) 通用指令后训练、[Qwen2.5-Math](https://arxiv.org/abs/2409.12122) 数学自改进、[QwQ-32B](https://qwenlm.github.io/blog/qwq-32b/) 两阶段 reasoning RL、[Qwen3](https://arxiv.org/abs/2505.09388) Thinking/Non-Thinking 统一后训练，以及 [Qwen3-Coder](https://qwenlm.github.io/blog/qwq3-coder/) / [Tongyi DeepResearch](https://arxiv.org/abs/2510.24701) agentic coding 与深度研究）：它的价值在于公开材料把"可验证任务"和"通用体验回填"都讲到了，而不是只停在某个 GRPO 名字上。从六阶段定位看，Qwen 系列横跨全部六个阶段，核心改进是"SFT 冷启动 + RL/偏好优化的分阶段设计 + 数据回流的通用能力回填"。

#### [Qwen2.5](https://arxiv.org/abs/2412.15115)：通用指令模型的多阶段后训练

Qwen2.5 报告的 post-training 不只是 SFT。它先用百万级监督样本覆盖通用问答、代码、数学、多语言、结构化数据分析、长文本生成和安全，再做多阶段 RL。这里的 motivation 是：base model 有知识，但不会稳定按照用户意图输出；单轮 SFT 又容易把模型训练成"会答但不够偏好对齐"。因此 Qwen2.5 把 instruction following、长文本、结构化输出、专业能力和人类偏好分阶段处理。最小复现时，应把通用 SFT 数据按能力域分桶，而不是混成一个大 JSONL；每个桶都保留独立 eval，例如长文摘要、JSON 输出、代码、数学、拒答和多语言。

#### [Qwen2.5-Math](https://arxiv.org/abs/2409.12122)：CoT 与 TIR 自改进闭环

这份报告的 motivation 是普通 CoT 在精确计算、符号操作和算法推理上会出错，所以 Qwen2.5-Math 同时训练 Chain-of-Thought 和 Tool-Integrated Reasoning。数据构造不是只收集人工解题，而是让模型生成候选 CoT / TIR 解法，再用答案解析器、Python executor、majority voting 和 reward model 筛选。Qwen2.5-Math-RM-72B 被用作数学 reward model，既服务 rejection sampling，也服务后续 RL。

TIR 的训练细节尤其重要。模型生成的 token 包括自然语言思考、Python 代码、工具返回和最终答案；其中 executor 返回不是模型自己写的内容，训练时要 mask 掉 executor output token 的 loss。否则模型会学到错误目标：把环境返回当作自己应生成的文本。RL 阶段使用 GRPO，在数学题上用规则 verifier / reward model 打分，并用 KL 系数约束策略漂移。报告里还强调 Qwen2.5-Math 主要面向中英文数学，不建议当通用聊天模型用，这说明专业后训练会带来能力偏置，之后若要产品化必须做通用能力回填。

最小复现路径是：选 GSM8K / MATH / OlympiadBench 子集；让强模型为每题生成 4-16 条 CoT 和 Python TIR 候选；用答案 parser、`sympy`、Python 执行和 majority vote 过滤；把正确轨迹做 SFT；再对每题采样 8 个响应，用规则 verifier 给 0/1 或部分分，做 GRPO。对 TIR 数据单独处理 loss mask，并记录工具错误率、格式错误率和最终答案正确率。

#### [QwQ-32B](https://qwenlm.github.io/blog/qwq-32b/)：基于结果奖励的两阶段 RL

QwQ-32B 的公开博客把 RL 拆得很清楚：从 cold-start checkpoint 出发，第一阶段只扩 math 和 coding RL，不用传统 reward model，而是数学 accuracy verifier 和代码执行服务器。数学 reward 看最终答案是否正确；代码 reward 看生成代码是否通过预定义测试。这个阶段的目标是把"能不能解题"推上去。

第二阶段再做 general capabilities RL，奖励来自 general reward model 和一部分 rule-based verifier，用少量步数提升 instruction following、人类偏好对齐和 agent performance，同时避免数学和代码明显回落。这是一个非常实用的 recipe：先在高置信可验证任务上放大能力，再用通用偏好和规则数据修复体验。最小复现时，第一阶段可以只用数学和代码，第二阶段混入 5%-20% 通用指令/偏好/安全数据，观察 math、code、chat、agent 四类 eval 是否互相挤压。

#### [Qwen3](https://arxiv.org/abs/2505.09388)：Thinking 与 Non-Thinking 的统一后训练

Qwen3 的关键不是单纯"用了 GRPO"，而是把思考模式做成产品可控能力。后训练前两阶段是 long-CoT cold-start 和 reasoning RL：cold-start 用少量高质量长思考数据教格式、推理组织和答案边界；reasoning RL 用 query-verifier pairs 做 GRPO。报告披露了筛 query-verifier pair 的四条标准：不能出现在 cold-start 数据里；对 cold-start 模型可学习；尽量有挑战；覆盖广泛子领域。最终收集 3,995 个 query-verifier pairs，并强调大 batch、每题多 rollout、off-policy 提升样本效率，以及通过控制 entropy 稳住探索和利用。

后两阶段解决"模型只会长想"的问题。Qwen3 把有 reasoning path 和没有 reasoning path 的数据合成统一训练集，让模型同时支持 thinking 和 non-thinking；最后用 general-domain RL 回填通用能力、安全、多语言和工具体验。这个顺序对应一个可复现模板：先训练 `<think>` 格式和长 CoT；再只在 verifier 高置信任务上做 RL；然后混入短回答、普通聊天和工具指令，让模型学会什么时候不展开长思考；最后评估平均输出长度、正确率、用户偏好和非思考模式质量。

#### [Qwen3-Coder](https://qwenlm.github.io/blog/qwq3-coder/) 与 [Tongyi DeepResearch](https://arxiv.org/abs/2510.24701)：从答案奖励到过程奖励

[Qwen3-Coder](https://qwenlm.github.io/blog/qwq3-coder/) 的训练对象是 repository-level action：读文件、定位 bug、写 patch、运行测试、处理失败、提交修改。reward 的主信号来自单测、静态检查、编译、issue 需求覆盖和 patch 合理性。[Tongyi DeepResearch](https://arxiv.org/abs/2510.24701) 的训练对象是 search / read / synthesize 过程：任务不是回答一个事实，而是搜索证据、去重来源、比较冲突信息、组织带引用的报告。它们共同说明 Qwen 的 agent 后训练已经把"prompt -> answer"改成"environment episode -> verified outcome"。可复现时先做小规模 SWE-bench Lite 或网页 QA：固定工具协议，保留成功轨迹做 SFT，再用测试通过率或 answer judge 做 RL。

#### [Moonshot Kimi](https://arxiv.org/abs/2501.12599)

Kimi 的公开资料分三条线（[Kimi k1.5](https://arxiv.org/abs/2501.12599) 长思考 RL 简化框架、[Kimi K2](https://arxiv.org/abs/2507.20534) Agentic Intelligence 数据与工具闭环、[Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/) 研究 Agent 奖励设计）：分别对应 reasoning scaling、开放 agentic model、research agent。其中 k1.5 回答"长思考 RL 怎么稳定"，Kimi-Researcher 回答"研究型 agent 怎么从端到端 RL 里长出来"。从六阶段定位看，Kimi 主要改进"RL 阶段的长思考稳定性 + 数据与环境的研究 agent 构造 + 奖励设计的长度/证据链约束"。

#### [Kimi k1.5](https://arxiv.org/abs/2501.12599)：简化长思考 RL 训练

k1.5 的 motivation 是训练 test-time compute scaling。它没有把系统做得特别复杂，而是强调一个简洁框架：policy 采样多个响应，reward 只看 outcome，policy optimization 在 KL 约束下把高 reward 轨迹概率推高。报告明确把它和 MCTS、value function、process reward model 区分开：重点不是训练一个逐步打分器，而是让模型在足够多 rollout 中自己探索更有效的推理路径。

数据上，k1.5 把任务分成可验证和偏好型两类。数学、代码、选择题更适合 rule / execution verifier；开放问答、写作、复杂偏好需要 reward model 或 judge。训练时同一 prompt 会采样多个候选，reward 后形成相对优劣，再用 policy mirror descent 更新。可复现的关键是让采样数足够大，因为单个响应的 reward 噪声很高；同题多采样才能看出"哪些推理路径更稳"。

#### 长度奖励：控制过度思考

k1.5 报告专门讨论 overthinking：模型学会长 CoT 后，可能把 token 写得越来越多，甚至在已经得到答案后继续绕。它的 length reward 不是无条件奖励短，而是在同一题的多个候选里比较：正确答案中更短的响应得到额外奖励；错误答案不会因为短而被奖励，甚至长且错会被惩罚。这个设计把"正确且高效"写进 reward，而不是只靠 generation max length。

length reward 还要 warm up。训练早期模型还不会稳定解题，如果过早惩罚长度，会压制探索，模型可能学不到完整推理；等准确率上来后再加入长度项，才能把长思考压缩成有效思考。最小复现实验可以这样做：前 30%-50% RL steps 只用 correctness reward；之后在每题的正确样本里按长度排序，把较短正确解加 0.1-0.3 奖励，把过长错误解扣分；同时监控准确率和平均 response length，确认不是靠变短牺牲正确率。

#### Long-to-Short：从完整推理到精简回答

k1.5 的 long-to-short 思路和 length reward 配套。第一阶段允许模型用很长推理拿到正确答案，得到复杂问题的策略；第二阶段通过蒸馏、偏好或长度奖励，把冗余步骤压掉。这个流程和"直接训练短答案模型"不同：短模型要保留长模型学到的搜索和自检能力，只减少无用表达。复现时可以保留长 CoT 成功轨迹，然后让强模型或同模型生成 concise solution，做一轮 SFT / DPO，再用 verifier 确认短解仍然正确。

#### [Kimi K2](https://arxiv.org/abs/2507.20534)：Agentic Intelligence 的数据与工具闭环

K2 的公开报告强调 open agentic intelligence，重点不是单 benchmark，而是让模型在工具、代码和复杂任务里具备行动能力。对应的后训练样本应包含任务目标、工具协议、观察、动作、错误恢复和最终结果。K2 的学习点是 agent 数据不能只靠人工写演示，必须结合真实任务、合成任务、工具执行结果、verifier 和 judge，不断筛出成功轨迹，再回流到 SFT / RL。

#### [Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/)：覆盖证据链的研究 Agent 奖励

Kimi-Researcher 面向长程研究任务。它的训练单位是一个 research episode：模型提出搜索计划，调用搜索/浏览工具，阅读多个来源，提取证据，合并冲突信息，写出带引用的回答。最终 reward 不能只看"答案像不像"，还要看引用是否存在、证据是否支持结论、来源是否覆盖关键角度、是否遗漏反例、是否重复搜索低价值网页。最小复现路径是：构造 200-500 个需要多网页证据的问题；用浏览器工具记录轨迹；让 judge 分别打 evidence coverage、citation correctness、answer faithfulness、redundant-search penalty；先 SFT 成功轨迹，再用 episode-level reward 做 GRPO / DPO。

#### [字节 Seed / Doubao](https://seed.bytedance.com/en/public_papers/dapo-an-open-source-llm-reinforcement-learning-system-at-scale)

字节 Seed 的公开材料（[Seed1.5-Thinking](https://arxiv.org/abs/2504.13914)、[VAPO](https://arxiv.org/abs/2504.05118)、[DAPO](https://seed.bytedance.com/en/public_papers/dapo-an-open-source-llm-reinforcement-learning-system-at-scale)（[GitHub](https://github.com/BytedTsinghua-SIA/DAPO)）、[UI-TARS](https://arxiv.org/abs/2501.12326)（[GitHub](https://github.com/bytedance/ui-tars)）、[UI-TARS-2](https://huggingface.co/papers/2509.02544)、[Seed Prover 1.5](https://seed.bytedance.com/en/blog/seed-prover-1-5-advanced-mathematical-reasoning-through-a-novel-agentic-architecture)、[Seed1.8](https://seed.bytedance.com/en/blog/official-release-of-seed1-8-a-generalized-agentic-model)）适合学习两类东西：reasoning RL 的工程补丁，以及 GUI / prover 这类环境型 agent 后训练。DAPO、VAPO、UI-TARS-2 都不是只给算法名，而是在回答"为什么大规模 rollout 会不稳"。从六阶段定位看，Seed/Doubao 主要改进"RL/偏好优化阶段的采样与优势估计 + 数据与环境的 GUI/证明器构造 + SFT 冷启动的反射调优"。

#### [Seed1.5-Thinking](https://arxiv.org/abs/2504.13914)：推理模型的基础训练配方

Seed1.5-Thinking 的目标是用 RL 提升数学、代码和复杂推理。它的任务构造仍以可验证题为主：数学看答案，代码看执行，逻辑题看规则 verifier。SFT 阶段先给模型长 CoT 冷启动，RL 阶段再通过 outcome reward 放大可验证能力。这个模式和 DeepSeek-R1、Qwen3 类似，但 Seed 系列后续报告更强调训练系统、采样和 advantage 处理。

#### [DAPO](https://seed.bytedance.com/en/public_papers/dapo-an-open-source-llm-reinforcement-learning-system-at-scale)：GRPO 的四项稳定性改进

DAPO 的 motivation 是开源社区复现大规模 reasoning RL 时，常见失败不是因为 GRPO 公式不会写，而是因为样本、clip、长度和梯度归一化细节没处理好。它在 GRPO 上加四个关键组件。

Dynamic Sampling 处理"没有学习信号"的 prompt。如果同一题采样的所有答案全对或全错，组内 reward 方差接近零，advantage 没意义。DAPO 会持续采样或过滤，直到 batch 里保留有非零 advantage 的组，把算力放在边界题上。Clip-Higher 处理探索被 PPO clip 压住的问题：长 CoT 中某些低概率 token 可能打开新解法，如果上界太紧，正确但罕见的推理路径无法被充分强化；因此它把 `eps_clip_high` 设得高于下界，例如常见配置是 low 0.2、high 0.28。

Token-Level Policy Gradient 处理长响应被样本级平均稀释的问题。普通 sequence-level loss 会让长 CoT 的关键 token 和无关 token 一起平均，信号变弱；DAPO 改成按 token 聚合，让长推理链中每个生成 token 都更直接参与优化。Overlong Reward Shaping 处理超过长度限制的噪声样本：如果模型写满上下文还没完成，不能简单把截断文本当正常失败样本，否则 reward 噪声很大；需要对过长响应做分段惩罚、遮罩或单独 shaping。

最小复现路径是：用 Qwen2.5-7B/32B base、AIME/MATH 类可验证题、每题 8-16 个 rollout；先跑普通 GRPO 作为 baseline；再依次加入 dynamic sampling、clip-higher、token-level loss、overlong shaping；记录有效 prompt 比例、entropy、平均长度、AIME pass@1 和训练崩溃次数。DAPO 的价值正是在这种 ablation 中体现。

#### [VAPO](https://arxiv.org/abs/2504.05118)：面向长 CoT 的价值模型与优势估计

VAPO 研究的是 value-model-based RL。长 CoT 下，GAE 容易把最终稀疏 reward 衰减到前面 token，长短响应的 advantage 尺度也不一致。报告的 ablation 很有信息量：移除 decoupled GAE 会导致奖励信号指数衰减并大幅掉点；Length-Adaptive GAE 根据序列长度调节 GAE 参数，让短响应和长响应都能收到合适 credit；token-level policy gradient 给长响应更合理权重；positive-example LM loss 用正样本的语言模型损失稳住策略；group sampling 用较少 prompt、更多 repetition 提高组内比较质量。报告中还给出类似 `epsilon_low=0.2`、`epsilon_high=0.28`、positive LM loss weight 0.1、512 prompts 每个 16 samples 这类可复现实验级参数。

VAPO 的小型复现不必先训练大 value model，可以先做一个简化实验：同一批数学题分别用 GRPO 和带 value baseline 的 PPO/VAPO 风格训练，比较长答案任务上 advantage 方差、reward 延迟衰减和最终正确率。重点不是追求 SOTA，而是观察长序列 RL 里 credit assignment 如何影响训练稳定性。

#### [UI-TARS](https://arxiv.org/abs/2501.12326)：通过轨迹与偏好学习 GUI 操作

UI-TARS 的输入是截图/界面状态、历史动作和任务目标，输出是点击、输入、滚动等 GUI action。它面临的数据问题是高质量 action trace 极少。公开资料里可学习的方法是用大量虚拟机探索真实软件任务，从构造指令出发生成轨迹，再做规则过滤、VLM 评分和人工复核。Reflection tuning 把错误恢复也纳入训练：标注员指出轨迹中哪一步错了，并给出纠正动作或恢复步骤，再用 DPO 这类偏好优化让模型偏向能修正错误的策略。

#### [UI-TARS-2](https://huggingface.co/papers/2509.02544)：多轮 RL、混合环境与数据飞轮

UI-TARS-2 的 motivation 是 GUI-only agent 不足以完成真实任务：很多工作流还需要文件系统、终端、下载文件、读取本地数据。它引入 hybrid GUI environment，把 GUI、file system、terminal 放在统一 sandbox 里，并用大规模 rollout 平台支持多轮 RL。数据 flywheel 的逻辑是：模型生成新轨迹，高质量轨迹进入 SFT，低质量但有学习价值的数据进入 continual pre-training 或后续探索；每轮模型变强后，再产生更难、更长的轨迹。

这类系统的 reward 需要多层：最终任务是否完成，界面状态是否达到目标，文件是否生成，终端命令是否成功，动作是否无效或越界，回合数是否过多，是否违反安全边界。最小复现可以用 MiniWoB / BrowserGym / OSWorld 子集：定义统一 action schema；每个任务提供 reset、observe、step、success check；用 200 条人工或强模型轨迹做 SFT；再用多轮 rollout + success reward 做 RL；额外收集失败轨迹训练 reflection。

#### [Seed Prover 1.5](https://seed.bytedance.com/en/blog/seed-prover-1-5-advanced-mathematical-reasoning-through-a-novel-agentic-architecture)：形式化证明环境中的 Agentic RL

形式化数学的环境是 theorem prover，而不是浏览器。动作是选择 tactic、生成 lemma、调用搜索器；reward 是 proof 是否通过、证明长度、搜索步数和中间 lemma 是否复用。它给 agent RL 的启发是：只要环境能验证，就可以把复杂任务变成可训练 episode。[Seed1.8](https://seed.bytedance.com/en/blog/official-release-of-seed1-8-a-generalized-agentic-model) 则把 reasoning、multimodal、tools 和 generalized agent 能力放进同一模型卡，说明后训练目标正在从"题库正确率"扩展到"多环境任务执行"。

#### [DeepSeek](https://arxiv.org/abs/2501.12948)

DeepSeek 的公开材料（[DeepSeekMath](https://arxiv.org/abs/2402.03300)、[DeepSeek-R1](https://arxiv.org/abs/2501.12948)、[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)）是理解 GRPO / RLVR 的主线之一。DeepSeekMath 先给出 critic-free 的组内相对优势，R1 再证明纯规则 reward 可以诱发长思考，V3.2 则把可验证任务推进到 agentic task synthesis。从六阶段定位看，DeepSeek 主要改进"RL 阶段的 GRPO 组内相对优势 + 数据回流的拒绝采样 + SFT 冷启动的长短思考策略"。

#### [DeepSeekMath](https://arxiv.org/abs/2402.03300)：GRPO 的最小可复现方案

DeepSeekMath-RL 从 DeepSeekMath-Instruct 7B 出发，使用约 144K 个与 GSM8K、MATH 相关的 CoT 问题做 RL。每个问题采样一组输出，用 reward model / 规则正确性打分，再用组内均值和标准差归一化形成 advantage。这样省掉 PPO 的 critic/value model，显存和训练复杂度更低。报告中的典型设置包括 policy learning rate 1e-6、KL coefficient 0.04、每题 64 个输出、max length 1024、batch size 1024，并在每轮 exploration 后做一次 policy update。

GRPO 的直觉是：模型不需要知道"这个答案绝对值是多少"，只需要知道"同一题的这一组答案里哪个更好"。数学题天然适合，因为同题多采样会产生正确、错误、格式错、半对等候选。最小复现可以用 7B 数学 SFT 模型、MATH 子集、每题 8-16 个 rollout、答案 parser 打分；先用组内 reward 标准化，再加 KL 到参考模型，观察 GSM8K/MATH 提升和通用能力损失。DeepSeekMath 的经验是，SFT 后已经很强的模型，仍能通过 RL 获得 out-of-domain reasoning 提升。

#### [DeepSeek-R1-Zero](https://arxiv.org/abs/2501.12948)：从基础模型直接进行规则 RL

R1-Zero 的 motivation 是验证长 CoT 是否必须来自人工 SFT。它从 base model 直接做 RL，奖励主要是 accuracy reward 和 format reward：数学/代码题看最终答案或执行结果，格式 reward 保证模型按约定输出。训练后出现反思、回溯、自我验证、延长思考等行为，说明一部分 reasoning pattern 可以由 outcome reward pressure 诱发。

R1-Zero 的局限也很重要：可读性差、语言混杂、输出格式不稳定。这说明"纯 RL 能探索能力上限"，但不等于产品 recipe。复现实验时，应把 R1-Zero 当成研究实验：从 base model 出发，只用可验证数学/代码题，避免开放问答；评估除了正确率，还要看格式失败、重复、语言混杂和平均长度。

#### [DeepSeek-R1](https://arxiv.org/abs/2501.12948)：冷启动、推理 RL 与拒绝采样

R1 正式版回到更工程化的四段式。第一步用少量高质量 cold-start 数据修正格式、可读性和基本长思考结构；第二步做 reasoning-oriented RL，继续在数学、代码、逻辑等可验证任务上强化；第三步用训练后的模型做 rejection sampling，生成更多 SFT 数据，同时混入写作、事实问答、角色扮演等通用数据，避免模型只会解题；第四步做最终 RL，同时优化 helpfulness、harmlessness 和 reasoning。

这条流水线的核心教训是"能力和体验分开塑造，再合并"。可验证 RL 推高数学/代码，但会带来啰嗦、风格漂移和通用聊天退化；rejection sampling 和 final RL 是能力回填。最小复现可以用 1K cold-start 长 CoT、20K 可验证题 RL、再从 RL 模型采样并筛 10K 通用 SFT 数据，最后混合安全/偏好数据做一轮 DPO 或 GRPO。评估要同时看 math/code、普通指令、拒答、平均长度和格式稳定性。

#### [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)：从答案验证器到 Agentic Verifier

V3.2 的方向是让模型在工具环境中合成和完成 agentic tasks。这里的训练样本不是单个答案，而是带工具调用、环境观察、失败恢复和最终交付的 episode。reward 不只看最终文本，还看工具调用是否成功、是否找到证据、代码是否通过测试、任务是否在环境里真的完成。它和 MiniMax M2.1、UI-TARS-2、LongCat 属于同一趋势：RLVR 正从 math verifier 扩展到 software / browser / GUI / tool verifier。

#### [智谱 Z.ai / GLM](https://arxiv.org/html/2602.15763v1)

GLM 的主线是 ARC：Agentic、Reasoning、Coding（[GLM-4.5](https://arxiv.org/abs/2508.06471)、[GLM-5](https://arxiv.org/html/2602.15763v1)）。GLM-4.5 先证明这三类能力可以放在一个 MoE 模型里共同优化；GLM-5 则把后训练 recipe 讲得更清楚。从六阶段定位看，GLM 主要改进"RL 阶段的异步单轨迹优化 + SFT/RL 跨阶段蒸馏防遗忘 + 分阶段推理/Agent/通用对齐"。

#### [GLM-4.5](https://arxiv.org/abs/2508.06471)：混合推理与专家模型迭代

GLM-4.5 支持 thinking 和 direct response 两种模式。这个设计的 motivation 和 Qwen3 类似：复杂题需要长思考，普通助手场景不能每次都展开冗长 CoT。后训练阶段通过 expert model iteration 和 RL 同时提升 agentic、reasoning、coding。expert iteration 可以理解为"先让专项强模型产生或筛选高质量数据，再回流训练统一模型"；RL 则在数学、代码、工具和 agent benchmark 上继续放大可验证能力。

#### [GLM-5](https://arxiv.org/html/2602.15763v1)：Reasoning RL、Agentic RL 与 General RL

GLM-5 的公开报告明确写出 progressive alignment：先 multi-task SFT，引入 interleaved thinking modes；再做 reasoning RL；再做 agentic RL；最后 general RL 做人类风格对齐。Reasoning RL 主要处理数学、逻辑、代码这类 outcome verifier 高置信任务，先把长链推理和自检能力推上去。Agentic RL 把模型接入多轮工具环境、文件系统、代码库和软件工程任务，让模型学习"观察 -> 行动 -> 环境反馈 -> 修正"。General RL 最后回填普通聊天、简洁性、安全、指令跟随和风格，减少前两段带来的啰嗦和能力偏置。

#### 异步 Agent RL：解耦生成与训练

GLM-5 引入新的 asynchronous RL infrastructure，并使用 slime 的可定制 rollout interface。长程 agent rollout 的耗时差异很大：有的任务只需一轮，有的要跑测试、调用工具、等待环境。同步 PPO/GRPO 会让训练端等最慢 episode。GLM-5 的思路是把 rollout generation、环境交互、verifier 分支和训练解耦，让不同任务的经验可以持续进入训练队列。slime 的 server-based rollout execution 允许为不同任务写 multi-turn loop、tool invocation、environment feedback handling、verifier-guided branching，而不改底层训练栈。

#### On-Policy Cross-Stage Distillation：缓解阶段间遗忘

分阶段 RL 的风险是：Reasoning RL 学到的能力在 Agentic RL 或 General RL 中被冲掉。GLM-5 用 on-policy cross-stage distillation 在后续阶段保留前阶段强能力：当前策略在线生成数据，前一阶段能力作为蒸馏目标或筛选信号参与训练。最小复现可以做三段小实验：MATH/代码 GRPO 得到 reasoning 模型；SWE-bench Lite 或工具任务 RL 得到 agentic 模型；最后混入通用指令做 DPO/GRPO，同时用第一阶段模型在数学题上的输出做 distillation，观察数学是否回退。

#### [腾讯混元 Hunyuan](https://tencent.github.io/llm.hunyuan.T1/README_EN.html)

混元公开资料（[Hunyuan-T1](https://tencent.github.io/llm.hunyuan.T1/README_EN.html)、[Hunyuan-A13B](https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf)、[Hunyuan-A13B-Instruct](https://huggingface.co/tencent/Hunyuan-A13B-Instruct)）可以拆成 T1 的 reasoning RL 和 A13B 的 fast/slow thinking instruct model。A13B 技术报告细节披露不如 MiniMax/MiMo 那么完整，但 T1 页面给出了若干训练稳定性线索，足够作为 reasoning RL 系统设计参考。从六阶段定位看，Hunyuan 主要改进"RL 阶段的课程学习与策略稳定性 + 数据回流的回放与重置机制 + SFT 的快慢思考混合数据"。

#### [Hunyuan-T1](https://tencent.github.io/llm.hunyuan.T1/README_EN.html)：以 RL 为主的后训练算力分配

T1 明确说 post-training 阶段 96.7% 的算力投入在 reinforcement learning，目标是提升纯推理和人类偏好对齐。任务来源覆盖数学、逻辑推理、科学、代码等 world science and reasoning problems，并结合 ground-truth feedback。这里的 reward 主体仍是可验证信号：数学答案、逻辑题规则、代码执行、科学题标准答案或 judge。公开资料没有给出完整 reward 公式，但说明它不是单一聊天偏好 RL，而是 reasoning-heavy RL。

#### 课程学习、上下文扩展与 Token 效率

T1 的训练计划使用课程学习：逐步提高数据难度，并逐步扩展上下文长度，让模型既提升推理能力，也学会更有效地使用 token。这个设计和 MiniMax-M1 / Qwen3 的长思考训练一致：不能一开始就把最大长度、最难题和复杂 reward 全部打开，否则训练早期会被噪声和超长输出拖垮。复现时可以把数学/代码题按难度分三段，先短 CoT + 中等题，再长 CoT + 难题，最后加入长度/效率评价。

#### 数据回放、策略重置与统一奖励

T1 公开材料提到参考 data replay 和 periodic policy resetting，使长期训练稳定性提升超过 50%。这说明混元在处理长时间 RL 的策略漂移：data replay 防止模型忘掉早期能力；policy reset 则在策略偏离过大或出现退化时，把模型拉回更稳定的 checkpoint / reference。偏好对齐阶段采用 self-reward + reward model：早期 T1-preview 作为 self-reward 评估器，对输出进行综合评分，再叠加 reward model，引导模型自我改进。可复现时可以维护一个 replay buffer，混入旧阶段高质量样本；每 N 步评估 entropy、格式错、平均长度和通用 eval，一旦退化就回滚或重置 reference。

#### [Hunyuan-A13B](https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf)：Fast 与 Slow Thinking

A13B-Instruct 模型卡展示了 slow-thinking 默认开启，也可以通过 `enable_thinking=False` 关闭 CoT。这说明后训练数据要同时包含两类响应：带 `<think>` 的慢思考轨迹，以及直接回答的快思考轨迹。否则模型要么每次都慢想，要么复杂问题推理不足。最小复现可以在同一批 prompt 上构造两份标签：复杂题保留思考过程，简单问答只给短答；SFT 后在 RL / DPO 中加入"是否需要 thinking"的偏好，评估复杂题正确率和简单题平均长度。

#### [百度 ERNIE](https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)

ERNIE 的公开资料（[ERNIE 4.5](https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)、[ERNIE 5.0](https://arxiv.org/abs/2602.04705)）披露很有结构：LLM post-training 是 SFT + RL，RL 阶段用 Progressive RL 和 Unified Preference Optimization；VLM post-training 则是三段 SFT + 一段 reasoning RL。ERNIE 的价值在于把多任务、多 reward、多模态后训练的"兼容性"问题讲得比较清楚。从六阶段定位看，ERNIE 主要改进"RL/偏好优化阶段的多 reward 统一与 Progressive RL + 数据与环境的多模态任务构造 + SFT 的多能力域分桶覆盖"。

#### SFT：覆盖任务域后进入 RL

ERNIE 4.5 的 SFT 覆盖通用指令、逻辑、数学、代码、专业任务、安全和多模态理解。这里的关键不是数据量，而是每个能力域要有可评测目标。对 LLM 来说，SFT 让模型学会基本回答格式和任务能力；对 VLM 来说，三段 SFT 分别强化视觉感知、复杂视觉推理，以及 thinking / non-thinking 数据混合。多模态模型如果直接进 RL，很容易把视觉识别错误和推理错误混在一起，所以先用 SFT 把 perception 打稳。

#### Unified Rewarding System：统一多类奖励信号

ERNIE 4.5 图示里列出 rule-based reward、RLLM、sandbox、RDRM、checklist-aware verifier、GRM、DRM 等组件。它要解决的是 reward 来源异构：数学题可能是规则答案；代码题是 sandbox 执行；开放问答是 generative reward model；安全/清单任务是 checklist verifier；偏好任务是 discriminative reward model。若不做 domain normalization，不同 reward 尺度会互相压制。ERNIE 的统一奖励系统可以抽象成三步：先按任务域选择 verifier / RM；再把 reward 归一化到可比较尺度；最后按训练阶段控制不同任务权重。

#### Progressive RL：从 Logic RL 到 General RL

ERNIE 4.5 把 LLM RL 分成 Stage 1 Logic RL、Stage 2 Reasoning RL、Stage 3 General RL。Logic RL 用更干净、规则性更强的任务稳定推理格式；Reasoning RL 扩展到数学、代码、复杂推理；General RL 回填普通指令、人类偏好和安全。这个顺序和 GLM-5 / Qwen3 的"先能力、再泛化"一致。最小复现可以按这个顺序组织数据：先 2K 逻辑/符号题，后 10K 数学/代码题，最后 10K 通用偏好题；每阶段单独评估前一阶段能力是否被覆盖。

#### UPO：多任务 RL 的尺度与稳定性

Unified Preference Optimization 的动机是混合 reasoning tasks 和 non-reasoning tasks 时，reward-format、domain normalization、informative prompt filtering 都会影响训练。数学/代码 0-1 reward、偏好分数、安全分数不能直接相加。UPO 的复现思路是：为每个任务域维护 reward normalization；过滤没有信息量的 prompt；对不同 reward source 做分域权重；训练时记录各域 reward 均值和方差，避免某一类任务主导更新。

#### [ERNIE 5.0](https://arxiv.org/abs/2602.04705)：统一多模态后训练

ERNIE 5.0 继续面向文本、图像、视频、语音统一模型。这里最大的难点是 reward 可比性和模态平衡：图像理解 reward、视频时序 reward、文本偏好 reward、语音任务 reward 的错误来源完全不同。复现时不要把多模态题简单拼成文本 JSON，而要为每种模态准备感知 eval、推理 eval 和偏好 eval，再统一做阶段式 SFT/RL。

#### [阶跃星辰 StepFun](https://stepfun.ai/research/en/step3)

StepFun 的公开资料（[Step3](https://stepfun.ai/research/en/step3)、[STEP3-VL-10B](https://huggingface.co/papers/2601.09668)、[Step-DeepResearch](https://arxiv.org/abs/2512.20491)）覆盖多模态 reasoning 和 deep research agent。STEP3-VL-10B 披露了一个紧凑 10B VLM 怎么靠 scaled post-training 追近更大模型；Step-DeepResearch 则属于研究型 agent 训练。从六阶段定位看，StepFun 主要改进"数据与环境的多模态/研究 agent 任务构造 + RL 奖励的视觉证据链与引用检查 + SFT 冷启动的多模态推理轨迹"。

#### [STEP3-VL-10B](https://huggingface.co/papers/2601.09668)：全参数预训练后的视觉推理 RL

这份报告的 motivation 是小模型也能在多模态复杂推理上接近大模型，但需要把视觉语言协同和后训练一起设计。模型先在 1.2T multimodal tokens 上做统一、完全解冻的预训练，把 perception encoder 和 Qwen3-8B decoder 对齐；post-training 阶段再做超过 1K iterations 的 reinforcement learning。这里的关键是：VLM 的 RL 不是只训文本答案，而是让视觉证据、文本推理和答案生成一起被 reward 约束。

#### RLVR 与 RLHF：分开处理视觉验证与开放偏好

视觉数学、OCR 后计算、图表读数、选择题、几何/空间题可以做 RLVR：答案能规则验证，或能由程序/标准答案检查。开放式图片描述、复杂审美、视觉安全和解释质量更适合 RLHF / judge reward。复现时可以把数据分成两桶：第一桶用 MathVista、ChartQA、OCR-VQA、几何题做 exact / numeric verifier；第二桶用多模态 judge 给 helpfulness、faithfulness、detail、safety 打分。不要把两类 reward 直接相加，先分域归一化。

#### PaCoRe：并行视觉假设与答案协调

Parallel Coordinated Reasoning 的目标是扩展 test-time compute。多模态任务里，错误常来自"看错图"而不是"不会推理"。PaCoRe 让模型探索多个视觉假设或推理路径，再合成更可靠的答案。训练上对应两个信号：候选路径要多样且有证据，最终整合要正确且不幻觉。小型复现可以做 self-consistency 的多模态版：同一张图采样多条证据链，用 verifier / judge 选出正确链，再 SFT 模型学习"列出候选视觉证据 -> 交叉检查 -> 给答案"。

#### [Step-DeepResearch](https://arxiv.org/abs/2512.20491)：训练完整研究过程

Deep research agent 的任务包括搜索、浏览、证据抽取、冲突比较、引用和长文组织。SFT 阶段应使用高质量 research trajectories，教模型如何规划 query、如何读来源、如何记录证据；RL 阶段的 reward 则要拆成 answer correctness、citation existence、evidence support、source coverage、redundant search penalty、final report structure。复现时可以用 300 个多来源问题、一个搜索 API、一个浏览器提取器和 citation checker，先训练轨迹格式，再对最终答案和引用证据做 episode-level reward。

#### [美团 LongCat](https://tech.meituan.com/2026/02/02/LongCat-Flash-Thinking-2601-techreport.html)

[LongCat-Flash-Thinking-2601](https://tech.meituan.com/2026/02/02/LongCat-Flash-Thinking-2601-techreport.html) 是一份很像"agent RL 工程系统设计说明"的报告。它的核心不是某个 reward 公式，而是环境扩展、强化学习扩展、噪声鲁棒训练和 heavy thinking。从六阶段定位看，LongCat 主要改进"RL 阶段的全异步流式训练系统 + 数据与环境的自动环境生成图谱 + SFT/RL 的噪声鲁棒扰动注入"。

#### 环境扩展：从领域定义生成可解工具环境

LongCat 的 motivation 是真实 agent 场景太多，手工适配 prompt、工具链和环境接口成本极高。它构建覆盖 20+ 领域、上万情境的环境生成系统：输入领域定义，自动合成 60+ 工具、数据库 schema、工具调用接口和验证逻辑。覆盖的场景包括文件管理、数据分析、电商零售、电信服务等。这个设计把"训练数据"变成"可交互环境图谱"。

环境生成最难的是一致性。一个复杂环境可能有几十个数据库和工具参数依赖，若随机生成任务，容易出现"看似可解、实际无解"。LongCat 使用可解路径优先：先随机采样一条长工具调用链作为黄金工具链；围绕这条链构造任务和数据库状态；再用 BFS 受控扩展环境子图，保证新工具的前序依赖已存在；根据环境复杂度和剩余工具动态加入新黄金链；如果工具数不足 20，就从全局工具库补一条中等规模可用链。这个方法的可复现重点是：先保证至少一条成功路径存在，再扩环境，而不是先造环境再祈祷任务可解。

#### 冷启动数据：真实轨迹与双路合成

LongCat 在 RL 前把预训练/微调目标重新定义为"给 RL 提供冷启动策略"。有真实数据的领域，如数学和编码，通过质量控制和可执行验证筛选高质量轨迹；缺真实数据的领域，如搜索和工具使用，使用文本驱动合成和环境锚定合成。文本驱动合成从任务描述出发生成轨迹；环境锚定合成从已有工具链和数据库状态出发生成任务，保证任务能被环境验证。复现时可以先做一个 5 个工具的小环境：订单查询、退款、库存、用户信息、日志；先采样黄金链，再让模型生成任务和轨迹。

#### DORA：全异步流式 RL

Agent rollout 耗时差异巨大，同步训练会浪费大量 GPU。DORA 支持多版本模型并行探索，不同版本产生的经验随产随收进样本队列；训练器无需等待所有任务完成。调度上拆成轻量 Rollout Manager 和多个 Rollout Controller，后者各自管理虚拟 rollout 组，通过数据并行处理环境交互。环境部署通过扩展 PyTorch RPC，把环境实例化到 CPU 空闲机器上。

为适配 5600 亿参数 MoE，DORA 还做 Prefill-Decode 解耦和 KV-cache 交换。PD 解耦把长上下文 prefill 和 decode 放到不同设备组，避免多轮交互中 prefill 阻塞 decode；KV-cache 以 chunk 级聚合、异步传输、计算重叠和 CPU 驻留方式动态交换，减少重复计算。资源分配上做双层平衡：整体按环境难度调 rollout 配额，批内保证任务域多样性。报告称这种系统达到传统同步训练 2-4 倍效率，并支持千步以上稳定训练。

#### 噪声鲁棒训练：在训练中注入真实扰动

LongCat 主动注入工具超时、工具报错、返回缺字段、数据库不一致、指令歧义、需求变更等扰动，让模型学习恢复。reward 不应只看最终成功，也要奖励错误检测、重新计划、换工具、向用户澄清。最小复现可以在工具环境里随机让 10%-30% 调用失败或返回部分字段，训练模型根据错误码重试、改参数或走备用链；评估 clean success rate 和 noisy success rate 的差距。

#### Heavy Thinking：同时扩展推理宽度与深度

LongCat 的重思考模式不是只把单条 CoT 拉长，而是先生成多条推理/行动路径，再用总结模型分析、筛选和整合。它适合复杂 agent 任务，因为单一路径一旦早期工具选择错，后面会越走越偏。小型复现可以让模型对同一工具任务采样 3-5 条计划，用 verifier / judge 选最佳计划或合并计划，再执行。训练时把"候选路径 -> 比较 -> 最终计划"的轨迹回流 SFT / RL。

#### [蚂蚁 Ling / Ring](https://ant-ling.medium.com/deep-insight-efficient-inference-introducing-the-trillion-parameter-ling-1t-model-77d6170e5e8e)

Ling / Ring 的公开材料（[Ling-1T](https://ant-ling.medium.com/deep-insight-efficient-inference-introducing-the-trillion-parameter-ling-1t-model-77d6170e5e8e)、[Ring-1T](https://ant-ling.medium.com/ring-1t-release-the-flow-state-of-insight-born-of-epiphany-c20e8e32817c)）更偏模型发布和推理效率，没有像 DeepSeek-R1、Qwen3 或 MiniMax M2.1 那样展开完整后训练流水线，因此更适合作为"产业信号"阅读，而不是直接复刻训练配方。从六阶段定位看，Ling/Ring 主要改进"评测阶段的推理效率与部署约束 + SFT/RL 的快慢思考数据设计，以及后训练与推理部署的联合目标定义"。

能明确学习到的是两点：第一，trillion-scale MoE 模型也会把 deep thinking / insight 类能力作为 post-training 目标，而不是只做聊天对齐；第二，长序列推理和高效推理部署要一起考虑。

复现时主要借鉴 fast/slow thinking 数据设计：为复杂数学、代码、分析题保留长思考轨迹，为普通问答保留短答；用偏好数据惩罚无意义长思考，并用 eval 同时看正确率和 token cost。

#### [华为 Pangu](https://github.com/pangu-tech/pangu-ultra)

Pangu 公开信息（[Pangu Ultra](https://github.com/pangu-tech/pangu-ultra)、[Pangu Pro MoE](https://arxiv.org/abs/2505.21411)、[盘古开源新闻](https://www.huawei.com/cn/news/2025/7/pangu-opensource)）的重点在昇腾原生训练、MoE 稀疏效率和开源模型体系，后训练细节没有像 R1/Qwen/MiniMax 那样展开。从六阶段定位看，Pangu 主要改进"目标定义阶段的硬件部署约束联合设计 + RL/评测阶段的 MoE 效率与吞吐权衡，以及后训练与昇腾原生训练系统的耦合优化"。

可学习点是硬件和训练 recipe 的耦合：如果模型要部署在 Ascend NPU 上，post-training 不能只看算法，还要考虑 MoE 路由、长上下文显存、推理吞吐和快慢思考的成本。

复现层面可以把它作为"工程约束型后训练"案例：同一个 reasoning 模型同时评估正确率、激活专家数、平均输出长度、吞吐和部署成本。

#### [01.AI Yi](https://arxiv.org/abs/2412.01253)

[Yi-Lightning](https://arxiv.org/abs/2412.01253) 披露的是传统产品级 LLM 后训练线：pre-training 之后做 SFT 和 RLHF，并强调 multi-stage training、synthetic data construction、reward modeling，以及 RAISE 安全框架贯穿 pre-training、post-training 和 serving。它不像 agent 报告那样提供工具环境 recipe，但适合学习"聊天模型如何被人类偏好拉齐"。从六阶段定位看，Yi 主要改进"SFT 的多阶段数据治理 + RL/偏好优化的 reward modeling 与安全框架贯穿，以及评测阶段的真实人类偏好对齐验证"。

可复现时可以做三段：用高质量中文/英文指令做 SFT；为同一 prompt 采样多个回答，人工或 judge 排序训练 reward model；再做 PPO/DPO，并单独评估 Chinese、Math、Coding、Hard Prompts 和 safety。

Yi-Lightning 还提醒一点：静态 benchmark 和真实人类偏好会有差距，后训练指标不能只看题库。

#### [InternLM / 上海 AI Lab](https://arxiv.org/abs/2403.17297)

[InternLM2](https://arxiv.org/abs/2403.17297) 是开源社区理解传统 RLHF 工程化的重要参考。它的重点不是长 CoT RLVR，而是数据治理、SFT、reward modeling 和 online RLHF。从六阶段定位看，InternLM 主要改进"RL/偏好优化阶段的条件化在线 RLHF + SFT/RM 的数据治理体系，以及评测阶段的分域多目标能力评估"。

COOL（Conditional Online RLHF）解决偏好优化让模型在不同任务域上漂移的问题：某些用户喜欢简洁，某些任务需要详细，安全场景又要保守。条件化训练让模型根据任务条件、偏好条件或数据域调整优化目标，而不是把所有偏好压成一个平均人。

最小复现可以这样做：为每条偏好数据标注 domain / style / safety 条件；训练 reward model 时把条件作为输入；在线 RLHF 时按条件采样 prompt 和 reward；评估时分域看 helpfulness、harmlessness、verbosity 和中文能力。即使没有可执行 verifier，偏好 RL 也要做数据分桶和条件控制，否则模型容易向单一风格塌缩。

#### [百川 Baichuan 与 360 智脑](https://arxiv.org/abs/2309.10305)

两家公司的公开资料（[Baichuan 2](https://arxiv.org/abs/2309.10305)、[360Zhinao](https://arxiv.org/abs/2405.13386)）代表了经典的中文开源对齐路线。从六阶段定位看，Baichuan 主要改进"SFT 冷启动 + RM 训练 + PPO 偏好优化的经典三段式闭环，以及数据回流阶段的 RM 数据过滤与重标工具化"。

[Baichuan 2](https://arxiv.org/abs/2309.10305) 是国内较早公开 SFT -> RM -> PPO 经典对齐流程的报告。SFT 阶段先让 base model 学会对话和指令；RM 阶段收集偏好比较，训练 reward model；PPO 阶段用 RM 分数优化策略并加 KL 约束。它适合放在课程里作为 InstructGPT 路线的中文/开源对照：在没有大规模可验证 RLVR 时，SFT/RM/PPO 仍是完整后训练闭环。

[360Zhinao](https://arxiv.org/abs/2405.13386) 的公开材料强调数据质量和数据治理。RM 不只是 PPO 的奖励器，也可以做 judge、过滤器和数据重标工具：对候选回答打分，筛掉低质量样本，发现重复模式，再回流 SFT。

可复现实验可以把同一批中文指令采样 4 个回答，用 judge/RM 打分，保留 top-1 做 rejection sampling SFT，再用 bottom/top pair 做 DPO。这个流程虽然不如 agent RL 酷，但非常接近大量真实产品模型的日常后训练。

#### [昆仑万维 Skywork 与 小米 MiMo](https://arxiv.org/abs/2505.07608)

Skywork-OR1 和 MiMo（[Skywork-OR1](https://huggingface.co/papers/2505.22312)、[MiMo](https://arxiv.org/abs/2505.07608)、[MiMo-VL-Miloco](https://arxiv.org/abs/2512.17436)）都适合学习"小模型 / 蒸馏模型继续做 RL"的问题。它们不像 frontier lab 只堆规模，而是关注 entropy collapse、数据难度、reward 稀疏和训练稳定性。从六阶段定位看，Skywork/MiMo 主要改进"RL 阶段的 entropy 动力学与 premature collapse 缓解 + 数据与环境的难度驱动采样 + 评测阶段的 entropy/重复率监控指标"。

[Skywork-OR1](https://huggingface.co/papers/2505.22312) 建在 DeepSeek-R1-Distill 系列之上。蒸馏模型已经会长 CoT，但继续 RL 时很容易过早收敛到少数表达和解题模式，entropy 下降后探索消失。报告的主线就是通过训练 pipeline 和 ablation 找出影响 entropy dynamics 的因素，并证明缓解 premature entropy collapse 对测试性能关键。公开结果显示 32B 平均准确率从 57.8% 到 72.8%，7B 从 43.6% 到 57.5%，并开源权重、代码和数据。

复现重点是监控 entropy，而不是只看 reward。用 R1-Distill-7B 做数学/代码 RL；每步记录 token entropy、response length、pass@1、重复 n-gram、格式错误率；尝试调整采样温度、KL、clip、数据难度和动态采样。如果 reward 上升但 entropy 快速塌缩，后期往往泛化差。

[MiMo](https://arxiv.org/abs/2505.07608)-7B 在 post-training 阶段构造 130K verifiable mathematics and programming problems 做 RL。数学题用答案 verifier；编程题用测试执行。它还提出 test-difficulty-driven code reward，缓解代码 reward 稀疏：不是所有测试通过/失败都等价，能通过更难测试或更多隐藏测试应有更细粒度奖励。Strategic data resampling 则用来稳定训练，把算力集中在既有挑战又可学习的样本上。

MiMo 的最小复现非常清楚：准备 80K 数学题和 50K 编程题，或更小的 5K/2K 版本；数学用 Math-Verify / parser 判答案；代码题为每题准备 easy/medium/hard tests，reward 按通过测试难度加权；每轮 RL 后统计哪些题全对、全错、半对，对全对/全错下采样，对半对题提高采样。这个方法对 7B 尤其重要，因为小模型训练预算有限，不能把 rollout 浪费在没有学习信号的样本上。

[MiMo-VL-Miloco](https://arxiv.org/abs/2512.17436) 延续了"小模型 + 高质量可验证数据 + 稳定 RL"的路线，但对象变成视觉语言。可学习点和 STEP3-VL 类似：视觉题要区分 perception 错误和 reasoning 错误；reward 需要同时覆盖答案正确性、视觉证据引用和输出格式。复现时可把数学图表/OCR/几何题作为 RLVR 数据，再混入开放图片描述偏好数据做回填。

#### [快手、商汤、讯飞](https://arxiv.org/abs/2507.01949)

这三家公司公开了多模态后训练（快手 [Kwai Keye-VL](https://arxiv.org/abs/2507.01949)）、原生理解生成（商汤 [SenseNova U1](https://www.sensetime.com/en/news/51170629)）和深度推理（讯飞 [Spark X1](https://news.cgtn.com/news/2025-01-15/China-releases-Spark-X1-deep-reasoning-model-that-packs-a-punch-1AbIq8PzzEI/index.html)）的动态，但缺乏完整 recipe。阅读时看它们分别押注哪些能力面：VLM、多模态生成、深度推理、中文场景、端到端产品体验。不要从发布材料反推出未披露的 SFT/RL 细节。从六阶段定位看，这三家主要在"目标定义阶段的中文产品场景定制 + 多模态/推理能力选型 + 评测阶段的本土场景适配"上有参考价值，完整训练流水线未公开。

:::

::: details 国外团队公开实践

### 国外团队

国外团队的公开材料则在通用偏好对齐、安全训练、端侧部署约束和企业级工具集成方面有更系统的讨论，正好可以作为国内方法在通用体验、安全和部署侧的补充对照。

#### [OpenAI](https://openai.com/index/openai-o1-system-card/)

OpenAI 的公开资料横跨三代后训练（[InstructGPT](https://arxiv.org/abs/2203.02155)、[GPT-4](https://arxiv.org/abs/2303.08774)、[o1](https://openai.com/index/openai-o1-system-card/)、[o3/o4-mini](https://openai.com/index/o3-o4-mini-system-card/)、[o3 Operator](https://openai.com/index/o3-o4-mini-system-card-addendum-operator-o3/)、[GPT-4.5](https://openai.com/index/gpt-4-5-system-card/)、[GPT-5](https://openai.com/index/gpt-5-system-card/)、[GPT-5.1](https://openai.com/index/gpt-5-system-card-addendum-gpt-5-1/)、[GPT-5.4 Thinking](https://openai.com/index/gpt-5-4-thinking-system-card/)、[GPT-5.5](https://openai.com/index/gpt-5-5-system-card/)、[GPT-5.5 Instant](https://openai.com/index/gpt-5-5-instant-system-card/)、[GPT-5-Codex](https://openai.com/index/gpt-5-system-card-addendum-gpt-5-codex/)、[GPT-5.1-Codex-Max](https://openai.com/index/gpt-5-1-codex-max-system-card/)、[GPT-5.2-Codex](https://openai.com/index/introducing-gpt-5-2-codex/)）：InstructGPT 的经典 RLHF、o-series 的 reasoning / deliberation、安全系统卡中的 deliberative alignment，以及 Codex / Operator 类 agent 模型。闭源系统卡不会披露完整 recipe，但方法边界很清楚。从六阶段定位看，OpenAI 主要改进"RL 阶段从 RLHF 到推理/Agent RL 的完整演进 + 目标定义阶段的安全与工具约束 + 评测阶段的对抗性安全评估"。

#### [InstructGPT](https://arxiv.org/abs/2203.02155)：RLHF 的最小闭环

InstructGPT 的流程可以直接复现成教学实验。第一步是 demonstration SFT：标注员写高质量回答，让 base model 先学会按指令完成任务。第二步是 reward modeling：对同一 prompt 采样多个回答，标注员排序，训练 reward model 预测人类偏好。第三步是 PPO：用 reward model 给 policy 输出打分，并用 KL penalty 约束 policy 不要偏离 SFT 模型太远。这里的关键是三类数据不同：SFT 数据是"好答案"，RM 数据是"偏好比较"，PPO 数据是"prompt + on-policy samples"。

最小复现可以用 5K 指令做 SFT；为 1K prompt 各采样 4 个回答，做 pairwise preference 训练 RM；最后用 PPO / DPO / IPO 任选一种偏好优化。评估不能只看 reward model 分数，还要人工或 LLM judge 看 helpfulness、truthfulness、toxicity、verbosity 和 instruction following，因为 RM 很容易被策略钻空子。

#### GPT-4 到 o-series：推理后训练扩展动作空间

[GPT-4](https://arxiv.org/abs/2303.08774) 技术报告只高层描述 post-training 和 safety；[o1](https://openai.com/index/openai-o1-system-card/)/[o3/o4-mini](https://openai.com/index/o3-o4-mini-system-card/) 系统卡更明确：模型通过强化学习学会在回答前进行更长 deliberation，并在需要时使用工具。这里的变化是 action 不再只是"下一个 token"，还包括何时写代码、何时浏览、何时调用图像/文件工具、何时停止、何时拒答。reward 也从人类偏好扩展到最终答案、工具结果、策略合规、安全边界和用户体验。

这种能力的可复现抽象是：选择一个有工具的任务族，例如代码执行数学题；模型输出思考和工具调用；环境返回 execution result；reward 同时检查最终答案、工具格式、调用次数和安全策略。先 SFT 少量成功工具轨迹，再做 RL。这样能复现 o-series 的方法形态，而不是复刻闭源配方。

#### Deliberative Alignment：将安全约束纳入推理

OpenAI 系统卡中反复出现的一个方向是让模型在困难安全问题上先推理策略，再决定回答方式。早期安全对齐容易变成拒答模板；deliberative alignment 更像把 policy spec、边界案例和安全评测做成训练任务：模型要识别请求类型，判断能否安全完成，必要时转换成安全替代方案。复现时可以构造安全 prompt、策略条款和正确处理示例，用 SFT 教模型引用策略，再用 preference/RL 奖励"安全完成"而不是盲拒。

#### Operator 与 Codex：真实环境中的 Agent 后训练

Operator 和 Codex 类模型把后训练扩展到 browser / software engineering episode。coding agent 的环境要包含仓库状态、测试命令、patch verifier、lint、用户指令层级和失败恢复；browser agent 的环境要包含页面状态、可点击元素、任务成功检查和安全沙箱。[GPT-5-Codex](https://openai.com/index/gpt-5-system-card-addendum-gpt-5-codex/) 的系统卡明确说它通过真实软件工程任务上的 RL 训练，学习贴近人类代码风格和 PR 偏好、严格遵循指令、反复运行测试直到通过；[GPT-5.1-Codex-Max](https://openai.com/index/gpt-5-1-codex-max-system-card/) 继续把训练对象扩展到跨多个上下文窗口的长程 agentic coding，通过 compaction 在百万 token 级任务中保持连贯；[GPT-5.2-Codex](https://openai.com/index/introducing-gpt-5-2-codex/) 则强调 SWE-Bench Pro、Terminal-Bench 2.0、Windows 原生环境、长上下文理解和可靠工具调用。最小复现可以用 SWE-bench Lite：checkout 仓库，给 issue，模型编辑文件，运行 tests，reward 为测试通过 + patch 合理性；也可以用 MiniWoB / BrowserGym：模型观察 DOM/截图，点击输入，reward 为任务完成和动作合法。

#### [Anthropic](https://arxiv.org/abs/2212.08073)

Anthropic 的公开资料（[Constitutional AI](https://arxiv.org/abs/2212.08073)、[Anthropic CAI overview](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)、[Claude 4 System Card](https://www.anthropic.com/claude-4-system-card)、[Claude Sonnet 4.5](https://www.anthropic.com/claude-sonnet-4-5-system-card)、[Claude Opus 4.5](https://www.anthropic.com/claude-opus-4-5-system-card)、[Claude Opus 4.6](https://www-cdn.anthropic.com/0dd865075ad3132672ee0ab40b05a53f14cf5288.pdf)）最值得学的是 Constitutional AI 和系统化安全评测。它的公开材料不提供 Claude 4 的完整训练 recipe，但 Constitutional AI 是可以复现的方法。从六阶段定位看，Anthropic 主要改进"RL/偏好优化阶段的 Constitutional AI 与 RLAIF + 评测阶段的系统化安全与对抗性评估 + SFT 阶段的原则引导安全对齐"。

#### [Constitutional AI](https://arxiv.org/abs/2212.08073)：用原则与 AI Feedback 扩展安全数据

传统 RLHF 需要大量人工比较。Constitutional AI 先定义一组原则，即 constitution；supervised phase 中，模型先生成回答，再根据 constitution 自我批改并重写，形成更安全的 SFT 样本。preference phase 中，AI 根据 constitution 比较两个回答，生成偏好数据，训练 preference model，最后用 RL 优化 policy。这就是 RLAIF：把人工从逐条偏好判断中部分移到原则设计和质量审计上。

最小复现路径是：写 20-50 条安全/诚实/隐私/无害原则；为有风险 prompt 采样回答；让强模型按原则指出问题并重写；用重写数据做 SFT；再让强模型对两条回答按原则排序，训练 DPO 或 reward model。评估时要单独看 over-refusal，因为安全原则过强会让模型拒绝正常请求。

#### Claude 系统卡：联合设计后训练与评测

[Claude 4](https://www.anthropic.com/claude-4-system-card) 系列系统卡关注 reward hacking、sabotage、sycophancy、alignment faking、hidden objectives、jailbreak、extended thinking 下的策略遵循等。这里的学习点不是某个 RL 公式，而是安全后训练必须有 adversarial evaluation。模型在训练 reward 下表现好，不代表不会在长上下文、工具调用、角色扮演或高压提示里偏离。

#### Extended Thinking 的安全风险

当模型有更长思考和工具能力时，安全训练不再只是"输出拒答"。模型可能在思考中制定规避策略，或在工具环境中完成不该完成的步骤。因此安全 reward 要覆盖策略遵循、工具限制、信息泄露、隐私、欺骗和拒答质量。复现时可以把工具任务和安全规则结合：例如要求模型处理文件，但禁止读取无关敏感文件；reward 同时检查任务成功和越权行为。

#### [Google DeepMind](https://arxiv.org/abs/2507.06261)

Google DeepMind 的公开资料（[Gemini 1.5](https://arxiv.org/abs/2403.05530)、[Gemini 2.5](https://arxiv.org/abs/2507.06261)、[Gemini 2.5 Deep Think](https://blog.google/products/gemini/gemini-2-5-deep-think)、[Gemini 2.5 Computer Use](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-computer-use-model/)、[Gemini 3.1 Pro](https://deepmind.google/models/model-cards/gemini-3-1-pro/)、[Gemma 3](https://arxiv.org/abs/2503.19786)）披露粒度不如开放论文，但方向非常明确：多模态、长上下文、工具、reasoning 和安全评测一起做。Gemini / Gemma 这条线适合学习"统一多模态模型怎么设计后训练任务"。从六阶段定位看，Google 主要改进"数据与环境的多模态长上下文任务构造 + RL 奖励的证据对齐与过程验证 + 评测阶段的跨模态多维度能力评估"。

#### Gemini 1.5 与 2.5：长上下文中的证据对齐

长上下文模型不是只把 context window 扩大。后训练要让模型在几十万 token、图片、视频、文档中定位证据，避免把不相关片段混进答案。任务构造应包含 needle-in-haystack、长文档 QA、多文档冲突、视频事件定位和跨模态引用。reward 不能只看最终答案，还要看证据位置是否正确、引用是否支持结论、是否忽略了无关干扰。

#### Deep Think：并行探索与答案汇总

[Gemini 2.5 Deep Think](https://blog.google/products/gemini/gemini-2-5-deep-think) 展示的是 test-time compute scaling 的另一种形态：生成多条候选思路，比较并整合。它和 LongCat heavy thinking、self-consistency 属于同族。训练上需要 reward 区分"有用的多样性"和"无意义发散"：候选路径要覆盖不同假设，整合答案要比单一路径更正确。小型复现可以对数学/视觉题采样 5 条推理，verifier 选正确路径，再训练模型输出"候选分析 + 最终合并答案"。

#### Computer Use：GUI 环境中的安全动作学习

[Gemini 2.5 Computer Use](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-computer-use-model/) 面向屏幕状态和动作序列：观察网页/桌面，输出点击、输入、滚动等动作，再根据环境反馈继续。reward 至少包含任务完成、动作有效、回合数、是否误点敏感控件、是否泄露信息、是否违反用户授权。复现时可用 BrowserGym / OSWorld：每个 task 提供 reset、observe、step、success check 和安全检查；先 SFT 成功轨迹，再 RL 学长期策略和错误恢复。

#### Gemma：小模型蒸馏与定向后训练

Gemma 系列提供更接近开源社区的路径：用强教师蒸馏和高质量数据过滤提升小模型，再针对数学、指令、多语言、安全做专项 post-training。它的意义是：不一定复刻 frontier 级 RL 系统，小模型也能通过数据质量、教师选择、能力分桶和 targeted preference optimization 得到实用能力。

#### [Meta Llama](https://arxiv.org/abs/2407.21783)

[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) 是开放模型里最适合作为产品级 chat model 后训练流水线的参考之一。它的价值不是某个单独算法，而是完整的数据治理、SFT、reward model、rejection sampling、preference optimization、安全对齐和评测闭环。从六阶段定位看，Meta 主要改进"SFT 阶段的数据治理与能力分桶 + 数据回流的拒绝采样自改进 + RL/偏好优化与安全对齐的完整开源基线"。

Llama 的 SFT 不应理解成"堆 instruction JSON"。数据需要覆盖通用问答、代码、数学、多语言、工具、安全和长上下文，并为每个域设置独立 eval。工程上通常要做去重、质量过滤、格式统一、拒答边界清洗、过长/过短样本控制。复现时先用小规模能力分桶，而不是一个混杂数据池。

对同一个 prompt 采样多个回答，用 reward model、规则 verifier 或 judge 选出最好的，加入下一轮 SFT。它介于 SFT 和 RL 之间：不直接更新策略梯度，但能把模型自己的高质量输出蒸馏回模型。对数学/代码可以用 verifier；对聊天/安全可以用 RM / judge。最小复现时，每个 prompt 采样 4-8 个回答，保留 top-1，同时保留 top/bottom pair 做 DPO。

Llama 的安全不是最后加拒答样本，而是在数据过滤、SFT、安全 RM、红队、发布阈值中持续出现。Preference optimization 进一步拉开好坏答案概率差，但也可能牺牲多样性和诚实性，所以需要 truthfulness、安全、拒答、helpfulness 同时评估。它是一条适合开源团队复现的基础线：即使没有 agent 环境，也能把 SFT、RS、DPO/RLHF、安全评测做完整。

#### [Microsoft Phi](https://arxiv.org/abs/2504.21318)

Phi 的公开资料（[Phi-4](https://arxiv.org/abs/2412.08905)、[Phi-4-reasoning](https://arxiv.org/abs/2504.21318)）重点是小模型 reasoning。[Phi-4-reasoning](https://arxiv.org/abs/2504.21318) 不是靠巨大参数硬推，而是靠高质量合成数据、teachable prompts 和一段 outcome-based RL，把 14B 级模型推到较强推理水平。从六阶段定位看，Microsoft Phi 主要改进"SFT 阶段的高质量合成推理轨迹 + 数据与环境的难度组织可学题集 + RL 阶段的 outcome-based 小模型推理强化"。

小模型容量有限，后训练数据必须可学、干净、有清晰监督。数学、科学、代码和逻辑题要按难度组织，过难题全错没有 RL 信号，过易题全对浪费 rollout。复现时应先构造 5K-20K 可教题集，保证 SFT 后模型能达到一定成功率，再进入 RL。

[Phi-4-reasoning](https://arxiv.org/abs/2504.21318) 的思路可以抽象为：先用高质量 synthetic reasoning traces 做 SFT，让模型会展开推理；再用 outcome reward 对可验证题做 RL，强化正确路径并控制无效长思考。小模型尤其要监控 average response length，因为一点 RL 就可能让输出变长但正确率不升。最小复现实验是用 Phi/Qwen 7B-14B、MATH/GPQA 子集、强教师生成 CoT，SFT 后每题采样 8 个，用答案 verifier 做 GRPO，并加入长度统计。

#### [NVIDIA Nemotron](https://arxiv.org/abs/2505.00949)

NVIDIA Nemotron 的公开资料（[Nemotron-4 340B](https://arxiv.org/abs/2406.11704)、[Llama-Nemotron](https://arxiv.org/abs/2505.00949)、[Llama Nemotron Ultra](https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra-open-model-delivers-groundbreaking-reasoning-accuracy/)、[Nemotron Agent Blog](https://developer.nvidia.com/blog/build-enterprise-ai-agents-with-advanced-open-nvidia-llama-nemotron-reasoning-models/)、[Nemotron-H](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/)、[Nemotron 3](https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/)）把后训练做成可复用资产：除了权重，还发布 synthetic data、preference data、reward model，以及 NIM、NeMo Gym 等部署栈。Llama Nemotron 把 reasoning、tool use、RAG、instruction following 和企业部署放在一起。从六阶段定位看，NVIDIA 主要改进"RL/偏好优化的企业级工具与 RAG 奖励 + 数据回流与 RM 资产化复用 + 评测阶段的部署成本/吞吐联合指标"。

[Nemotron-4 340B](https://arxiv.org/abs/2406.11704) 的方法是用强模型和规则生成候选数据，通过质量过滤和偏好标注训练 RM，再做 RLHF / preference optimization。可复现时可以把 RM 当作独立产物维护：它既用于 PPO/DPO，也用于 rejection sampling、数据过滤和自动评测。

[Llama-Nemotron](https://arxiv.org/abs/2505.00949) 的公开博客描述了三段式：从 Llama 底座出发，先剪枝提升效率，再蒸馏改善能力，然后用 post-training 数据和 RL 强化 reasoning、instruction following、function calling 和 chat。Llama-Nemotron-Post-Training Dataset 覆盖 math、coding、general reasoning、instruction following；OpenCodeReasoning 等数据强化代码推理。[Llama Nemotron Ultra](https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra-open-model-delivers-groundbreaking-reasoning-accuracy/) 还支持 reasoning on/off，说明它也要处理长思考成本和普通交互体验。

NVIDIA 强调蒸馏能搬运老师能力，但要进一步提升，需要 curriculum-driven RLVR。企业 agent 场景的 reward 来自工具调用正确性、RAG 忠实性、function calling schema、代码执行和用户意图对齐。公开资料还提到使用 REINFORCE 和 heuristic based verifiers 做 instruction following / function calling 增强，再用 HelpSteer2 等偏好数据做 RLHF。复现时可以做两桶 RL：一桶 math/code verifier，一桶 function calling verifier；最后混入 chat/RAG 偏好回填。

post-training 评估应包含 latency、throughput、function calling 成功率、RAG 引用忠实性和推理开销；企业模型如果只看 AIME，会忽略上线最常见的失败点。

#### [Mistral](https://arxiv.org/abs/2506.10910)

[Magistral](https://arxiv.org/abs/2506.10910) 的公开摘要很值得放在 reasoning RL 章节：Mistral 明确说它使用自家的 scalable RL pipeline，不依赖已有实现或从其他模型蒸馏来的 RL traces，而是 ground-up 做 pure RL。从六阶段定位看，Mistral 主要改进"RL 阶段的无蒸馏纯 RL 探索 + SFT 阶段的大小模型冷启动路径设计 + 数据与环境的推理语言/格式约束"。

蒸馏能快速得到长 CoT 格式，但也会继承老师的风格和错误。Magistral 的方向是从自家 checkpoint 出发，用 RL 自己探索推理能力。公开资料还提到 [Magistral](https://arxiv.org/abs/2506.10910) Medium 基于 Mistral Medium 3 只用 RL 训练 reasoning，Magistral Small 则包含来自 Medium 的 cold-start 数据。这对应两种复现路径：大模型直接 RL 探索；小模型先蒸馏大模型 cold-start，再 RL。

多语言模型做 reasoning RL 时可能出现推理语言混杂。Magistral 提到一种 simple method to force reasoning language，说明后训练不仅要管答案正确，还要管推理语言和输出风格。复现时可以在 prompt / template 中显式指定 reasoning language，并用 format reward 检查推理段语言、答案段语言和结构。

Magistral 的一个有意思结论是，只在 text data 上做 RL 仍能保持或提升 multimodal understanding、instruction following 和 function calling。这说明 RL 不一定必然破坏通用能力，但需要持续评估。最小复现时，做完数学/文本 RL 后，要同时跑函数调用、普通指令、多语言和视觉文本化任务，确认没有被 reasoning 数据拉偏。

:::

::: details 补充实践要点索引

### 补充索引

下面的团队不再展开完整案例，只列出每个报告回答了哪一个具体工程问题。你遇到类似问题时可以按关键词快速跳转到对应资料。

前面的深度案例（MiniCPM 5、Kimi K3、Qwen-AgentWorld、GLM-5.2）覆盖了推理 RL、Agent 环境、异步训练等主线方法。下面按团队补充其他公开报告中的关键实践点，每个都回答了一个具体工程问题。

**Apple Foundation Models**（[技术报告](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)）回答端侧模型的后训练如何与部署接口联合设计：guided generation、constrained tool calling 和 LoRA 支持被纳入训练目标，奖励拆成偏好 RM、规则 verifier 和 tool schema checker 三类混合，并按端侧/云端目标分别加权；端侧评测还要加入简洁性、延迟、内存和能耗。从六阶段定位看，Apple 主要改进"目标定义阶段的端侧部署接口联合设计 + RL 奖励的端云异构加权方案 + 评测阶段的延迟/内存/能耗端侧专属指标"。复现时可以让小模型按 JSON schema 训练工具调用，用 constrained decoder 评测 schema success rate。

**xAI Grok 与 Pi**（[Grok](https://x.ai/news/grok-4-1/)、[Pi](https://pi.ai/)）都把产品人格变成可评测的多目标奖励，而不是依赖 system prompt：personality、truthfulness、emotional intelligence、safety 四个目标分开训练 reward head，再做 multi-objective DPO/RL。人格评估不能只看单一分数，尤其要单独测 sycophancy，即模型是否为了讨好用户而承认错误事实。从六阶段定位看，xAI/Pi 主要改进"RL/偏好优化阶段的多目标人格奖励建模 + 评测阶段的谄媚度专项指标"。复现时构造多维 judge rubric，用偏好数据训练 multi-objective DPO/RL。

**IBM Granite**（[Granite 3.3/4.0/4.1](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)）说明 GRPO/TPO 这类 reasoning RL 可以下沉到企业级小模型，并支持切换 thinking 模式。评测要同时覆盖 RAG 引用忠实性、函数调用成功率、拒答正确性和 thinking on/off 的差异。从六阶段定位看，IBM 主要改进"RL 阶段的企业级小模型 reasoning RL 下沉 + 评测阶段的 RAG/工具调用企业场景指标 + 数据回流的专家模型合并"。复现路线是 SFT→GRPO 强化可验证推理→偏好优化忠实性→model merging 融合领域专家。

**Salesforce xLAM / SFR-RL**（[博客](https://www.salesforce.com/blog/large-action-model-ai-agent/)）把工具调用奖励拆成 schema validity、tool selection accuracy、argument exact match、execution success、groundedness 五段，并用 Pipelined Synchronous RL 让 rollout 与 training 两阶段共享 GPU 集群交替执行。Inference gateway 负责自动检测 crash、重建 engine、恢复权重并重排 in-flight work，工具执行尽量本地化。从六阶段定位看，Salesforce 主要改进"RL 阶段的五段式工具奖励拆分 + RL 训练系统的流水线同步调度 + 数据与环境的工具执行本地化"。复现时准备 50 个 mock API 生成请求和调用链，用函数执行结果做 verifier，并实现 timeout/retry/失败标记，避免系统错误污染数据。

**Amazon Nova Forge**（[介绍](https://aws.amazon.com/nova/forge/)）把企业私有验证器做成 Remote Reward Functions（Reward as a Service）：训练系统通过 API 调用企业自己的 CI、仿真、业务流程和真实 API 判分，企业奖励逻辑不必进入训练系统。平台支持从 pre-training/mid-training/post-training 任意 checkpoint 进入，企业私有数据与 Nova-curated 数据混合训练。复现时用本地服务模拟私有 API verifier，模型生成 SQL/代码补丁/客服动作，训练器通过 HTTP 调用奖励。

**Cohere Command A**（[技术报告](https://cohere.com/research/papers/command-a-technical-report.pdf)）处理多能力企业模型如何避免能力覆盖：先训核心 instruct 模型，再为 code/safety/RAG/math/multilingual/long-context 分别训练 expert，最后参数合并（Expert Soup）并做 best-of-N polishing。每个 expert track 有独立的数据配方、偏好目标和评测标准，offline preference 与 online RL 交替迭代。复现时复制 3-6 个 expert 各用不同数据做 DPO/RLVR，model soup 合并后再用通用偏好数据 polish。

**Databricks DBRX Instruct 与 AI21 Jamba 1.5a**（[DBRX](https://huggingface.co/databricks/dbrx-instruct)、[Jamba 1.5a](https://www.ai21.com/research/jamba-1-5a/)）代表企业模型的两种务实路线。DBRX 不追求长 CoT 和复杂 Agent，评估指标从一开始就与企业场景绑定：函数调用成功率、引用忠实度、JSON 输出稳定性和业务术语准确率。Jamba 1.5a 在通用 instruct 完成后，用合成偏好 + RLAIF 轻量对齐企业行为准则，不重跑整个后训练流水线；复现时写一组公司政策/产品原则，生成正负回答对做一轮 DPO，并单独评测 over-refusal，避免准则过强导致拒绝正常请求。

**Cursor Composer 2**（[技术报告](https://cursor.com/blog/composer-2-technical-report)）把训练单位从"prompt→answer"变成"仓库 episode→可验证修复结果"：环境包含完整仓库状态、issue、编辑器动作、终端输出、测试结果和 patch 验证器，奖励看是否解决 issue、不破坏已有测试、不修改无关文件。复现从 SWE-bench Lite 入手：checkout PR 前状态，模型编辑文件，运行测试，以测试通过率作为主奖励。

**LG EXAONE 4.0 与 NAVER HyperCLOVA X THINK**（[EXAONE 4.0](https://www.lgresearch.ai/data/cdn/upload/EXAONE_4_0.pdf)、[HyperCLOVA X THINK](https://huggingface.co/papers/2506.22403)）说明后训练配方不会自动跨语言迁移：数学/代码奖励与语言无关，但开放问答、写作、文化常识、安全和思考表达都依赖本土语言。它们用韩语重新做 SFT 和 reasoning RL，thinking/non-thinking 模式单独做本土评测。复现时不能只翻译英文 prompt，需要重新构造本土语言的冷启动数据、偏好标注和验证集。

**AI2 Tulu 3**（[论文](https://openreview.net/forum?id=i1uGbfHHpH)）的价值在透明度而非 SOTA：数据、代码、recipe 和评测全公开，覆盖 SFT→偏好学习→RLVR 三阶段。它清楚公开 prompt 混合方式、偏好构造、可验证奖励接入、超参数和评测组织，最适合作为后训练复现的第一站：先跑通全开源流程，再逐个加入 DAPO/VAPO/MiniMax 多 scaffold、Qwen 分阶段训练等技巧做 ablation，而不是一开始就堆砌改进。从六阶段定位看，Allen AI Tulu 3 主要改进"全流程公开透明的可复现基线 + SFT/偏好/RLVR 三阶段标准 recipe + 评测组织的标准化参考"。

:::

---

## 共同规律

现在把前面二十余个团队的报告收回来，看它们的共同点。学习这一节的目标不是记公司名称，而是能从任何一份新的工业技术报告里，迅速定位它的改进属于哪一类规律，评估它适合放在自己后训练闭环的哪个位置。

综合上述二十余个团队的公开技术报告，可以归纳出后训练流水线的几条共同规律：

1. **奖励从偏好到验证：从主观分走向客观分、再走向环境分。** 这是一个清晰的三段式演变，每一步都在解决前一阶段奖励信号不稳定的问题。

早期 RLHF 时代（InstructGPT、Llama 2）的奖励本质是"**主观偏好分**"。标注员面对的问题通常是"回答 A 和回答 B 哪一个更有帮助"，需要根据个人偏好、语言习惯和对 helpfulness 的主观理解做出判断。这种奖励的问题是不一致性：不同标注员标准不同，同一标注员在不同时间判断也会波动，甚至"语气更流畅"的回答可能事实完全错误。2023 年很多开源聊天模型就出现过这种现象：RLHF reward 曲线一路上涨，但在数学 benchmark 上的准确率反而比 SFT 模型低——reward model 学会了奖励"流畅的语气和讨好的表达"，而不是"正确的推理和事实"，训练方向自然就偏了。

R1/Qwen/Seed/Mistral 这一代团队把奖励推进到"**规则验证分**"阶段。DeepSeek-R1 证明了一个关键结论：纯规则 reward 就足以诱发长思考能力。数学奖励不看推理写得漂不漂亮，只看 $\boxed{}$ 里的答案是否与标准答案精确字符串匹配；代码奖励不看注释规不规范，只看 pytest 的 exit code 是否为 0。这一步变化影响深远——reward 越客观、越没有解释空间，几万步甚至几十万步的长训练就越稳定。QwQ-32B 的第一阶段 RL 甚至完全不用 reward model，只用数学 accuracy verifier 和代码执行服务器，就把"能不能解题"的核心能力推了上去，彻底避开了传统 RM 的主观偏差问题。

现在 MiniMax/Kimi/LongCat 这一代团队把奖励进一步推进到"**真实环境执行分**"。奖励不再只看最终答案对错，还要看中间过程和环境状态的真实迁移：MiniMax M2.1 的 SWE-Resolve 不仅检查新增测试是否通过，还要检查原有测试是否被破坏、是否修改了无关文件、有没有硬编码绕过验证器；Kimi-Researcher 的 reward 要看引用是否真实存在、证据是否真的支持结论、来源是否覆盖了关键角度；LongCat 甚至把工具超时、报错、返回缺字段这些环境扰动本身也纳入 reward 设计。这三步连起来就是 reward 的完整演进：从"人喜欢哪个回答"的主观分，到"答案对不对"的客观分，再到"任务在真实环境里是否真的完成"的环境分——reward 越接近真实世界的成功标准，长训练就越稳定，模型能力的迁移性就越强。

2. **数据从静态样本到可交互环境：从问答对到可回放的完整 episode。** 数据形态的演变同样经历了三个阶段，每个阶段的数据复杂度和工程要求都上了一个台阶。

早期 SFT 时代的数据是静态的"人工写好的问答对"。一条样本就是一个 JSON 对象：`{"prompt": "用户问题", "response": "标准答案"}`，长度通常几百到几千 token，训练时模型只需要"看到问题，背诵答案"。这种数据构造简单、存储方便，但只能覆盖团队已经想到的情况——模型记住了固定问答模式，遇到训练集之外的新问题就不会了。2023 年之前的大部分开源聊天模型都停留在这个阶段，数据主要靠人工标注和强模型生成，没有环境、没有交互、也没有自动验证。

R1 之后，数学和代码任务的数据进入"**verifier + 拒绝采样**"阶段。数据不再是一次性写死的问答对，而是可以反复采样、自动筛选的动态产物：对同一个 prompt 让模型生成 8-16 个候选回答，用规则 verifier 自动判分，留下正确的轨迹做 SFT 或 RL，错误的丢弃或分析原因补数据。DeepSeek-R1 的 rejection sampling、Qwen 的自改进、Kimi 的 long-to-short 都是这个思路：模型自己生成数据，verifier 自动筛选，成功的轨迹回流到下一轮训练。这时候一条样本已经包含了 prompt、多个候选、每个候选的 reward、最终筛选出的正确轨迹——数据量比纯问答对大了一个数量级，但还不需要持久化外部环境。

现在 Agent 任务的数据已经变成了"**持久化环境、可回放、可组合的 scaffold**"。一个代码修复样本不再是 500B 的 issue 文本加 2KB 的 patch，而是几十 MB 的 Docker 镜像快照、仓库初始状态、依赖锁定版本、完整的工具调用序列、每一步的环境观察和中间 reward；一个网页 agent 样本不再是问题和答案，而是浏览器初始 DOM 状态、每一帧截图、点击输入序列、页面状态变化和最终任务成功判定。Kimi K3 更进一步，把 scaffold（执行框架、工具配置、上下文管理方式）本身也变成可组合的变量，同一个任务可以用不同的 harness 运行，避免模型过拟合单一格式。GitHub PR、Docker、Playwright、数据库快照、工具图谱、搜索网页历史——所有这些都变成了后训练数据的一部分，数据工程的复杂度已经远远超出了"整理 JSONL"的范畴。

3. **训练顺序越来越分段：从两步走到多阶段精细化拆分。** 后训练不是一个"做完就发布"的一次性流程，而是多个阶段顺序衔接、每个阶段有明确目标的流水线，阶段划分越来越细是近两年最明显的趋势之一。

早期后训练很简单，就是"SFT → RL"两步走：先拿一批高质量数据做监督微调让模型学会基本格式，再用 RLHF 或 PPO 做偏好对齐就发布。InstructGPT、Llama 1/2 以及 2023 年的大部分开源模型都是这个路线。这种两阶段方案的问题是目标混杂：SFT 既要教格式、又要教知识、还要教风格；RL 既要提升能力、又要对齐偏好、还要保证安全——多个目标混在一起训练，很容易互相挤压，结果往往是"聊天更流畅了，但数学和代码能力掉了"。

Qwen/DeepSeek/GLM 这一代把训练拆成了更清晰的三到四阶段：第一步是 **cold-start SFT**，只用少量高质量长思考数据教格式、推理结构和基本行为，不追求覆盖所有能力；第二步是 **reasoning RL**，只在数学、代码、逻辑等高置信可验证任务上训练，把"能不能解题"的核心能力推上去；第三步是 **general RL** 回流，混入通用指令、偏好、安全、短回答数据，把前一阶段训练带来的"啰嗦、格式怪、通用能力退化"这些问题修复回来。QwQ-32B、DeepSeek-R1、GLM-5、ERNIE 4.5、Hunyuan-T1 虽然算法名称不同，但这个三阶段顺序高度一致——先在可验证任务上放大能力，再用通用数据修复体验，这个顺序不能反过来，否则模型会被偏好数据拉到"会说话但不会解题"的状态。

最近的方案在三阶段基础上又往前拆了一步。Qwen-AgentWorld 在 SFT 之前加了 continue pretraining 阶段，先让模型学习环境知识——"什么是文件系统、pytest 报错是什么意思、DOM 结构怎么读"——这些环境常识先通过持续预训练灌进去，再教 SFT 动作格式，最后才做 RL；Kimi K3 和 MiniCPM5 还用 OPD（On-Policy Distillation）把多个领域专家模型的能力合并回一个统一模型，相当于在数据回流阶段加了一步"多专家能力整合"。阶段拆分越来越细的本质原因是：不同能力需要不同的数据、不同的 reward、不同的训练火候，混在一起训谁都训不好——像教学生一样，先学基础概念，再专项突破，最后综合模拟，这个顺序反了就事倍功半。

4. **训练系统本身成为竞争力：从改公式到做工程。** 最容易被开源社区低估的一点是：当任务从短文本数学题扩展到长程 Agent 任务时，决定能不能训成的往往不是算法公式，而是训练系统的工程实现。

早期 reasoning RL 大家主要在改算法公式：从 PPO 到 GRPO，怎么算 advantage、怎么加 KL 约束、怎么设计 reward 公式——这些是论文里能看到的部分。DAPO 和 VAPO 开始触及工程细节：DAPO 改了动态采样（没有学习信号的 prompt 持续过滤）、clip 范围（高概率 token 探索空间放大）、token-level 梯度归一化（长响应不被平均稀释）、过长响应 shaping（截断样本不当成正常失败）；VAPO 改了长度自适应 GAE、解耦价值估计、正样本 LM loss 稳定策略。这些改动看起来都是"小补丁"，但少了任何一个，大规模长 CoT RL 就会崩——这也是很多人复现 R1 失败的根本原因：公式抄对了，但工程细节没处理。

到了 Agent RL 阶段，工程系统的差异进一步拉大。GLM-5.2 的 SAO（异步单轨迹优化）解决了同步 GRPO 的 GPU 闲置问题：同一 batch 里短任务 30 秒做完、长任务跑 15 分钟，同步等最慢的那条会浪费 70% 以上算力，异步让每条轨迹做完就进训练队列，做好一条训一条；LongCat 的 DORA（流式 RL）更进一步，支持多版本模型并行探索、Prefill-Decode 解耦、KV-cache 跨步骤交换和 CPU 驻留，把同步训练效率提升了 2-4 倍，还支持千步以上稳定训练；MiniMax 的 multi-scaffold 调度、Qwen-AgentWorld 的多环境统一接口，这些都不是"把 GRPO 公式写出来就能跑"的代码层面事情，而是 rollout 调度策略、环境生命周期管理、硬件资源分配、KV 缓存管理、数据版本追踪、失败自动恢复这些一整套基础设施。

闭源团队在这些系统工程上投入的人力往往几倍于算法工程师。能训多长的轨迹、能不能支持千步以上稳定训练、多卡扩展效率是 50% 还是 90%、rollout 崩溃了能不能自动恢复、环境不一致能不能自动检测——这些工程细节决定了你的系统能不能训长任务、能不能扩到大规模、能不能在多轮迭代中保持稳定。异步 rollout、PD 解耦、KV-cache 交换、环境调度、失败恢复、reward 服务化、可执行 verifier——这些都是"后训练实践"本身的一部分，而不是外围工程。很多时候不是算法不对，而是你的系统根本撑不到算法起效的时候。

如果把上面的公司实践抽象成一个可复现的小型项目，可以按下面的顺序做：

1. **先选一个可验证任务族。** 数学题最简单，代码题次之，网页/GUI/研究 agent 最难。开放聊天任务的 reward 较难定义，不适合作为第一版 RL 实验。这一步容易踩的坑是一开始选开放聊天导致 reward 模糊，先从数学/代码这种规则验证题切入最快。
2. **把任务封装成环境。** 数学环境需要 answer parser 和 verifier；代码环境需要仓库 checkout、依赖安装、测试命令和 patch 检查；网页环境需要浏览器、状态记录和证据抽取；GUI 环境需要截图、动作空间和可复位 sandbox。这一步容易踩的坑是环境依赖没锁定版本，同一份代码今天能跑明天跑不通，reward 全是噪声。
3. **先做 SFT 冷启动。** 收集或生成成功轨迹，让模型学会输出格式、工具协议、思考结构和停止条件。没有冷启动就直接 RL，容易先在格式和工具调用上乱掉。这一步容易踩的坑是 SFT 数据太杂、格式不统一，模型学完后工具调用格式五花八门，RL 前几万步全在修格式。
4. **再做采样和筛选。** 对每个 prompt 采样多个输出，用 verifier / judge / reward model 选出正确、简洁、过程合理的轨迹。这个阶段就是 Qwen、DeepSeek、Kimi、MiniMax 都在反复做的 rejection sampling / self-improvement。这一步容易踩的坑是只留最终答案正确的轨迹，把前面对、最后一步错的高价值片段全部丢掉，样本效率极低。
5. **最后做 RL。** 简单题可以用 GRPO / DAPO 风格的组内相对优势；需要 value model 的任务可以参考 PPO / VAPO；长程 agent 要额外处理异步 rollout、失败恢复、工具噪声和 token-level credit assignment。这一步容易踩的坑是一上来就把长度上限拉满、lr 调太大，训练前几百步 reward 就崩掉或 entropy 快速塌缩。
6. **训练后做能力回填。** 用通用指令、安全、短回答、风格和非思考模式数据再对齐一次，防止模型被 reasoning RL 训练得又长又慢。这一步容易踩的坑是回填数据太多、学习率没降，把前面 reasoning RL 好不容易训出来的长链推理能力直接冲掉。

一个很小但完整的练习可以是：用 5K 数学题或 1K 代码修复题做 SFT，采样 8 个候选，用规则 verifier 过滤，再用 GRPO 训练一轮，最后评估正确率、平均输出长度、格式错误率和通用聊天退化。这样的闭环比只停留在算法名称层面更能揭示后训练的真实难点。

::: details 小闭环的成本估算
很多读者关心：跑通上面这条全链路闭环，最低需要多少成本？我们以 7B 模型、8 卡 A100（或 8 卡 H800）、5K 数学题 + 2K 代码题的规模为例，给出一个量级估算。

- **冷启动 SFT**：约 2K-4K step，在 8 卡 A100/H800 上大约 3 小时，模型学会基本格式和解题套路。
- **每轮 GRPO 训练**：每题采样 8-16 个候选，完整 3-5 epoch 约 8-12 小时；如果是代码题还要算上沙箱环境执行和测试的等待时间，实际 wall clock 可能更长。
- **3 轮完整训练 + 评测**：SFT + 3 轮 GRPO + 每轮之间的数据分析和失败回流，总共约 2-3 天。
- **云服务器成本**：按当前主流云厂商的 8 卡 A100 按时计费（约 $20-30/小时）或 8 卡 H800（约 $35-50/小时），3 天 72 小时的纯算力成本大约在 **$1,500 到 $3,500** 的量级区间；如果用竞价实例或包月折扣，可以压缩到 $1,000 左右。

需要特别提醒的是：**真正贵的不是算力，而是构造环境 + 数据 + verifier 的人力**。为 2K 道代码题写好可重复启动的沙箱、可判分的验证器、防止 reward hacking 的辅助规则，再加上 3 轮训练之间的失败分析和数据修补，一位有经验的工程师全职投入可能需要 2-4 周。人力成本往往是纯算力成本的 5-10 倍。这也是为什么工业界后训练团队的配置里，数据工程师和平台工程师的人数通常多于算法工程师。
:::

## 本节总结

回到开头的代码修复模型。从 issue 收集、环境封装、SFT 冷启动，到 RL、独立评测和失败回流，一轮结束时的失败样本就是下一轮的数据来源。

工业后训练的核心是**闭环**：定义目标、构造数据与环境、SFT 冷启动、RL 提升、独立评测、失败回流，六个阶段首尾相接，形成持续迭代的循环。线性流程的问题在于，第一轮数据永远只覆盖团队已经想到的情况。

模型进入真实场景后暴露的边界失败，包括沙箱不一致、验证器漏洞、不会用的工具，只能由独立评测按原因分类，送回对应环节：该补数据的补数据、该修环境的修环境、该升级验证器的升级验证器。模型能力正是在一轮一轮的修正中建立起来的。

多跑几轮只是表象，闭环真正建立的是从失败中持续学习的能力。

六个阶段各有定位，缺一不可：

- **目标定义**：写清成功标准、安全边界和成本约束，这一步模糊，后面所有环节都会跟着模糊。
- **数据与环境**：把原始事件变成可启动、可回放、带验证器的训练样本，环境不一致，一切训练信号都不可靠。
- **SFT 冷启动**：教模型基本的行为格式和操作协议，让模型知道怎么用工具、怎么观察环境反馈、什么时候算任务完成。
- **RL/偏好优化**：在大量尝试中把高奖励行为概率推高，让模型真正学会决策。
- **独立评测**：在没见过的新任务、接近部署的配置下检验提升是否真实，防止模型过拟合训练环境或钻 verifier 空子。
- **失败回流**：把评测发现的问题分类送回前面的环节，完成闭环的最后一环。

六阶段的产物也不只是模型权重：可复用的数据集、版本化的沙箱环境、逐步完善的验证器、能发现真实失败的评测体系，这些资产的长期价值往往超过单次训练得到的权重本身。

> **闭环真正建立的是从失败中持续学习的能力。**

当任务从数学题扩展到代码修复、网页操作这类需要和外部环境交互的任务时，六阶段的每一步都需要相应的扩展。

具体来说，**数学 RL** 的环境只需要题目文本和 20 行答案验证器，不需要启动外部进程，也不需要持久化中间状态。但**代码修复沙箱**要带 200MB 依赖、十几个启动命令和完整 git history，光拉起环境就要一分钟。

这种差异体现在三个维度。**episode 长度**从百 token 级扩展到万 token 级，**reward 延迟**从输出完立即判分变成几十步交互后才知道结果，**rollout 失败**从格式错扩展到沙箱崩溃、依赖装不上、工具超时。

六阶段各自的新要求：

- **目标定义**要加工具权限、时间预算和安全边界
- **数据与环境**要携带环境快照和可恢复 checkpoint
- **SFT** 要教基于环境观察的决策
- **RL** 要处理异步经验和多轮轨迹奖励
- **评测**要监控成本和安全回归
- **失败回流**要做细粒度的轨迹片段筛选和验证器版本更新

这些差异说明，数学 RL 的流程不能直接照搬到 Agent 任务。

把前面二十余个团队的公开实践放在一起，可以看到四条反复出现的规律：

1. **奖励从主观到客观**：reward 越接近真实世界的成功标准，长训练越稳定，能力迁移性越强。
2. **数据从静态到交互**：容器镜像、数据库快照、工具调用序列和环境观察日志都成为训练数据的一部分。
3. **训练从两步到多阶段**：冷启动 SFT、reasoning RL、agentic RL、通用能力回流分别使用不同的数据、奖励和训练节奏。
4. **系统成为核心竞争力**：从 DAPO/VAPO 的采样和 clip 补丁，到 SAO 异步、DORA 流式 RL，工程系统决定了训练能跑多长的任务、能扩展到多少张卡、能稳定坚持多少步。

训练跑到中途，奖励曲线还在上升，独立评测的分数却停在原地；另一条曲线上，熵在快速下滑。这些现象分别指向哪一层的问题，下一节沿着这条闭环继续讨论。[18.3 训练稳定性](./modern-industrial-practice) 介绍监控面板上各条曲线的读法，以及出现异常时从哪一层开始排查。
