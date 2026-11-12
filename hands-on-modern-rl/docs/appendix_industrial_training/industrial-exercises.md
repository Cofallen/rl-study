# C.4 工业实践练习

> 这一页不再把练习写成抽象算法题，而是从公开岗位要求倒推：真实的后训练岗位、强化学习岗位每天在做什么，候选人应该拿什么项目证明自己会做。

本页参考了 2026 年 5 月前后公开招聘页中的岗位描述。招聘信息会变化，所以不要背具体公司名称；要看清岗位背后的稳定能力：数据、训练、奖励、评测、系统、产品闭环。

## 一句话总结

工业界的后训练/RL 岗位通常不是让你“会 PPO 公式”就结束，而是让你交付这几类结果：

- 把业务问题改写成可训练、可评测的模型行为目标。
- 构建 SFT、偏好、奖励、RLVR 或 Agent 轨迹数据。
- 选择 SFT、DPO、RM、PPO、GRPO、RLOO、RLVR 等训练路线，并解释为什么。
- 跑稳定的训练，监控 KL、entropy、reward、pass rate、长度、吞吐和回归指标。
- 找出 reward hacking、能力退化、数据污染、评测失真、训练变慢等问题。
- 把模型改进落到产品指标、用户体验、安全合规或真实环境任务上。

## 岗位样本到能力地图

| 地区 | 典型岗位名称                                                                        | 岗位更关心什么                                                                  | 你要能交付的东西                                                                |
| ---- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 中国 | 大模型后训练算法工程师、垂域大模型训练算法工程师、AI 搜索/元宝/混元后训练算法工程师 | SFT/RM/RL 全流程、数据合成、线上反馈数据飞轮、垂域应用效果、国产/混合算力训练栈 | 数据清洗和配比方案、SFT/DPO/RLHF 实验、评测报告、Badcase 归因、业务场景迭代方案 |
| 中国 | 强化学习算法工程师、机器人/自动驾驶/推荐/营销 RL 工程师                             | 业务建模、仿真环境、离线 RL、PPO/SAC/GRPO 调优、sim-to-real 或在线效果          | MDP 建模、reward 设计、仿真训练、离线评估、上线风险控制、效果复盘               |
| 美国 | Post-Training Research Engineer / Scientist                                         | 面向大规模产品模型做最终后训练，强调模型能力、安全、评测和产品落地              | 端到端训练实验、鲁棒 eval、模型行为诊断、训练代码调试、产品反馈闭环             |
| 美国 | Agentic Post-Training / Synthetic RL / RL Engineering                               | Agent、工具调用、代码能力、合成环境、grader、RL 系统吞吐和可靠性                | Agent 任务环境、可复现 grader、轨迹存储、RLVR/GRPO 训练、训练 pipeline 性能优化 |
| 美国 | Reward Models / Alignment Research                                                  | 人类偏好、奖励模型、LLM-as-Judge、rubric、reward hacking                        | 偏好数据方案、RM 训练与校准、judge 一致性评估、反作弊数据集                     |
| 欧洲 | AI Scientist / Research Engineer / Research Platform                                | frontier 模型、企业定制、研究平台、HPC、评测、隐私合规                          | 训练/评测工具链、企业数据适配、可靠部署、实验管理和可复现报告                   |
| 欧洲 | Robotics / Autonomous Systems / RL Research Engineer                                | 机器人、自动驾驶、防务、自主系统、仿真和现实环境对齐                            | 仿真环境、策略训练、场景库、安全评测、sim-to-real 误差分析                      |

这个表的重点是：后训练岗位通常围绕“语言模型行为改进”，强化学习岗位通常围绕“策略在环境中做决策”。LLM 时代二者会重叠，特别是在 Agent、代码、数学、工具调用、机器人 VLA 和自动驾驶方向。

## 地区差异怎么看

**中国岗位：更重业务落地和数据飞轮。**

公开岗位里常见关键词是“应用场景”“垂直领域”“数据合成”“线上反馈”“评测体系”“训练效率”“实际用户体验”。这意味着你不能只展示一个开源 benchmark 分数，还要说明：用户问题从哪里来，坏例怎么进数据池，SFT/偏好/RL 数据怎么配比，训练后如何证明业务体验变好了。

**美国岗位：更重 frontier post-training 和大规模系统。**

OpenAI、Anthropic 这类岗位把 post-training 放在最终产品模型发布前的关键环节，常见任务包括 eval、grader、reward model、RL 系统、agentic model、工具使用、代码能力、训练 pipeline 稳定性。候选人要能在研究、工程、产品边界之间切换。

**欧洲岗位：更重研究平台、企业场景和具身/自主系统。**

欧洲的大模型公司会有企业定制、研究平台、HPC、评测和合规约束；同时机器人、自动驾驶、防务自主系统里的 RL 岗位更强调仿真、现实系统、安全边界和工程验证。练习时不要只做聊天模型，也要保留一个“环境交互型 RL”项目。

## 真实工作拆解

### 1. 后训练算法工程师

你具体要做：

- 把产品问题翻译成训练目标，例如“回答更有帮助”要拆成事实性、覆盖度、结构、拒答边界、引用质量。
- 构建 SFT 数据、偏好数据、reward 数据、RL prompt 池和回归评测集。
- 选择 SFT、DPO、RM+PPO、GRPO、RLVR 或混合 loss，并解释算力、数据和风险取舍。
- 跑训练并监控 loss、reward、KL、entropy、长度、重复率、拒答率、pass rate。
- 分析 Badcase，判断问题来自数据、奖励、采样、训练超参、prompt 模板还是评测。

可以练：

- 选一个垂域，比如法律、教育、金融研报、代码助手或企业客服，做 500 条 SFT 数据、200 条偏好对、100 条回归评测。
- 用 1B/3B/7B 开源模型跑 LoRA SFT，再用 DPO 或 ORPO 做一轮偏好优化。
- 写一份训练报告：数据来源、过滤规则、训练配置、指标变化、Badcase、下一轮数据计划。

### 2. Reward Model / Judge / Grader 工程师

你具体要做：

- 定义什么叫“好回答”，把模糊偏好拆成 rubric。
- 设计人工标注、合成偏好、LLM-as-Judge、规则 verifier 的混合打分方案。
- 训练或校准 reward model，检查它是否偏爱啰嗦、套话、过度拒答或固定格式。
- 给 RL 训练提供稳定奖励，并持续监控 reward hacking。

可以练：

- 做一个数学或代码 verifier：答案正确给主奖励，格式、长度、执行安全给辅助奖励。
- 做一个 LLM-as-Judge rubric：事实性、指令遵循、完整性、安全性各 1-5 分，抽样检查 judge 与人类标注一致率。
- 构造 50 条“看起来很礼貌但没信息量”的反例，测试 reward model 是否被骗。

### 3. RL / GRPO / RLVR 训练工程师

你具体要做：

- 为数学、代码、工具调用或 Agent 任务设计可验证奖励。
- 控制 on-policy 数据、采样组大小、KL 惩罚、优势归一化、训练 batch 和 rollout batch。
- 诊断 reward 上升但真实指标下降、KL 飙升、entropy 坍缩、输出变长或变短。
- 在有限算力下做稳定训练，而不是只追求一次性高分。

可以练：

- 用 GSM8K、MATH 子集或 HumanEval 子集搭一个 RLVR 训练：每个 prompt 采样 4-8 个回答，用规则或执行结果给 reward。
- 对比三组配置：低 KL、中 KL、高 KL，画出 reward、pass rate、KL、entropy、平均输出长度。
- 写一页事故复盘：如果 reward 上升但 pass@1 下降，你如何定位和回滚。

### 4. Agentic Post-Training 工程师

你具体要做：

- 搭建工具调用环境，例如浏览器、代码执行器、文件系统、数据库、API。
- 记录完整轨迹：observation、action、tool call、tool result、reward、失败原因。
- 设计 grader，判断任务是否真的完成，而不是只看最终文本漂亮不漂亮。
- 让模型学会多轮规划、函数调用、错误恢复和终止。

可以练：

- 做一个小型 coding agent 训练环境：输入 issue，模型修改代码，运行测试，测试通过给 reward。
- 做 30 个网页/文件操作任务，记录轨迹并写 grader，例如“找到价格并填入表格”“修复一个 failing test”。
- 对每个失败轨迹标注失败类型：没读懂任务、工具调用参数错、不会恢复、提前停止、幻觉文件路径。

### 5. RL 系统 / 训练基础设施工程师

你具体要做：

- 优化 rollout、reward、training、buffer、weight sync 的吞吐。
- 处理 policy version、staleness、长尾 completion、GPU 利用率、失败重试。
- 建立训练健康检查，让错误在小集群先暴露，而不是几天后才发现大训练坏了。
- 支持新算法接入，并保证稳定、快速、可复现。

可以练：

- 写一个简化版 rollout buffer：记录 prompt、response、reward、old logprob、policy_version。
- 模拟异步训练：让 rollout worker 使用旧策略生成数据，观察 staleness 对指标的影响。
- 给训练脚本加 10 个监控指标：tokens/s、samples/s、reward latency、KL、entropy、OOM 次数、重试次数、队列长度、policy lag、eval 回归。

### 6. 机器人 / 自动驾驶 / 自主系统 RL 工程师

你具体要做：

- 把任务建模成 MDP：状态、动作、奖励、终止条件、约束。
- 在仿真中训练策略，再分析仿真和现实的差距。
- 做安全评估：碰撞、越界、能耗、舒适性、鲁棒性、极端场景。
- 和感知、控制、仿真、硬件团队一起调试。

可以练：

- 用 Gymnasium、MuJoCo、Isaac Gym 或 PyBullet 做一个连续控制任务，比较 PPO 和 SAC。
- 对 reward 做三版设计：只奖励任务成功、加入能耗约束、加入安全约束，观察策略差异。
- 写一份 sim-to-real 风险清单：传感器噪声、延迟、动力学误差、执行器限制、环境分布偏移。

## 八个工业练习

### 岗位反推能力矩阵

**目标**：训练你从岗位描述里读出真实工作。

**任务**：

1. 各找 3 个中国、美国、欧洲的后训练或 RL 岗位。
2. 把每个岗位拆成 6 列：算法、数据、评测、系统、产品/业务、安全/合规。
3. 标出你已经做过的证据项目，以及还缺的证据项目。

**交付物**：一张能力矩阵表，加一段 300 字总结：你要投哪类岗位，下一步最该补哪个项目。

### 垂域后训练最小闭环

**场景**：你要做一个金融投研助手或法律问答助手，目标是回答更专业、更可靠。

**任务**：

1. 构造 500 条 SFT 数据，明确数据过滤规则。
2. 构造 200 条偏好对，说明 chosen/rejected 的判定标准。
3. 跑一轮 SFT，再跑一轮 DPO/ORPO。
4. 建 100 条回归评测，至少包含事实性、指令遵循、安全拒答、格式稳定性。
5. 做 20 条 Badcase 归因。

**交付物**：数据卡、训练配置、评测表、Badcase 表、下一轮数据配比方案。

### RLVR 数学或代码训练

**场景**：你要提升模型在数学或代码任务上的可验证正确率。

**任务**：

1. 选择 GSM8K/MATH 子集或 HumanEval/MBPP 子集。
2. 写 verifier：数学检查最终答案，代码运行单元测试。
3. 每个 prompt 采样 4-8 个回答，用 GRPO/RLOO/RLVR 跑一轮小训练。
4. 记录 reward、pass@1、pass@k、KL、entropy、平均长度。
5. 构造 10 条 reward hacking 样例，解释如何修复 verifier。

**交付物**：训练曲线、配置对比、verifier 代码、reward hacking 复盘。

### Reward Model 和 Judge 校准

**场景**：你的 RM 分数持续上升，但人工评测说模型越来越空洞。

**任务**：

1. 设计一个 4 维 rubric：事实性、完整性、帮助性、安全性。
2. 标注或合成 300 对偏好数据。
3. 训练一个小 RM，或用 LLM-as-Judge 做打分。
4. 检查长度偏置、套话偏置、过度拒答偏置。
5. 加入 50 条反例重新评测。

**交付物**：rubric、偏好数据样例、judge/RM 一致性报告、偏置分析、修复方案。

### Agent 任务环境

**场景**：你要训练一个能修复小型代码仓库 bug 的 Agent。

**任务**：

1. 收集 20 个小 issue，每个 issue 都有可运行测试。
2. 定义 action space：读文件、改文件、运行测试、提交答案。
3. 记录每条轨迹的工具调用和测试结果。
4. 设计 reward：测试通过、修改范围、失败恢复、安全约束。
5. 分析 10 条失败轨迹。

**交付物**：任务环境说明、grader、轨迹样例、失败类型统计、下一轮训练数据。

### 训练系统健康检查

**场景**：RL 训练跑到第 3 天突然变慢，模型指标也开始波动。

**任务**：

1. 画出 rollout、reward、buffer、trainer、weight sync 的数据流。
2. 加入 policy_version 和队列长度监控。
3. 设计 5 个自动报警：KL 异常、entropy 坍缩、reward latency 过高、eval 回归、policy lag 过大。
4. 模拟一个长尾 completion 拖慢 batch 的例子。
5. 写出回滚和重启策略。

**交付物**：系统图、指标面板、报警规则、故障复盘。

### 练习七：中国岗位模拟 与 产品数据飞轮

**场景**：一个 AI 搜索或教育产品希望模型回答更准、更贴合用户。

**任务**：

1. 从用户日志抽样，定义“可用于训练”的过滤条件。
2. 把日志分成 SFT、偏好、评测、红队四类。
3. 设计数据合成方案，但说明如何防止合成数据污染评测。
4. 给出每周迭代节奏：采样、标注、训练、评测、上线、回收反馈。
5. 说明隐私、版权和安全边界。

**交付物**：数据飞轮流程图、数据配比表、上线灰度指标、风险清单。

### 练习八：欧美岗位模拟 与 大规模 Agent/RL 项目计划

**场景**：你在做一个会使用工具的 frontier agent，目标提升代码、浏览器和多工具协作能力。

**任务**：

1. 定义 3 类任务环境：代码修复、网页信息查找、API 工具调用。
2. 为每类任务写 grader，并说明哪些信号可以自动化，哪些必须人工抽检。
3. 设计训练阶段：SFT 轨迹模仿、偏好优化、RLVR/GRPO、多轮 Badcase 数据回灌。
4. 设计 eval gate：上线前哪些指标必须不退化。
5. 写出 4 周项目排期和算力预算。

**交付物**：项目计划、任务环境、grader 设计、eval gate、训练风险和缓解方案。

## 面试时怎么展示

一个有说服力的项目，不是“我跑过某框架”，而是能讲清楚这五件事：

1. **目标**：你要改善哪种模型行为，为什么这个目标重要。
2. **数据**：数据从哪里来，怎么清洗，怎么配比，怎么避免泄漏。
3. **训练**：为什么选这个算法，关键超参是什么，如何保证稳定。
4. **评测**：离线指标、人工评测、回归测试、产品指标怎么互相验证。
5. **复盘**：失败在哪里，Badcase 如何归因，下一轮怎么改。

如果你准备中国后训练岗位，优先展示“垂域数据 + 后训练 + 评测 + 产品闭环”。如果你准备美国 frontier post-training/RL 岗位，优先展示“训练系统 + eval/grader + agentic RL + 大规模实验习惯”。如果你准备欧洲岗位，优先展示“研究工程能力 + 企业/物理/机器人场景 + HPC/合规/安全验证”。

## 参考岗位样本

- OpenAI 的 Post-Training 岗位强调把预训练模型改进到 ChatGPT、API 等真实产品中，并要求构建 eval、调试研究栈和做产品驱动研究。[^openai-post-training]
- OpenAI 的 Agentic Post-Training 岗位强调 factuality、instruction following、function calling、tool use、grading stack、user-data flywheel 和大规模 RL/post-training 基础设施。[^openai-agentic]
- OpenAI 的 Synthetic RL 岗位强调 synthetic data、environment、feedback、self-play、simulator 和训练动态分析。[^openai-synthetic-rl]
- Anthropic 的 Production Model Post-Training 岗位强调完整 post-training stack、Constitutional AI、RLHF、评测 pipeline、训练调试和可复现。[^anthropic-post-training]
- Anthropic 的 RL Engineering 岗位强调 RLHF 训练系统的速度、可靠性、易用性、pipeline profiling、健康检查和新算法实现。[^anthropic-rl-engineering]
- Anthropic 的 Reward Models 岗位强调偏好学习、LLM-based grading、rubric、reward hacking 和 reward model 泛化。[^anthropic-reward-models]
- 上海人工智能实验室的大模型训练岗位强调 CPT、SFT、RLHF/DPO、数据清洗、Megatron-LM、veRL、LLaMA-Factory、训练监控、Badcase 分析和垂域效果。[^shlab-training]
- 腾讯公开转载岗位样本显示，国内后训练岗位常强调 PostTraining、SFT/RM/RL、奖励系统、数据合成、线上反馈数据飞轮、个性化、长期记忆和全维度评测。[^tencent-yuanbao][^tencent-hunyuan]
- Mistral AI 的 Forge 产品岗位把 fine-tuning、reinforcement learning 和 post-training workflow 做成企业可用产品；其 EMEA Applied Scientist/Research Engineer 岗位强调仿真数据、训练评测、agent/RAG 与工程场景结合。[^mistral-forge][^mistral-ai4engineering]
- Helsing 的欧洲 RL 岗位和 Project Centaur 公开材料显示，具身/自主系统 RL 会强调仿真、现实系统、安全和任务环境中的策略能力。[^helsing-rl][^helsing-centaur]
- Google DeepMind 的 Careers 页面把 Research Engineer 描述为连接理论和实现、构建可扩展系统、测试和评估新想法的角色，这类能力也对应欧洲/英国研究工程岗位。[^deepmind-careers]

[^openai-post-training]: OpenAI, "Research Engineer / Research Scientist, Post-Training", <https://openai.com/careers/research-engineer-research-scientist-post-training-san-francisco/>

[^openai-agentic]: OpenAI, "Researcher, Agentic Post-Training", <https://openai.com/careers/researcher-agentic-post-training-san-francisco/>

[^openai-synthetic-rl]: OpenAI, "Researcher, Synthetic RL", <https://openai.com/careers/researcher-synthetic-rl-san-francisco/>

[^anthropic-post-training]: Anthropic, "Research Engineer, Production Model Post-Training", <https://www.anthropic.com/careers/jobs/4613592008>

[^anthropic-rl-engineering]: Anthropic, "Machine Learning Systems Engineer, RL Engineering", <https://www.anthropic.com/careers/jobs/4952051008>

[^anthropic-reward-models]: Anthropic, "Senior Research Scientist, Reward Models", <https://www.anthropic.com/careers/jobs/5024835008>

[^shlab-training]: 上海人工智能实验室, "大模型训练算法工程师", <https://www.shlab.org.cn/joinus/detail/7615234376275773734?mode=social>

[^tencent-yuanbao]: 牛企直聘转载腾讯官网岗位, "元宝-大模型后训练算法工程师", <https://jobs.niuqizp.com/job-vyU55n5n5.html>

[^tencent-hunyuan]: 牛企直聘转载腾讯官网岗位, "混元大语言模型后训练算法工程师", <https://jobs.niuqizp.com/job-vmU55NnaZ.html>

[^mistral-forge]: Mistral AI, "Product Manager, Forge"（职位已下线）

[^mistral-ai4engineering]: Mistral AI, "Applied Scientist / Research Engineer, AI4Engineering - EMEA"（职位已下线）

[^helsing-rl]: Helsing, "AI Research Engineer - Reinforcement Learning", <https://helsing.ai/jobs/4676357101>

[^helsing-centaur]: Helsing, "Helsing Announces Project Centaur: Autonomy for Air Combat", <https://helsing.ai/newsroom/helsing-announces-project-centaur-autonomy-for-air-combat>

[^deepmind-careers]: Google DeepMind, "Careers", <https://deepmind.google/careers/>
