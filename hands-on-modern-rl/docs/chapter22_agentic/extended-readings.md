# 补充阅读：Agentic RL 延伸阅读索引

本章前面的六节覆盖了 Agentic RL 的核心理论、工程实践和工业实战。但 Agentic RL 的技术版图远不止于此——2025–2026 年，RL 正在被应用于越来越多的智能体场景：从角色扮演到创意写作，从科学发现到情感对话。本页按主题整理了超过 120 篇代表性工作，方便你按兴趣深入探索。

::: tip 使用建议
每个主题按"综述 → 方法 → 系统"的顺序排列。建议先读综述类工作建立全局认知，再按需深入具体方向。标记为 **[开源]** 的工作附有 GitHub 链接，可以直接动手实验。
:::

## 综述与理论基石

Agentic RL 的理论基础正在快速成型。本节收录的综述从不同角度梳理了这一新兴领域的全貌：有的聚焦训练配方与工程实践，有的将 LLM 重新定义为自主决策者并围绕六大核心能力综述了 500+ 篇工作，还有的专门为深度研究系统或智能体搜索任务撰写 RL 基础。如果你想快速建立对 Agentic RL 全局版图的认知，从这里开始。

| 工作名称                                                     | 核心亮点                                                   | 链接                                                                                  |
| ------------------------------------------------------------ | ---------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Adaptation of Agentic AI: A Survey                           | 综述 AI 智能体的后训练、记忆与技能适应技术                 | [arXiv](https://arxiv.org/abs/2512.16301)                                             |
| Training Recipes for Agentic RL in LLMs                      | 系统整理 Agentic RL 的"训练配方"，包括环境、采样策略等     | [TechRxiv](https://www.techrxiv.org/doi/full/10.36227/techrxiv.173816128.89654321/v1) |
| The Landscape of Agentic RL for LLMs: A Survey               | 将 LLM 视为自主决策者，围绕六大核心能力综述超过 500 篇工作 | [arXiv](https://arxiv.org/abs/2509.02547)                                             |
| A Comprehensive Survey on RL-based Agentic Search            | 综述强化学习在智能体搜索任务中的应用                       | [arXiv](https://arxiv.org/abs/2510.16724)                                             |
| Meta-Thinking in LLMs via Multi-Agent RL                     | 探讨如何通过多智能体 RL 实现 LLM 的元思考能力              | [arXiv](https://arxiv.org/abs/2504.14520)                                             |
| Reinforcement Learning Foundations for Deep Research Systems | 首篇专为深度研究系统 RL 基础撰写的综述                     | [arXiv](https://arxiv.org/abs/2509.06733)                                             |

## 深度研究与信息整合

深度研究智能体（Deep Research Agent）是 Agentic RL 最热门的应用方向之一。与简单的搜索-总结不同，它需要模型在真实网络环境中进行多轮、长程的信息搜索、交叉验证和综合分析。本节收录了从端到端 RL 框架到引用感知奖励的多种方案，覆盖了从 7B 小模型到 30B 大模型的不同规模。

| 工作名称                       | 核心亮点                                                                                               | 链接                                                            |
| ------------------------------ | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| DeepResearcher **[开源]**      | 端到端 RL 框架，在真实网络环境中进行搜索交互                                                           | [GitHub](https://github.com/GAIR-NLP/DeepResearcher)            |
| Tongyi DeepResearch **[开源]** | 阿里通义实验室的 30.5B MoE 模型（3.3B 激活），采用"Agentic Mid-training + Post-training"两阶段训练流程 | [arXiv](https://arxiv.org/abs/2510.24701)                       |
| IntentRL                       | 训练智能体在开始长程研究前，主动澄清用户模糊意图                                                       | [arXiv](https://arxiv.org/abs/2602.03468)                       |
| DR Tulu / RLER                 | 采用演化评分标准 (RLER) 的 RL 训练方案，提升长文研究能力                                               | [AllenAI Blog](https://allenai.org/blog/dr-tulu)                |
| EigentSearch-Q+                | 引入结构化推理工具 (Q+)，增强深度研究智能体的能力                                                      | [arXiv](https://arxiv.org/abs/2604.07927)                       |
| Fathom-DeepResearch            | 由 Search 和 Reason 两个 4B 模型组成的多智能体系统，生成 DUETQA 数据集                                 | [arXiv](https://arxiv.org/abs/2509.24107)                       |
| PokeeResearch-7B **[开源]**    | 7B 参数量的开源深度研究智能体                                                                          | [HuggingFace](https://huggingface.co/PokeeAI/pokee_research_7b) |
| SFR-DeepResearch               | Salesforce 出品，专注于自主单智能体的持续 RL 训练                                                      | [arXiv](https://arxiv.org/abs/2509.06283)                       |
| CaRR / C-GRPO **[开源]**       | 引入引用感知的评分奖励，遏制模型产生幻觉                                                               | [GitHub](https://github.com/THUDM/CaRR)                         |

## 强化推理与代码生成

RLVR（Reinforcement Learning from Verifiable Rewards）天然适配代码生成任务——代码是否能通过测试、是否能正确执行，都是客观可验证的信号。本节的工作围绕这一核心优势展开：有的将代码执行反馈直接整合进多轮训练，有的探索无真值监督下的 RLVR，还有的发现模型会自发学会生成并执行代码，并揭示了其中的 Scaling Law。

| 工作名称                                    | 核心亮点                                                 | 链接                                                     |
| ------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------- |
| rStar2-Agent **[开源]**                     | 基于 GRPO 的 14B Agent RL 算法，在数学推理上展现强竞争力 | [arXiv](https://arxiv.org/abs/2508.20722)                |
| Murphy                                      | 多轮 RLVR 框架，将代码执行反馈直接整合进训练             | [arXiv](https://arxiv.org/abs/2511.07833)                |
| ZeroCoder                                   | 探索在没有真值监督的情况下，通过 RLVR 提升代码生成能力   | [arXiv](https://arxiv.org/abs/2604.07864)                |
| SARL                                        | 通过奖励推理拓扑结构，实现无标签的推理能力提升           | [arXiv](https://arxiv.org/abs/2603.27977)                |
| Agentic RL Scaling Law / ZeroTIR **[开源]** | 发现模型能自发学会生成并执行代码，并揭示其训练规律       | [GitHub](https://github.com/yyht/openrlhf_async_pipline) |
| Agnostics                                   | 语言无关的代码 RL 训练框架                               | [Project](https://agnostics.abgru.me)                    |
| ReLook                                      | 基于视觉反馈（渲染截图）的 RL 来优化网页前端代码生成     | [arXiv](https://arxiv.org/abs/2510.11498)                |
| Agentic Code Reasoning                      | 通过半形式化推理，为 RL 提供低成本、无风险的奖励信号     | [arXiv](https://arxiv.org/abs/2603.01896)                |
| Code-Space Response Oracles                 | 使用 LLM 作为代码生成预言机，替代传统 RL 预言机          | [arXiv](https://arxiv.org/abs/2603.10098)                |

## GUI 与网页智能体

GUI 智能体让 AI 能像人一样操作图形界面——点击按钮、填写表单、在网页上导航。RL 在这里的价值在于：SFT 只能教会模型"模仿点击"，而 RL 能让模型学会"根据目标选择最优操作路径"。本节覆盖了从网页到移动端、从 3B 小模型到持续学习框架的多种方案。

| 工作名称                                      | 核心亮点                                                     | 链接                                                           |
| --------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------- |
| WebAgent-R1 **[开源]**                        | 端到端多轮 RL 框架，将 3B 模型成功率从 6.1% 提升至 33.9%     | [GitHub](https://github.com/weizhepei/WebAgent-R1)           |
| Web-Shepherd **[开源]**                       | 首个网页导航专用步骤级奖励模型，能评估每一步交互             | [GitHub](https://github.com/kyle8581/Web-Shepherd)             |
| CRAFT-GUI                                     | 结合课程学习与 GRPO，提升 GUI 智能体性能                     | [arXiv](https://arxiv.org/abs/2508.11360)                      |
| MobileRL **[开源]**                           | 移动端在线 RL 框架，使用 ADAGRPO 算法                        | [GitHub](https://github.com/THUDM/MobileRL)                    |
| Co-EPG                                        | 通过协同进化框架，同时优化 GUI 智能体的规划与接地能力        | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40981) |
| Continual GUI Agents                          | 定义并解决 GUI 智能体在持续变化环境下的学习问题              | [arXiv](https://arxiv.org/abs/2601.20732)                      |
| WebFactory                                    | 全自动闭环 RL 流程，将 LLM 的智能"压缩"到高效的 GUI 智能体中 | [OpenReview](https://openreview.net/forum?id=HaIEP2PD4S)       |
| ZeroGUI                                       | 零人工成本的在线 GUI 智能体学习框架                          | [arXiv](https://arxiv.org/abs/2505.23762)                      |
| UI-S1                                         | 结合离线与在线数据优势的半在线 RL 训练方法                   | [arXiv](https://arxiv.org/abs/2509.11543)                      |
| Generalization in Online RL for Mobile Agents | 研究移动智能体在线 RL 中的泛化问题，证明 RL 能超越 SFT 基线  | [OpenReview](https://openreview.net/forum?id=INoDyme6wS)       |

## 具身智能与机器人

当 RL 从数字世界走向物理世界，智能体面对的不再是文本或图像，而是连续的控制信号和不确定的物理环境。本节的工作探索了如何让 LLM 直接参与机器人推理与控制：有的用 RL 优化空间推理过程使 7B 模型超越 GPT-4o，有的在像素级世界模型中训练出自我纠错能力，还有的研究跨形态迁移和持续学习中的"认知身份"保持问题。

| 工作名称                         | 核心亮点                                                       | 链接                                            |
| -------------------------------- | -------------------------------------------------------------- | ----------------------------------------------- |
| Robot-R1                         | 用 RL 直接优化机器人的推理过程，7B 模型空间推理能力超越 GPT-4o | [arXiv](https://arxiv.org/abs/2506.00070)       |
| WMPO **[开源]**                  | 在像素级视觉世界模型中进行 RL 训练，涌现出自我纠错能力         | [GitHub](https://github.com/WM-PO/WMPO) |
| ViVa                             | 使用预训练视频生成模型作为价值函数估计器，评估状态价值         | [arXiv](https://arxiv.org/abs/2604.08168)       |
| RoboAgent                        | 通过组合基础能力，实现具身任务规划                             | [arXiv](https://arxiv.org/abs/2604.07774)       |
| Cross-Embodiment Offline RL      | 通过形态学分组策略，实现跨不同形态机器人的离线 RL              | [arXiv](https://arxiv.org/abs/2602.18025)       |
| Sensory-Motor Control with LLMs  | 通过迭代策略精炼，使 LLM 能直接生成连续控制策略                | [arXiv](https://arxiv.org/abs/2506.04867)       |
| RM-RL                            | 提出"榜样模型"RL，用于实现精准的机器人操作                     | [arXiv](https://arxiv.org/abs/2510.15189)       |
| Learning Without Losing Identity | 研究具身智能体在持续学习新能力时，如何保持"认知身份"的稳定     | [arXiv](https://arxiv.org/abs/2604.07799)       |

## 多智能体系统与协作

多个智能体协作的难度远超单智能体——当你学新策略时队友也在变，环境是非平稳的；团队成功了功劳归谁，失败了责任在谁。本节的工作从多个角度应对这些挑战：将 GRPO 扩展至多智能体场景、通过知识蒸馏实现去中心化协调、用数字孪生解决上下文漂移问题，以及端到端联合优化采样与训练的大规模 MARL 框架。

| 工作名称                   | 核心亮点                                                                  | 链接                                         |
| -------------------------- | ------------------------------------------------------------------------- | -------------------------------------------- |
| MAPoRL                     | 多智能体协作训练新范式                                                    | [arXiv](https://arxiv.org/abs/2502.18439)    |
| M-GRPO                     | 将 GRPO 算法扩展至多智能体场景                                            | [arXiv](https://arxiv.org/abs/2511.13288)    |
| SAGE                       | 闭环自进化多智能体 RL 框架                                                | [arXiv](https://arxiv.org/abs/2603.15255)    |
| MARTI **[开源]**           | 多智能体辩论框架                                                          | [GitHub](https://github.com/TsinghuaC3I/MARTI) |
| KD-MARL                    | 通过知识蒸馏，将集中式专家的协调行为迁移到轻量级去中心化智能体中          | [arXiv](https://arxiv.org/abs/2604.06691)    |
| Value-Guidance MeanFlow    | 用于离线多智能体 RL 的价值引导流模型                                      | [arXiv](https://arxiv.org/abs/2604.08174)    |
| FlexMARL                   | 首个端到端训练框架，联合优化采样、训练及其编排，用于大规模 LLM-based MARL | [arXiv](https://arxiv.org/abs/2602.09578)    |
| TwinLoop                   | 提出仿真在环数字孪生框架，解决上下文变化导致的多智能体性能下降问题        | [arXiv](https://arxiv.org/abs/2604.06610)    |
| Equivariant Multi-agent RL | 用于多模态车路协同系统的等变多智能体 RL                                   | [arXiv](https://arxiv.org/abs/2604.06914)    |

## 世界模型与基于模型的 RL

无模型 RL 的核心瓶颈是样本效率——智能体必须通过大量试错才能学习。世界模型提供了一条绕过瓶颈的路径：先学会"脑内模拟环境"，再在想象中生成训练数据。本节收录了从扩散世界模型到对象中心表征的多种方案，核心思路都是让策略模型与世界模型交互，在"想象"中完成多步规划与训练。

| 工作名称                | 核心亮点                                         | 链接                                        |
| ----------------------- | ------------------------------------------------ | ------------------------------------------- |
| GIRL                    | 通过信息论幻觉控制，实现生成式想象 RL            | [arXiv](https://arxiv.org/abs/2604.07426)   |
| World4RL                | 扩散世界模型，用于机器人操作的策略精炼           | [arXiv](https://arxiv.org/abs/2509.19080)   |
| Dreamer-CDP             | 无需重建原始像素观察的 Dreamer 变体              | —                                             |
| RLVR-World              | 使用 RLVR 直接优化世界模型                       | [arXiv](https://arxiv.org/abs/2505.13934)   |
| OC-STORM                | 利用对象中心表征增强世界模型，实现样本高效的 RL  | [arXiv](https://arxiv.org/abs/2501.16443)   |
| Imagine-then-Plan (ITP) | 让策略模型与世界模型交互，在"想象"中生成多步轨迹 | [arXiv](https://arxiv.org/abs/2601.08955)   |

## 角色扮演与人格模拟

角色扮演不只是"假装是某个人"——它要求模型在长对话中保持一致的人格特征、思维方式和行为模式。RL 在这里的价值在于：通过可验证的角色意识奖励，强化模型对"我是谁"的持续感知。本节的工作从双层思考框架（区分角色视角和模型视角）到多角色自博弈，探索了如何让 AI 真正"入戏"并保持角色一致性。

| 工作名称                               | 核心亮点                                                                                                | 链接                                                        |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| HER (Human-like Reasoning)             | 提出双层思考框架，区分角色的第一人称思维和 LLM 的第三人称思维（注：非经典 Hindsight Experience Replay） | [arXiv](https://arxiv.org/abs/2601.21459)                   |
| OMAR                                   | 通过多轮自博弈 RL，培养 AI 的社交智能                                                                   | [arXiv](https://arxiv.org/abs/2602.03109)                   |
| R4                                     | 赋予奖励模型和角色扮演智能体推理与检索能力                                                              | [ICLR Poster](https://iclr.cc/virtual/2026/poster/10007049) |
| VeriRole                               | 通过可验证的提示引导 RL 提升角色意识                                                                    | [OpenReview](https://openreview.net/forum?id=lW7kMpMj9K)    |
| SPELL                                  | 多角色自博弈 RL 框架，用于长上下文推理                                                                  | [arXiv](https://arxiv.org/abs/2509.23863)                   |
| Consistently Simulating Human Personas | 提出评估和改进 LLM 角色一致性的统一框架                                                                 | [OpenReview](https://openreview.net/forum?id=A0T3piHiis)    |
| CPO                                    | 针对角色扮演对话中奖励模糊问题的比较策略优化                                                            | [arXiv](https://arxiv.org/abs/2508.09074)                   |
| RAIDEN-R1                              | 提出可验证的角色意识奖励 (VRAR)，强化模型对自身角色的感知                                               | [arXiv](https://arxiv.org/abs/2505.10218)                   |

## 创意与长文写作

创意写作对 RL 提出了独特挑战：奖励不像代码执行那样客观可验证，"好"的写作是主观的、多维度的。本节的工作探索了如何设计能捕捉创意质量的奖励信号——从生成式奖励模型对故事偏好进行多维推理，到通过交替 RL 优化基于评分标准的奖励模型，再到用 RLAIF 比较不同奖励策略以激发小模型的创意能力。

| 工作名称                                        | 核心亮点                                                              | 链接                                                           |
| ----------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------- |
| Writer-R1                                       | 记忆增强的重放策略优化（Memory-augmented Replay Policy Optimization） | [arXiv](https://arxiv.org/abs/2603.15061)                      |
| R2-Write                                        | 对开放域写作进行系统性研究，提出反思与修正框架                        | [arXiv](https://arxiv.org/abs/2604.03004)                      |
| DPWriter                                        | 通过多样化规划分支，解决 RL 训练中输出多样性降低的问题                | [arXiv](https://arxiv.org/abs/2601.09609)                      |
| RLMR                                            | 首次在在线 RL 训练中结合主观偏好与客观验证                            | [arXiv](https://arxiv.org/abs/2508.18642)                      |
| Rewarding Creativity                            | 开发生成式奖励模型，对故事偏好进行多维分析和显式推理                  | [arXiv](https://arxiv.org/abs/2601.07149)                      |
| Alternating RL for Rubric-Based Reward Modeling | 通过交替 RL 优化基于评分标准的奖励模型，在多个写作基准上达到 SOTA     | [arXiv](https://arxiv.org/abs/2602.01511)                      |
| Igniting Creative Writing in SLMs               | 在 RLAIF 框架下比较两种奖励策略，激发 7B 小模型的创意写作             | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.868/) |

## 情感智能与共情对话

共情不只是"理解情绪"——它需要在恰当的时机表达恰当的回应，同时保持对话的逻辑连贯。RL 在这里的价值是让模型学会平衡"情感支持"与"认知推理"。本节的工作从可验证情感奖励到基于心理学的共情奖励建模，探索了如何为 RL 提供更扎实的奖励信号。

| 工作名称                          | 核心亮点                                                         | 链接                                      |
| --------------------------------- | ---------------------------------------------------------------- | ----------------------------------------- |
| RLVER                             | 利用可验证情感奖励训练 LLM 的高阶共情能力                        | [arXiv](https://arxiv.org/abs/2507.03112) |
| CARE                              | 认知推理增强的 RL，提升情感支持对话的逻辑性与支持质量            | [arXiv](https://arxiv.org/abs/2510.05122) |
| COMPEER                           | 统一过程-结果 RL，实现结构化共情推理                             | [arXiv](https://arxiv.org/abs/2508.09521) |
| DialogXpert                       | 基于在线价值 RL 的对话规划，在谈判、情感支持等任务上成功率超 94% | [arXiv](https://arxiv.org/abs/2505.17795) |
| EILS                              | 受生物情绪启发的内稳态学习信号框架，用于构建自适应自主智能体     | [arXiv](https://arxiv.org/abs/2512.22200) |
| SAGE (Steering Dialog Generation) | 使用隐变量控制对话生成的长期行为，用于构建情感智能聊天机器人     | [arXiv](https://arxiv.org/abs/2503.03040) |
| PERM                              | 基于心理学的共情奖励建模，为 RL 提供更扎实的奖励信号             | [arXiv](https://arxiv.org/abs/2601.10532) |

## 艺术与视觉创作

RL 进入艺术领域是一个有趣的跨界——它将"审美判断"建模为可优化的奖励信号。本节的工作覆盖了从图像生成优化到分层绘画、从个性化手绘到艺术风格学习的多种应用。核心思路包括：协调多个专家模型迭代优化图像生成、通过逆 RL 从笔触数据中学习艺术家风格，以及用分层 RL 实现高层规划与低层绘制的分离。

| 工作名称                | 核心亮点                                                           | 链接                                                                                 |
| ----------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| Image-POSER             | 反思式 RL 框架，协调多个专家模型，根据复杂文本提示迭代优化图像生成 | [arXiv](https://arxiv.org/abs/2511.11780)                                            |
| HRL-Painter             | 基于分层 RL 的绘画方法，高层规划区域，低层执行绘制                 | [Neurocomputing](https://doi.org/10.1016/j.neucom.2025.129972)                       |
| PersonaSketch-RL        | 基于 RL 的策略，用于优化个性化手绘插图生成                         | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1875952125001338) |
| RMLer                   | 将跨类别概念融合建模为 RL 问题，用于合成新颖物体                   | [arXiv](https://arxiv.org/abs/2512.19300)                                            |
| Sequential Art Creation | 深度 RL 框架，用于创作视觉形式上与输入不同的序列艺术作品           | [UTA Thesis](https://mavmatrix.uta.edu/cse_theses/539/)                              |
| MVAEx-RL                | 基于 RL 的多模态艺术元素提取和动态适应策略，用于环境设计           | [Springer](https://link.springer.com/article/10.1007/s44163-025-00712-z)             |
| DailyArt                | 将关节估计问题建模为合成介导的推理问题，从单张静态图像中推断动力学 | [arXiv](https://arxiv.org/abs/2604.07758)                                            |

## RL 训练基础设施与算法创新

Agentic RL 的工程复杂度远超标准 LLM RL——你需要同时管理 GPU 上的模型训练、CPU 上的工具执行和网络上的环境交互。本节聚焦于支撑这些复杂训练流程的基础设施和算法创新：从全异步训练系统到可扩展的合成学习环境，从检索增强的策略优化到将推理计算转化为训练信号的新范式。

| 工作名称                 | 核心亮点                                                        | 链接                                                       |
| ------------------------ | --------------------------------------------------------------- | ---------------------------------------------------------- |
| AReaL v1.0 **[开源]**    | 蚂蚁与清华联合开源，实现"Agent 一键接入 RL 训练"                | [GitHub](https://github.com/inclusionAI/AReaL)             |
| RollArt / RollArc        | 通过解耦基础设施 (RollArc) 最大化多任务 Agentic RL 的训练吞吐量 | [arXiv](https://arxiv.org/abs/2512.22560)                  |
| SparrowRL                | 在商用网络上实现无损稀疏增量同步的高性能 RL 训练系统            | [arXiv](https://arxiv.org/abs/2602.11456)                  |
| Laminar                  | 基于全解耦架构的可扩展、鲁棒的异步 RL 后训练系统                | [arXiv](https://arxiv.org/abs/2510.12633)                  |
| SCALER                   | 合成可扩展的自适应学习环境，为 RL 训练提供无限可验证的推理环境  | [arXiv](https://arxiv.org/abs/2601.04809)                  |
| L-Zero (L0)              | 低成本、可扩展的端到端通用智能体训练流程                        | [arXiv](https://arxiv.org/abs/2506.23667)                  |
| Compute as Teacher (CaT) | 将推理时的并行采样计算转化为 RL 训练的监督信号                  | [arXiv](https://arxiv.org/abs/2509.14234)                  |
| RAPO                     | 检索增强策略优化，在训练期间显式扩展智能体的探索空间            | [arXiv](https://arxiv.org/abs/2603.03078)                  |
| LLM-Explorer **[开源]**  | 清华出品，可增强各种 RL 算法探索能力的插件                      | [GitHub](https://github.com/tsinghua-fib-lab/LLM-Explorer) |

## 科学发现与工业应用

RL 正在走出实验室，进入化学、材料科学、医学和工业制造等真实应用场景。本节的工作将科学问题建模为 MDP：先导化合物优化变成在合成约束下的搜索问题，材料设计变成利用形成能反馈的优化问题，工业异常检测变成了数据合成的策略学习问题。这些应用展示了 RL 作为"通用决策优化器"的潜力。

| 工作名称                             | 核心亮点                                                                | 链接                                                                                                      |
| ------------------------------------ | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| MolReAct                             | 将先导化合物优化建模为 MDP，用 RL 在合成约束下进行高效搜索              | [arXiv](https://arxiv.org/abs/2604.07669)                                                                 |
| PolyRL                               | 使用 RL 引导的多目标聚合物生成与发现                                    | [RSC](https://pubs.rsc.org/en/content/articlelanding/2026/dd/d5dd00272a)                                  |
| Helix                                | 面向开放式科学问题求解的分层进化 RL 框架                                | [arXiv](https://arxiv.org/abs/2603.07642)                                                                 |
| RLFEF                                | 利用形成能反馈的 RL 来微调材料扩散模型，提升晶体稳定性                  | [dblp](https://dblp.org/rec/journals/nn/HuangXJY26.html)                                                  |
| AnomalyAgent                         | 工业异常数据合成智能体，通过 RL 优化生成高真实感的异常样本              | [arXiv](https://arxiv.org/abs/2604.07900)                                                                 |
| Autonomous Adaptive Solver Selection | 使用约束 RL 框架，在化学积分过程中自主选择求解器                        | [arXiv](https://arxiv.org/abs/2604.00264)                                                                 |
| PPO-based Surface Reconstruction     | 基于 PPO 的深度 RL 框架，用于 AgPd 合金催化剂的表面重构                 | [AIP PDF](https://pubs.aip.org/aip/jap/article-pdf/doi/10.1063/5.0295785/20878476/045001_1_5.0295785.pdf) |
| MedVR                                | 针对医学 VQA，提出熵引导视觉重定位（EVR）和共识驱动信用分配两种 RL 机制 | [arXiv](https://arxiv.org/abs/2604.08203)                                                                 |

---

> **提示：** 以上工作均为 2025–2026 年发表或预印的论文/项目。部分 arXiv 论文可能已更新版本，建议通过论文标题在 [arxiv.org](https://arxiv.org) 或 [Semantic Scholar](https://www.semanticscholar.org) 搜索获取最新版本。
