# 23.2 视觉反思 RL：让模型带着证据再回答

先看一道图表题。柱状图中，A 类为 42，B 类为 57，C 类为 39。问题问"B 比 A 高多少"。模型如果只抓住最高的柱，很容易直接回答 57；如果先读出两根柱的数值，再计算 $57-42$，答案才是 15。

这里缺少的能力并不神秘：模型需要在输出最终答案前，保留视觉证据、完成计算，并回到图像检查一次。我们把这种"观察—推理—核验—回答"的过程称为**视觉反思（Visual Reflection）**。它的目标是减少看错、漏看和凭语言先验猜答案，而不是单纯让回答变长。

本节以 [Qwen3-VL 技术报告](https://arxiv.org/abs/2511.21631)为主线，解释视觉反思依赖哪些模型结构、怎样通过后训练形成，以及为什么"多想几步"仍然可能失败。Qwen3-VL 于 2025 年 9 月开始发布模型，技术报告在 2025 年 11 月公开；它与 2025 年 4 月发布的纯文本 Qwen3 不是同一次发布。

![GeoQA 示例](./images/geoqa-example.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 1：几何视觉问答示例。模型需要先识别图中的几何关系（视觉证据），再完成计算（中间推理），最后给出答案。来源：<a href="https://github.com/ChallenAI/EasyR1" target="_blank" rel="noopener noreferrer">EasyR1 项目</a></em>
</div>

## 什么是视觉反思？

视觉反思是指**模型在输出最终答案前，显式保留视觉证据、完成中间推理、并回看图像核验的过程**。它不是让回答变长，而是让推理过程可观察、可验证。

这个概念来自认知科学的一个基本洞察：人类的复杂视觉任务不是"看一眼就答"，而是反复观察、计算、核验。读一张收据时，你会先定位小计和税额，读取两个数字，完成加法，再回到收据确认读数来自正确区域。视觉反思把这个过程模型化——让 AI 也通过"观察—推理—核验"来完成任务。

![VLM-R1 IoU 可视化](./images/ref-vlm-r1-iou.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 2：VLM-R1 的视觉定位可视化。模型在推理过程中显式输出边界框坐标，将视觉证据转化为可验证的轨迹。来源：<a href="https://github.com/River-Zhang/VLM-R1" target="_blank" rel="noopener noreferrer">VLM-R1 项目</a></em>
</div>

## 视觉推理的三类对象

给模型一张收据，问"含税总额是多少"。描述任务只要求识别"这是一张收据"；推理任务需要定位小计与税额，读取两个数字，再完成加法。只要其中一个数字来自错误区域，后面的计算即使完全正确，最终答案仍会失败。

因此，一条视觉推理轨迹至少包含三类对象：

- **视觉证据**：图中哪些区域、文字、物体或时间片段与问题有关。例如柱状图中 A 柱顶部（42）和 B 柱顶部（57）。
- **中间推理**：证据之间怎样比较、计算或建立因果关系。例如读出 42 和 57，计算 $57-42=15$。
- **最终答案**：按任务要求输出数字、选项、坐标或自然语言。例如输出 15。

普通结果奖励只检查最终答案。它会把"看错后算对"和"看对后算错"都记为失败，也会把"完全没有看图但碰巧猜对"记为成功。[23.1 视觉奖励设计](./vlm-challenges)已经说明这种标量奖励的局限；视觉反思的价值，就是把部分中间过程变成可观察、可验证的轨迹。

```mermaid
flowchart LR
    I["图像 / 视频"] --> O["定位相关证据"]
    Q["问题"] --> O
    O --> R["基于证据推理"]
    R --> V["回看证据并核验"]
    V --> A["最终答案"]
    V -->|"证据不足"| O
```

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 3：视觉反思流程。模型先定位证据，再基于证据推理，然后回看核验，证据不足时重新观察。</em>
</div>

回到开头的柱状图：模型需要先定位 A 柱和 B 柱（视觉证据），读出 42 和 57（中间推理），计算差值（中间推理），再回到图像确认读数来自正确位置（核验），最后输出 15（最终答案）。任何一步出错，答案都会失败。

## 视觉反思与普通 VLM 的区别

你可能已经发现，视觉反思用的也是 VLM 的框架——图像、问题、回答。那它和前面学的普通 VLM 推理有什么本质区别？

| 维度 | 普通 VLM | 视觉反思 VLM |
|------|---------|-------------|
| 推理过程 | 隐式，直接从图像到答案 | 显式，保留中间证据和计算步骤 |
| 可验证性 | 只看最终答案 | 可检查中间证据是否正确 |
| 失败模式 | 看错、漏看、猜对都混在一起 | 可定位是哪一步出错（证据/推理/核验） |
| 训练目标 | 结果奖励 | 结果 + 过程奖励（证据命中率、工具调用有效性） |
| 工具使用 | 无 | 可调用放大、搜索等工具重新观察 |
| 适用任务 | 简单识别、分类 | 复杂推理、多步计算、长视频理解 |

核心差异可以归结为一点：**在普通 VLM 中，你只关心最终答案对不对；在视觉反思中，你需要关心中间证据是否被正确使用。**

## 视觉证据怎样进入语言推理

反思能力首先受模型结构限制。如果视觉细节在进入语言模型前已经丢失，再长的思维链也无法恢复原始信息。Qwen3-VL 仍采用"视觉编码器—视觉语言合并层—语言模型"三部分结构，同时加入三项与证据保留直接相关的改动[^qwen3vl]。

![Qwen2.5-VL 架构](./images/qwen2.5-vl-architecture.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 4：Qwen2.5-VL 架构示意。视觉编码器提取特征，经过视觉语言合并层后送入语言模型。Qwen3-VL 在此基础上增加了 DeepStack 和多级位置编码。来源：<a href="https://arxiv.org/abs/2502.13923" target="_blank" rel="noopener noreferrer">Qwen2.5-VL 技术报告</a></em>
</div>

### DeepStack：把不同深度的视觉特征送入语言模型

视觉编码器的浅层更容易保留边缘、纹理和局部位置，深层更偏向物体与语义。只取最后一层特征，细小文字或局部几何关系可能已经被压缩。

DeepStack 会从视觉编码器的多个层级取出特征，经独立的合并模块处理后，注入语言模型前几层。这样做不需要把更多视觉 token 塞进上下文，却能让语言模型同时接触局部细节与高层语义。它解决的是"证据有没有进入推理过程"，并不保证模型一定使用了正确证据。

### Interleaved-MRoPE：把时间、宽度和高度写进位置

一张图片中的视觉 token 具有二维位置，视频还多出时间轴。Qwen3-VL 的 interleaved-MRoPE 把时间、高度和宽度位置交错分配到旋转位置编码中，使模型能区分"左上角的表头""右下角的数值"和"第 12 秒出现的物体"。

这项结构改动对空间关系和长视频尤其重要。若位置编码无法稳定表达时间与空间，模型可能识别出两个对象，却弄错它们的先后与相对位置。

### 文本时间戳：让视频证据可以被说出来

视频任务常问"某个动作发生在什么时候"。Qwen3-VL 将时间位置显式写成文本时间戳，使回答可以引用"3.0 秒附近"这样的证据。这样，时间定位从隐藏向量变成了可检查的文本对象。

三项改动共同提供了视觉反思的底座：DeepStack 尽量保留细节，位置编码保存空间与时间关系，文本时间戳让视频证据能够进入语言推理。官方报告还给出原生 256K 交错多模态上下文，并提供 2B、4B、8B、32B 稠密模型以及 30B-A3B、235B-A22B MoE 模型，以覆盖不同延迟与质量需求[^qwen3vl_repo]。

## Thinking 版本怎样通过后训练形成

Qwen3-VL 同时发布 Instruct 与 Thinking 版本。Instruct 版本更偏向直接回答；Thinking 版本会在复杂任务上生成较长的中间推理。二者共享多模态底座，但后训练目标不同。

根据技术报告，Thinking 路线依次使用长思维链冷启动、强模型到弱模型蒸馏、推理强化学习和通用强化学习[^qwen3vl]。这条链可以从一道几何题理解。

![GRPO 训练流程](./images/illustrated-grpo.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 5：GRPO 训练流程示意。Thinking 版本的推理 RL 使用类似机制，在同一条件下的多个推理轨迹中比较相对优势。来源：<a href="https://github.com/ChallenAI/EasyR1" target="_blank" rel="noopener noreferrer">EasyR1 项目</a></em>
</div>

![EasyR1 GRPO  diagram](./images/easyr1-grpo-diagram.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 6：EasyR1 的 GRPO 实现架构。模型在多个并行环境中采样轨迹，通过组内归一化计算相对优势。来源：<a href="https://github.com/ChallenAI/EasyR1" target="_blank" rel="noopener noreferrer">EasyR1 项目</a></em>
</div>

冷启动阶段先给模型少量结构完整的示范：读出图形关系，写出中间等式，再给答案。它解决的是输出格式与基本推理习惯。

蒸馏阶段让更强模型为较小模型提供高质量轨迹。小模型先学会"可行的推理大致长什么样"，再进入强化学习探索，减少从完全随机的长回答中寻找正确路径的成本。

推理强化学习覆盖文本与多模态任务，包括数学、代码、逻辑、视觉定位和视觉谜题。可验证任务可以使用答案、坐标、边界框或工具结果作为奖励。通用强化学习随后补充指令遵循、交互质量与安全等目标，防止模型只会做有标准答案的题。

这和"在提示词里写一句 please think step by step"有本质差别。提示词只改变一次推理的上下文；后训练会提高整类轨迹在策略中的概率，使模型在没有手写五步模板时也可能产生观察、计算与核验行为。

## 证据不足时重新观察

有些图像细节小到模型一次前向很难看清。例如，在一张 4K 电路图里寻找某个标号，或在长截图中核对一行金额。继续生成文本不会增加图像分辨率，此时缺少的是"重新观察"的动作。

Qwen3-VL 的 Thinking with Images 把图像放大与搜索工具接入推理过程。模型可以先判断证据不足，再调用 `image_zoom_in_tool` 裁出局部区域，读取新的视觉观察后继续推理。官方仓库将这一能力单独列为 cookbook，技术报告则描述了冷启动 SFT 与工具集成 RL 的训练流程[^qwen3vl_repo]。

```mermaid
sequenceDiagram
    participant M as Thinking 模型
    participant T as 图像工具
    participant E as 奖励 / 验证器
    M->>M: 判断现有证据不足
    M->>T: 放大表格右下区域
    T-->>M: 返回高分辨率局部图
    M->>M: 读取数值并计算
    M->>E: 提交答案与工具轨迹
    E-->>M: 答案、格式与工具有效性反馈
```

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 7：Thinking with Images 工具调用流程。模型判断证据不足时调用放大工具，获取高分辨率局部图后继续推理。</em>
</div>

为了理解这类训练，可以写一个教学化的复合奖励：

$$
R = R_{\text{answer}} + \lambda_f R_{\text{format}}
  + \lambda_t R_{\text{tool}} - \lambda_c C_{\text{tool}}.
$$

$R_{\text{answer}}$ 检查最终答案，$R_{\text{format}}$ 检查输出能否解析，$R_{\text{tool}}$ 检查工具参数与返回值是否有效，$C_{\text{tool}}$ 计算不必要的调用成本。这个公式是用于解释设计空间的教学简化，并非 Qwen3-VL 论文公布的精确训练目标。

加入成本项很重要。若只奖励最终正确，模型可能对每道题都反复放大整张图；成功率提高了，延迟与调用成本却不可接受。工具让模型获得新证据，同时也把"何时重新看、看哪里、看几次"变成新的策略学习问题。

## 运行一个最小的视觉反思检查

下面的代码使用官方 `transformers` 接口加载 4B Thinking 模型。它只演示推理接口，不代表完成了 RL 训练。显存需求受图片分辨率、精度和注意力实现影响，运行前应按本机条件选择模型与量化方式。

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "Qwen/Qwen3-VL-4B-Thinking"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "url": "./chart.png"},
        {"type": "text", "text": "读出 A、B 两根柱的数值，计算 B-A，并给出证据。"},
    ],
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=1024)
answer = processor.batch_decode(
    output_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True,
)[0]
print(answer)
```

检查结果时不要只看最终数字。至少记录四项证据：是否读出了 A 与 B，两个读数是否来自正确位置，计算是否正确，换成一张数值不同但版式相同的图后答案是否随图变化。

最后一项是反事实检查。若换图后中间证据和答案几乎不变，模型很可能依赖题型先验。此时继续奖励长思维链只会让猜测写得更像推理。

## 视觉反思仍然会怎样失败

**反思可能建立在错误观察上。** 模型把 42 看成 47 后，可以写出一条完全自洽的减法过程。修复这类问题需要视觉定位、OCR 或工具验证，单纯延长思维链没有帮助。

![EasyR1 训练曲线](./images/easyr1-geoqa-curves.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 8：EasyR1 在 GeoQA 任务上的训练曲线。即使使用 GRPO，模型仍可能在某些子任务上出现奖励黑客或证据使用不充分的问题。来源：<a href="https://github.com/ChallenAI/EasyR1" target="_blank" rel="noopener noreferrer">EasyR1 项目</a></em>
</div>

**推理长度可能代替证据质量。** 奖励模型若偏爱详细回答，模型会学会增加步骤和措辞。评测应同时报告正确率、视觉证据命中率、输出 token 数和工具调用成本。

**Thinking 并非所有任务都需要。** 对简单 OCR 或直接定位任务，长推理会增加延迟，并可能在正确观察之后引入新的计算错误。2026 年的 Perception-RFT 在文档问答实验中甚至观察到，4B 模型的无显式推理训练优于 reasoning 变体[^perception_rft]。这项结果不否定视觉推理；它说明"先看还是先想"取决于任务瓶颈。

**工具调用成功不等于任务成功。** 放大区域正确，只能证明动作有效；还要检查模型是否使用返回结果更新了结论。轨迹评测必须同时保存调用参数、工具观察和最终答案。

## 与前面章节的联系

视觉反思不是孤立的概念，它把前面章节的多个思想串联起来：

- **多模态奖励设计（23.1）**：视觉反思把部分中间过程变成可验证的轨迹，让奖励不仅检查结果，还能检查证据使用。
- **GRPO 的组内归一化（18 章）**：Thinking 版本的推理 RL 使用类似机制，在同一条件下的多个推理轨迹中比较相对优势。
- **工具调用 RL（22 章 Agent）**：Thinking with Images 把放大与搜索变成动作，让模型在证据不足时重新观察。
- **RLHF 的多维奖励（15 章）**：视觉反思使用多维奖励（答案、证据、工具、成本），防止模型只讨好单一指标。
- **反事实检查（25 章）**：换图检查是否依赖题型先验，与奖励黑客的识别方法一脉相承。

## 前沿：从视觉反思到音频接地推理

视觉与音频面对同一个根问题：推理链必须锚定到当前模态的证据。视觉模型可能绕开图片直接猜答案，音频模型也可能把声音先粗略转成文字，再只围绕文字推理。

[Step-Audio-R1](https://arxiv.org/abs/2511.15848) 把这一问题称为声学接地不足，并提出 MGRD（Modality-Grounded Reasoning Distillation，模态接地推理蒸馏）。MGRD 通过迭代蒸馏、监督微调与可验证奖励强化，让推理显式引用音高、节奏、音色等声学证据。它不是 DPO 的多模态版本；完整方法与奖励设计放在 [24.1 音频奖励设计](../chapter27_audio_rl/reward-design)。

这组对照留下一个可以迁移到所有模态的判断：推理变长之前，先确认模型是否获得并使用了正确证据。

## 小结

视觉反思的核心是把"观察—推理—核验—回答"变成可观察、可验证的轨迹。Qwen3-VL 的架构改动（DeepStack、位置编码、时间戳）解决证据怎样进入语言推理；后训练管线（冷启动、蒸馏、推理 RL、通用 RL）解决反思行为怎样形成；工具集成解决证据不足时怎样重新观察。但反思仍可能建立在错误观察上，推理长度可能代替证据质量，工具调用成功不等于任务成功。验证视觉反思是否有效，需要检查中间证据、反事实换图、工具轨迹和成本，而不能只看最终答案。

## 延伸阅读与参考资料

[^qwen3vl]: Qwen Team, [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631), 2025。架构、后训练与 Thinking with Images 的主要来源。

[^qwen3vl_repo]: QwenLM, [Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL)。模型版本、官方推理接口与 cookbook 索引。

[^perception_rft]: Harikrishnan P M, et al., [Stop Thinking, Start Looking: Efficient Post-Training for Multimodal Document Question Answering via Reasoning-Free Alignment](https://arxiv.org/abs/2607.14682), 2026。用于说明显式推理并非所有视觉任务都受益。

- [QVQ-72B-Preview：To See the World with Wisdom](https://qwenlm.github.io/blog/qvq-72b-preview/)：Qwen 团队在 Qwen3-VL 之前对视觉长思维链的探索。
- [QVQ-Max：Think with Evidence](https://qwenlm.github.io/blog/qvq-max-preview/)：展示增加思考预算对视觉数学任务的影响与边界。
