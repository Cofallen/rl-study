# 23.3 动手：多模态 GRPO

> **本节目标**：手写一个最小多模态 GRPO 训练循环，让视觉语言模型根据图片回答计数问题，并用规则奖励优化答案准确率。

> **学习路径**：[15.1 GRPO](../chapter18_grpo/grpo-practice-and-mechanism) → [23.1 视觉奖励设计](./vlm-challenges) → [23.2 视觉反思 RL](./qwen3-vl-reflection) → **23.3 多模态 GRPO** → [23.4 GeoQA](./easyr1-geoqa)

> **本节代码与资源**：[vlm_grpo_train.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter11_vlm_rl/vlm_grpo_train.py)

本页代码用于解释多模态 GRPO 的数据流。仓库脚本使用模板模拟不同质量的回答，不会下载或更新真实视觉语言模型。

在仓库根目录运行：

```bash
cd code/chapter11_vlm_rl
pip install -r requirements.txt
python vlm_grpo_train.py
```

运行后会输出组内奖励、相对优势和模拟训练曲线。它能检查奖励计算与组内归一化，不能证明真实视觉模型已经学会计数。

第 15 章已经用 GRPO 训练纯文本模型做数学推理。这里把输入换成“图像 + 问题”：模型先编码图片，再生成多条回答，规则奖励检查计数结果，组内相对优势决定更新方向。实验使用合成几何图形，因此每张图的正确数量在生成时已经确定，可以直接检验训练前后准确率。

![VLM-R1 IoU Reward Curve](./images/ref-vlm-r1-iou.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>图 1：VLM-R1 训练日志中的 IoU reward 曲线。它把视觉 grounding 的质量转成可优化的奖励信号，和本节“规则奖励 + GRPO”的动手实验是同一类训练思路。来源：<a href="https://github.com/om-ai-lab/VLM-R1" target="_blank" rel="noopener noreferrer">VLM-R1 GitHub</a></em>
</div>

这张图的价值不在于曲线本身有多漂亮，而在于它提醒我们：VLM RL 不是“给图片加几个 token”这么简单。只要 reward 能刻画视觉 grounding，GRPO 就可以把“看对哪里”这件事变成可优化的训练信号。下面的几何图形计数实验，就是这个思想的最小版本。

## 23.3.1 数据集：几何图形计数

我们选择一个简单的视觉问答任务——几何图形计数。这个任务的好处是有客观标准答案，可以用规则奖励来评估，不需要训练额外的 RM。

每张图片包含若干个基本几何图形（三角形、圆形、正方形），问题形如"图中有几个圆形？"。模型的理想回答流程是：先描述看到了什么（"我看到图片中有 3 个三角形、2 个圆形和 1 个正方形"），然后推理出答案（"所以圆形的数量是 2"）。

```python
# ==========================================
# 几何图形计数
# ==========================================
from datasets import Dataset
import random

def generate_shape_image(num_triangles, num_circles, num_squares, seed=None):
    """生成包含指定数量几何图形的图片"""
    from PIL import Image, ImageDraw

    if seed is not None:
        random.seed(seed)

    img = Image.new('RGB', (256, 256), 'white')
    draw = ImageDraw.Draw(img)

    # 随机放置三角形
    for _ in range(num_triangles):
        x, y = random.randint(20, 236), random.randint(20, 236)
        size = random.randint(15, 35)
        draw.polygon([(x, y - size), (x - size, y + size), (x + size, y + size)],
                     fill='red', outline='darkred')

    # 随机放置圆形
    for _ in range(num_circles):
        x, y = random.randint(20, 236), random.randint(20, 236)
        r = random.randint(10, 25)
        draw.ellipse([(x - r, y - r), (x + r, y + r)],
                     fill='blue', outline='darkblue')

    # 随机放置正方形
    for _ in range(num_squares):
        x, y = random.randint(20, 236), random.randint(20, 236)
        s = random.randint(12, 28)
        draw.rectangle([(x - s, y - s), (x + s, y + s)],
                       fill='green', outline='darkgreen')

    return img


def generate_dataset(num_samples=500):
    """生成几何图形计数数据集"""
    data = []
    for i in range(num_samples):
        # 随机生成 1-5 个各种图形
        n_tri = random.randint(1, 5)
        n_cir = random.randint(1, 5)
        n_sqr = random.randint(1, 5)

        img = generate_shape_image(n_tri, n_cir, n_sqr, seed=i)

        # 随机选择一个问题
        questions = [
            f"图中有几个三角形？",
            f"图中有几个圆形？",
            f"图中有几个正方形？",
        ]
        answers = [str(n_tri), str(n_cir), str(n_sqr)]
        q_idx = random.randint(0, 2)

        data.append({
            'image': img,
            'question': questions[q_idx],
            'answer': answers[q_idx],
            'ground_truth': {
                'triangles': n_tri,
                'circles': n_cir,
                'squares': n_sqr,
            }
        })

    return Dataset.from_list(data)

# 生成训练集和验证集
train_dataset = generate_dataset(500)
val_dataset = generate_dataset(100)
```

## 23.3.2 奖励设计：三个评估维度

这个任务的奖励函数包含三个维度，每个维度都有明确的评分标准：

| 奖励维度              | 分值 | 评估标准                         | 类型     |
| --------------------- | ---- | -------------------------------- | -------- |
| 正确性（Correctness） | +1.0 | 最终答案与 ground truth 一致     | 规则奖励 |
| 推理质量（Reasoning） | +0.5 | 回答中包含对图片内容的描述       | 规则奖励 |
| 格式规范（Format）    | +0.2 | 回答遵循"描述 → 推理 → 答案"格式 | 规则奖励 |

这个奖励设计背后的思路是：正确答案最重要（+1.0），但我们不只想要"猜对答案"的模型——我们想要"看图 → 描述 → 推理 → 答案"的完整链路。所以推理质量（+0.5）和格式规范（+0.2）作为辅助奖励，引导模型形成正确的推理习惯。

```python
# ==========================================
# 三维评估
# ==========================================
import re

def compute_reward(response, ground_truth, target_shape):
    """
    计算三维奖励分数
    - response: 模型生成的回答
    - ground_truth: {'triangles': n, 'circles': n, 'squares': n}
    - target_shape: 本次问题的目标图形 ('triangles'/'circles'/'squares')
    """
    reward = 0.0

    # 1. 正确性奖励：提取最终答案，检查是否正确
    correct_answer = str(ground_truth[target_shape])
    # 尝试从回答末尾提取数字
    numbers = re.findall(r'\d+', response)
    if numbers and numbers[-1] == correct_answer:
        reward += 1.0

    # 2. 推理质量奖励：检查是否描述了图片内容
    shape_keywords = {
        'triangles': ['三角形', '红色', '三角'],
        'circles': ['圆形', '蓝色', '圆'],
        'squares': ['正方形', '绿色', '方块'],
    }
    has_description = any(kw in response for kw in shape_keywords[target_shape])
    if has_description:
        reward += 0.5

    # 3. 格式规范奖励：检查是否包含推理关键词
    reasoning_keywords = ['所以', '因此', '总共', '数量是', '答案是']
    has_reasoning = any(kw in response for kw in reasoning_keywords)
    if has_reasoning:
        reward += 0.2

    return reward
```

## 23.3.3 训练前后对比

训练前的模型可能只输出一个计数结果，没有给出可检查的视觉描述。训练后，我们希望模型先描述图形，再给出计数答案。下面的 GRPO 循环使用格式、描述和答案三部分奖励推动这种变化。

```python
# ==========================================
# VLM GRPO 训练循环
# ==========================================
def vlm_grpo_train(model, tokenizer, dataset, num_epochs=3, group_size=4, lr=1e-6):
    """
    用 GRPO 训练 VLM
    - group_size: 每个 prompt 生成多少个回答（组内比较）
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    normalizer = RewardNormalizer()

    for epoch in range(num_epochs):
        for batch in DataLoader(dataset, batch_size=8):
            all_log_probs = []
            all_rewards = []

            for prompt_img, prompt_text, ground_truth, target_shape in batch:
                # 对每个 prompt 生成 group_size 个回答
                group_responses = []
                group_log_probs = []
                group_rewards = []

                for _ in range(group_size):
                    # VLM 前向传播：输入图像 + 文本
                    response, log_prob = model.generate_with_log_prob(
                        image=prompt_img,
                        text=prompt_text,
                        max_new_tokens=128,
                        temperature=0.8
                    )

                    # 计算奖励
                    reward = compute_reward(response, ground_truth, target_shape)

                    group_responses.append(response)
                    group_log_probs.append(log_prob)
                    group_rewards.append(reward)

                all_log_probs.append(group_log_probs)
                all_rewards.append(group_rewards)

            # GRPO 核心：计算组内相对优势
            # 回顾第 15 章：组内相对优势 Advantage = (R_i - mean) / std
            rewards_tensor = torch.tensor(all_rewards)
            mean_r = rewards_tensor.mean(dim=-1, keepdim=True)
            std_r = rewards_tensor.std(dim=-1, keepdim=True) + 1e-8
            advantages = (rewards_tensor - mean_r) / std_r

            # 策略梯度损失
            log_probs_tensor = torch.stack([torch.stack(lp) for lp in all_log_probs])
            loss = -(log_probs_tensor * advantages.detach()).mean()

            # 加入 KL 惩罚（回顾第 8 章）
            kl_penalty = compute_kl_penalty(model, ref_model, batch)
            loss = loss + 0.05 * kl_penalty

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
```

下面是用于说明评估方式的示例。它表示预期的行为变化，不是仓库中已保存的实测输出。训练前，模型对“图中有几个圆形？”可能只回答：

> **训练前**：“3。”（只给出答案，没有可检查的视觉描述）

训练后，模型的回答变成：

> **训练后**："我看到图片中有 2 个红色三角形、**3 个蓝色圆形**和 1 个绿色正方形。问题是关于圆形的数量。所以圆形的数量是 3。"

若训练后的回答稳定出现这种结构，说明格式奖励和描述奖励产生了预期影响。最终答案是否正确仍要在独立验证集上统计，不能只凭一条示例判断。

## 23.3.4 训练指标分析

训练 VLM 时，除了第 13 章提到的标准指标（奖励、KL 散度、响应长度）之外，还有几个多模态特有的指标值得关注：

**注意力热力图变化。** 注意力图可以显示不同文本位置与图像区域之间的权重分布。它可以用于发现模型是否完全忽略图像，但不能单独证明模型完成了正确计数；最终判断仍以独立样本的答案准确率为准。

**推理长度与准确率的关系。** 统计回答中推理部分的长度和最终答案准确率的关系。理想情况是一个倒 U 形曲线——适度的推理长度效果最好。太短意味着模型没有仔细看图（猜答案），太长可能意味着模型在"过度思考"甚至产生视觉幻觉。

**跨泛化测试。** 在训练集之外的新图形组合上测试模型的表现。如果模型真的学会了"看图计数"的能力，它应该在从未见过的图形组合上也能正确回答——比如训练时最多 5 个图形，测试时给 7 个。

<details>
<summary>思考题：为什么 VLM GRPO 的学习率（1e-6）比纯文本 GRPO（通常 5e-7 到 1e-5）的范围更窄？</summary>

VLM 包含两个组件——视觉编码器（ViT）和文本解码器（Transformer）。如果学习率太大，RL 的梯度可能会破坏视觉编码器已经学到的特征（图像理解能力），导致模型"失明"——虽然还在输出文字，但已经"看不懂"图片了。如果学习率太小，文本解码器的策略更新太慢，训练效率极低。

实践中，一个常见的做法是对视觉编码器和文本解码器使用不同的学习率——视觉编码器用更小的学习率（比如文本解码器的 1/10），甚至完全冻结视觉编码器。这样可以在保持视觉理解能力的同时，让文本生成部分通过 RL 充分优化。下一节会详细讨论这个策略选择。

</details>

这个实验保留了文本 GRPO 的组内相对优势计算，同时把图像加入输入，并把视觉描述加入奖励。下一节 [GeoQA 几何推理](./easyr1-geoqa) 会把同一思路放到真实几何数据和训练框架中。

## 本节小结

- 多模态 GRPO 仍按组采样回答并计算相对优势，输入中额外包含视觉编码。
- 合成计数任务提供精确答案，能够分别检查回答格式、推理描述和最终计数。
- 训练后必须在未见图形组合上评估，并监控视觉编码器是否因过大的更新而退化。

## 参考资料

- [VLM-R1 GitHub](https://github.com/om-ai-lab/VLM-R1) —— 提供了 VLM-R1 的训练曲线、grounding reward 示例与开源实现，可作为本节实验的真实项目参照。
