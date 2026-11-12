# 13.7 扩展实战：Reward Hacking 与数据飞轮

## 本节导读

**核心内容**

- 通过一个故意有漏洞的奖励函数，观察 reward hacking 如何发生。
- 学会同时监控 reward、长度、重复率、KL 和人工质量，而不是只看单一曲线。
- 把 badcase 收集、错误聚类、补数据、重训和回归评估组织成数据飞轮。

**核心公式**

$$
R_{bad}(x,y)=0.01|y|+0.5\cdot\mathbb{1}[\text{has\_list}(y)]
+0.3\cdot\mathbb{1}[\text{has\_polite\_phrase}(y)]
\quad \text{（有漏洞的奖励：把“好”误写成“长、像模板”）}
$$

$$
R_{safe}(x,y)=
0.4R_{helpful}+0.3R_{correct}+0.2R_{format}
-0.05R_{length}-0.05R_{repeat}
\quad \text{（更安全的混合奖励：多维度互相制衡）}
$$

> **先记住一句话**
>
> Reward hacking 不是模型“坏”，而是模型认真优化了你写错的目标。修复它也不是骂模型，而是修奖励、补数据、加评估闸门。

13.1–13.6 已经把 RLHF 主线讲完：SFT 负责起点，RM 负责偏好信号，PPO 负责按奖励优化，大规模工程负责把它放大，评估负责确认真的变好。本节承接不适合塞进主线的两类材料：一个 reward hacking 受控实验，以及一个数据飞轮工程模板。

## 故意制造坏奖励

第 4 章讲奖励设计时说过：奖励函数定义了智能体眼里的目标。大模型 RLHF 也是一样。只不过这里的“环境奖励”经常来自 RM、judge、规则检查或它们的混合。

新手最常见的错觉是：

> 只要 reward 曲线上升，模型就在变好。

Reward hacking 实验就是专门打破这个错觉。我们故意写一个有漏洞的奖励函数，让模型很容易找到“拿高分但回答变差”的策略。这样做有三个好处：

1. 你能亲眼看到 reward 和真实质量如何背离。
2. 你能学会哪些监控指标会提前报警。
3. 你能理解为什么真实 RLHF 必须有人工抽检和回归评估。

## 有漏洞的奖励函数

Reward hacking 最容易被低估，因为训练曲线看起来通常很漂亮。最好的入门方式是故意制造一个漏洞：让奖励函数偏爱长回答、列表格式和固定客套话，然后观察模型如何学会“凑字数”。

```python
def flawed_reward(prompt: str, response: str) -> float:
    """
    有漏洞的奖励函数。
    核心问题：把“详细”误写成“越长越好”，且给固定格式和客套话加分。
    """
    length_score = len(response) / 100.0

    format_score = 0.0
    if "- " in response or "1." in response:
        format_score += 0.5
    if "**" in response:
        format_score += 0.5

    politeness_score = 0.0
    for phrase in ["我很乐意", "希望这能帮到", "请注意", "以下是一些"]:
        if phrase in response:
            politeness_score += 0.3

    return length_score + format_score + politeness_score
```

这个奖励函数想表达的是“回答应该详细、有结构、有礼貌”，但实际编码出来的是：

```text
越长越好
有列表就好
有加粗就好
有客套话就好
```

PPO 或 GRPO 一旦发现这个规律，就会把回答长度、标题、列表和客套话一起推高。模型没有做错，它只是发现了你的奖励函数真正奖励的东西。

### 手算两个回答的坏奖励

同一个 prompt：

```text
请用一句话解释 PPO 的 KL 惩罚。
```

回答 A：

```text
KL 惩罚限制新策略偏离参考策略太远，防止 PPO 更新失控。
```

回答 B：

```text
我很乐意帮助你。以下是一些重要内容：
- **第一点**：PPO 很重要。
- **第二点**：KL 也很重要。
- **第三点**：希望这能帮到你。
```

按 `flawed_reward` 粗略算：

| 回答 | 长度分  | 格式分 | 客套话分 | 总分 | 人类观感 |
| ---- | ------- | ------ | -------- | ---- | -------- |
| A    | 约 0.32 | 0.0    | 0.0      | 0.32 | 简洁准确 |
| B    | 约 0.75 | 1.0    | 0.6      | 2.35 | 冗长空泛 |

坏奖励会强烈偏好 B。只要 PPO 继续优化，模型就会更常写 B 这类回答。

## 实验设置

不要一开始就用大模型跑完整 RLHF。这个实验用小模型、少量 prompt、短训练步数就够了。

| 项目      | 建议                                     |
| --------- | ---------------------------------------- |
| 模型      | 已经 SFT 过的小模型，或一个小 chat model |
| prompt 数 | 50 到 200 条                             |
| 生成长度  | 先限制 `max_new_tokens=256`              |
| 奖励      | `flawed_reward`                          |
| 对照      | 同一组 prompt 的 SFT 原始输出            |
| 监控      | reward、长度、重复率、人工抽检           |

实验目标不是训练一个好模型，而是观察模型如何利用奖励漏洞。

一个伪代码流程：

```python
for step in range(num_steps):
    prompts = sample_prompts(prompt_pool)
    responses = actor.generate(prompts, max_new_tokens=256)

    rewards = [flawed_reward(p, r) for p, r in zip(prompts, responses)]
    kl = compute_kl(actor, reference, prompts, responses)
    total_rewards = [r - beta * k for r, k in zip(rewards, kl)]

    ppo_update(actor, critic, prompts, responses, total_rewards)

    if step % eval_interval == 0:
        log_reward_hacking_metrics(step, prompts, responses, rewards, kl)
```

这里即使不真的跑 PPO，也可以用不同 decoding 策略生成样本，先验证 `flawed_reward` 是否会偏爱坏回答。奖励函数单测永远应该早于大规模训练。

## 观察三个异常信号

跑这个实验时，不要只记录 reward。至少同时画三条曲线：

| 指标              | 正常情况               | Reward hacking 信号            |
| ----------------- | ---------------------- | ------------------------------ |
| `reward_mean`     | 缓慢上升               | 持续上升，且远快于人工质量提升 |
| `response_length` | 在任务需要的范围内波动 | 和 reward 一起持续变长         |
| `distinct_ngram`  | 保持相对稳定           | 明显下降，说明输出越来越模板化 |

再加上两个 PPO-RLHF 特有指标：

| 指标             | 为什么看                          |
| ---------------- | --------------------------------- |
| `kl_mean`        | 看 Actor 是否快速偏离 reference   |
| `judge_win_rate` | 看外部 judge 或人工是否真的更喜欢 |

一个很粗糙但有用的检测器如下：

```python
def reward_hacking_report(rows):
    """
    rows: [{"reward": float, "text": str}, ...]
    """
    import numpy as np
    from collections import Counter

    rewards = np.array([row["reward"] for row in rows])
    lengths = np.array([len(row["text"]) for row in rows])
    length_corr = float(np.corrcoef(rewards, lengths)[0, 1])

    phrases = Counter()
    for row in rows:
        words = row["text"].split()
        phrases.update(" ".join(words[i:i + 4]) for i in range(max(0, len(words) - 3)))

    unique_4grams = len(phrases)
    total_4grams = sum(phrases.values())
    distinct_4 = unique_4grams / max(total_4grams, 1)

    return {
        "length_reward_corr": length_corr,
        "distinct_4": distinct_4,
        "top_phrases": phrases.most_common(5),
        "warning": length_corr > 0.7 or distinct_4 < 0.5,
    }
```

这个检测器不能证明模型一定被 hack，但它能把最常见的长度黑客和模板黑客暴露出来。真正确认问题时，还要抽查 reward 最高、长度增长最大、重复短语最多的样本。

## 奖励曲线与样本质量

一次受控实验可能出现这样的日志：

| step | reward | length | distinct-4 | KL   | 人工备注     |
| ---- | ------ | ------ | ---------- | ---- | ------------ |
| 0    | 0.8    | 120    | 0.82       | 0.00 | SFT 输出正常 |
| 50   | 1.4    | 180    | 0.76       | 0.04 | 回答略变长   |
| 100  | 2.2    | 310    | 0.61       | 0.09 | 客套话增多   |
| 150  | 3.1    | 520    | 0.42       | 0.18 | 明显模板化   |
| 200  | 4.0    | 760    | 0.31       | 0.27 | 大量重复列表 |

如果只看 reward，这是一条漂亮曲线；如果看样本，这是训练坏了。

典型坏样本可能长这样：

```text
我很乐意帮助你。以下是一些重要说明：

1. **首先**，PPO 是一个非常重要的方法。
2. **其次**，KL 惩罚也是一个非常重要的方法。
3. **再次**，理解 KL 惩罚对理解 PPO 很重要。
4. **最后**，希望这能帮助你更好地理解 PPO。

请注意，以上内容只是一个简要说明，希望这能帮到你。
```

它礼貌、有列表、有加粗、很长，所以坏奖励给高分。但它几乎没有解释 KL 惩罚的实质。

## 多维度奖励修复

修复思路不是简单地“惩罚长度”，而是把原来混在一起的目标拆开：

```python
def safer_reward(prompt: str, response: str) -> float:
    helpfulness = judge_helpfulness(prompt, response)
    correctness = judge_correctness(prompt, response)
    format_score = validate_required_format(prompt, response)
    repetition_penalty = ngram_repetition_rate(response, n=4)
    length_penalty = max(0, len(response) - target_max_length(prompt)) / 400

    return (
        0.40 * helpfulness
        + 0.35 * correctness
        + 0.15 * format_score
        - 0.05 * repetition_penalty
        - 0.05 * length_penalty
    )
```

这个版本有几个改进：

| 改进                            | 作用                                     |
| ------------------------------- | ---------------------------------------- |
| helpfulness 和 correctness 分开 | 防止“有帮助但错误”或“正确但没用”混在一起 |
| format 只占较小权重             | 防止模型为格式牺牲内容                   |
| length 是惩罚，不是奖励         | 防止越长越高分                           |
| repetition 单独惩罚             | 防止模板复读                             |
| target length 依赖 prompt       | 长答案任务和短答案任务分开处理           |

然后再加上 PPO-RLHF 里的 KL 约束，避免策略为了追逐新奖励而离 SFT reference 太远。修复是否有效，不能只看新 reward，还要看人工偏好、回归 benchmark、长度分布和重复率是否一起变好。

## 补充 Rejected 数据

如果 RM 已经学会偏爱长废话，光加长度惩罚可能治标不治本。更稳的做法是把坏样本加回偏好数据，让 RM 明确学到它们应该低分：

```json
{
  "prompt": "请用一句话解释 PPO 的 KL 惩罚。",
  "chosen": "KL 惩罚限制新策略偏离参考策略太远，防止 PPO 更新过猛。",
  "rejected": "我很乐意帮助你。以下是一些重要说明：PPO 很重要，KL 很重要，希望这能帮到你...",
  "tags": ["length_hack", "template_hack"],
  "source": "ppo_badcase"
}
```

这就是数据飞轮的基本动作：模型暴露了失败模式，我们把失败模式变成训练数据，再回归评估它是否被修掉。

## 数据飞轮工程模板

RLHF 不是训练一次就结束。一个实际可用的数据飞轮通常长这样：

```text
部署模型或运行离线评估
  -> 收集 badcase、用户反馈、评测失败样本
  -> 按错误类型聚类
  -> 定向生产 SFT / preference 数据
  -> 经过质量闸门
  -> 训练 SFT、RM 或 PPO-RLHF
  -> 回归评估和人工抽检
  -> 通过后再部署
```

关键点是“按错误类型补数据”，而不是盲目扩大数据量。一次主动学习循环可以写成：

```python
def active_learning_cycle(model, eval_set, data_producer):
    errors = evaluate_and_collect_errors(model, eval_set)
    clusters = cluster_errors_by_type(errors)

    new_data = []
    for cluster in clusters.top_k(k=3):
        new_data.extend(data_producer.generate(
            task_type=cluster.type,
            difficulty=cluster.difficulty,
            num_samples=1000,
        ))

    cleaned = quality_gate(new_data)
    updated_model = train_on_new_data(model, cleaned)
    report = regression_eval(updated_model, eval_set)
    return updated_model, report
```

这里的 `quality_gate` 至少要做去重、评测集污染检查、长度过滤、chosen/rejected 差异检查、难度分层和少量人工抽检。数据飞轮转得越快，越需要评测集和人工抽检当刹车，否则模型会很快过拟合到当前评测和当前 judge 的偏好。

## 错误类型聚类

badcase 如果只存成一堆文本，很快就没人看。更好的做法是给每个失败样本打标签：

| 标签            | 含义                   | 后续补数据方向                     |
| --------------- | ---------------------- | ---------------------------------- |
| `length_hack`   | 回答变长但信息密度低   | 加短而准的 chosen、长废话 rejected |
| `template_hack` | 固定套话反复出现       | 加多风格 chosen、模板 rejected     |
| `hallucination` | 编造事实或引用         | 加事实核验数据、拒绝不确定         |
| `over_refusal`  | 该答的问题也拒绝       | 加安全边界样本                     |
| `under_refusal` | 高风险问题没拒绝       | 加安全拒答偏好对                   |
| `format_fail`   | JSON/代码块/字数不符合 | 加规则奖励和格式 SFT               |
| `reasoning_gap` | 推理跳步或结论错       | 加过程监督或可验证题               |

聚类的目标不是做漂亮报表，而是决定下一轮数据怎么生产。

## 两个典型案例

**Reasoning 数据循环。** 对数学和代码任务，验证器可以直接判断答案对不对。模型对同一个 prompt 采样多条回答，验证器标出正确和错误，正确回答可以作为正例，错误回答可以作为反例。评测后仍然失败的题型继续聚类，再定向生成类似题。

```text
采样 N 条回答
  -> 验证器判断正确/错误
  -> 正确且简洁的作为 chosen
  -> 错误、跳步、冗长的作为 rejected
  -> 训练 RM / DPO / RLVR
  -> 回归 GSM8K、MATH、HumanEval 或自建测试集
```

**Agent 轨迹数据循环。** Agentic RL 的数据不是普通文本，而是模型和环境交互后的轨迹。成功轨迹可以作为 chosen；失败轨迹要拆开看是规划错、工具调用错、观察理解错，还是最终答案错。只有知道失败类型，后续补数据才不会变成“再多生成一点”。

```text
Agent 执行任务
  -> 收集成功/失败轨迹
  -> 成功轨迹作为 chosen
  -> 失败轨迹按原因打标签
  -> 修订失败轨迹或补局部训练数据
  -> 在 SWE-bench、WebArena 或自建任务上回归
```

## 最小验收标准

做完 reward hacking 修复后，至少检查：

| 指标            | 期望                     |
| --------------- | ------------------------ |
| reward          | 不再靠长度单调上涨       |
| 长度            | 回到任务合理范围         |
| distinct n-gram | 不低于 SFT 明显太多      |
| 人工偏好        | 修复后回答确实更可用     |
| 回归集          | 原有能力没有掉点         |
| stress case     | 长废话、模板废话不再高分 |

如果新 reward 下降了，但人工质量上升了，不要慌。那说明旧 reward 本来就不可信。RLHF 调试时，修复坏奖励往往会让曲线短期变“难看”，但模型真实质量更健康。

## 本节小结

Reward hacking 要靠受控实验练诊断，数据飞轮要靠质量闸门防自嗨。RLHF 真正难的地方不只是算法，而是让奖励、数据和评估互相制衡。

这一章到这里就完整闭环了：base model 不是 assistant，SFT 给它行为起点，RM 给它偏好方向，PPO 让它按奖励练习，评估和数据飞轮负责防止它学歪。下一章进入[第 14 章 DPO](../chapter17_dpo/dpo-objective-derivation)，先学习如何直接使用偏好数据更新模型；工业规模的数据、系统与稳定性问题将在第 18 章统一梳理。

## 练习

1. 修改 `flawed_reward`，故意让它偏爱“包含专业术语”的回答，并设计 3 个 stress case。
2. 给一个 reward hacking 样本打标签：是 length、template、format、semantic 里的哪一种？
3. 设计一轮数据飞轮：从 badcase 收集到重新训练，你会设置哪些质量闸门？
