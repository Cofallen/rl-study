# 16.5 Adaptive Reasoning

Section 16.4 introduced the concept of a mode switch and a reasoning budget for the model, but the switch still needs to be set by the user or the router. In real-world services, the difficulty of a request is not always evident from the surface: a short mathematical problem may require a long proof, while a long meeting transcript may only need to extract dates.

Adaptive reasoning allows the model to decide, based on the task and current progress, how much computation to invest. The system then uses a hard upper bound to control the worst-case latency. This section first introduces the concept of a continuous budget, then uses a research-oriented task to illustrate how difficulty can change during execution. We then discuss cost and safety constraints, and finally explain why this mechanism connects to process rewards and the Agent.

<img src="../../chapter19_reasoning/images/adaptive-thinking-budget.svg" alt="The model uses current evidence to decide whether to continue computing or stop and answer, subject to a hard resource limit">

## 1. How Does the Model Judge Task Difficulty?

The model is not required to judge a fixed "difficulty label," but rather whether the next segment of computation is still useful. If the current evidence is already sufficient, further generation will only increase length. If a tool just returned conflicting results, the system may need to perform additional search and comparison. A continuous budget is used to describe this gradual decision.

Hybrid Thinking chooses between "direct answer" and "entering reasoning." Adaptive reasoning goes a step further by deciding how much computation to invest after entering reasoning. To describe this continuous variation, we can introduce a concept variable $\tau \in [0,1]$. This variable is used for explanation in this section and does not represent a publicly available parameter in all products:

- $\tau = 0$: Direct answer
- $\tau = 0.5$: Medium reasoning (hundreds of tokens of CoT)
- $\tau = 1.0$: Use the highest reasoning budget currently allowed by the system

In practice, the system does not necessarily need to explicitly output $\tau$. The router can choose the budget, or the model can implicitly decide the amount of computation by the timing of stopping generation. $\tau$ simply places these different implementations on the same "from low budget to high budget" axis.

### 1.1 Difference Between Fixed Thinking Budget and Adaptive Thinking

A fixed thinking budget is defined by the caller through a token upper limit; adaptive thinking allows the model to decide whether to enter extended thinking and for how long based on the request. The caller can still influence the model's tendency through effort levels, and use `max_tokens`, task budget, or timeout to control the worst-case cost.

Each approach has its own advantages:

- **Thinking Budget**: The boundaries are clear, making it easier to estimate the cost of a single request, but each request must use the preset budget;
- **Adaptive Thinking**: It can skip simple tasks or extend complex tasks, making the cost distribution harder to predict.

In actual deployment, it is common to allow the model to stop adaptively within the permitted range, and then set hard boundaries using total tokens, time, and tool usage counts.

### 1.2 Understanding the Control Layer from Claude's Public Interface

Anthropic has not publicly released the full training details. Based on the [official blog](https://www.anthropic.com/news/claude-opus-4-6) and the [Extended Thinking documentation](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking), we can observe how the interface controls the budget; whether an independent difficulty estimator is used internally cannot be directly determined from the interface.

#### Teaching Derivation: Allocating Computation Based on Prompt Difficulty

Anthropic has not published the complete objective used for adaptive thinking. To understand how quality and cost can be balanced, we can write a generic cost-sensitive objective. The formula below is a teaching derivation, not a published Claude training recipe:

$$
r_{\text{total}} = r_{\text{task}} - \lambda C,
$$

where $r_{\text{task}}$ measures whether the task is completed, $C$ represents the cost of reasoning tokens, tool calls, or runtime, and $\lambda$ determines how much the system is willing to trade off cost for quality. This formula does not specify a particular training algorithm, but it illustrates how two types of pressure enter the objective simultaneously:

- Thinking too much on simple tasks → Wasting computational power → Punishment
- Thinking too little on difficult tasks → Answering incorrectly → Punishment

As the computation continues to increase on simple tasks, the task reward no longer rises significantly, and the cost term will encourage the model to stop early. In complex tasks, as long as additional computation can significantly improve the success rate, the benefit may outweigh the cost.

#### Adaptive Thinking API

Claude 4.6 and later versions support the model using `thinking: {type: "adaptive"}` and use `effort` to regulate the model's tendency to use extended thinking. Below is the interface structure:

```python
# Enable adaptive thinking
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=16000,
    messages=[{"role": "user", "content": "..."}],
    thinking={"type": "adaptive"},
    output_config={"effort": "high"},
)
```

Earlier versions of Extended Thinking used `budget_tokens` to manually set the target budget; this approach has been deprecated in version 4.6, and versions 4.7 and later use adaptive thinking and effort instead. The evolution of the interface precisely illustrates the distinction between two layers of control: the model decides how much thinking is actually used for the current request, while the developer determines the overall level of effort and resource limits.

#### Thinking Signatures

The thinking block in the Claude API includes a `signature` field, which is used to verify and return the previous thinking content unchanged across multiple rounds or tool calls. Some hidden thinking is returned as encrypted content, allowing the API to reuse it in the next round without requiring the client to read the original text. The primary purpose of the signature is to maintain the integrity of the context, and it should not be interpreted as "the visible structure of the user's reasoning" or as an automatic security filter.

## 2. How Adaptive Thinking is Used in Research Tasks

A typical math question usually ends after generating the answer. However, research tasks are more complex, as their difficulty changes with environmental feedback: a code's first run fails, experimental results contradict the hypothesis, or conflicting sources are retrieved, all of which force the model to reallocate computation. Anthropic's model evaluations include scaled-down AI research tasks, which can be used to understand why budgets change during execution. Evaluation scores depend on tools, runtime, and scoring criteria, and cannot be directly converted into the efficiency of general research work.

### 2.1 LLM Training: Experimental Results Determine the Next Step

The model first selects data and training configurations, then writes code, runs small-scale training, and reads the loss and evaluation results. If the loss diverges immediately, the next computation should be used to check the data, learning rate, and numerical settings; if the training proceeds normally but the two approaches are very similar, the system may need to increase the sample size or design an experiment that better distinguishes between hypotheses. The budget is adjusted iteratively based on the observed results, and cannot be determined solely by the length of the initial prompt.

### 2.2 Text-RL: Reward Increase Does Not Equal Task Completion

In text environments, the model must define states, actions, and rewards, implement algorithms, and compare strategies. After the training curve increases, it is essential to check whether the agent has truly completed the task or merely exploited loopholes in the reward rules. When the latter occurs, continuing training with the same reward function becomes meaningless; the computation should instead focus on failure samples, reward modifications, and re-evaluation.

### 2.3 Quadruped-RL: Simulation Feedback Changes Search Direction

Controlling a quadruped requires setting gait goals, action penalties, and stability constraints, followed by running simulations to observe the policy. If the robot moves forward but frequently falls, it indicates that the speed reward has overshadowed stability. If the actions are smooth but the robot remains stationary, it suggests that the constraints are too strong. Each simulation provides new states for the next iteration, so the budget must cover code modifications, training, and behavior checks—not just generating longer text.

All three tasks share a common process: first, create an executable plan, read the environment feedback, and then decide whether to continue optimizing, modify the goals, or stop. Adaptive thinking here controls the entire trajectory of computation and tool budget.

### 2.4 How the Constitution Constrains Observable Behavior

Anthropic publicly released the new [Claude Constitution](https://www.anthropic.com/news/claude-new-constitution) in 2026, outlining the values and behaviors they hope the model will follow. While it does not directly expose the model's internal reasoning step-by-step, it can be transformed into training samples, evaluation criteria, and external testing.

#### The Constitution as a Reasoning Constraint

The Constitution covers honesty, helpfulness, safety, and the interests of various stakeholders. Training systems can use it to evaluate the final answer and observable actions. If the system retains reasoning summaries or tool logs, these can also be checked for rule conflicts. However, the hidden internal states cannot be directly validated solely based on the documentation.

#### How the Constitution Enters Training

The Constitution can be converted into judgment criteria and preference samples during the training phase, without needing to include the full text in each inference prompt:

1. Decompose the Constitution into executable judgment criteria.
2. Use these criteria to generate a large number of "Constitution-aligned" preference data.
3. Use this data for RLHF / DPO training.

These samples increase the probability of responses that align with the criteria, but they cannot guarantee that the model will follow the criteria in all new scenarios. Therefore, independent safety evaluations are still required.

#### The Constitution and Interpretability

Making the principles public allows external evaluators to design more specific tests, such as checking how the model chooses when a user's request conflicts with safety rules. It increases the transparency of the behavior goals, but the principles, training implementation, and actual behavior remain at three different levels.

## 3. How to Control Safety, Latency, and Cost

Allowing the model to extend reasoning on its own can improve the completion rate of complex tasks, but it also makes the length of a single request and the number of tool calls harder to predict. The system therefore needs to simultaneously monitor whether "additional computation improves task outcomes" and "how much resource is consumed in the worst case."

Adaptive reasoning delegates the length of a request and the scope of execution to the model to dynamically decide. Therefore, the following three constraints must be considered in evaluation.

### 3.1 Distinguishing Effective Reasoning from Invalid Length

The length of reasoning itself does not equate to the quality of reasoning. The model may repeatedly check conclusions that have already been determined or generate text unrelated to the answer; this increases cost without improving success rate.

During training, correctness and cost can be jointly included in the objective. During deployment, the system can compare the success rate of tasks before and after increasing the budget. Relying solely on language fluency is insufficient to determine whether a segment of reasoning is useful; it requires outcome verifiers, ablation studies, or process evaluations to check whether additional steps change the conclusion.

### 3.2 Controlling Latency and Cost Fluctuation

The same prompt may use different lengths across different sampling runs, so latency and cost form a distribution. The service needs to record the median and tail latency, and also separately observe whether failed requests are more likely to exhaust the budget.

During deployment, the system typically sets both a model autonomy stop condition and a hard budget upper limit. The former adjusts the computation based on the task, while the latter controls the worst-case latency and cost. A too small upper limit may truncate the reasoning for difficult problems, while a too large upper limit may widen the cost fluctuation.

### 3.3 Handling the Attack Surface in Reasoning Chains

Tool results and external documents continuously enter long trajectories, which may contain instructions unrelated to the user's task. If the model treats these low-priority contents as system requirements, subsequent searches, file accesses, and answers may deviate from the original task.

This is the problem addressed by [OpenAI's Instruction Hierarchy](https://openai.com/index/the-instruction-hierarchy/) (2025)—clarifying the priority of the system prompt, user prompt, and tool return results to prevent low-priority content from hijacking high-priority behaviors.

### 3.4 Comparison of Fixed, Hybrid, and Adaptive Thinking

The differences among the three control approaches are as follows:

| Dimension      | Fixed Budget                                            | Dual-Mode                                         | Adaptive Thinking                                                          |
| -------------- | ------------------------------------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------- |
| Control Method | Uses a preset budget for each request                   | Switches between direct answer and thinking mode  | The model decides whether to continue within effort and a hard upper limit |
| Advantages     | Delay and cost are easy to estimate                     | Simple tasks can skip lengthy thinking            | Can adjust computation based on task progress                              |
| Risks          | Wastes on simple tasks, insufficient for difficult ones | Routing errors may select the wrong mode          | Cost fluctuations, harder to evaluate stopping conditions                  |
| Use Cases      | Batch tasks with stable difficulty                      | General conversation and reasoning mixed services | Research, code, and long tool tasks                                        |

From fixed depth to Hybrid Thinking, and then to adaptive thinking, the control granularity gradually increases. The finer the granularity, the more reliable stopping conditions, cost monitoring, and regression testing are required.

## 4. How to Integrate Process Alignment with Agents

When a model only outputs a single answer, the system primarily evaluates the final output. However, when a model autonomously thinks and invokes tools over a longer period, the intermediate steps also influence the cost and safety. Adaptive budgets, therefore, cannot rely solely on token counts; they also require process feedback to determine which steps are worth continuing.

### 4� Process Alignment

Process rewards can assess whether intermediate steps are correct and whether they violate tool or safety constraints. They can detect issues earlier than end rewards, but they also require the evaluator to understand the current context. Chapter 17 will introduce three types of feedback separately:

- PRM scores reasoning steps;
- Monitoring models check for abnormal behavior in long trajectories;
- Rules or constitutions write allowed actions as training and evaluation standards.

### 4.2 A Safe Sandbox for Reasoning Models

A safe sandbox primarily isolates tools, files, and network permissions. Even if a model generates an incorrect plan, as long as the execution interface performs permission checks, parameter validation, and result auditing, the erroneous text will not automatically become a real action. Whether the reasoning content is shown to the user is the issue addressed in [16.6](./cot-visibility-alignment).

### 4.3 Alignment in Reasoning Scaling

After the budget increases, the monitoring cost also increases with the trajectory length and the number of tool calls. Evaluation cannot be limited to fixed short answers; it must also cover long contexts, budget exhaustion, tool failures, and mid-trajectory recovery.

### 4.4 Integration of Reasoning and Agentic Behavior

Research tasks connect reasoning with execution: the model writes code, runs experiments, reads results, and then decides whether to continue. At this point, the thinking budget must be unified with the environment budget, for example, the maximum number of tool calls, the longest running time, and the number of allowed retries. This is precisely what [Agentic RL](../chapter22_agentic/overview) addresses in long-trajectory decision-making.

## Summary

Adaptive thinking extends the binary pattern selection to a continuous computational allocation: the model determines the depth of reasoning based on the task, and the system uses a budget cap to control latency and cost. Research tasks demonstrate that this mechanism can cover longer analysis and execution processes, while also making process monitoring, prompt injection protection, and cost prediction more critical.

[16.6 Display and Alignment of Reasoning Chains](./cot-visibility-alignment) then addresses a direct question: after the model generates internal reasoning, what content should be displayed, what should be monitored or hidden.
