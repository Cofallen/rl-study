# 16.4 Hybrid Thinking and Budget Control

Section 16.3 explains that the same model can achieve higher success rates by using more candidates, revisions, or search. Now, consider two requests placed in the same service queue: the first is "Translate 'hello' into Chinese," and the second is "Prove that a sequence converges." If both are processed in the long-think mode, the translation request will unnecessarily increase latency. If both are answered directly, the proof question is prone to errors in intermediate steps.

**Hybrid Thinking** enables a single model to support both direct answering and deep thinking behaviors. The problem is then divided into three layers: how to coexist two modes during training, how to choose during inference whether to allocate the budget to a long chain or multiple short answers, and how to compress redundant steps once the model has learned long thinking. Finally, the deployment system must ensure that mode switching, answer consistency, and graceful shutdown when the budget is exhausted are all under control.

<img src="../../chapter19_reasoning/images/hybrid-thinking-routing.svg" alt="A request is routed between direct answering and extended reasoning under a budget constraint">

## 1. How a Single Model Can Support Two Modes

The most straightforward approach is to maintain two separate models: one for fast answering and one for complex reasoning. However, this approach leads to two sets of weights, two sets of deployment, and potential synchronization issues between the two models. Hybrid Thinking instead shares a single set of parameters and uses training samples or control tokens to specify the current mode.

### 1.1 How DeepSeek V3.1 Provides Two Modes

[DeepSeek V3.1](https://api-docs.deepseek.com/news/news250821) (2025.08) publicly demonstrates this interface: a single model supports both "Think" and "Non-Think" modes, with the API exposed through `deepseek-reasoner` and `deepseek-chat`. The public documentation confirms that the model supports two modes, with improvements in thinking efficiency and tool capabilities. However, it does not disclose the specific training data or loss details for each mode.

- **Single Model**: Does not maintain two independent "think model" and "non-think model"
- **Mode Switching**: Requests are routed to either the think or non-think interface
- **Shared Capabilities**: Both modes are derived from the same model version, and knowledge and tool capabilities can be synchronized

To train such a model, the data must cover both long reasoning and direct answering. Otherwise, one mode may dominate and suppress the other. Whether the mode is selected by the user, a template, or the model itself depends on the product interface. It cannot be inferred solely from the "dual-mode" result what the specific training process was.

### 1.2 How Qwen3 Trains Two Output Types

The [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (2025.05) proposes a more systematic Hybrid Thinking approach. All Qwen3 model series (from 0.6B to 235B) support two modes:

Qwen3 learns two types of outputs using the same set of model parameters. The chat template or user setting tells the model whether to generate a long reasoning process or a direct answer:

- Training data contains a mixture of thinking and non-thinking samples.
- Thinking samples: complete long Chain-of-Thought (CoT) plus the answer.
- Non-thinking samples: short direct answers.
- The template controls whether thinking is enabled.

Sharing parameters avoids maintaining two completely independent models, but the choice of mode still requires request settings or upper-level routing. The model cannot decide to make the most cost-effective decision simply because "it looks like a math problem." Section 16.5 will continue to address this issue.

### 1.3 How the Thinking Budget Limits the Length of Reasoning

The Qwen3 technical report further discusses the **thinking budget**: the deployment side allocates a token budget for the reasoning process, allowing the same model to form different quality-delay trade-off points. The parameter names and implementations may vary across different inference services, but the following pseudo-code illustrates the interface meaning:

```python
# Qwen3 thinking budget example
response = client.chat.completions.create(
    model="qwen3-235b-a22b",
    messages=[{"role": "user", "content": "Prove that √2 is irrational"}],
    extra_body={"thinking_budget": 2000}  # Service-side illustrative parameter
)
```

This upper limit directly affects three deployment metrics:

- A small budget shortens the waiting time for high-frequency requests.
- The token limit restricts the worst-case cost of a single call.
- A large budget leaves more room for checking and revising difficult problems.

Budget control also introduces truncation issues: if the system stops generating in the middle of a proof, the remaining tokens may not be sufficient to write the final answer. Therefore, training and deployment must separately address length efficiency, stopping conditions, and answer preservation. One cannot equate an API limit with the model having learned to correctly conclude at any truncation point.

## 2. Choosing Between Long Reasoning and Parallel Short Answers

Once entering "tasks requiring more computation," the next decision is how to spend the computation. For knowledge-based question-answering, multiple short answers voting may already be sufficient; for problems requiring continuous proofs, a long chain that retains intermediate states may be more valuable. The NoThinking experiment compares these two approaches under the same budget.

In April 2025, Ma et al. compared long reasoning with NoThinking in their paper [Reasoning Models Can Be Effective Without Thinking](https://arxiv.org/abs/2504.09858). The paper used the DeepSeek-R1-Distill-Qwen series of models and found that, under low token budgets or low latency conditions, multiple shorter answers can be a competitive approach.

### 2.1 How to Compare the Two Approaches Under the Same Budget

Experiments control the total token usage or wall-clock delay, comparing two approaches: Thinking uses a longer trajectory to complete one or a few attempts; NoThinking skips explicit long reasoning and independently generates more complete answers, then uses a task validator or confidence score to select the best answer. The paper covers seven types of tasks, including mathematics, formal proofs, and code. The conclusions focus on the low-budget regime and cannot be generalized to "NoThinking is better on all tasks."

### 2.2 Why NoThinking Works on Some Tasks

A long chain concentrates the budget on a single solution path. When the early steps go wrong, the subsequent calculations depend on the incorrect prefix. Multiple short answers disperse the budget across different starting points, covering a richer set of methods. It also relies on two conditions: the short answers themselves must have a certain success rate, and the system must be able to judge which answer is better. When the single-trial success rate is too low or the task lacks a validator, increasing the number of short answers may not be helpful.

These experiments compare two ways of allocating computation: using the budget to extend a single reasoning chain or to generate more independent candidates. For easy questions where the single-trial success rate is already high, more sampling helps cover the correct answer. For tasks requiring continuous derivation, long reasoning is more valuable. The choice depends on both the task difficulty and the reliability of the validator.

## 3. How Long2Short Compresses Reasoning Chains

Long reasoning training first prioritizes success rate, and the model may thus learn to repeatedly check and restate. When deploying, it is not sufficient to simply use hard truncation: if the truncation point falls before the conclusion, the model may not even have the chance to output an answer. The goal of Long2Short is to enable the model to learn, during training, to retain necessary steps and remove redundant ones.

Reasoning with reinforcement learning may lead to longer responses, as longer trajectories provide more opportunities for trial, checking, and recovery. This has three direct reasons:

- **Reward signals encourage "getting the answer right"**, and longer CoT means more opportunities for checking, thus increasing the probability of getting the answer right.
- **Reflection and verification behaviors are reinforced**, and the model learns to "check again."
- **There is no explicit length constraint**, so the model has no motivation to compress CoT.

Length increases are only valuable if they improve accuracy; repeated restatements and ineffective checks directly increase deployment costs. This leads to the long2short problem: how to transfer the planning and correction capabilities learned in long reasoning to shorter responses while maintaining the task success rate?

[Kimi k1.5](https://arxiv.org/abs/2501.12599) compares four specific approaches: model weight merging, Shortest Rejection Sampling, DPO, and long2short RL. They respectively approach the problem from model parameters, supervised samples, preference data, and RL rewards.

### 3.1 Getting Short Responses from Long Models

When multiple samples are drawn for the same question, the length of the correct answer often varies. Shortest Rejection Sampling selects the shortest correct answer from multiple correct responses as the SFT sample; DPO treats shorter correct answers as preference samples and longer or incorrect answers as rejected samples. These two methods first ensure the answer is correct, then learn the difference in length, avoiding the model learning to shorten responses by omitting steps.

Model weight merging takes a different approach: it directly averages the weights of long and short CoT models, observing whether the long reasoning capabilities can be partially transferred to the short model. It does not generate new compressed text, nor does it guarantee that each capability of the merged model lies between those of the two models. Therefore, it still requires full evaluation.

### 3.2 Comparing Answer Lengths Within a Group in RL

In long2short RL, the model continues training from a model that has already achieved a good balance between correctness and length. It shortens the allowed rollout length. For Kimi k1.5, the length reward is computed as a relative comparison within a group of answers: shorter correct answers receive higher length rewards; shorter incorrect answers do not receive positive rewards.

Let the length of the $i$-th answer be $\operatorname{len}(i)$, and let $l_{\min}$ and $l_{\max}$ be the shortest and longest lengths within the group, respectively. We first compute

$$
\lambda_i = 0.5 - \frac{\operatorname{len}(i) - l_{\min}}{l_{\max} - l_{\min}}.
$$

$\lambda_i$ decreases as the answer length increases. The final length reward is determined by combining $\lambda_i$ with the correctness of the answer. If all answers in the group have the same length, the length term is set to 0. This design learns to "prefer shorter answers among those that can solve the problem," without requiring a fixed target length for all questions in advance.

For three correct responses of 800, 1,200, and 2,000 tokens, the normalized lengths are $0$, about $0.33$, and $1$. Their length coefficients are therefore $0.5$, about $0.17$, and $-0.5$. The comparison is relative to the current response group rather than to one universal target length.

### 3.3 Effect Before and After Compression

The paper reports that the long2short RL model achieves a Pass@1 of 60.8 on AIME 2024, with an average of 3,272 tokens used. This number only indicates the quality–length trade-off in the given experiment; when selecting a deployment model, one should plot the full curve rather than comparing only a single score.

### 3.4 How to Coordinate Training Compression with Inference Budget

Long2short and thinking budget are complementary:

- **Long2short** is **training-stage compression**—it modifies the model to prefer generating shorter CoT.
- **Thinking budget** is **inference-stage control**—it does not change the model, but limits the computation per request by setting a service ceiling.

During deployment, the two controls can be combined:

```text
Training stage: long2short RL trains the model to generate shorter CoT
Inference stage: thinking budget sets a safe upper limit
```

## 4. How to Control Modes and Budget During Deployment

After training to obtain two modes and a shorter reasoning chain, the service interface is the final step. Each request must sequentially make three decisions: whether to enter the thinking mode, the maximum allowed computation, and how to produce a complete answer once the budget is exhausted.

Once the mode switch is determined, training and deployment must address three issues: how to select the mode, how to evaluate the capability differences between the two modes, and how to wrap up the process when the budget is exhausted.

### 4.1 Determining Whether to Think

Users can explicitly choose the mode, or the router can select it based on the task, historical success rate, and latency goals. If the model is involved in the decision, the choice should still be recorded, and the cost of mistakenly enabling long thinking and mistakenly disabling thinking should be separately evaluated.

### 4.2 Ensuring Consistency Between the Two Modes

When a deterministic question appears in two modes and produces conflicting answers, it indicates that at least one path is unreliable. During training, shared data or consistency objectives can be introduced. During deployment, the two modes should be evaluated separately, and it should not be assumed that shared parameters will naturally ensure identical answers. For open-ended questions, it is acceptable for the answers to differ, with the evaluation focusing on factual accuracy and task outcomes rather than exact word-by-word consistency.

### 4.3 Ending the Reasoning When the Budget is Exhausted

Hard truncation may stop the response in the middle of a step. A more reliable interface should reserve tokens for the final answer and trigger a convergence when the budget is nearly exhausted. Training data can also cover different budget conditions. If the budget is exhausted without a reliable conclusion, the system should return an incomplete status or a degraded solution, and should not package any intermediate conclusion as a definitive answer.

## Summary

Hybrid Thinking uses mode switching to avoid unnecessary long reasoning for simple tasks, while the thinking budget sets a computational limit for complex tasks. long2short shortens the reasoning chain from the training stage, and the two can be used in combination.

The four case studies address issues at different levels:

- **V3.1**: A single model provides both Think and Non-Think interfaces.
- **Qwen3**: Thinking Mode Fusion + thinking_budget parameter.
- **Kimi k1.5**: long2short RL actively compresses CoT.
- **NoThinking Study**: Compares using the same budget for a single long chain versus multiple short answers.

Fixed mode switches still require users or routers to pre-judge the task difficulty. [16.5 Adaptive Thinking](./adaptive-thinking) further allows the model to dynamically decide the depth of reasoning based on the input.
