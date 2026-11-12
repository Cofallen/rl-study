# 16.1 From Language Models to Reasoning Models

Let us first consider a scenario where a standard language model is likely to fail. When faced with a competition math problem, it may read the question and immediately generate the most familiar formula; if the first step is chosen incorrectly, the subsequent tokens only continue along this erroneous path. The same applies to code tasks: the model can write a patch in one go, but it may not first identify the problem, run tests, and then revise the plan based on the results of failed attempts.

The reasoning model changes this execution process. Before providing the final answer, the model can expand intermediate steps, compare candidate methods, invoke tools, and check the results. Additional computation is valuable only when these behaviors improve the success rate, so the training objective, reasoning budget, and result validation must all be adjusted accordingly.

Chapter 16 follows this transformation: Section 16.1 first explains why reasoning models emerge; Section 16.2 enters the pure RL training of R1-Zero; Sections 16.3 and 16.4 and 16.5 address how to increase reasoning computation while keeping parameters fixed; Sections 16.6 further discusses how to present and supervise the reasoning content. This section first answers four questions: what changes in the product form, what reasoning behaviors have been observed in public experiments, what aspects are reinforced by RL, and how the system distinguishes reasoning from the final answer.

<img src="../../chapter19_reasoning/images/reasoning-model-loop.svg" alt="Comparison between direct generation and a reasoning model's problem-solving loop">

## 1. o1 How to Bring Reasoning Capabilities into the Product

Still using the math problem as an example, a direct answer model only performs a single continuous generation. In contrast, a reasoning model allows for a process of "listing conditions—trying methods—checking—revising" before arriving at the answer. The significance of o1 lies in embedding this process into a callable product, enabling users to trade more waiting time for higher task success rates.

### 1.1 o1 How to Train Longer Reasoning Processes

OpenAI highlighted in the [o1 release note](https://openai.com/index/learning-to-reason-with-llms/) the emphasis on training and reasoning computation: the model learns, through large-scale reinforcement learning, how to organize longer internal reasoning. As the training computation and per-problem thinking time increase, the evaluation performance also improves. Since the official release did not disclose the full reward composition, we cannot simplify o1 as "only rewarding the final answer." What can be determined, however, is that reinforcement learning directly optimizes the effectiveness of reasoning chains in complex tasks.

At the time of its release, o1 significantly outperformed GPT-4o on several reasoning benchmarks:

- **AIME 2024**: GPT-4o achieved approximately 12% on single responses, while o1 reached about 74%. When using majority voting across 64 responses, o1 achieved approximately 83%;
- **Codeforces**: o1's Elo rating was 1673, placing it in the 89th percentile among evaluated human players;
- **GPQA Diamond**: o1 achieved approximately 77% on single responses, while GPT-4o achieved about 51%.

The reasoning computation used for single responses, majority voting, and re-ranking differs. When presenting metrics together, it is essential to retain the evaluation methodology to prevent readers from conflating "model strength" with "more sampling."

For the missing comparison point, GPT-4o reached about 56% with 64-response consensus on AIME 2024. If a model's single-sample accuracy is $p=0.12$, 64 samples contain at least one correct answer with probability $1-(1-0.12)^{64}\approx99.97\%$. Finding a correct candidate and selecting it are different problems: correlated errors and answer aggregation still limit majority voting. o1's approximately 74% single-response result therefore cannot be explained simply by candidate coverage.

o1 does not display its internal Chain-of-Thought (CoT) to users, only returning the final answer or a curated explanation. This approach reduces the risk of internal reasoning being directly copied and prevents users from inspecting the model's actual reasoning process step by step. Therefore, the system also needs to establish credibility through verifiable answers, citations, and external evaluations.

### 1.2 Reasoning Budget as a Product Interface

After models can benefit from more thinking, the server also needs to decide how much computation to allocate per request. OpenAI introduced the `reasoning_effort` parameter when officially releasing o1 in December 2024. Developers can choose between a lower or higher reasoning investment. This parameter does not guarantee a fixed number of tokens; instead, it expresses a tier in terms of quality, latency, and cost.

- **Low effort:** short reasoning and low latency for translation, rewriting, simple questions, and format conversion.
- **Medium effort:** moderate checking for ordinary coding and medium-difficulty mathematics.
- **High effort:** longer derivations for competition problems, multi-constraint planning, and complex debugging.

In the same month, OpenAI previewed o3, and it was officially released in April 2025. The official report on o3 continued to expand the computational resources allocated to RL training compared to inference computation, and it outperformed o1 under the same latency and cost conditions. The important change is not in a particular number on an ever-updating leaderboard, but rather in the product interface, which now allows the system to adjust the reasoning investment according to the task.

Sections 16.3–16.5 will break down this interface: how additional computation can be used, how a fixed budget can be set, and how the model dynamically selects a budget based on the task.

### 1.3 o3 and o4-mini Bring Reasoning into Tool Calls

Released in April 2025 were [o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/), and there was no "complete o4" released at the same time. These two models integrated tools such as web search, Python execution, file and image analysis into the same problem-solving process:

- The model can actively call tools (search, code execution, image analysis) during the reasoning process.
- The results of the tool calls influence the next step of reasoning.
- This forms a loop of "think → call a tool → think again → call a tool again."

This working style is typically referred to as **agentic reasoning**: the model reads feedback from the external environment during multiple rounds of reasoning and decides the next action. It is directly related to the tool calls and environment interaction discussed in [Chapter 19 on Agentic RL](../chapter22_agentic/overview).

## 2. How Competitive Programming Validates Reasoning Training

Baseline scores can only indicate that results have improved, but they cannot explain what the model does during the problem-solving process. Competitive programming provides a more observable environment: whether the code compiles, whether the tests pass, and whether the results improve after modifications, can all be recorded by tools.

After the release of o1, a direct question arises: do these behaviors come from a large number of manually written reasoning chains, or can they be reinforced through result feedback?

In February 2025, OpenAI published [Competitive Programming with Large Reasoning Models](https://arxiv.org/abs/2502.06807) (arXiv:2502.06807). The paper did not disclose the full training details, so the following discussion will focus only on behaviors and computational allocations that can be directly observed from the experiments.

### 2.1 Comparison of End-to-End RL with Specialized Pipelines

Traditionally, tasks on platforms like Codeforces typically use specialized pipelines:

```text
Problem → Compiler + Test Case Generation → Program Synthesis → Selecting the Optimal Solution
```

Each step in this pipeline is handled by a dedicated module, such as search algorithms, program generation, and test filtering. The reasoning model in the paper incorporates all these behaviors into a single generation trajectory and achieves better results on the reported Codeforces setup. This comparison indicates that end-to-end training can learn to organize candidate generation, execution testing, and solution modification. However, the conclusions are still constrained by the model, the task, and the computational budget.

### 2.2 How Multi-Stage Reasoning Behaviors Emerge

The paper also documents the multi-stage reasoning behaviors exhibited by o1/o3 when solving Codeforces problems, including:

- **Generating Multiple Candidate Solutions**: The model generates multiple solutions and compares their strengths and weaknesses.
- **Execution Verification**: Using tools to execute code and check whether the results match expectations.
- **Self-Correction**: Re-deriving solutions after discovering errors.
- **Strategy Switching**: Switching from greedy algorithms to dynamic programming, and then to backtracking.

The training process did not explicitly encode the "try greedy first, then dynamic programming" as a fixed procedure. However, high-quality pre-training data may have already exposed the base model to these methods. The role of reinforcement learning (RL) is to increase the probability of trajectories that pass tests, making the comparison of candidates, execution verification, and error correction more stable. Clear reward signals provide selection pressure, and the base model's existing knowledge and sampling coverage determine which behaviors can be reinforced.

### 2.3 How Training and Inference Compute Power Are Divided

The paper also reports a key trade-off:

- **Increasing Training Compute**: The model's foundational capabilities improve, but the number of reasoning tokens per problem remains largely unchanged.
- **Increasing Inference Compute** (letting the model think longer): With fixed training compute, performance can be further improved.

Training compute determines which methods the model has already mastered, while inference compute determines how many attempts and checks the model can make on a given problem. Section 16.3 will further compare these two types of compute.

## 3. Is RL Creating or Activating Capability?

After observing that the model checks and corrects its own reasoning, we must also address the source of its capabilities. If all changes are attributed to RL "creating reasoning," we would overlook the knowledge the base model has already learned from pre-training data, including mathematical, coding, and language patterns. The "emergence" reported by o1 and R1-Zero needs to be understood in two layers:

**Meaning One: Reasoning Behaviors Never Explicitly Appeared in Training Data**

R1-Zero has never seen any manually annotated CoT (Chain-of-Thought) in training data, yet after training, it autonomously generates long reasoning chains, performs reflection, and conducts verification. This "emergence" is relative to the training data — the model was not explicitly taught to do so.

**Meaning Two: Is All Reasoning Ability Created by the RL Phase?**

Subsequent replication experiments (e.g., SimpleRL-Zoo, Open-R1) show that the base model could already generate correct reasoning with a certain probability before RL. RL reallocates the probabilities of these trajectories based on rewards, making effective methods appear more frequently, and may also form new strategies by combining existing behaviors. Therefore, when analyzing the effects of RL, we must simultaneously examine the base model's capabilities, the sampling coverage, and the changes in post-training probabilities.

This distinction is important in practice:

- First, perform multiple sampling on the base model to estimate whether correct trajectories already exist.
- Then, compare the success rate and behavior distribution before and after RL to determine whether the issue is a lack of capability or whether effective trajectories are too rare.
- Finally, decide whether to supplement with more pre-training or SFT (Supervised Fine-Tuning) data, or to increase RL data and compute.

The experiments with DeepSeek-R1 also support this distinction: pre-training provides knowledge and basic reasoning capabilities, while RL makes the reasoning behaviors that can receive rewards appear more stably.

For example, sample one mathematics problem 100 times from the base model. Suppose 72 responses jump directly to an incorrect answer, 22 contain reasoning with a calculation error, and 6 contain a complete correct derivation with verification. The base model is not wholly incapable: correct trajectories already have 6% probability. Group-based RL can use those six trajectories as positive evidence and raise their probability. If no correct trajectory appears even under broad sampling, the missing capability must first be supplied by pretraining or SFT.

## 4. How Reasoning Models Organize Output and Tools

After training the model to perform verification and correction, the system must organize these behaviors into a stable interface. At least three types of content need to be distinguished: the reasoning state used for further computation within the model, the actions that can be passed to tools for execution, and the final answer returned to the user. When all these are mixed in a single text stream, the verifier finds it difficult to extract the answer, and the tools struggle to determine when to execute.

Public models adopt different interfaces, which can be compared based on three independent questions:

| Interface Question                  | Representative Approach                                                                      | Issues to be Further Addressed                                                           |
| ----------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Whether to engage in long reasoning | DeepSeek V3.1, Qwen3 support both reasoning and non-reasoning modes within the same model    | How to determine which mode is needed for the current task                               |
| How much computation to invest      | OpenAI `reasoning_effort`, Qwen3 thinking budget, Claude effort                              | How to align the budget with the task difficulty                                         |
| What to show to the user            | OpenAI provides reasoning summaries, while models like DeepSeek-R1 can output reasoning text | Whether the displayed content can be checked, and whether it leaks sensitive information |

Model names and interfaces may change, but these three questions remain stable. They correspond respectively to Section 16.4 on modes and budgets, Section 16.5 on adaptive allocation, and Section 16.6 on the display of reasoning chains.

The display method affects user verification, data distillation, and security monitoring, which will be discussed separately in Section 16.6. Here, we first examine how the training format separates the reasoning segment from the answer segment.

### 4.1 How to Separate Reasoning Tokens from Answer Tokens

Traditional LLM outputs consist of a single token sequence: `<answer token>`. In contrast, reasoning models output an additional segment of "reasoning tokens": `<reasoning token> <delimiter> <answer token>`.

In engineering implementation, this is typically achieved through special tokens:

```text
<|begin_of_thought|>
The user's question is to solve... Let me first understand the problem...
Possible approaches include...
First approach: ... This approach has the issue of...
Second approach: ... This approach seems feasible...
Let me verify it...
Yes, the second approach yields the correct result.
<|end_of_thought|>
<|begin_of_solution|>
The final answer is X.
<|end_of_solution|>
```

After splitting, the system can extract only the answer to be passed to the outcome verifier, or apply length constraints or process evaluations on the reasoning segment. The delimiter itself does not automatically provide correctness labels, but it establishes clear data boundaries for Hybrid Thinking, thinking budget, and long2short.

## Summary

o1 makes "investing more reasoning computation" a callable product capability, while o3 and o4-mini further integrate tool feedback into the reasoning process. Competitions in programming experiments further demonstrate that end-to-end RL can reinforce behaviors such as candidate generation, execution validation, and error correction; the pre-trained model's existing knowledge and coverage range still limit what RL can reinforce.

[16.2 R1-Zero Pure Reinforcement Learning Reasoning](./r1-zero-pure-rl-reasoning) will elaborate on rewards, intra-group comparisons, and behavioral changes, specifically explaining how these reasoning behaviors can be formed during training without going through SFT cold start.
