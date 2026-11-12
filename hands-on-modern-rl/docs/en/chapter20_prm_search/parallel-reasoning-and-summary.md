# 17.6 Parallel Reasoning and Answer Aggregation

Section 17.5 follows a search tree by sharing prefixes and gradually selecting paths. It is suitable for tasks where intermediate states are easy to save and score, but it requires frequent waiting for "generate one step—score—then generate again." If computational resources are sufficient, multiple complete reasoning paths can be run in parallel, and their results can be compared at the end.

For example, the same math problem can simultaneously try factoring, the quadratic formula, and completing the square. Majority voting only counts the three final answers, while a coordination model can check the reasoning behind each path and integrate results when one path has correct arithmetic and another has a clearer explanation.

This section first compares serial depth with parallel breadth, then uses PaCoRe to illustrate how a coordinator reads multiple paths, followed by extending this coordinator to GenRM and LLM-as-Judge. Finally, we return to the verifier and reasoning strategies from Chapter 17 and place them into a unified selection order.

<img src="../../chapter20_prm_search/images/parallel-reasoning-aggregation.svg" alt="Several reasoning paths are generated in parallel, normalized, and aggregated into one answer">

## 1. How to Allocate Reasoning Compute to Depth and Breadth

After fixing the total token budget, we can either let one path think longer or generate multiple shorter paths. The former retains continuous states, while the latter increases method diversity. Whether the task has multiple feasible solutions, whether the paths can be executed in parallel, and whether there is a reliable aggregator at the end will collectively determine the choice.

In [Chapter 16 Test-time Compute Scaling](../chapter19_reasoning/test-time-scaling), we discussed two ways to spend reasoning compute:

- **Serial Depth**: The model generates a long CoT
- **Parallel Breadth**: The model generates multiple independent CoTs and aggregates them in some way

| Method           | Representative                      | Compute Allocation                         | Suitable Tasks                                                              | Main Cost                                                                                 |
| ---------------- | ----------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Serial Depth     | Long reasoning, sequential revision | One path retains continuous state          | Proof, code, and tool tasks where the next step depends on the previous one | Increased rounds accumulate wall-clock delay, and early errors propagate                  |
| Parallel Breadth | Best-of-N, Self-Consistency         | Multiple paths are generated independently | Tasks with multiple independent solutions and easy-to-compare results       | Total computation increases with the number of paths, and a reliable aggregator is needed |

Parallel breadth can shorten waiting time with more hardware, but does not reduce the total computation. When resources are insufficient, multiple paths will also queue up for execution.

**PaCoRe** generates multiple inferences in parallel, compresses them into messages, and then allows the next round of parallel paths to read these messages. It differs from majority voting in that: multiple lines of thought from the previous round enter the next round of inference, and the final system does not only count the number of times an answer appears.

### 1.1 How PaCoRe Coordinates Multiple Inferences

[PaCoRe](https://github.com/stepfun-ai/PaCoRe) (StepFun, 2026 Technical Report) extends parallel inference through multiple rounds of message passing. Let $K_r$ be the number of parallel paths per round, for example, a configuration of `[32, 4]` indicates that 32 paths are generated in the first round, and 4 coordinated paths are generated after compression in the second round.

#### How Information is Exchanged in Each Round

```text
┌─────────────────────────────────────────────────────┐
│ Step 1: Generate $K_r$ inferences in the r-th round  │
│   - Input the original question and compressed messages from the previous round │
│   - Each path explores independently                  │
├─────────────────────────────────────────────────────┤
│ Step 2: Compress the current path results into coordinated messages │
│   - Retain candidate methods, conflicts, and intermediate conclusions │
│   - Control message length to ensure the next round can still fit into the context │
├─────────────────────────────────────────────────────┤
│ Step 3: Continue if not the last round; generate answers in the last round │
│   - Use outcome reward to evaluate the final result during training │
└─────────────────────────────────────────────────────┘
```

#### Difference Between Coordinator and Voting

PaCoRe differs from Best-of-N + Majority Vote in how information flows:

- **Majority Vote**: Selects the answer that appears most frequently (simple counting)
- **PaCoRe**: Compresses methods and conflicts from multiple inferences, allowing subsequent rounds to continue refining and synthesizing

The coordinator may bring two benefits:

- **Handling Equivalent Answers**: When multiple paths arrive at equivalent answers through different formulations, the coordinator can compare semantic meanings; majority voting requires first normalizing the answers
- **Comparing Reasoning Basis**: When two paths reach the same answer, the coordinator can still examine intermediate steps; the quality judgment is still limited by the coordinator model's capability

#### Training PaCoRe

PaCoRe uses outcome rewards to train coordination and synthesis capabilities, without requiring step-by-step PRM labels. The reward only indicates whether the final answer is correct, and the training algorithm must also estimate which paths and messages contributed to the result.

```python
def pacore_reward(prompt, target_answer, round_widths):
    message = ""
    for round_id, width in enumerate(round_widths):
        reasonings = [model.generate(prompt, message) for _ in range(width)]
        if round_id < len(round_widths) - 1:
            message = coordinator.compact(prompt, reasonings)
        else:
            final_answer = coordinator.answer(prompt, reasonings)

    reward = 1.0 if final_answer == target_answer else 0.0
    return reward
```

This pseudocode only illustrates the data flow. In practice, the implementation would distinguish between intermediate coordination messages and the final answer, and record the corresponding generation probabilities; an outcome reward would not be automatically and equally distributed to all paths.

## 2. Choosing Among the Three Reasoning Structures

The coordinator can utilize multiple complete paths, but it does not reuse common prefixes; tree search can reuse nodes, but it increases serial scoring. Below, we place PaCoRe, parallel thinking, and MCTS in the same table to compare their computational structures rather than only their final scores.

PaCoRe reports using the same 8B model to form three reasoning configurations: `[4]`, `[16]`, and `[32,4]`. The high-level configuration achieves 93.7% on AIME 2025, with a total test-time compute of approximately 1.87 million tokens; on HMMT 2025, it achieves 94.5%, with about 1.8 million tokens. As a baseline for training, RLVR-8B achieves 84.1% and 75.4%, but only uses about 50,000 tokens.

These results show that trained coordination models can still benefit from broader parallel computation, and also indicate that improvements are not cheap: the token count used by the high-level configuration is more than an order of magnitude higher than that of the single-chain baseline. When comparing approaches, one must report accuracy, total tokens, wall-clock delay, and parallel hardware together.

### 2.1 System Differences Among the Three Reasoning Structures

Comparison of the three reasoning structures:

| Dimension           | PaCoRe                                                        | Public Deep Think Interface                                                           | MCTS                                                      |
| ------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Reasoning Structure | Multiple parallel paths with compressed messages              | Parallel consideration of multiple hypotheses, internal algorithm not fully disclosed | Explicit tree state with repeated visits                  |
| Budget Control      | Number of paths per round and number of rounds                | Product effort or mode                                                                | Number of iterations, expansion count, and rollout length |
| Feedback            | Training mainly uses outcome rewards                          | Public materials do not disclose full structure                                       | Results, PRM, or external verifier                        |
| Main Cost           | Large number of complete trajectories and message compression | Difficult to reproduce internal coordination                                          | Serial selection, state maintenance, and multiple scoring |

### 2.2 Tasks Suitable for PaCoRe

PaCoRe is suitable for tasks that can try multiple approaches in parallel and where the final answer can be reliably scored. It does not require step-by-step PRM, but instead requires training the model to learn compression and synthesis. As the number of parallel paths increases, the total computational cost approximately grows, and performance does not guarantee linear improvement. If a path must frequently access tool states or intermediate errors need to be immediately stopped, the tree search with step-by-step verification, such as 17.5, is usually more appropriate.

## 3. How Generative Models Evaluate and Aggregate Answers

The coordinator needs to perform two tasks: judge the issues of each response, and compress multiple pieces of evidence into a conclusion. If it only outputs a scalar, the system finds it difficult to check the basis of aggregation; allowing the model to generate evaluation text and then obtaining scores from judgment tokens leads to a more general **GenRM (Generative Reward Model)**.

Traditional reward models directly output a scalar. GenRM first generates evaluation text or judgment tokens, then derives scores from the generated probabilities, thus being able to retain both numerical judgments and explanations.

### 3.1 Output Forms of GenRM

```text
Input: prompt + response + "Please evaluate this answer"
Output: Natural language evaluation + [GOOD/BAD]
```

GenRM can read the probability of the GOOD token at the judgment position and convert the generated judgment into a numerical reward:

$$\text{GenRM}(q, o) = P(\text{"good"} | q, o, \text{prompt})$$

Here, $q$ is the question, $o$ is the response to be evaluated, and the last `prompt` indicates the evaluation instruction. The higher the probability that the model assigns to GOOD at the GOOD/BAD position, the higher the reward. When comparing two responses, the model can also output A or B, and the probabilities of the two tokens can be normalized. Natural language explanations are used for checking the basis of judgment, while the training process uses comparable numerical values or labels.

### 3.2 GenRM and Discriminative RM

| Dimension | Discriminative RM                                      | GenRM                                                           |
| --------- | ------------------------------------------------------ | --------------------------------------------------------------- |
| Output    | Scalar, category, or preference probability            | Evaluation text and judgment token                              |
| Training  | Regression, classification, or ranking loss            | Next token prediction, compatible with preference data          |
| Debugging | Mainly depends on score distribution and error samples | Can read evaluation rationale, but still needs validation       |
| Inference | A single forward pass yields the score                 | Requires generating several evaluation tokens, typically slower |

### 3.3 Relationship between GenRM and PRM

GenRM is a broader concept — it can perform ORM (evaluation of the entire response) or PRM (evaluation of each step of reasoning).

[Section 17.3: ThinkPRM](./generative-prm) in [17.3] is a representative of GenRM doing PRM. Other GenRM works:

- **Generative Verifiers** ([Zhang et al.](https://arxiv.org/abs/2408.15240)): Use Chain-of-Thought evaluation
- **LLM-as-Judge** ([Zheng et al.](https://arxiv.org/abs/2306.05685)): Use GPT-4 to evaluate outputs of other models

### 3.4 LLM-as-Judge and Self-Rewarding

LLM-as-Judge is a common application of GenRM: using a stronger LLM to evaluate candidate outputs.

#### Applications of LLM-as-Judge

- **Benchmark Evaluation**: Using GPT-4 as the judge for MT-Bench, AlpacaEval, etc.
- **Training Data Filtering**: Using LLM to filter high-quality training data
- **Alternative to RLHF**: Using LLM to replace human-annotated preference data (RLAIF)

#### Self-Rewarding Language Models

[Self-Rewarding LM](https://arxiv.org/abs/2401.10020) (Meta 2024) allows the model to participate in evaluating its own candidate responses:

```python
def self_reward_training(prompt, model):
    # 1. Generate multiple responses
    responses = [model.generate(prompt) for _ in range(N)]

    # 2. Let the model evaluate itself
    rewards = [model.judge(prompt, r) for r in responses]

    # 3. Use self-evaluation for RL (DPO or PPO)
    model = rl_update(model, prompt, responses, rewards)
```

This approach does not require separately training a fixed reward model, but still needs evaluation prompts, initial supervision, and independent evaluation. When generation and evaluation come from similar models, self-evaluation may amplify existing biases: the model is less likely to recognize errors it has not yet mastered.

## 4. How to Combine Verifier with Reasoning Strategies

At this point, the method selection can return to a specific decision line: first determine what results can be reliably verified, then judge whether intermediate errors need to be located, and finally decide whether additional computation is needed for the depth of the tree or the breadth of parallel paths.

These four variations occur at different levels:

### 4.1 From Discrimination to Generation

Discriminative PRM outputs fixed labels, while generative PRM provides reasoning justifications, reusing the current model for self-evaluation. The more the system relies on its own judgment, the more external testing is needed to check for systematic biases.

### 4.2 From Depth to Breadth

Tree of Thoughts and MCTS reuse intermediate nodes, while PaCoRe generates complete paths in parallel across breadth. The former is suitable for tasks where step-by-step feedback is reliable, while the latter is suitable for tasks where multiple complete solutions can be compared with each other.

### 4.3 From Static to Dynamic

Fixed PRM can score each step; adaptive search further decides how to allocate subsequent budgets based on score differences.

### 4.4 From Single to Mixed

Outcome rewards, process rewards, formal checks, and LLM evaluations cover different types of errors. When using them in combination, it is important to clarify which layer each score applies to and handle dimensionality and conflicts.

## Summary of This Chapter

Chapter 17 begins with the example of "missing a square in the sixth step, yet the final answer may coincidentally be correct," and gradually fills in the feedback needed for training and reasoning:

- **17.1 Outcome Rewards and Process Rewards**: Sparse rewards and credit assignment
- **17.2 Discriminative PRM**: Let's Verify and PRM800K
- **17.3 Generative PRM**: Natural language evaluation and label efficiency of ThinkPRM
- **17.4 Formal Verifier**: AlphaProof, Lean4, and DeepSeek-Prover-V2
- **17.5 Reasoning with Search**: Beam Search, ToT, MCTS, and AlphaCodium
- **17.6 Parallel Reasoning and Answer Aggregation**: PaCoRe, GenRM, and LLM-as-Judge

When facing a new task, one can ask the following five questions in sequence:

1. Can the final answer be automatically verified? When verification is possible, first establish a reliable ORM.
2. Will errors propagate along long trajectories? When propagation is likely, consider step feedback.
3. Can steps be stably split and judged? They can be categorized, generated with explanations, or handed over to a formal checker.
4. Are intermediate states worth reusing? When they are, use Beam Search, ToT, or MCTS.
5. Are multiple complete solutions easier to parallelize? If so, consider Best-of-N, PaCoRe, or a generative coordinator.

This order also explains why PRM is not the default answer for all reasoning tasks. Evaluators themselves can make mistakes, search can amplify these biases, and parallel coordination consumes a lot of tokens. Only by first establishing reliable feedback and then expanding the reasoning computation can longer or more trajectories be converted into real capabilities.

**Next Chapters**:

- [Chapter 18: Industrial Practice](../chapter16_llm_rl_industrial/industrial-post-training) — Integrating Preference Optimization, RLVR, Reasoning Models, and PRM into a Complete Training Loop
- [Chapter 19: Agentic RL](../chapter22_agentic/overview) — PRM in Multi-Step Trajectories
- [Chapter 25: Reward Hacking](../chapter30_alignment_failures/classical-failures) — Reward Hacking Issues in PRM
