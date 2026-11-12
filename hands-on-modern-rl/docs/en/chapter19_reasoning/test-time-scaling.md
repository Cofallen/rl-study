# 16.3 Test-Time Scaling

In Section 16.2, we trained models that tend to perform more extensive reasoning. However, during deployment, a new issue arises: the same model may face two different tasks — translating "hello" into Chinese and solving an AIME problem — and if both are generated with 10,000 tokens, the former may waste computation, while the latter may not be sufficient.

Once the parameters are fixed, the system has three ways to increase computation: generating more candidate answers in parallel, revising a single answer repeatedly, or searching across multiple intermediate steps. All of these fall under **Test-Time Compute Scaling**. This section continues with a single math problem, first explaining why additional computation may be effective, then examining the three approaches separately, and finally discussing when the gains may plateau and how to set stopping conditions.

<img src="../../chapter19_reasoning/images/test-time-compute-options.svg" alt="Test-time compute can be allocated to parallel candidates, iterative revision, or search over intermediate steps">

## 1. Why Increase Computation During Reasoning

A model's failure to generate a correct answer in one attempt may have two causes: it either lacks the required knowledge or it has the correct method but selected the wrong path or made a calculation error in this particular sampling. Increasing reasoning computation can only alleviate the second type of problem. To determine where to allocate the additional budget, we must first distinguish between training computation and single-task computation.

Training computation and inference computation occur in different stages. Training computation updates parameters to enable the model to learn new knowledge and strategies; inference computation keeps the parameters fixed and increases sampling, checking, or searching on the current problem. These two types of computation cannot be directly interchanged based on FLOPs.

Early language models typically generate a single, relatively short answer at a time. Reasoning models, however, allow for generating longer trajectories or more candidates on the same problem, transforming inference computation from a fixed cost into a budget that can be allocated. The question then becomes: once the model's capabilities are fixed, which allocation strategy is most effective in improving the success rate of the current task?

### 1.1 The Budget Experiment by Snell et al.

[Snell et al. 2024](https://arxiv.org/abs/2408.03314) ("Scaling LLM Test-Time Compute Optimally") systematically compares different reasoning budgets and allocation strategies:

**Experimental Setup**: A base model (Llama-3-8B-Instruct) is fixed, and the performance is compared on different difficulty levels of math problems using two approaches:

- **Approach A**: Use more reasoning compute — let the model generate N candidate solutions, and select the best one using a verifier (best-of-N)
- **Approach B**: Switch to a stronger model, and compare the gains from increased model capability versus increased reasoning compute

The experiment yielded three main results:

1. On the given model and math data in the paper, appropriately allocating reasoning compute can allow a smaller model to outperform a baseline with higher compute but poor allocation;
2. For problems where the base model rarely generates correct candidate solutions, further gains from increasing a limited budget will diminish;
3. The appropriate strategy varies with problem difficulty and verifier quality, and a single strategy cannot be universally applied to all problems.

Once training is complete, model parameters are fixed, and the reasoning phase can still adjust the number of candidates and the number of revisions. Therefore, the same model can be allocated a smaller budget for simple tasks and a larger budget for more difficult tasks. The experiment also shows that this adjustment is constrained by the base model's capability: when the model cannot generate the correct approach even once, the benefit of repeated sampling is minimal.

## 2. Where Can Reasoning Compute Be Spent

Once a budget is fixed, the most direct choice is either "trying more paths" or "continuing to refine along a single path." Tree search combines both approaches: first generate multiple intermediate branches, and then concentrate the budget on the paths with higher scores.

### 2.1 Parallel Sampling

Let the model independently generate $ N $ candidate solutions, and then use a verifier to select the best one. This is the idea behind the "best-of-N" approach.

Assume that the model generates the correct answer with probability $ p $ each time. The probability of at least one correct answer among $ N $ samples is:

$$
P(\text{At least one correct}) = 1 - (1 - p)^N.
$$

This is because the probability of all $ N $ samples being incorrect is $ (1 - p)^N $, and we subtract this from 1. For example, when $ p = 0.2 $ and $ N = 5 $, the probability of at least one correct answer is approximately $ 1 - 0.8^5 = 67.2\% $. This is just the coverage probability; the final step still requires the verifier to select the correct candidate from the pool.

At $p=0.2$, coverage is $20.0\%$ for $N=1$, $36.0\%$ for $N=2$, $59.0\%$ for $N=4$, $83.2\%$ for $N=8$, and $97.2\%$ for $N=16$. The marginal gain from one additional candidate falls from $16.0$ percentage points after the first sample to about $0.6$ points after sixteen samples. Parallel sampling therefore has sharply diminishing returns even before verifier errors are considered.

```python
# Parallel sampling illustration
candidates = [model.generate(prompt) for _ in range(N)]
scores = [verifier.score(prompt, c) for c in candidates]
best = candidates[argmax(scores)]
```

When the candidates are generated independently, they can be sampled in parallel, increasing the chances of covering the correct answer. However, the cost is that both generation and scoring computations scale with $ N $, and the system must have a reliable verifier to select the correct candidate. This is precisely the issue addressed in [Chapter 17 on PRM](../chapter20_prm_search/outcome-vs-process).

### 2.2 Sequential Refinement

Parallel sampling starts from scratch each time, thus failing to utilize the work already completed in the previous version. Sequential refinement first generates an initial solution, then asks the model to identify issues in it and produce the next version:

```python
# Sequential Refinement Example
solution = model.generate(prompt)
for _ in range(K):
    feedback = model.critique(prompt, solution)
    solution = model.revise(prompt, solution, feedback)
```

Sequential refinement reuses the solution from the previous version, making it suitable for tasks where errors can be identified and local modifications are cheaper than generating from scratch. Each round depends on the previous one, so the delay accumulates with the number of rounds; if the critique itself is incorrect, subsequent versions may be further away from the correct answer. Actual systems can still incorporate an external verifier to check whether the modifications are worth accepting.

### 2.3 Tree Search

Parallel sampling repeats the same prefix, while sequential refinement may get stuck on the same erroneous path. Tree search expands the reasoning process into nodes: each node stores an intermediate step, and the system can attempt multiple subsequent steps from the same prefix, using a verifier to decide which paths to retain. The specific algorithm is placed in [17.5 Inference-Time Search](../chapter20_prm_search/inference-time-search), and here we just remember that it addresses the question of "how to reuse intermediate results."

## 3. Deep Think: How to Turn Parallel Reasoning into a Product

The first two sections treated parallel candidates as multiple independent calls. In practice, systems must also address another issue: how multiple paths share GPU resources, when to stop, and who integrates the final conclusions. Deep Think can serve as a case study for the productization of parallel reasoning.

Google's public description of [Gemini Deep Think](https://blog.google/products-and-platforms/products/gemini/gemini-2-5-deep-think/) provides a product case: the system considers multiple hypotheses simultaneously using parallel thinking, and allows these candidates to be revised or combined before the final answer is generated. However, the public materials do not disclose the full algorithm, so we cannot directly conclude whether the internal implementation uses Best-of-N, tree search, or some cross-path attention mechanism.

Three system functions can be abstracted from the interface behavior:

- Generating multiple independent reasoning paths in parallel;
- Comparing or combining information from different paths;
- Deciding when to stop and generate the final answer within a budget.

If $N$ paths are assigned to independent computing resources, they can be generated in parallel; however, the total computational cost still increases with the number of paths. The end-to-end speed also depends on scheduling, aggregation, and hardware utilization, and cannot be directly scaled by a factor of $N$.

### 3.1 Why Product Scores Cannot Directly Explain the Algorithm

Google's report on the dedicated Deep Think version achieved a gold-level performance in the 2025 IMO, and subsequent versions also published results on HLE, ARC-AGI-2, and Codeforces. These results indicate that parallel thinking can serve as an effective reasoning budget strategy, but they cannot alone prove whether the gains come from a certain number of paths or a particular aggregation algorithm. Evaluations may also involve code execution, search, and different budgets, so when comparing models, these conditions must be recorded simultaneously.

For the course, a more important conclusion is that parallel paths reduce wall-clock waiting time, but do not eliminate the total computational cost. The more paths there are, the more the system needs to handle GPU scheduling, candidate deduplication, evaluator throughput, and stopping conditions.

### 3.2 How to Terminate Parallel Paths

It is easiest to implement a fixed generation of $N$ paths, but this can lead to simple questions that have already reached a consistent answer continuing to consume the budget. A dynamic system can check three pieces of information after each batch of candidates: whether a verifiable answer has already appeared, whether high-scoring candidates are consistent with each other, and whether adding another batch of candidates can bring new methods. The system can terminate when the answer has already been verified externally and the candidates tend to be consistent. When candidates are still conflicting and the verifier cannot distinguish them, the value of continuing sampling is typically low. In such cases, it is more necessary to supplement feedback with tools, search, or human rules.

This termination logic cannot be automatically obtained by the phrase "parallel thinking." It must be designed together with the task verifier, maximum token limit, wall-clock timeout, and failure fallback.

## 4. When to Stop Adding Inference Computation

The number of candidates, the number of revisions, and the number of search nodes can continue to increase, but the accuracy will not always rise at the same rate. Deployment requires comparing "how much more success the next computation can bring" with "how much more delay and cost it will add."

Additional computation increases the number of generated tokens, the number of verifier calls, and hardware usage, but the delay does not necessarily increase in proportion to the total computation: parallel sampling may use more GPUs to achieve shorter waiting times, while sequential revisions will directly increase the number of serial rounds. When deploying, it is essential to simultaneously record the task success rate, wall-clock delay, total tokens, the number of verifier calls, and the cost.

The task name alone cannot directly determine the budget. For the same coding problem, parallel candidate generation is possible when there is a complete test suite; however, when there is no test suite, further sampling may only produce more indistinguishable answers. Before setting the budget, one should first determine whether the verifier is reliable, whether the success rate of a single attempt is sufficient, and the cost of task failure.

This also forms the engineering motivation for [16.4 Hybrid Thinking and Budget Control](./hybrid-thinking): choosing the inference mode and budget based on the task difficulty.

### 4.1 Why Rewards Gradually Decrease

The additional reward calculated depends on whether "further attempts can still find new correct paths." For simple questions, the model has already answered correctly after a small number of computations, and further generation only repeats the same result. For medium-difficulty questions, more candidates or a single revision may correct occasional errors. For difficult questions beyond the base model's capability, the model repeatedly generates similar errors. The budget experiments by Snell et al. show that the optimal allocation varies with the difficulty of the question and the verifier, and a fixed increase in the number of candidates does not maintain the same marginal gain.

Therefore, test-time compute scaling cannot fully replace training compute scaling. They are complementary:

- **Training compute** determines the **upper bound of capability**
- **Test-time compute** determines how **close the model gets to that upper bound**

If the base model does not already possess the knowledge or skills required to solve the problem, increasing the number of samples and revisions will not generate the correct path. Therefore, reasoning computation relies on the capabilities already present in the base model.

## Summary

Test-time Compute Scaling provides three directly adjustable resources: the number of parallel candidates, the number of revisions per reasoning chain, and the size of the search tree. The harder the task, the more likely increasing these resources will yield gains; however, the gains will quickly diminish when the base model is insufficient or the task is already simple.

The next section will apply this principle to deployment: how a single model can switch between direct answering and deep thinking, while using a thinking budget to limit latency and cost.
