# 17.2 Discriminative PRM

The proof in Section 17.1 missed a square in step six. To enable the model to detect such errors, the most direct approach is to collect many "problem—reasoning prefix—current step" instances, and have annotators determine whether the current step can be continued. After training, the evaluator sees new steps and outputs the probability of `good`, `bad`, or `neutral`.

This is the discriminative PRM: rewriting the step-by-step checking into a classification task. This section follows a data production line—first, split the full reasoning into annotatable steps, then train a classifier, use the step probabilities for candidate sorting or reinforcement learning, and finally revisit how annotation noise and step boundaries might alter the results.

<img src="../../chapter20_prm_search/images/discriminative-prm-pipeline.svg" alt="A discriminative PRM pipeline from step segmentation and annotation to training, reranking, and reinforcement learning">

## 1. How to Collect Process Supervision Data

The classifier needs to first see labeled steps. Take the proof in Section 17.1 as an example: the annotator reads the problem and the first five steps, then determines whether the sixth step's equation transformation is valid. Showing only the current sentence would lose the context, while showing the entire subsequent reasoning might let the final answer influence the judgment. Therefore, the input is usually truncated at the current step.

[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (Lightman et al. 2023) compared two approaches: labeling only the final answer and step-by-step labeling. The difference lies not in whether the model is "larger," but in where the supervision signal falls—on the end or on each step.

The paper compared two types of supervision:

| Supervision Type     | Annotation Content              | Annotation Cost                            |
| -------------------- | ------------------------------- | ------------------------------------------ |
| **ORM** (Outcome RM) | Correctness of the final answer | One result label per solution              |
| **PRM** (Process RM) | Validity of intermediate steps  | One solution contains multiple step labels |

Both types of evaluators are used to rank the model's generated candidate solutions. The paper found that, as the model and annotation scale increase, process supervision consistently outperforms outcome supervision in the given experiments. A process-supervised model trained on full data solved 78% of the problems on a representative subset of the MATH test. This 78% includes multiple candidate generation and evaluator selection, so it cannot be treated as a single-generation accuracy rate, nor can it be directly extrapolated to other tasks.

### 1.1 How PRM800K Records Step Labels

[PRM800K](https://github.com/openai/prm800k) is a step-level annotated dataset released by OpenAI. Its scale is as follows:

- **800K** step-level annotations
- Model problem-solving processes for questions from the MATH dataset
- Each step is annotated as one of three states:
  - **good** (correct): The reasoning is reasonable and can continue
  - **bad** (incorrect): The reasoning is wrong and should stop or backtrack
  - **neutral** (neutral): It may be a transitional step, and no judgment is made on correctness

The data collection can be carried out step by step along a problem-solving trajectory:

1. Let the base model generate candidate steps or the full problem-solving process;
2. Organize the candidates into steps that can be independently evaluated according to the data specification;
3. Labelers assign a score of $+1$, $0$, or $-1$ to the current candidate based on the original question and previous steps;
4. Once the first error is found, stop or rewrite subsequent candidates, and leave ambiguous samples for the next round of active learning.

The repository interprets these values as correct, no obvious error but no progress, and error, respectively. They are commonly written as good, neutral, and bad, and training should be based on the data fields and annotation descriptions.

#### Understanding the Cost from the Annotation Scale

OpenAI has not publicly disclosed the actual annotation cost of PRM800K. The following is a simplified estimation to illustrate what "800,000 steps" means in terms of workload:

- 800K steps × 30 seconds per step = 24,000 hours (approximately 12 person-years)
- If each step takes an average of 30 seconds to check, the total workload is approximately 24,000 hours

Actual projects also need to account for training, review, dispute resolution, and quality sampling, so the above hours cannot be directly converted into the real cost of PRM800K. What can be determined is that step-by-step supervision requires more human judgment than checking the final answer. Teams must allocate their annotation budget to the steps that are most error-prone and most discriminative among candidates.

## 2. How to Train a Step Classifier

With the step labels in place, the next step is to train the classifier. The model must simultaneously read the original problem, the previous reasoning, and the current step, because "so $p$ is even" on its own has no meaning; whether it is valid depends on what conditions have already been derived.

### 2.1 Model Architecture

The classifier can leverage the language model's ability to read long prefixes, and then output a category or scalar score at the end of the current step. The input consists of `<Problem> <Solution up to step $i$>`, and the output represents the probability that the current step is correct, incorrect, or neutral. Public papers focus on the comparison between process supervision and result supervision, and there is no need to guess an unpublicized proprietary base model as a version of GPT-4.

Subsequent work (e.g., [Math-Shepherd](https://arxiv.org/abs/2312.08935)) also uses open-source language models to construct process rewards. The size of the base model, the reward head, and the source of the labels can vary, but the classifier must always read the context before the current step.

### 2.2 Training Objective

For the $i$-th step, the model outputs probabilities for three categories: good, bad, and neutral. The training objective is to maximize the probability of the human-labeled category using cross-entropy:

$$\mathcal{L}_{\text{PRM}} = -\sum_{i} \sum_{c \in \{good, bad, neutral\}} y_{i,c} \log p_\theta(c | q, o_{\leq i}, s_i)$$

Here, $c$ iterates over the three categories, $y_{i,c}$ is a one-hot label, and $p_\theta(c\mid q,o_{\le i},s_i)$ is the model's predicted category probability based on the problem and the context up to the current step. If the label for the $i$-th step is good, only the term corresponding to good has $y_{i,c}=1$, and this term becomes $-\log p_\theta(\text{good}\mid\cdots)$. The smaller the loss, the closer the model's probability of good is to 1.

For a step labeled good, predicted probabilities $0.95$, $0.70$, and $0.10$ produce losses of about $0.05$, $0.36$, and $2.30$. Cross-entropy therefore gives a mild penalty to an uncertain but plausible judgment and a much stronger penalty to a confident misclassification.

### 2.3 Data Augmentation for Training

When human labeling is limited, more training samples can be constructed from existing answers. Common approaches include:

- **Automatic Labeling**: Use a stronger evaluation model or result rollout to estimate unlabelled steps, obtaining noisy silver labels.
- **Synthetic Data**: Generate "look-like-wrong" steps from known correct problem-solving processes as negative samples.
- **Data Mixing**: Combine datasets such as PRM800K and Math-Shepherd with others.

## 3. How to Use for Answer Selection and RL

The classifier outputs the probability of each step's category. The system must then decide how to use these probabilities. During inference, multiple complete solutions can be ranked. During training, step scores can be converted into rewards. Both usages share the same PRM, but they differ in aggregation methods and risk profiles.

The most direct use is **reranking**: generate several candidate solutions, score their intermediate steps, and select one candidate. A PRM can also provide training rewards, but the system must then decide how step scores become token- or trajectory-level signals.

The workflow of Re-ranking is as follows:

1. **Generation**: Let the base model generate N candidate solutions for a math problem (N is typically 4–64).
2. **Scoring**: Use PRM to score each step of each candidate solution, obtaining the total score for the entire problem-solving process.
3. **Selection**: Select the candidate solution with the highest total score as the final answer.

PRM is responsible for ranking and does not replace the generation model. Increasing the number of candidates improves the likelihood of at least one correct solution, but also increases the cost of generation and scoring. Therefore, the paper compares ORM and PRM using the Best-of-$N$ curve, rather than writing a multi-candidate result as a single-model accuracy. On a representative subset of MATH, the full process supervision model achieves 78% in the paper's large-scale candidate setting and outperforms the corresponding result supervision model under different candidate budgets.

### 3.1 Token-level and Step-level Re-ranking

PRM Re-ranking has two scoring methods:

**Token-level**: Each token receives a score, and the total score of a response is the aggregation (mean, sum, min) of all token scores.

**Step-level**: Each reasoning step receives a score, and the total score of a response is the aggregation of all step scores.

PRM800K corresponds to the step-level judgment: the evaluator assigns a score at the end of each step, and these scores are combined to form the ranking criterion for the entire solution.

Choice of aggregation method:

- **Mean**: The average score of all steps. Robust, but may dilute the impact of critical error steps.
- **Min**: The minimum score of all steps. Conservative, tends to "reject the entire response if any step is wrong."
- **Product**: The product of all step scores. More strict than min.

In this mathematical experiment, the minimum step score is an effective aggregation method, because a single incorrect equation may invalidate the entire proof. Open writing does not have such strict logical dependencies, so this choice cannot be directly applied; the aggregation rule should be determined on the validation set of the target task.

For example, response A with step scores $[0.9,0.8,0.2,0.9]$ has mean $0.70$, minimum $0.2$, and product $0.130$. Response B with $[0.6,0.6,0.6,0.6]$ has mean $0.60$, minimum $0.6$, and the same product $0.130$. Mean favors A despite one severe error, minimum favors the consistently valid B, and product cannot distinguish them. Aggregation therefore encodes a task-dependent assumption about how one bad step affects the whole solution.

### 3.2 Using Step Scores for RL Training

Re-ranking is an inference-time application. PRM can also be used for RL training—using the step-level scores of PRM as a dense reward for RL.

Specific approach:

```python
# Using PRM as the reward function for RL training
def prm_reward(prompt, response):
    # Split the response into reasoning steps
    steps = split_into_steps(response)

    # Score each step
    step_scores = [prm(prompt, steps[:i+1]) for i in range(len(steps))]

    # Aggregate to get the reward for the entire response
    # Choosing min: reject the entire response if any step is wrong
    return min(step_scores)
```

This reward function can replace the ORM in RLHF and be used for PPO / GRPO training.

[Math-Shepherd](https://arxiv.org/abs/2312.08935) trains a reward model using automatically constructed process labels and uses it for candidate selection and reinforcement learning. The specific gains depend on the base model, dataset, and sampling settings, so it should be compared with ORM under the same computational budget.

## 4. How Annotation and Step Splitting Affect Performance

At this point, the data flow of PRM is complete: split into steps, annotate labels, train a classifier, and aggregate scores. The risks correspond to these four steps. Annotators may make misjudgments, domain shifts can cause the classifier to lose knowledge, and splitting steps too long or too short can alter the meaning of labels.

The performance of discriminative PRM is constrained by the following conditions:

### 4.1 Annotation Cost

Step-by-step annotation requires reading the context, judging the current step, and handling disputes, which is significantly more costly than simply verifying the final answer. New code or biomedical tasks do not necessarily require replicating a PRM800K. One can first annotate a small number of high-value samples, measure where the classifier fails, and then decide whether to expand the data.

### 4.2 Cross-Domain Generalization

Equation transformations in mathematics and state modifications in code require different knowledge, so a mathematical PRM cannot be used for code without evaluation. When transferring, one must recheck step splitting, label definitions, and classifier accuracy; it may be necessary to supplement with labels from the target domain.

[Generative PRM](./generative-prm) (next section) can explain the reasoning in natural language, making it easier to integrate into new tasks. However, this interface does not guarantee that it naturally achieves stronger cross-domain accuracy.

### 4.3 Annotation Noise

Even with PRM800K, annotation is not 100% accurate:

- Different annotators may judge the same step differently
- The correctness of complex reasoning steps is inherently subjective
- The boundary of the neutral category is ambiguous

Repeated annotation, dispute review, and model ensembling can reduce the impact of individual labels or individual evaluators.

### 4.4 Step Splitting

Discriminative PRM needs to split the response into "steps"—but how to split is itself a problem:

- Split by newline: Too mechanical, a "step" may span multiple lines.
- Split by period: Too fragmented, a complete reasoning may contain multiple periods.
- Use an LLM to split: Introduces new LLM invocation costs.

When steps are too short, the classifier cannot see the full operation; when steps are too long, a single label may cover multiple judgments. The splitting rule should correspond to the verifiable actions in the task and be fixed as part of the data specification.

### 4.5 How to Combine Process Scores and Outcome Scores

The approaches supported in the open literature can be summarized into three categories. The first category retains both ORM and PRM: the outcome score prevents the model from only writing "seemingly reasonable" processes, while the step score identifies intermediate errors. The second category trains or calibrates the evaluators by domain, making the label meanings consistent with mathematical, coding, or tool tasks. The third category first uses a strong model or rules to generate candidate labels, then manually samples difficult cases for inspection.

If we combine the two types of rewards with weights, it can be written as:

$$r = \alpha r_{\text{process}} + \beta r_{\text{outcome}}$$

Here, $r_{\text{process}}$ is the aggregation of step scores, $r_{\text{outcome}}$ is the final result, and $\alpha$ and $\beta$ control the weights of the two components. This formula does not specify the optimal weights; it reminds us to separately monitor "whether the process score increases" and "whether the task is truly completed," to avoid the evaluator's bias being amplified by the policy.

## Summary

Discriminative PRM maps the problem and reasoning prefix to step category probabilities. The data production process sequentially involves step segmentation, manual or automatic annotation, classification training, and score aggregation; any change in the definition at any stage will affect the final ranking and training reward.

It is suitable for tasks with clear step boundaries that require rapid evaluation of a large number of candidates. Step-by-step annotation and domain transfer remain costly, so [17.3 Generative PRM](./generative-prm) instead uses natural language evaluation, allowing the system to provide explanations of errors alongside conclusions.
