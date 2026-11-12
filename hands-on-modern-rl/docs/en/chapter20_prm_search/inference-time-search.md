# 17.5 Search During Reasoning

Sections 17.2 through 17.4 have already established the ability to determine whether a middle step is worth continuing. Now, we reintroduce the evaluator into the generation process: when a model is in the middle of writing, if the current equation is already incorrect, there is no need to spend thousands of tokens to complete the entire answer; if two candidates share a correct prefix, there is no need to regenerate from scratch twice.

Search during reasoning saves intermediate states and distributes computation among "which path to expand, how many branches to keep, and when to backtrack." This section observes four approaches on the same quadratic equation: how independent sampling wastes prefixes, how beam search fixes the number of paths to keep, how tree of thoughts allows backtracking, and how Monte Carlo Tree Search (MCTS) balances known high-scoring paths with paths that have not been sufficiently explored. Finally, we use code to test when additional search is worth the cost.

<img src="../../chapter20_prm_search/images/inference-search-tree.svg" alt="Inference-time search reuses shared prefixes and scores branches to expand, retain, or stop">

## 1. Why Intermediate Reasoning Needs to Be Reused

Suppose the model has correctly written $(x+2)(x+3)=0$, and the next step is simply to set each factor to zero. Another path, however, calculates the discriminant $25-24$ as $-1$. Independent sampling would generate both paths to the end; with intermediate evaluation, the system can continue expanding the first path and stop the second path early.

Consider a math problem:

```text
Solve $x^2 + 5x + 6 = 0$
```

The model can generate multiple reasoning paths:

```text
Path A: Using the quadratic formula
  → $x = \frac{-5 \pm \sqrt{25 - 24}}{2} = \frac{-5 \pm 1}{2}$
  → $x = -2$ or $x = -3$

Path B: Using factoring
  → $x^2 + 5x + 6 = (x+2)(x+3) = 0$
  → $x = -2$ or $x = -3$

Path C: Trying completing the square
  → $x^2 + 5x = -6$
  → $x^2 + 5x + \frac{25}{4} = \frac{25}{4} - 6 = \frac{1}{4}$
  → $(x + \frac{5}{2})^2 = \frac{1}{4}$
  → $x + \frac{5}{2} = \pm \frac{1}{2}$
  → $x = -2$ or $x = -3$
```

All three paths lead to the correct answer. However, if the model makes a mistake on one path (e.g., path A miscalculates the square root), the result of a single sample will be incorrect.

Best-of-N addresses this issue by generating multiple independent paths and selecting the best one using PRM. However, Best-of-N has limitations:

- **Lacks reuse of path similarities**: If two paths share the same prefix, Best-of-N will generate them independently.
- **Cannot backtrack in the middle**: If a path discovers a mistake halfway through, Best-of-N can only start over from the beginning.
- **Low search efficiency**: N independent samples are equivalent to brute-force enumeration.

Search during reasoning transforms these redundant efforts into an explicit state tree:

- **Shared prefixes**: The same prefix is only computed once.
- **Intermediate evaluation**: Use PRM to evaluate intermediate states, deciding whether to continue or backtrack.
- **Resource allocation**: Allocate search computation to the most promising directions.

The algorithms that follow mainly differ in two decisions: how many nodes to keep after each round, and whether a path that is temporarily pruned can return later.

## 2. How Beam Search and ToT Extend the Reasoning Tree

**Beam Search** adopts the most straightforward rule: at any moment, only maintain $K$ highest-scoring partial inferences. In each round, these nodes are expanded, re-scored using PRM, and the new top $K$ are retained.

### 2.1 Algorithm of Beam Search

```python
def beam_search_thoughts(prompt, model, prm, K=4, expansions=2, max_steps=10):
    # Initial beam: only one empty state
    beams = [{"thought": "", "score": 1.0}]

    for step in range(max_steps):
        # Expand each beam: let the model generate the next inference
        candidates = []
        for beam in beams:
            for _ in range(expansions):
                next_thought = model.generate_next(prompt, beam["thought"])
                score = prm.score(prompt, beam["thought"] + next_thought)
                candidates.append({
                    "thought": beam["thought"] + next_thought,
                    "score": score
                })

        # Select top-K as new beams
        beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:K]

        # If a complete answer is found, stop
        if any(is_complete(b["thought"]) for b in beams):
            break

    return beams[0]["thought"]  # Return the best beam
```

### 2.2 Characteristics of Beam Search

The fixed width makes Beam Search easy to implement, and $K$ paths can be expanded in parallel. The cost also comes from the fixed $K$: simple problems may maintain redundant paths, while difficult problems may prematurely eliminate branches that later prove valuable. Eliminated nodes will not re-enter the beam, so early PRM misjudgments will continue to affect the results.

### 2.3 Applicable Scenarios for Beam Search

Beam Search is a suitable starting point when the steps are clearly defined, the per-step scores are reliable, and only a small number of candidates need to be retained at each level. If errors are only revealed much later, the current scores are difficult to determine early pruning, and the risk of fixed beam becomes larger.

### 2.4 How ToT Preserves Branches and Provides Backtracking Opportunities

[Tree of Thoughts](https://arxiv.org/abs/2305.10601) (Yao et al. 2023) is an extension of Beam Search — it supports **branching, backtracking, and a mix of DFS/BFS**.

#### Core Structure of ToT

```text
                Root
              /      \
            A1        A2
           /  \      /  \
         B1   B2   B3   B4
        / \    |    |   / \
       C1  C2  C3   C4 C5  C6

       Search Algorithm: BFS (Breadth-First Search) or DFS (Depth-First Search)
       Evaluation: Each step is scored using PRM
       Backtracking: Low-scoring nodes are pruned
```

#### Algorithm of ToT

```python
def tree_of_thoughts(prompt, model, prm, max_depth=10, breadth=4):
    # Start DFS from the root
    def dfs(thought, depth):
        if depth >= max_depth:
            return [{"thought": thought, "score": prm.score(prompt, thought)}]

        # Generate N candidate next steps
        candidates = []
        for _ in range(breadth):
            next_thought = model.generate_next(prompt, thought)
            full_thought = thought + next_thought
            score = prm.score(prompt, full_thought)
            candidates.append({"thought": full_thought, "score": score})

        # Sort by score and prune low-scoring ones
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:breadth // 2]  # Prune half

        # Recursively process the retained candidates
        results = []
        for c in candidates:
            results.extend(dfs(c["thought"], depth + 1))

        return results

    return dfs("", 0)
```

#### Characteristics of ToT

ToT allows the system to first break down reasoning into coarser thoughts, and then expand them using BFS, DFS, or a constrained beam search. It can revisit un-deleted intermediate nodes to try new subsequent steps, but it does not inherently use less computation than Best-of-N; the benefits depend on whether the shared prefixes and intermediate scores are truly useful. If each layer fully retains $B$ branches, the number of nodes grows exponentially with depth, so implementations must set constraints on width, depth, or total node budget.

#### Experimental Results of ToT

On the [24 Game](https://arxiv.org/abs/2305.10601) (24-point game) task:

| Method                                     | Success Rate |
| ------------------------------------------ | ------------ |
| Greedy decoding                            | 7.3%         |
| CoT prompting                              | 4.0%         |
| Self-consistency (multi-sampling + voting) | 9.0%         |
| **Tree of Thoughts**                       | **74.0%**    |

In this task and prompt setup, GPT-4 combined with ToT increased the success rate from single-digit percentages to 74%. The intermediate states in the 24-point game are easy to evaluate, making it particularly suitable for search. This level of improvement cannot be directly extrapolated to open-ended tasks without clear states and validation rules.

## 3. How MCTS Uses Verifier Feedback to Select Paths

Beam Search only considers the current score at each level, and once a path is underestimated early on, it is discarded and never revisited. MCTS increases the number of visits, allowing the system to both utilize high-scoring nodes and preserve opportunities for nodes that have not been sufficiently explored.

**Monte Carlo Tree Search (MCTS)** distributes the budget by repeatedly visiting the tree. In LLM reasoning, the model can propose the next step, and then use rollout results, PRM, or an external checker to update node values:

- Evaluate nodes using outcome rewards, PRM, or an external verifier
- Use the model as a policy (recommend the next step)
- Balance exploration and exploitation using the UCB formula

### 3.1 The Four Steps of MCTS

Each iteration performs the following:

1. **Selection**: Starting from the root, use the UCB rule to select child nodes until reaching a leaf.
2. **Expansion**: Generate $N$ children from the leaf.
3. **Simulation**: Roll out a complete candidate solution from a child node.
4. **Backpropagation**: Propagate the rollout reward to the ancestor nodes.

### 3.2 The UCB Formula

When selecting nodes, we need to consider two things: nodes with high average scores are worth further exploitation, and nodes with low visit counts should also be tried. UCB combines these two aspects:

$$\text{UCB}(n) = Q(n) + c \cdot \sqrt{\frac{\ln N(p)}{N(n)}}$$

Where:

- $Q(n)$: The average reward of node $n$ (from PRM)
- $N(n)$: The number of times node $n$ has been visited
- $N(p)$: The number of times the parent node has been visited
- $c$: The exploration constant

The first term $Q(n)$ represents the observed average value. The second term increases with the parent node's visit count $N(p)$, but decreases with the current node's visit count $N(n)$, thus prioritizing the exploration of less-visited child nodes. A larger $c$ makes the search more willing to try new branches, while a smaller $c$ focuses the search on the current high-scoring branches. In practice, we often first visit unexplored child nodes, or add a small smoothing term to the denominator to avoid division by zero when $N(n) = 0$.

For example, if two child nodes have the same average score of 0.6, one has been visited 20 times, and the other only 2 times. The second term will give the node with 2 visits a higher UCB until it accumulates enough evidence. Here, the $Q$ value can come from the final result, PRM, or a combination of both.

Take $c=1.414$ and a parent visited 100 times. Child A has $Q=0.8$ and $N=50$, giving an exploration term of about $0.43$ and UCB $1.23$. Child B has $Q=0.7$ and $N=30$, giving about $0.55$ and UCB $1.25$. Child C has only $Q=0.6$ but $N=5$, giving about $1.36$ and UCB $1.96$. The underexplored child is selected despite its lower observed score.

### 3.3 Characteristics of MCTS

The number of visits allows MCTS to allocate more budget to high-value branches while continuing to explore less-visited branches. The asymptotic properties rely on assumptions such as a finite action space, sufficient exploration, and reliable rewards. In open-text generation, action candidates are truncated by the model, and evaluators can also make errors, so classical guarantees cannot be directly considered as guarantees of correctness. Its main cost includes multiple rollouts, state caching, and value updates.

### 3.4 Representative Works

- **rStar** ( [arXiv:2408.06195](https://arxiv.org/abs/2408.06195) ): MCTS + self-play, used for mathematical reasoning
- **AlphaProof** ( [DeepMind 2024](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) ): AlphaZero-style reinforcement learning, proof search, and Lean verifier
- **RAP** ( [Reasoning via Planning](https://arxiv.org/abs/2305.14992) ): MCTS + LLM as world model

### 3.5 Code Generation Search in AlphaCodium

[AlphaCodium](https://arxiv.org/abs/2401.08500) (2024.01) organizes code generation into an iterative process of "understanding the problem—generating tests—writing a draft—executing—fixing":

- Code tasks can be automatically checked using **existing tests**; new tests generated by the model can only supplement coverage and cannot guarantee complete validation
- Iterative search is used: generate → test → fix → retest

#### The Process of AlphaCodium

```text
1. Problem Understanding: Let the LLM extract key information and generate test cases
2. Preliminary Solution: Generate a candidate solution
3. Iterative Fixing:
   a. Run the test cases
   b. If failed, analyze the error
   c. Let the LLM fix the error
   d. Repeat until all tests pass
4. Output the Final Solution
```

#### Characteristics of AlphaCodium

- Existing tests can directly serve as a result verifier; however, when test coverage is insufficient, errors in the implementation may still be missed
- Iterative (not tree search) — simple and efficient
- The paper reports improvements over direct generation on benchmarks such as CodeContests, with the magnitude of improvement varying depending on the model and evaluation setup

## 4. When is Search Worth the Computation Cost

Search does not improve accuracy for free. Every node expansion requires calling a generative model, and many methods also require calling a PRM, executing a test, or running a proof checker. Whether to adopt search depends on the reliability of intermediate feedback and the cost of a single failure.

Different methods can estimate the cost by considering the "number of generated branches or expansions":

| Method                   | Main Budget Item                                                                              |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| Greedy decoding          | One full generation                                                                           |
| Best-of-N                | $N$ full generations and $N$ result evaluations                                               |
| Beam Search ($K,D$)      | Approximately $K \times D$ node expansions and layer-wise evaluations                         |
| Tree of Thoughts ($B,D$) | Fully expanded to $O(B^D)$, actually controlled by pruning and node budget                    |
| MCTS                     | Number of iterations, number of expansions per iteration, rollout length, and evaluation cost |

If Tree of Thoughts retains all $B$ branches at each layer, the number of nodes grows as $B^D$ when the depth reaches $D$. Therefore, practical systems must implement pruning or set a node budget. MCTS does not expand the entire tree; its computational cost is mainly determined by the number of iterations and the number of expansions per iteration. Both methods involve more state maintenance and step-by-step evaluations compared to independent sampling. Therefore, whether to use search depends on whether the intermediate feedback can offset these additional costs.

Scientific computing, formal proof, and competitive programming typically have executable checkers. When search expands each path and receives relatively reliable feedback, the additional computation is more likely to translate into higher success rates. Without a reliable verifier, search may repeatedly expand along paths with incorrect evaluations.

### 4.1 Search During Training and Inference

It is also necessary to decide whether the search occurs during training or during inference.

**Using Search Results During Training** (e.g., the reinforcement learning loop in AlphaProof):

- Use the search results as training data
- Let the model increase the probability of high-value steps
- During deployment, search can still be performed as needed for the task

**Search During Inference** (e.g., ToT, MCTS):

- Use search during inference to improve performance
- The search budget can be changed without retraining

These two approaches can also be combined:

- Light search during training (to accelerate convergence)
- Search during inference based on the task's difficulty

This aligns with the idea in [Chapter 16: Test-time Compute Scaling](../chapter19_reasoning/test-time-scaling) — where to allocate computational resources is an engineering trade-off.

## Summary

PRM can provide process rewards during training and score partial paths during inference. Search algorithms use these scores to decide which intermediate steps to retain, expand, or discard.

Main methods:

- **Beam Search**: Simple and parallel, suitable for moderate tasks
- **Tree of Thoughts**: Supports backtracking and pruning, suitable for complex tasks
- **MCTS**: Allocates the budget between exploration and exploitation based on visit counts
- **AlphaCodium**: Specialized for code, using unit tests as the verifier

The number of nodes fully expanded in the search tree grows exponentially with depth; Beam Search and MCTS control the cost by limiting the number of retained paths or sampling steps. Therefore, practical systems should choose between Best-of-N, constrained search, or direct generation based on the task's value and the verifier's cost.

[17.6 Parallel Reasoning and Summary Aggregation](./parallel-reasoning-and-summary) continues the discussion of another allocation strategy: generating multiple reasoning paths in parallel and then having the model or verifier exchange information and aggregate the results.
