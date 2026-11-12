# 19.7 Multi-Agent Collaboration

> [19.6 Code Interpreter RL](./industrial-practice) trains a **single** agent to complete programming tasks within a tool invocation loop. However, when the task escalates from "write a function" to "refactor an entire codebase + run tests + write documentation + submit a PR," the single agent's context window, attention bandwidth, and error recovery capabilities are all overwhelmed. **Multi-agent collaboration** (multi-agent collaboration) is a key extension of agentic RL in 2025-2026: breaking a complex task into multiple subtasks for different agents, each focusing on a specific subtask, and coordinating through explicit communication protocols. This section will clarify three things: (1) the fundamental differences between LLM-era multi-agent systems and classical MARL; (2) mainstream collaboration paradigms (Orchestrator-Worker, Debate, Swarm); and (3) RL training methods for multi-agent systems.

## From Classical MARL to LLM-Era Multi-Agent Systems

[Chapter 12, Section 12.2](../chapter14_exploration_marl_hierarchical/marl) discussed classical MARL: CTDE framework, MADDPG, MAPPO. These algorithms deal with **homogeneous** agents learning **Nash equilibrium** in a **fixed** environment — for example, multiple robots pursuing or evading, or multi-agent StarCraft micro-management. LLM-era multi-agent systems are completely different:

| Dimension           | Classical MARL                 | LLM-Era Multi-Agent Systems                                  |
| ------------------- | ------------------------------ | ------------------------------------------------------------ |
| Number of Agents    | 2-20 agents                    | 2-10 agents (constrained by cost)                            |
| Agent Heterogeneity | Homogeneous (same policy)      | Highly heterogeneous (different roles: planner/coder/tester) |
| Communication       | Implicit via environment state | Explicit natural language communication                      |
| Task Type           | Zero-sum / cooperative game    | Long-term software tasks (PR, research, operations)          |
| Training Goal       | Nash equilibrium / team reward | Task completion rate (end-to-end verifiable)                 |
| Training Algorithm  | MAPPO / QMIX                   | GRPO + multi-trajectory reward allocation                    |

The most critical difference lies in **heterogeneity** and **explicit communication**. In classical MARL, all agents share the same policy $\pi_\theta(a \mid s)$, and they influence each other only through the environment state. In LLM multi-agent systems, each agent has a different system prompt (e.g., "You are a code reviewer," "You are a test engineer"), and they coordinate with each other through **natural language messages**. This makes the communication bandwidth explode — a single coordination might consume several thousand tokens — but it also makes the collaboration semantically richer.

## Three Main Architectures

### Orchestrator-Worker Pattern

**The simplest and most commonly used** collaboration paradigm. An **Orchestrator agent** (the coordinator) is responsible for task decomposition, subtask assignment, and result aggregation; multiple **Worker agents** (workers) each perform their own subtasks.

```
[User: "Fix GitHub Issue #123"]

    ↓
[Orchestrator]
    ├── 1. Read issue → Call Worker-A: "Locate the bug file"
    ├── 2. Worker-A returns file.py:42
    ├── 3. Call Worker-B: "Write a fix patch for file.py:42"
    ├── 4. Worker-B returns patch.diff
    ├── 5. Call Worker-C: "Run tests + write changelog"
    └── 6. Aggregate → Submit PR
```

Anthropic's internal research published in 2025 measured that the **Orchestrator-Worker pattern achieves a 90.2% speedup and an 18–32% increase in success rate on SWE-bench Verified** compared to a single agent. The key reason is not "two agents are stronger than one," but rather **task decomposition prevents the context window from being overwhelmed** — a single agent handling the entire PR process must focus on "finding the file / writing code / running tests / writing documentation" all at once; decomposition allows each worker to focus on a single task.

The policy of the Orchestrator can be formalized as a hierarchical MDP:

$$\pi_\theta^{\text{orch}}(w_t, m_t \mid q, h_{1:t})$$

where $w_t \in \{1, \ldots, K\}$ is the worker to which the $t$-th step is assigned, $m_t$ is the message sent to that worker, and $h_{1:t}$ is the history of interactions.

### Debate Mode

Multiple agents **debate with each other** to converge on a more reliable answer. The theoretical foundation of this paradigm is AI Safety via Debate by Anthropic (Irving et al., 2018); DeepMind's 2024 Scaling Inference paper validates the effectiveness of LLM Debate on math problems.

The MDP for Debate:

$$\pi_\theta(a_t^{(i)} \mid q, a_{1:t-1}^{(1)}, a_{1:t-1}^{(2)}, \ldots, a_{1:t-1}^{(K)})$$

The $i$-th agent observes the historical responses of all other agents and outputs its response for this round, $a_t^{(i)}$. The final answer is selected by an **external judge** (a human or another LLM).

The training objective for Debate is **truth convergence**: making the honest agent win after multiple rounds of debate. This is significantly more challenging than Orchestrator-Worker training—requiring **adversarial training** (adversarial training): intentionally training a "lying agent" and then training an "honest agent" to defeat it.

### Agent Swarm Mode

**Kimi K2.5 (2026.01)** and **Step 3.7 Flash Advisor Mode** push the multi-agent paradigm to its extreme: **dozens of heterogeneous agents** are online simultaneously, dynamically scheduled by a meta-controller. This is essentially an A2A (Agent-to-Agent) protocol plus an RL scheduler.

Key differences of Swarm:

- **Agent pool** rather than a fixed set of workers: the meta-controller dynamically selects agents from the pool based on the task
- **A2A communication protocol**: agents communicate through structured protocols (e.g., Anthropic A2A, OpenAI Function Calls)
- **Credit assignment across agents**: which agent contributed the most? Requires SHAP or attention attribution

Formalization:

$$\pi_\theta^{\text{swarm}}(a_t \mid q, \text{pool}, h_{1:t})$$

where $a_t = (\text{select-agent}, \text{message}, \text{route-to})$.

::: warning Cost Explosion in Swarm Mode
Swarm mode consumes 10–50 times more tokens than a single agent. Kimi K2.5 paper reports: processing a SWE-bench task on average consumes 280K tokens (single agent baseline is 18K). This is why industrial deployment in 2026 still primarily uses Orchestrator-Worker—cost-controlled, with performance close to Swarm.
:::

## Reinforcement Learning for Multi-Agent Systems

### From Team Rewards to Individual Attribution

The most challenging issue in multi-agent reinforcement learning (RL) is **credit assignment** (credit assignment). When a task is successfully completed, who should receive the reward?

**Approach 1: Team-Average Reward Distribution**

All agents receive the same reward $ r / K $, where $ K $ is the number of agents:

$$ R^{(i)} = \frac{1}{K} \sum_t r_t $$

This approach is simple but can lead to the **free-rider problem**: a worker who does not contribute may still receive a reward if the team succeeds.

**Approach 2: Shapley Value Attribution**

The Shapley value from game theory measures the marginal contribution of each agent:

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}}
\frac{|S|!(N-|S|-1)!}{N!}
\left[v(S\cup\{i\})-v(S)\right]
$$

Here, $ v(S) $ is the success probability of the subset $ S $ completing the task. The term $ N! $ requires **counterfactual evaluation** — removing agent $ i $ from the team and checking whether the task can still be completed. Although this method is the fairest, it is computationally expensive.

**Approach 3: Heuristic Attribution via Orchestrator**

An orchestrator outputs weights $ w_i $ in the final reward, and agent $ i $ receives a reward of $ w_i \cdot R $:

$$ R^{(i)} = w_i \cdot R^{\text{team}}, \quad \sum_i w_i = 1 $$

This is the approach used in Kimi K2.5 — it is cost-effective and interpretable, but relies on the orchestrator's ability to attribute rewards (essentially, training the orchestrator to learn attribution through RL).

### Multi-Trace GRPO

Standard GRPO samples $G$ trajectories for the same prompt, and normalizes the advantage:

$$\hat{A}_j = \frac{R_j - \text{mean}(R_{1:G})}{\text{std}(R_{1:G})}$$

The multi-agent version is called **Multi-Agent GRPO (MA-GRPO)**: each trajectory is not generated by a single agent, but is instead **generated collaboratively by a team**. $G$ trajectories = $G$ team collaborations.

Key engineering implementation:

```python
def ma_grpo_step(prompts, team_size):
    # For each prompt, sample G team collaboration trajectories
    trajectories = []
    for prompt in prompts:
        for g in range(G):
            # 1. Orchestrator decomposes the task
            subtasks = orchestrator.decompose(prompt)
            # 2. Workers execute in parallel
            worker_outputs = [workers[i](subtasks[i]) for i in range(team_size)]
            # 3. Orchestrator aggregates the results
            final_answer = orchestrator.aggregate(worker_outputs)
            # 4. Calculate reward
            r = verifier(prompt, final_answer)
            trajectories.append({
                'prompt': prompt,
                'final': final_answer,
                'reward': r,
                'orch_logp': orchestrator.logp(...),
                'worker_logp': [w.logp(...) for w in workers]
            })

    # GRPO advantage normalization
    rewards = [t['reward'] for t in trajectories]
    advantages = (rewards - mean(rewards)) / (std(rewards) + eps)

    # Calculate loss for orchestrator and workers separately
    orch_loss = -mean(a * t['orch_logp'] for a, t in zip(advantages, trajectories))
    worker_losses = [-mean(a * lp for a, lp in zip(advantages, t['worker_logp']))
                     for t in trajectories]

    total_loss = orch_loss + sum(worker_losses)
    return total_loss
```

## Engineering Details in the Kimi K2.5 Training Framework

### Agent Swarm in Kimi K2.5

Kimi K2.5 (2026.01, arXiv:2602.02276) is the first industrial model to publicly disclose the training details of the Swarm mode:

- **Agent Pool**: 32 heterogeneous agents (coder, tester, planner, reviewer, debugger, etc.)
- **A2A Protocol**: Structured messages based on JSON Schema
- **Training Data**: 12M team collaboration trajectories covering SWE / DeepResearch / Customer Service
- **Reward**: Verified tasks use RLVR, open tasks use LLM-as-Judge
- **Scheduling RL**: Meta-controller trained with PPO, aiming to minimize token consumption and maximize success rate

Reported metrics:

- SWE-bench Verified: 68.3% (single agent baseline 49.1%)
- BrowseComp: 72.1% (single agent 51.4%)
- Average token consumption: 280K (baseline 18K, 15.6×)

### Step 3.7 Flash Advisor Mode

Step 3.7 Flash's Advisor Mode follows a different design: a **Conservative Orchestrator-Worker** architecture with a dedicated **Advisor agent** for reflection and error correction.

```
[Orchestrator] → [Worker-A: code] → [Advisor: review] → [Orchestrator] → [Worker-B: test]
```

The Advisor does not execute tasks directly; it only reviews Worker outputs. After seeing the Advisor's comments, the Orchestrator decides whether the work should be revised. This dumbbell-shaped collaboration pattern costs only about one fifth as much as Swarm while achieving comparable results.

Reported Metrics:

- SWE-bench Verified: 62.4% (between single agent and Swarm)
- Average token consumption: 52K (about 1/5 of Swarm)

## Correspondence with [Chapter 26: Self-Play](../chapter32_selfplay/self-play-outlook/)

Multi-agent collaboration has a special form: **multiple agents are different instances of the same policy**, competing with each other. This is the core idea of AlphaGo / AlphaZero / Constitutional AI Self-Critique. See [Chapter 26: Self-Play](../chapter32_selfplay/self-play-outlook/) for details.

Key Differences:

- **Multi-agent Collaboration**: Heterogeneous agents, explicit communication, team tasks
- **Self-Play**: Homogeneous agents (same policy), interacting through environment, zero-sum or cooperative game

Both approaches are beginning to converge in the LLM era — for example, Constitutional AI's Self-Critique can be viewed as "two agents collaborating (one generating, one criticizing), but using the same policy."

## Failure Patterns in Multi-Agent Collaboration

Having covered the theory, let us return to engineering — several typical failure patterns observed in multi-agent systems in production environments.

### Amplification of Communication Errors

In a single-agent system, an error only affects the agent itself. However, in a multi-agent system, an error from one agent becomes the input for other agents, leading to exponential amplification of the error.

```
Worker-A (erroneous) → Outputs "bug in file_X.py:42"
    ↓
Orchestrator assigns Worker-B to fix file_X.py:42
    ↓
Worker-B fixes a non-existent bug, introducing a new bug
    ↓
Orchestrator assigns Worker-C to test, discovers the new bug
    ↓
...and so on in an infinite loop...
```

Internal data from Anthropic: the "chain of errors" rate in multi-agent systems is 2.7 times that of single-agent systems.

**Countermeasure**: Each agent should output with an associated **confidence score**; low-confidence outputs trigger the Orchestrator to perform a secondary verification.

### Groupthink

After multiple agents influence each other, the system may converge to a wrong consensus — especially in Debate mode. If one agent uses an incorrect premise, other agents may accept it based on "politeness" or "conformity."

**Countermeasure**: Introduce a "devil's advocate" agent — an agent specifically tasked with challenging the mainstream view. Anthropic's Debate system enforces that at least one agent must hold a dissenting position.

### Free Rider

When team rewards are averaged, a worker may learn to "contribute minimally"—only outputting responses that appear reasonable but lack substantive content. The team can still succeed.

**Solutions**:

- **Shapley value attribution** (computationally expensive)
- **Orchestrator explicit scoring** (depends on the capability of the Orchestrator)
- **Individual worker evaluation during testing** (most rigorous but most costly)

### Context Redundancy

In a multi-agent system, each worker needs to "understand the global context" to function. However, the global information (task description, progress made) is repeated in the prompt of each worker—leading to a token cost explosion.

```
Task: "Fix GitHub Issue #123"
Context (seen by each worker):
  - Full issue description: 500 tokens
  - Relevant code files: 2000 tokens
  - Progress from existing workers: 1500 tokens
Total: 4000 tokens × 5 workers = 20K tokens for context alone
```

**Solutions**: Hierarchical context—Orchestrator maintains the full context, and workers only see a condensed summary.

## Open-Source Frameworks and Tools

To reproduce multi-agent RL training, the following open-source tools are available:

| Framework        | Source      | Features                                                             |
| ---------------- | ----------- | -------------------------------------------------------------------- |
| **AutoGen**      | Microsoft   | Multi-agent dialogue framework, supports various collaboration modes |
| **CrewAI**       | CrewAI Inc. | Role-based agents (planner/researcher/writer)                        |
| **MetaGPT**      | DeepWisdom  | Multi-agent system driven by SOP (Standard Operating Procedure)      |
| **LangGraph**    | LangChain   | Multi-agent orchestration based on state graphs                      |
| **Agency Swarm** | VRSEN       | Open-source implementation of "agent swarm" in the literal sense     |

However, these frameworks are mostly **inference-time** tools—they define how agents converse, but do not involve RL training. **Very few open-source frameworks are capable of training multi-agent systems using RL**, mainly including:

- **OpenRLHF** (ByteDance): Supports multi-agent PPO/GRPO with customizable reward allocation
- **verl** (ByteDance): A distributed RL framework supporting joint training of heterogeneous agents
- **OpenResearcher**: Specialized for Deep Research, includes a simple Orchestrator-Worker

Industrial-grade Swarm training (e.g., Kimi K2.5) currently has **no complete open-source implementation**—this remains a core barrier for top Chinese and American labs.

## Summary of This Section

| Paradigm            | Communication Style  | Training Objective            | Representative System | Cost Factor |
| ------------------- | -------------------- | ----------------------------- | --------------------- | ----------- |
| Single Agent        | N/A                  | Task completion rate          | Baseline              | 1×          |
| Orchestrator-Worker | One-way dispatch     | Team reward                   | Anthropic internal    | 3–5×        |
| Debate              | Bidirectional debate | Truth convergence             | Anthropic / DeepMind  | 5–10×       |
| Agent Swarm         | Fully connected A2A  | Team + individual attribution | Kimi K2.5             | 15–30×      |

Core challenges of RL training for LLM-era multi-agent systems: **credit assignment** and **token cost**. The former determines whether training can converge, while the latter determines commercial viability. In 2026, the mainstream approach is Orchestrator-Worker with explicit attribution, while Swarm remains in the research stage.

The next chapter [Chapter 20: Code Agent Reinforcement Learning](../chapter23_rl_based_swe/swe-bench-and-rlvr) applies this collaborative framework to SWE tasks—you will see how SWE-Agent trains a single-agent code intelligence agent using Orchestrator-Worker, and how DeepSWE trains multi-agent collaboration using self-play.
