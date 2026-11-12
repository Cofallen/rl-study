#!/usr/bin/env python3
"""A tiny offline Deep Research RL benchmark.

The point of this script is not to train a language model. It isolates the
research policy that decides which query to issue and which evidence page to
open, then optimizes that small policy with REINFORCE. The same environment and
reward can later be reused around a small LLM with LoRA/GRPO.

Run:
    python docs/chapter10_agentic_rl/code/deep_research_rl_benchmark.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re
from typing import Callable


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def overlap(a: str, b: str) -> float:
    left = tokens(a)
    right = tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    answer: str


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    gold_answer: str
    support_doc_id: str
    query_options: tuple[str, ...]


DOCS = [
    Document(
        "deepresearcher-paper",
        "DeepResearcher reinforcement learning framework",
        (
            "DeepResearcher trains an autonomous research agent directly in a "
            "real web environment. The paper reports up to a 28.9 point "
            "improvement over prompt-engineering baselines and observes "
            "emergent planning, cross-validation, and self-reflection."
        ),
        "DeepResearcher",
    ),
    Document(
        "browsecomp-paper",
        "BrowseComp hard browsing benchmark",
        (
            "BrowseComp is an OpenAI benchmark for hard information-seeking "
            "tasks. It contains 1,266 challenging problems designed so that "
            "answers require persistent web browsing and verification."
        ),
        "BrowseComp",
    ),
    Document(
        "openresearcher-paper",
        "OpenResearcher synthetic trajectory pipeline",
        (
            "OpenResearcher builds a fully open long-horizon deep research "
            "trajectory synthesis pipeline. It uses an offline 15M document "
            "corpus and produces more than 97K trajectories."
        ),
        "OpenResearcher",
    ),
    Document(
        "openresearcher-tools",
        "OpenResearcher browser primitives",
        (
            "OpenResearcher simulates browser interaction with three "
            "reproducible primitives: search, open, and find. The environment "
            "is stable enough for algorithm iteration without web API noise."
        ),
        "search, open, and find",
    ),
    Document(
        "tongyi-paper",
        "Tongyi DeepResearch model scale",
        (
            "Tongyi DeepResearch uses a 30.5B-parameter mixture-of-experts "
            "architecture, while only 3.3B parameters are activated per token. "
            "The training recipe combines agentic mid-training and post-training."
        ),
        "Tongyi DeepResearch",
    ),
    Document(
        "sfr-paper",
        "SFR DeepResearch autonomous single agent",
        (
            "SFR-DeepResearch studies autonomous single-agent deep research. "
            "The reported SFR-DR-20B system reaches 28.7 percent on Humanity's "
            "Last Exam under its evaluation setting."
        ),
        "SFR-DR-20B",
    ),
    Document(
        "web-shepherd-paper",
        "Web-Shepherd process reward model",
        (
            "Web-Shepherd trains process reward models for web agents. It "
            "shows that step-level supervision can improve web-agent behavior "
            "by 10.9 percentage points on its benchmark setting."
        ),
        "Web-Shepherd",
    ),
    Document(
        "rstar2-paper",
        "rStar2 Agent GRPO training",
        (
            "rStar2-Agent applies an efficient GRPO-style training recipe to "
            "agentic reasoning. It demonstrates that careful reinforcement "
            "learning can let a smaller model compete with much larger systems."
        ),
        "rStar2-Agent",
    ),
    Document(
        "carr-paper",
        "Citation-aware rubric rewards",
        (
            "CaRR, Chaining the Evidence, scores deep search agents with "
            "citation-aware rubric rewards. It checks whether cited evidence "
            "supports the answer instead of merely counting citations."
        ),
        "CaRR",
    ),
    Document(
        "longwriter-paper",
        "LongWriter Zero report generation",
        (
            "LongWriter-Zero shows that reinforcement learning with composite "
            "rewards can unlock ultra-long generation behavior without relying "
            "only on supervised long-form demonstrations."
        ),
        "LongWriter-Zero",
    ),
    Document(
        "distractor-browser",
        "Browser automation benchmark notes",
        (
            "General web automation benchmarks measure clicking, typing, and "
            "page navigation. They do not necessarily evaluate citation "
            "faithfulness or long research synthesis."
        ),
        "web automation",
    ),
    Document(
        "distractor-rag",
        "Classic RAG pipeline notes",
        (
            "A classic RAG pipeline usually retrieves a small set of passages "
            "once, then asks a model to answer. It lacks the multi-turn search "
            "and verification loop used by deep research agents."
        ),
        "classic RAG",
    ),
]


TASKS = [
    Task(
        "t1",
        "Which RL framework for deep research reports up to a 28.9 point improvement over prompt-engineering baselines?",
        "DeepResearcher",
        "deepresearcher-paper",
        (
            "deep research RL 28.9 point improvement prompt engineering",
            "deep research agent benchmark",
            "classic RAG prompt engineering",
        ),
    ),
    Task(
        "t2",
        "Which OpenAI benchmark has 1,266 hard browsing problems?",
        "BrowseComp",
        "browsecomp-paper",
        (
            "OpenAI 1266 hard browsing problems benchmark",
            "web automation benchmark notes",
            "deep research evaluation",
        ),
    ),
    Task(
        "t3",
        "Which open pipeline uses a 15M document corpus and produces more than 97K trajectories?",
        "OpenResearcher",
        "openresearcher-paper",
        (
            "15M document corpus 97K trajectories open pipeline",
            "synthetic data for agents",
            "long text generation reinforcement learning",
        ),
    ),
    Task(
        "t4",
        "What are the three reproducible browser primitives used by OpenResearcher?",
        "search, open, and find",
        "openresearcher-tools",
        (
            "OpenResearcher reproducible primitives search open find",
            "browser interaction tools",
            "citation-aware rewards primitives",
        ),
    ),
    Task(
        "t5",
        "Which deep research model has 30.5B total parameters and 3.3B activated parameters?",
        "Tongyi DeepResearch",
        "tongyi-paper",
        (
            "30.5B 3.3B activated parameters deep research model",
            "mixture of experts training recipe",
            "small model deep research benchmark",
        ),
    ),
    Task(
        "t6",
        "Which autonomous single-agent system reports 28.7 percent on Humanity's Last Exam?",
        "SFR-DR-20B",
        "sfr-paper",
        (
            "28.7 percent Humanity's Last Exam autonomous single agent",
            "single agent deep research",
            "BrowseComp hard browsing benchmark",
        ),
    ),
    Task(
        "t7",
        "Which method demonstrates 10.9 percentage point gains from step-level web-agent supervision?",
        "Web-Shepherd",
        "web-shepherd-paper",
        (
            "10.9 percentage point step-level supervision web agents",
            "process reward model web benchmark",
            "citation reward deep search",
        ),
    ),
    Task(
        "t8",
        "Which agentic reasoning report uses a GRPO-style recipe to make smaller models competitive?",
        "rStar2-Agent",
        "rstar2-paper",
        (
            "GRPO style recipe smaller model competitive agentic reasoning",
            "deep research trajectory synthesis",
            "classic RAG multi turn loop",
        ),
    ),
    Task(
        "t9",
        "Which reward method checks whether cited evidence supports an answer instead of counting citations?",
        "CaRR",
        "carr-paper",
        (
            "citation-aware rubric rewards cited evidence supports answer",
            "citation count benchmark",
            "web automation clicking benchmark",
        ),
    ),
    Task(
        "t10",
        "Which RL writing system shows composite rewards can unlock ultra-long generation?",
        "LongWriter-Zero",
        "longwriter-paper",
        (
            "composite rewards ultra long generation reinforcement learning",
            "long form report generation",
            "OpenResearcher browser primitives",
        ),
    ),
]


class LinearSoftmaxPolicy:
    """Shared feature policy for query selection and evidence selection."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.query_weights = {
            "bias": 0.0,
            "question_overlap": 0.0,
            "specificity": 0.0,
            "number_overlap": 0.0,
        }
        self.doc_weights = {
            "bias": 0.0,
            "rank_bonus": 0.0,
            "query_title_overlap": 0.0,
            "question_text_overlap": 0.0,
        }

    def score(self, weights: dict[str, float], features: dict[str, float]) -> float:
        return sum(weights[name] * features[name] for name in weights)

    def distribution(
        self, weights: dict[str, float], feature_list: list[dict[str, float]], temperature: float
    ) -> list[float]:
        logits = [self.score(weights, features) / temperature for features in feature_list]
        max_logit = max(logits)
        exp_logits = [math.exp(logit - max_logit) for logit in logits]
        total = sum(exp_logits)
        return [value / total for value in exp_logits]

    def choose(
        self,
        weights: dict[str, float],
        feature_list: list[dict[str, float]],
        temperature: float,
        greedy: bool,
    ) -> tuple[int, list[float]]:
        probs = self.distribution(weights, feature_list, temperature)
        if greedy:
            return max(range(len(probs)), key=lambda idx: probs[idx]), probs

        draw = self.rng.random()
        running = 0.0
        for idx, prob in enumerate(probs):
            running += prob
            if draw <= running:
                return idx, probs
        return len(probs) - 1, probs

    def update(
        self,
        weights: dict[str, float],
        feature_list: list[dict[str, float]],
        chosen_idx: int,
        probs: list[float],
        advantage: float,
        lr: float,
    ) -> None:
        expected = {name: 0.0 for name in weights}
        for prob, features in zip(probs, feature_list):
            for name in weights:
                expected[name] += prob * features[name]

        chosen = feature_list[chosen_idx]
        for name in weights:
            weights[name] += lr * advantage * (chosen[name] - expected[name])


def query_features(question: str, query: str) -> dict[str, float]:
    question_numbers = {token for token in tokens(question) if token.isdigit()}
    query_numbers = {token for token in tokens(query) if token.isdigit()}
    return {
        "bias": 1.0,
        "question_overlap": overlap(question, query),
        "specificity": min(len(tokens(query)) / 8.0, 1.0),
        "number_overlap": 1.0 if question_numbers & query_numbers else 0.0,
    }


def doc_features(question: str, query: str, doc: Document, rank: int) -> dict[str, float]:
    return {
        "bias": 1.0,
        "rank_bonus": 1.0 / (rank + 1),
        "query_title_overlap": overlap(query, doc.title),
        "question_text_overlap": overlap(question, doc.text),
    }


def search(query: str, rng: random.Random, limit: int = 4) -> list[Document]:
    scored = []
    for doc in DOCS:
        score = 1.6 * overlap(query, doc.title) + overlap(query, doc.text)
        score += rng.uniform(-0.015, 0.015)
        scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:limit]]


def reward(task: Task, answer: str, doc_id: str) -> float:
    exact = float(answer.lower() == task.gold_answer.lower())
    citation = float(doc_id == task.support_doc_id)
    return 0.75 * exact + 0.25 * citation


def run_episode(
    policy: LinearSoftmaxPolicy,
    task: Task,
    rng: random.Random,
    temperature: float = 1.0,
    greedy: bool = False,
) -> tuple[float, dict[str, object]]:
    q_features = [query_features(task.question, query) for query in task.query_options]
    q_idx, q_probs = policy.choose(policy.query_weights, q_features, temperature, greedy)
    query = task.query_options[q_idx]

    results = search(query, rng)
    d_features = [doc_features(task.question, query, doc, rank) for rank, doc in enumerate(results)]
    d_idx, d_probs = policy.choose(policy.doc_weights, d_features, temperature, greedy)
    doc = results[d_idx]

    score = reward(task, doc.answer, doc.doc_id)
    trace = {
        "query_idx": q_idx,
        "query_probs": q_probs,
        "query_features": q_features,
        "doc_idx": d_idx,
        "doc_probs": d_probs,
        "doc_features": d_features,
        "query": query,
        "doc": doc,
        "answer": doc.answer,
    }
    return score, trace


def evaluate(
    policy: LinearSoftmaxPolicy,
    tasks: list[Task],
    rng_factory: Callable[[], random.Random],
    greedy: bool = True,
) -> dict[str, float]:
    rewards = []
    exact = 0
    citations = 0
    for task in tasks:
        score, trace = run_episode(policy, task, rng_factory(), temperature=0.7, greedy=greedy)
        doc = trace["doc"]
        rewards.append(score)
        exact += int(str(trace["answer"]).lower() == task.gold_answer.lower())
        citations += int(isinstance(doc, Document) and doc.doc_id == task.support_doc_id)
    count = len(tasks)
    return {
        "reward": sum(rewards) / count,
        "answer_exact": exact / count,
        "citation_support": citations / count,
    }


def train(policy: LinearSoftmaxPolicy, tasks: list[Task], epochs: int = 220) -> list[float]:
    baseline = 0.25
    history = []
    for epoch in range(epochs):
        shuffled = tasks[:]
        policy.rng.shuffle(shuffled)
        epoch_rewards = []
        temperature = max(0.45, 1.15 - epoch / 260.0)

        for task in shuffled:
            score, trace = run_episode(policy, task, policy.rng, temperature=temperature)
            baseline = 0.92 * baseline + 0.08 * score
            advantage = score - baseline
            epoch_rewards.append(score)

            policy.update(
                policy.query_weights,
                trace["query_features"],
                int(trace["query_idx"]),
                trace["query_probs"],
                advantage,
                lr=0.24,
            )
            policy.update(
                policy.doc_weights,
                trace["doc_features"],
                int(trace["doc_idx"]),
                trace["doc_probs"],
                advantage,
                lr=0.18,
            )

        history.append(sum(epoch_rewards) / len(epoch_rewards))
    return history


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    print(f"  reward             : {metrics['reward']:.3f}")
    print(f"  answer_exact       : {metrics['answer_exact']:.3f}")
    print(f"  citation_support   : {metrics['citation_support']:.3f}")


def main() -> None:
    train_tasks = TASKS[:7]
    test_tasks = TASKS[7:]
    seed = 7

    before_policy = LinearSoftmaxPolicy(random.Random(seed))
    before = evaluate(before_policy, test_tasks, lambda: random.Random(seed), greedy=False)
    print_metrics("Before RL on held-out benchmark", before)

    policy = LinearSoftmaxPolicy(random.Random(seed))
    history = train(policy, train_tasks)
    after = evaluate(policy, test_tasks, lambda: random.Random(seed + 1), greedy=True)
    print_metrics("\nAfter RL on held-out benchmark", after)

    print("\nTraining reward checkpoints")
    for idx in [0, 9, 49, 99, 149, 219]:
        print(f"  epoch {idx + 1:>3}: {history[idx]:.3f}")

    print("\nLearned query weights")
    for name, value in policy.query_weights.items():
        print(f"  {name:<18} {value:+.3f}")

    print("\nExample held-out trace")
    task = test_tasks[0]
    _, trace = run_episode(policy, task, random.Random(seed + 2), temperature=0.7, greedy=True)
    doc = trace["doc"]
    assert isinstance(doc, Document)
    print(f"  question : {task.question}")
    print(f"  query    : {trace['query']}")
    print(f"  open     : {doc.title}")
    print(f"  answer   : {trace['answer']}")
    print(f"  reward   : {reward(task, str(trace['answer']), doc.doc_id):.3f}")


if __name__ == "__main__":
    main()
