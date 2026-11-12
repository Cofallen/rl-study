---
title: 15.2 DeepSeek-R1-Zero and DAPO
---

# 15.2 DeepSeek-R1-Zero and DAPO

In the previous section, we saw how GRPO replaces the Critic with within-group normalization. This section zooms out to two breakthroughs that reshaped RL for LLMs in 2025: DeepSeek-R1-Zero showed that pure RL can work without any SFT, and DAPO pushed GRPO's engineering efficiency further with four targeted improvements.

## DeepSeek-R1-Zero

Before DeepSeek-R1, the alignment pipeline had an almost unquestioned rule: **SFT first, RL second**. The reasoning was straightforward. A base model only knows how to continue text; it does not know how to answer questions. If you apply RL directly to a base model, the outputs mix languages, ignore format, and look like noise. The reward model has nothing meaningful to score, so RL training has no stable starting point.

In January 2025, the DeepSeek team broke this assumption. They found that **in domains with rule-verifiable rewards—math answer matching, code compilation—SFT is not necessary for cold start**. A base model trained directly with large-scale GRPO can organize its latent knowledge into usable reasoning strategies. This produced DeepSeek-R1-Zero: a pure RL model trained with no SFT data at all.

Why does this work? When the reward is a binary check—correct or incorrect—the model does not need to first learn "how to answer." It only needs to find output patterns that receive high scores through trial and error. Even if the initial outputs are messy, as long as the model occasionally produces a correct answer and receives reward, RL reinforces that path. After enough steps, the model discovers a coherent reasoning format on its own.

### Emergence and the Aha Moment

The most surprising finding during R1-Zero training was **emergent behavior**. Without any human demonstrations, the model independently developed:

- **Chain-of-thought reasoning**: the model shifted from "give the answer directly" to "analyze the problem, write formulas, and compute step by step," with no explicit instruction to do so.
- **Self-reflection**: when an answer was wrong, the model learned to go back, inspect its reasoning, find mistakes, and correct them.
- **Strategy switching**: for different problem types, the model automatically selected different solving approaches.

These abilities were not manually designed. They emerged because the model discovered that they produce higher rule-based rewards. The DeepSeek team called this the **"Aha Moment"**: at a certain training stage, the model suddenly seemed to "get it" and began showing reasoning abilities it had never displayed before.

The emergence follows a clear timeline:

- **Early training** (0–100 steps): outputs are short and messy, often jumping directly to an incorrect numeric answer.
- **Middle training** (100–500 steps): simple calculation steps appear, but errors are frequent.
- **Aha moment** (~500–1000 steps): the model starts checking its own calculations, producing behaviors like "wait, let me recalculate."
- **Late training** (1000+ steps): a stable "analyze → compute → verify" pattern emerges.

This raises a deeper question: **where does the reasoning ability come from?** The likely answer is that pretraining already supplies the raw materials—logic, mathematics, language knowledge—and RL merely organizes them into usable strategies. This also explains why 1-shot RLVR works: the model already has reasoning ability, and RL only activates it.

### Open-Source Reproduction: SimpleRL-reason

A natural follow-up question: is the R1-Zero phenomenon exclusive to DeepSeek's large-scale systems? With open-source base models, smaller data, and public frameworks, can we still observe "RL-activated reasoning without SFT"?

[SimpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason) and the follow-up paper [SimpleRL-Zoo](https://arxiv.org/abs/2503.18892) answer this directly. They do not propose a new algorithm. Instead, they shrink R1-Zero's hypothesis into a reproducible setting: start from a base model, skip SFT, skip the reward model, and use only verifiable math problems with rule-based rewards.

The training loop:

```text
Math problem x
→ base model generates solution y
→ extract final answer from y
→ compare with ground truth
→ get 0/1 rule reward
→ update model with PPO / RL
```

The spirit is the same as R1-Zero: rewards come from objective answer verification, not from human preference or a reward model. The early SimpleRL-reason implementation used OpenRLHF with Ray for distributed scheduling and vLLM for efficient sampling. SimpleRL-Zoo extended the experiments to multiple model families and sizes, including Llama, Mistral, DeepSeek-Math, Qwen2.5-Math, and Qwen2.5 at different scales.

The pedagogical value of this case is that it decomposes "is pure RL feasible?" into concrete questions:

| Question                                              | What SimpleRL-Zoo reveals                                                       |
| ----------------------------------------------------- | ------------------------------------------------------------------------------- |
| Does the base model choice matter?                    | Different families have very different zero-RL starting points and ceilings.    |
| Is more data always better?                           | Small amounts of high-quality, verifiable math data still produce clear signal. |
| Does longer chain-of-thought mean stronger reasoning? | Response length, accuracy, and self-verification must be examined separately.   |
| Is a "simple recipe" really simple?                   | The reward is simple, but distributed sampling and length control are not.      |

SimpleRL-reason is best understood as an open-source reference experiment for R1-Zero, not a separate contribution. It confirms that zero RL is not just a slogan and not exclusive to closed-source systems. As long as the base model has latent ability and the task provides stable verifiable rewards, RL can organize those abilities into more reliable strategies.

The boundaries should also be clear. "Simple" refers to the training signal and recipe, not the hardware cost. Public reproductions still require multi-GPU training, parallel rollout, and standard evaluation benchmarks. It proves that the R1-Zero idea can be open-sourced and studied experimentally—not that a single answer-matching function on a personal laptop reproduces full reasoning emergence.

### R1-Zero's Limits and the Engineering Compromise

R1-Zero proved the feasibility of pure RL, but it had an obvious weakness: **poor language quality**. Without SFT, the model's answers often mixed languages, used messy formatting, and were hard to read. The reasoning was strong, but the presentation looked like a brilliant student who cannot express ideas clearly.

The final released DeepSeek-R1 therefore adopted a multi-stage compromise:

1. **Cold start**: a small amount of high-quality SFT data teaches the model a basic output format.
2. **Large-scale GRPO**: strengthens reasoning ability. This is the core stage.
3. **Rejection sampling**: filters high-quality data from the GRPO-trained model.
4. **SFT fine-tuning**: uses the filtered data to further improve format and language quality.
5. **Second RL stage**: combines an RM and GRPO for final alignment.

## DAPO

GRPO proved that RL can work without a Critic, but it still has engineering pain points. DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) addresses them directly and was accepted as a NeurIPS 2025 poster.

### DAPO's Four Improvements

| Improvement                 | GRPO's problem                                                                   | DAPO's solution                                                             | Effect                          |
| --------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------- |
| **Clip-Higher**             | Symmetric upper/lower clipping over-suppresses low-probability actions.          | Decouple clipping ranges and give low-probability actions more upward room. | Better exploration              |
| **Dynamic sampling**        | All prompts participate in training, wasting compute on solved problems.         | Filter out prompts the model already solves.                                | 2–3x better training efficiency |
| **Token-level loss**        | Sequence-level reward normalization ignores differences between tokens.          | Token-level policy gradients for finer credit assignment.                   | Better long-sequence training   |
| **Overlong Reward Shaping** | Overlong answers are truncated and penalized, producing discontinuous gradients. | Smooth length penalty function.                                             | More stable training            |

**Clip-Higher.** GRPO clips the policy ratio symmetrically, for example to $[0.8, 1.2]$. This is fine for high-probability actions that the model already prefers. But for a promising action whose current probability is only 0.01, the lower bound of 0.8 lets it be pushed down to 0.008—almost completely suppressed. DAPO decouples the upper and lower clipping ranges, giving low-probability actions more room to grow.

**Dynamic sampling** solves the "graduation problem." Late in training, many prompts have near-zero within-group variance because the model already solves them reliably. These prompts provide no gradient signal—everyone in the group is correct. DAPO filters them out and keeps only prompts that still produce useful gradients. On the AIME 2024 math competition, DAPO reached a score of 50 using **half the training steps** of DeepSeek-R1.

**Token-level loss** addresses another blind spot. Standard GRPO normalizes the entire sequence: a correct answer is fully reinforced, a wrong answer is fully suppressed. But in a wrong answer, the first 80% of reasoning steps may be correct—only the final calculation fails. Token-level loss lets GRPO distinguish which tokens contributed to the error and which did not. This connects directly to the [credit assignment problem in Chapter 8](../chapter10_ppo/gae-reward-model): in long sequences, we need per-token contributions to the final outcome.

**Overlong Reward Shaping** tackles response length runaway. The model may learn that "more tokens = more likely to contain correct reasoning," producing verbose 2000+ token answers. The original GRPO approach truncates beyond a maximum length and assigns a penalty. But truncation is a hard boundary: a 499-token answer is fine, a 501-token answer is penalized. This discontinuity destabilizes the gradient signal. DAPO replaces the hard cutoff with a smooth penalty function, letting the model learn length control naturally.

```python
# ==========================================
# DAPO dynamic sampling sketch
# ==========================================
def dynamic_sampling(prompts, model, reward_fn, threshold=0.95):
    """
    Filter out prompts the model has already mastered.
    """
    useful_prompts = []

    for prompt in prompts:
        # Sample each prompt multiple times and compute accuracy.
        correct_count = 0
        num_samples = 8
        for _ in range(num_samples):
            response = model.generate(prompt)
            reward = reward_fn(prompt, response)
            if reward >= 1.0:  # Correct answer.
                correct_count += 1

        accuracy = correct_count / num_samples
        # Keep only prompts whose accuracy is below the threshold.
        if accuracy < threshold:
            useful_prompts.append(prompt)

    print(f"Before filtering: {len(prompts)} problems")
    print(f"After filtering:  {len(useful_prompts)} problems")
    print(f"Filtered out:     {len(prompts) - len(useful_prompts)} mastered")
    return useful_prompts
```

---

DeepSeek-R1-Zero and DAPO show the potential of pure RL training: no SFT, no Critic, as long as the reward signal is clear enough. But this raises a prerequisite question: **where does the reward come from?** In the next section, we turn to RLVR and see how verifiable rewards can fully replace the reward model.
