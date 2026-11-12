---
title: '23.1 Visual Reward Design'
---

# 23.1 Why a VLM Can Guess Correctly Without Looking: Visual Rewards and Hallucination

Start with a small counting task. An image contains three circles, and the question asks how many circles it contains. The model answers three and receives an outcome reward of 1. That score cannot distinguish two behaviors: the model may have counted the circles, or it may have ignored the image and guessed the most common answer in the training set.

In a text task, a correct answer often supplies a strong learning signal. An image adds a perceptual chain: the model must locate the relevant region, recognize objects or text, and then reason. A final answer fails when any link fails; even a correct answer can rest on incorrect visual evidence. **VLM reinforcement learning must therefore reward both correctness and whether the answer is grounded in the image.**

We will follow that problem from reward attribution to gradient flow, visual shortcuts, autonomous driving, and training diagnostics. The next section, [23.2 Visual Reflection RL](./qwen3-vl-reflection), continues with models that gather and verify visual evidence during reasoning.

![VISTA-Gym Overview](../../chapter26_vlm/images/ref-vista-gym-overview.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 1: VISTA-Gym places visual QA, tool use, trajectory rewards, and policy updates in one loop. Source: <a href="https://www.eigenai.com/blog/vista-gym-vista-r1" target="_blank" rel="noopener noreferrer">VISTA-Gym / VISTA-R1 Blog</a>.</em>
</div>

Moving RL from text to multimodal models is not "just add image tokens." Once you train seriously, you run into a set of problems that do not appear in text-only RL:

1. **Who is responsible for an error?** If the answer is wrong, was the vision encoder wrong, or was the language reasoning wrong?
2. **Should the vision encoder be updated by RL?** Update too aggressively and you can degrade vision (the model "goes blind"); freeze it completely and you cannot improve visual ability.
3. **Will the model pretend it saw the image?** If guessing can get reward, RL can reinforce visual hallucinations.
4. **How does vision connect to action?** In driving, robotics, and GUI agents, visual outputs affect real decisions. Safety and latency become training constraints.

::: tip Prerequisites

- [GRPO](../chapter18_grpo/grpo-practice-and-mechanism): group-based optimization without a critic
- [The RLHF pipeline](../chapter15_rlhf/standard-rlhf-pipeline): rules vs model rewards, hacking risks
- [PPO-RLHF loop](../chapter15_rlhf/ppo-rlhf-loop): KL penalty, clipping, reference model
  :::

## VLM RL vs Text-Only RL

In text-only RL, inputs and outputs are tokens. If an answer is bad, we usually ask one question: did the generated tokens match the reward target?

In VLM RL, there is a visual pipeline:

image -> vision encoder -> visual tokens -> multimodal fusion -> language reasoning -> output.

So training becomes "see correctly, then reason correctly." A single scalar reward does not automatically tell you where the failure came from. This is the core credit-assignment problem in multimodal RL.

![VISTA-Gym Main Results](../../chapter26_vlm/images/ref-vista-gym-workflow.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>Figure 1: VISTA-Gym / VISTA-R1 main results table. The w/o Tools and w/o Reasoning ablations show that capability gains in visual tasks come not only from scaling the model but also from the combined effect of tool verification, reasoning trajectories, and reward design. Source: <a href="https://www.eigenai.com/blog/vista-gym-vista-r1" target="_blank" rel="noopener noreferrer">VISTA-Gym / VISTA-R1 Blog</a></em>
</div>

## Reward Attribution: Visual Tokens and Text Tokens

In text-only RL, reward attribution is not a problem — the entire response is generated from text tokens, so the reward naturally belongs to the whole generation process. But in a VLM, a response's quality depends on two stages of capability: **visual understanding** (did the model correctly "see" the image content) and **text reasoning** (did the model make a reasonable derivation based on correct visual information).

### Where the Gradient Actually Flows

To see where gradients can flow, follow one image through the forward pass. The vision encoder divides it into patches, maps each patch to a visual token, projects those tokens to the language-model dimension, and concatenates them with the question. With a simplified $14\times14$ patch size, a $448\times448$ image contains $32\times32=1024$ patches. Real token counts also depend on patch merging, special tokens, and dynamic-resolution rules, so 1024 is an order-of-magnitude example rather than the context length of a particular VLM.

The policy loss is summed over generated response tokens:

$$
\mathcal L(\theta)=-\sum_{t\in\text{response}}A_t
\log\pi_\theta(y_t\mid y_{<t},x_{\text{text}},x_{\text{image}}).
$$

Visual tokens are inputs rather than sampled actions, so they receive no direct policy-gradient term. Each response probability still attends to those inputs, however, and the gradient can travel through the projector into the vision encoder. Whether the encoder changes is therefore controlled by which parameter groups are unfrozen, not by the scalar reward alone.

### One Group, Two Kinds of Error

For example, show the model an image with 3 circles and 2 triangles, and ask "How many circles are in the image?" The model answers "There are 2 circles in the image." This answer is wrong. But the cause of the error could be one of two things:

- **Visual error**: The model "saw" wrong, identifying 3 circles as 2. In this case, the reward signal should tell the visual encoder "you need to look more carefully."
- **Reasoning error**: The model "saw" correctly (its internal representation did identify 3 circles) but said the wrong number when generating text. In this case, the reward signal should tell the text decoder "your reasoning is wrong."

Now sample four responses to the same question. A sees three circles and answers 3; B sees two and answers 2; C sees three but loses one during reasoning and answers 2; D ignores the image and guesses 3 from a language prior. An outcome reward assigns A and D the same positive advantage and B and C the same negative advantage. Yet B needs a perceptual correction, C needs a reasoning correction, and D should not be reinforced at all. This is the ambiguity created by one scalar reward.

The problem is that in current VLM RL frameworks, we typically have only one scalar reward score and cannot naturally distinguish between these two cases. A more practical approach is to decompose an error into several observable checkpoints:

| Checkpoint         | What to Observe                                                  | Possible Training Action                                    |
| ------------------ | ---------------------------------------------------------------- | ----------------------------------------------------------- |
| Visual grounding   | Whether attention, selected regions, or IoU are reasonable       | Add grounding reward or visual consistency checks           |
| Text reasoning     | Whether the visual description is right but the derivation fails | Strengthen reasoning format, process reward, or verifier    |
| Cross-modal fusion | Whether image evidence and text conclusion agree                 | Differential learning rates, freeze ViT, or staged training |

The current mainstream approach is **holistic attribution** — distributing the reward across the entire sequence (visual tokens + text tokens), letting gradients update both the visual encoder and text decoder simultaneously. This is simple and direct, but has a fundamental problem: if the visual encoder's parameters are damaged by RL gradients, the model may lose image understanding ability — like someone being shoved hard while solving a math problem and then being unable to even read the question.

Another approach is **freezing the visual encoder** — RL only updates the text decoder's parameters. This guarantees that visual understanding is not degraded, but the cost is that the model cannot improve visual understanding through RL. In the geometric figure experiment from the previous section, this could mean the model can never accurately distinguish overlapping figures — because the visual encoder has no opportunity to learn this capability.

| Strategy                    | Strength                           | Weakness                            | Best Fit                                          |
| --------------------------- | ---------------------------------- | ----------------------------------- | ------------------------------------------------- |
| Full update                 | Jointly optimizes vision + text    | Visual encoder may be damaged       | Tasks where visual understanding must improve     |
| Frozen encoder              | Protects visual ability            | Cannot improve visual understanding | Scenarios with a strong pretrained visual encoder |
| Differential learning rates | Balances protection and adaptation | More hyperparameter tuning          | General recommended setting                       |

Differential learning rates is the most common compromise — the visual encoder uses 1/10 the learning rate of the text decoder. This both protects visual features and allows moderate visual optimization.

```python
# ==========================================
# Differential learning-rate configuration
# ==========================================

def setup_optimizer_with_lr_decay(model, text_lr=1e-6, vision_lr=1e-7):
    """Use different learning rates for visual encoder and text decoder"""
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters()
                       if 'vision' in n or 'vit' in n],
            'lr': vision_lr,  # Visual encoder: smaller learning rate
            'weight_decay': 0.01,
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if 'vision' not in n and 'vit' not in n],
            'lr': text_lr,    # Text decoder: normal learning rate
            'weight_decay': 0.01,
        },
    ]
    return torch.optim.AdamW(param_groups)
```

### From Outcome Rewards to Process Rewards

One route is to make perception outputs structured enough for deterministic checking. [Visual-RFT](https://arxiv.org/abs/2503.01785) combines three verifiable signals for detection: mean IoU between predicted and labeled boxes, confidence calibration for matched and unmatched boxes, and a format reward for a parseable template. None requires a separately trained reward model.

In the paper's few-shot setting, one-shot fine-grained classification with roughly 100 samples exceeds the supervised fine-tuning baseline by 24.3 percentage points, while two-sample COCO detection improves by 21.9 points. [VLM-R1](https://github.com/om-ai-lab/VLM-R1) applies related IoU rewards to referring-expression comprehension and open-vocabulary detection. These results show how a visual task can use rule-based rewards when its evidence can be represented as boxes or coordinates.

A second route makes evidence gathering part of the trajectory. Qwen3-VL's [Thinking with Images](https://github.com/QwenLM/Qwen3-VL/tree/main/cookbooks) lets the model zoom into a region during reasoning and then continue from the enlarged evidence. The tool call turns “where did the model look again?” into a recorded action that a trajectory evaluator can inspect.

## Visual Hallucination: The Model “Sees” Things That Are Not There

Visual hallucination is one of the most troublesome problems for VLMs. It refers to the model describing content in its response that simply does not exist in the image. For example, the image contains only one red triangle, but the model says "I see 3 red triangles and 2 blue circles in the image."

Visual hallucination does not exist in text-only RL — because a text-only model does not "see" anything; all its outputs are generated from text input. But a VLM's input includes an image, and the model must make judgments about the image's content, and those judgments can be wrong.

In RL training, visual hallucination can appear in a particularly insidious way. If one of the model's hallucinations happens to receive a high reward (e.g., it "fabricated" the correct number of figures), RL will reinforce this behavior — the model learns that "guessing" is more cost-effective than "looking." This is essentially the same as the reward hacking discussed in Chapter 25, but with an additional dimension: the model can cheat not only in text generation but also in visual understanding.

Several strategies for addressing visual hallucination:

**Strategy 1: Visual grounding checks.** Add visual consistency checks to the reward function — does the model's description match the image? This requires an additional verification model, or cross-validation using OCR/object detection tools.

**Strategy 2: Uncertainty penalties.** If the model is overly certain about visual content (e.g., saying "there are 3 circles" rather than "there seem to be 2-3 circles") and the description does not match reality, apply an additional penalty. Encourage the model to express uncertainty when unsure.

**Strategy 3: Multi-turn verification.** First have the model describe the image content, then use another model (or rule system) to verify the description's accuracy. Only responses that pass verification receive full reward. This essentially embeds a "fact-checking" step in the reward function.

**Strategy 4: Counterfactual comparison.** Pair a question with the real image and with a replacement image—blank, noisy, or a visually similar scene whose critical count or attribute differs. If the answer barely changes, the policy is probably using language priors rather than visual evidence.

```python
def image_sensitivity_bonus(answer_real, answer_shuffled, correctness):
    """Reward a correct answer only when it responds to changed visual evidence."""
    if not correctness:
        return 0.0
    if answer_real.strip() == answer_shuffled.strip():
        return -0.3
    return 0.1
```

The replacement must resemble the original scene while changing the decisive object or relation. Otherwise the model can pass the test from a superficial background difference.

Two standard evaluations can track hallucination before and after training. [POPE](https://arxiv.org/abs/2305.10355) converts object hallucination into yes/no questions and constructs random, popular, and adversarial negative samples. [CHAIR](https://arxiv.org/abs/1809.02156) measures fabricated objects in generated captions at sentence and object-mention levels. Running both periodically reveals whether a rising task reward is accompanied by worsening visual grounding.

<details>
<summary>Exercise: Why is visual hallucination more likely to worsen in RL training than in SFT training?</summary>

In SFT training, the model's output is constrained within the range of human-annotated "standard answers" — if the standard answer is "there are 2 circles in the image," the model is trained to say "there are 2 circles in the image." Human annotations serve as a "safety net."

But in RL training, the model discovers high-reward behaviors through trial and error. If it occasionally "fabricates" a coincidentally correct answer and receives a high reward, that behavior gets reinforced. Worse, RL's exploration mechanism encourages the model to try various strategies — including the "don't look at the image, just guess" strategy. If this strategy happens to work (on simple tasks, the probability of guessing correctly is not low), it gets rapidly reinforced.

This is why reward function design in VLM RL is more critical than in text-only RL — you must evaluate not only "is the answer correct" but also "did the model actually look at the image."

</details>

![PickScore compares candidate images through pairwise preference](../../chapter26_vlm/images/ref-pickscore-ranking.png)

The ranking illustrates why a preference score alone is insufficient: a candidate may match common aesthetic preferences while contradicting a particular image or prompt. Grounding checks must remain a separate signal.

![VISTA-R1 ablations](../../chapter26_vlm/images/ref-vista-gym-results.png)

The VISTA-R1 ablations separate gains from visual tools, explicit reasoning, and reward design. They are evidence that a larger model is not the only source of improvement; the observation and verification loop matters.

## VLM-RL in Autonomous Driving

VLM RL is not just an academic experiment; it has already shown tremendous potential in real-world applications. Autonomous driving is one of the most prominent directions.

Imagine an autonomous driving system architecture: the VLM receives images of the road ahead, generates an understanding of the current scene ("50 meters ahead a pedestrian is crossing, a truck in the left lane is changing lanes"), and then generates driving decisions based on this understanding. RL's role is to train the VLM to produce better scene understanding and decisions — not by training with human-annotated "standard answers" (because you cannot enumerate all possible scenarios) but by using reward signals to guide the model's learning.

In engineering practice, this closed loop is typically not written as simply "image → action → reward" but decomposed into a more conservative chain:

| Stage                 | Main Question                             | Safety Constraint                                               |
| --------------------- | ----------------------------------------- | --------------------------------------------------------------- |
| Scene understanding   | Were key traffic participants identified? | Trigger fallback on low confidence                              |
| Action candidates     | Is the current action reasonable?         | Hard-filter dangerous actions                                   |
| Simulation replay     | Are the consequences stable?              | Cover edge cases in simulation                                  |
| Reward update         | Does long-term safety improve?            | Prioritize safety reward over comfort and efficiency            |
| Deployment monitoring | Did out-of-distribution scenes appear?    | Allow only controlled online updates, no direct trial-and-error |

In autonomous driving, reward function design is much more complex than geometric figure counting. It typically includes three dimensions:

**Safety reward.** This is the most important dimension — any behavior leading to collision or danger should be severely punished. Safety constraints are typically implemented as **hard constraints**: certain actions (running red lights, driving the wrong way) never receive positive rewards, regardless of what justification the VLM provides.

**Comfort reward.** Driving should be not only safe but also comfortable — hard braking and sharp turns make passengers uncomfortable. Comfort constraints are typically implemented as **soft constraints**: as a regularization term in the reward function, traded off against the safety reward.

**Efficiency reward.** While ensuring safety and comfort, reach the destination as quickly as possible. Efficiency rewards encourage the model to choose shorter routes and more reasonable speeds.

```python
# ==========================================
# Autonomous-driving VLM-RL reward function
# ==========================================

def driving_reward(scene_description, action, telemetry):
    """
    Autonomous driving reward function
    - scene_description: VLM's description of the scene
    - action: driving action (steering, acceleration, braking)
    - telemetry: sensor data (speed, distance, lane position)
    """
    reward = 0.0

    # 1. Safety (hard constraint)
    if telemetry['collision_risk'] > 0.8:
        return -10.0  # High collision risk → large penalty

    if telemetry['red_light_violation']:
        return -10.0  # Running red light → large penalty

    if telemetry['speed_limit_exceeded']:
        reward -= 5.0  # Speeding → heavy penalty

    # 2. Comfort (soft constraint)
    jerk = abs(telemetry['acceleration_change'])  # Acceleration change rate
    reward -= 0.1 * jerk  # Hard acceleration/braking → small penalty

    lateral_error = abs(telemetry['lane_deviation'])  # Lane deviation
    reward -= 0.05 * lateral_error

    # 3. Efficiency (positive reward)
    if telemetry['speed'] > 0:  # Moving
        reward += 0.1  # Encourage forward progress
    if telemetry['distance_to_goal'] < telemetry['prev_distance']:
        reward += 0.2  # Approaching destination → reward

    # 4. Scene understanding quality (if VLM's description matches sensors)
    if scene_matches_sensors(scene_description, telemetry):
        reward += 0.3  # VLM correctly understood the scene

    return reward
```

Autonomous-driving VLM-RL has a fundamental conflict between safety and exploration. Dangerous exploration belongs in logged replay and simulation. A real vehicle should move through shadow mode and controlled validation before its execution scope expands. Simulation lowers exploration cost but introduces a Sim-to-Real gap.

The same issue can be expressed as constrained optimization. Let $c_t$ be a safety cost such as collision risk or a red-light violation, and let $d$ be the accepted cost budget:

$$
\max_\pi\;\mathbb{E}\!\left[\sum_t r_t\right]
\quad\text{s.t.}\quad
\mathbb{E}\!\left[\sum_t c_t\right]\le d.
$$

A Lagrange multiplier can increase the safety penalty when the policy exceeds the budget and relax it when margin remains. This formulation is easier to validate against a hard requirement such as a collision-rate ceiling than burying safety inside an undifferentiated scalar reward.

Latency is another constraint. At 108 km/h, or 30 m/s, a vehicle travels 60 meters in two seconds. This calculation only shows the scale of perception-to-decision delay; actual safety distance also depends on braking, road conditions, and redundant systems. In practice, large-model trajectories are distilled into smaller policies, online output length is bounded, and long reasoning is reserved for offline labeling, replay analysis, and reward computation.

## Architecture Choices for Multimodal Policies

Finally, let us summarize the architectural choices for VLM RL:

| Architecture                     | Visual Encoding                  | Fusion Method     | RL Update Range      | Best Fit                    |
| -------------------------------- | -------------------------------- | ----------------- | -------------------- | --------------------------- |
| ViT + Transformer                | Independent ViT                  | Cross-attention   | Full or differential | General VLM RL              |
| Unified Transformer              | Shared Transformer               | Patch embedding   | Full                 | Resource-limited settings   |
| Frozen ViT + lightweight decoder | Pretrained ViT (frozen)          | Linear projection | Text decoder only    | Fast iteration              |
| Multiple visual encoders         | Multiple ViTs (different scales) | Attention fusion  | Selective update     | High-precision visual tasks |

"ViT + Transformer" is currently the most mainstream architecture — the ViT encodes images into visual tokens, which then interact with text tokens through cross-attention. During RL training, you can choose full updates or differential learning rates.

"Frozen ViT + lightweight decoder" suits an early reward-design experiment. Freezing removes vision-side gradients and optimizer state, but the actual speedup depends on vision forward cost, response length, group size, and communication. It should not be summarized with one fixed multiplier.

"Multiple visual encoders" suits tasks requiring extremely high visual precision — such as medical image analysis or satellite image interpretation. Multiple ViTs process visual information at different scales or modalities (e.g., one handles overall layout, another handles fine texture), then integrate through an attention fusion layer. During RL updates, you can choose to update only the fusion layer, preserving each ViT's independent feature extraction capability.

```python
# ==========================================
# VLM RL architecture comparison: training speed
# ==========================================

# This table describes update scope without inventing hardware-bound timings.
architectures = {
    "Full update": {
        "train_vision": True,
        "vision_lr_scale": 1.0,
        "note": "Update vision and language; monitor visual degradation",
    },
    "Differential LR": {
        "train_vision": True,
        "vision_lr_scale": 0.1,
        "note": "Allow visual adaptation with smaller vision updates",
    },
    "Frozen ViT": {
        "train_vision": False,
        "vision_lr_scale": 0.0,
        "note": "Protect existing features but cannot repair perception with RL",
    },
}

for name, config in architectures.items():
    print(f"Strategy: {name}")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
```

## Training-Monitoring Checklist

Most failures discussed in this section leave measurable traces:

- If task accuracy stalls while reward rises, inspect the reward distribution for shortcuts.
- If accuracy with a blank or shuffled image approaches accuracy with the real image, add an image-sensitivity check.
- If a held-out general visual probe degrades, reduce the vision learning rate or freeze the encoder.
- If response length grows without an accuracy gain, inspect length-related rewards and truncation.
- If format reward falls, exploration may be leaving the required output schema.
- If IoU or grounding reward plateaus, the bottleneck may require new data or a tool rather than a larger policy update.
- If KL divergence spikes, reduce update size or smooth an overly sharp reward.

The blank-image control and general visual probe require dedicated held-out sets. Prepare them before training; after degradation appears, it is too late to reconstruct an uncontaminated baseline.

## A Diagnostic Sequence for VLM RL

When training reward rises but real visual performance does not, follow the causal chain.

**Reward attribution** fails because visual and text modules share one scalar; process rewards and differential learning rates trade adaptation against protection. **Visual hallucination** comes from describing absent content; grounding checks, uncertainty penalties, and counterfactual images trade compute for reliability. **Visual shortcuts** arise when language priors earn reward without the image; image-sensitivity and tool-trajectory rewards make evidence use observable. **Encoder degradation** occurs when RL damages pretrained features; freezing or a smaller vision learning rate protects perception at the cost of adaptation. **Unsafe exploration** belongs in simulation with hard constraints and shadow deployment. **Latency** calls for distillation and bounded reasoning. **Visual-token cost** calls for resolution limits or token pooling, with a corresponding loss of detail.

These failures propagate. An answer-only reward encourages a visual shortcut. Unfreezing the encoder to remove that shortcut can damage general perception. Adding high-resolution evidence and tools then increases rollout cost and latency. A useful training report therefore includes task accuracy, blank-image controls, general visual probes, trajectory cost, and safety metrics together.

The next section, [23.2 Visual Reflection RL](./qwen3-vl-reflection), asks how a model can preserve, check, and revisit visual evidence during reasoning.

## References

- [VISTA-Gym / VISTA-R1 Blog](https://www.eigenai.com/blog/vista-gym-vista-r1) — shows ablation results for tools, reasoning trajectories, and reward design in visual QA tasks.
- [VLM-R1 GitHub](https://github.com/om-ai-lab/VLM-R1) — provides grounding reward curves, useful for understanding how visual rewards enter VLM RL training.
- [Visual-RFT](https://arxiv.org/abs/2503.01785) — applies verifiable IoU, confidence, and format rewards to visual classification, detection, and grounding.
- [POPE](https://arxiv.org/abs/2305.10355) — turns object hallucination into binary visual questions.
- [CHAIR](https://arxiv.org/abs/1809.02156) — measures fabricated objects in image captions at sentence and object levels.
