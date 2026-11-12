# 22.1 GUI Agent Training: From Screenshots to Executable Actions

Start with a one-step task. A web form already contains an email address and verification code; only the **Submit** button remains. A person naturally moves the pointer to the button. The model receives a pixel matrix. It must first identify the button region, then translate “submit” into a coordinate action such as `click(214, 209)`.

The task does not end after the click. The page may show a success screen, or an expired-code error. The model must read the next screenshot, decide whether the action worked, and then stop, wait, or recover. **GUI-agent training learns this closed loop: find task-relevant interface information, produce executable actions, and use environment feedback to correct later behavior.**

<img src="../../chapter25_computer_use/images/gui-agent-one-step.svg" alt="A GUI operation proceeds from goal recognition through coordinate grounding to outcome verification">

The figure separates one click into four judgments. Training only button location stops at grounding; training only final success cannot identify whether recognition, localization, or verification caused a failure. Demonstrations, online RL, curriculum sampling, and progress rewards supply learning signals to different parts of this chain.

[Chapter 19: Agentic RL](../chapter22_agentic/overview) usually exposes tools with explicit function names and arguments. A GUI exposes screenshots plus mouse and keyboard input. The same “submit this form” objective can therefore generate different trajectories under different window sizes, themes, pop-ups, and loading states.

<img src="../../chapter25_computer_use/images/gui-agent-training-loop.svg" alt="The GUI-agent training loop connects screenshots, actions, environment transitions, verifiers, and rewards">

Each action changes the next screenshot, and a verifier judges whether the task advanced. This observation–action–new-observation loop is the starting point for formulating GUI operation as reinforcement learning.

## Step 1: Treat the GUI as an Interactive Environment

The tools in [Chapter 19: Tool Use and Trajectory](../chapter22_agentic/tool-use-and-trajectory) are **structured APIs** — `def search(query): return results`, with input and output as strings. However, in the real world, many software applications have only one interface: the **GUI**. Browsers, Excel, enterprise internal OA systems, Photoshop, and games — none of them expose public APIs; they only provide screens and mouse and keyboard events.

The **Computer Use** paradigm treats the entire operating system as the agent's environment:

- **Observation**: A screen capture $o_t \in \mathbb{R}^{H \times W \times 3}$, optionally accompanied by window, cursor, or accessibility-tree information
- **Action**: Atomic GUI events (mouse movement, click, scroll, keyboard input, wait)
- **Reward**: A binary signal indicating task completion ("whether a flight was successfully booked")

This MDP is entirely different from traditional RL benchmarks. CartPole has a 4-dimensional state, 2-dimensional actions, and dense per-step rewards; in the Computer Use paradigm, the state is millions of dimensions of pixels, the action space is mixed-type, and the reward is sparse, only given at the final step.

### What the Representative Systems Address

[Anthropic Computer Use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/computer-use-tool) packages screenshots, mouse input, and keyboard input as model tools. Browser products such as Operator and Project Mariner demonstrate complete task execution. These systems first answer an engineering question: how can a model receive screenshots and control software under explicit safety boundaries?

[UI-TARS](https://arxiv.org/abs/2501.12326), [UI-TARS-2](https://arxiv.org/abs/2509.02544), and [AutoGLM](https://arxiv.org/abs/2411.00820) are useful for understanding training. UI-TARS studies a native screenshot-only GUI agent; UI-TARS-2 extends it with multi-turn RL and parallel sandboxes; AutoGLM emphasizes planning and grounding interfaces and an online curriculum that changes with policy capability.

### Core Action Space

The primitive actions for Anthropic Computer Use are defined as follows (similar to OpenAI Operator and Google Mariner):

```python
ACTIONS = {
    "click":      {"x": int, "y": int, "button": "left|right|middle"},
    "double":     {"x": int, "y": int},
    "drag":       {"start": [x,y], "end": [x,y]},
    "type":       {"text": str},
    "key":        {"keys": "ctrl+c|enter|tab"},   # Combination keys
    "scroll":     {"x": int, "y": int, "dy": int},
    "wait":       {"ms": int},
    "screenshot": {},
    "done":       {"summary": str},
}
```

Note three key design choices:

1. **Action type and position must be predicted together.** A `click` is discrete, while $(x,y)$ is a location. Implementations may regress coordinates or discretize them into special tokens.
2. **Unobserved changes occur between screenshots.** Loading, animation, and background processes can change the interface, so a sequence of screenshots is usually only a partial state.
3. **Waiting is itself a decision.** Clicking before a page finishes loading creates new errors; waiting too long increases latency and step count.

### MDP Formalization

Define the Computer Use MDP as $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, T)$:

$$\mathcal{S} = \{\text{screenshots}\}, \quad \mathcal{A} = \{\text{click, type, scroll, key, wait, done}\}$$

The task description (e.g., "Help me convert this PDF into Markdown") is appended as an initial prompt $q$ before each observation. The policy is a conditional distribution:

$$\pi_\theta(a_t \mid q, o_{1:t}, a_{1:t-1})$$

The reward $R$ is typically sparse and binary: $r_T = \mathbb{1}[\text{task completed}]$, and intermediate steps $r_{t<T} = 0$. This makes credit assignment extremely difficult — a single browser automation task may involve 50 actions, with only the last step receiving a reward, making it impossible to determine which steps were correct or incorrect.

::: warning The central RL difficulty
High-dimensional screenshots, mixed actions, long trajectories, and sparse rewards amplify one another. An early misclick changes every later observation, while terminal reward does not identify the first wrong branch. Training therefore usually begins with demonstrations and then moves into resettable, verifiable environments for online RL.
:::

## Step 2: Ground a Text Goal in Screen Coordinates

The first challenge in computer use is not decision-making, but **grounding**: how does the model know where the "submit" button is on the screen at coordinate $(x, y)$?

### Set-of-Mark Prompting

Yang et al. 2023 propose the **Set-of-Mark (SoM)** prompting approach: first, use OCR or object detection to box all interactive elements on the screen, labeling them as $1, 2, \ldots, K$. When the agent outputs actions, it only needs to refer to the labels:

```
[Screen shot + Box 1: Input field "Username", Box 2: Input field "Password", Box 3: Button "Login"]

Agent: type("alice") → click(Box 1) → type("***") → click(Box 2) → click(Box 3)
```

This transforms the problem of continuous coordinate prediction into a **discrete selection** problem. However, the cost is the reliance on external detectors, and the agent is helpless when the detector misses elements.

### Visual Grounding

Models such as UI-TARS and CogAgent take a different approach: **let the VLM directly output coordinates**. [Set-of-Mark prompting](https://arxiv.org/abs/2310.11441) makes the alternative explicit: an external detector assigns discrete labels to interface regions, reducing coordinate prediction to label selection but inheriting detector misses. An end-to-end grounding model instead has two conceptual outputs:

$$\text{VLM}(o_t, q) \to \underbrace{(\text{thought}, \text{action token})}_{\text{language head}} + \underbrace{(x, y) \in [0,1]^2}_{\text{grounding head}}$$

The grounding head is typically an MLP that outputs normalized coordinates $(x, y) \in [0, 1]^2$, which are then scaled to pixel coordinates by multiplying with the screen size.

In the opening form, the center of the blue button is approximately $(214,209)$. A prediction of $(214,160)$ is syntactically valid but lands in the verification-code field. A GUI trajectory must therefore preserve both which action was chosen and where it was applied.

Training the grounding head uses **supervised imitation**: human-labeled "center points" $(x_i, y_i)$ of buttons, with loss defined as:

$$\mathcal{L}_{\text{ground}} = \frac{1}{N}\sum_i \|\hat{p}_\theta(o_i) - p_i\|_2^2$$

However, pure supervision has a problem: the model might output **empty space**. Supervision only learns "where the button is," not "the button should be pressed." Here, reinforcement learning comes into play.

### Joint RL for Grounding and Decision Making

We combine grounding and action selection into a single PPO objective:

$$\mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \gamma^t r_t\right] - \beta \cdot \mathcal{L}_{\text{ground}}(\theta)$$

The second term is a supervised loss for grounding, acting as a regularizer. This **SFT + RL joint training** is the standard recipe for GUI Agents — first imitate to learn basic operations, then use RL to optimize task success rate.

UI-TARS-2 takes this idea to its extreme: it outputs three parts — thought, action, and coordinate — as a **single sequence**, and optimizes them all simultaneously with RL:

```python
def ui_tars_forward(self, screenshot, task):
    # Encode image
    visual_tokens = self.vision_encoder(screenshot)  # [B, N_vis, d]

    # Concatenate prompt
    prompt = f"<task>{task}</task>\n<image>{visual_tokens}</image>\n"

    # Autoregressively generate thought + action + coord
    # Key: coord is wrapped with special tokens <coord_x> <coord_y>
    output = self.llm.generate(prompt, max_new_tokens=256)

    # Parse output: "<thought>...</thought>\n<action>click</action>\n<coord>(0.45, 0.62)</coord>"
    thought, action, coord = parse_action(output)
    return thought, action, coord
```

### From Demonstrations to Online Trajectories

Demonstrations teach basic operations but cannot cover the combinations of window sizes, pop-ups, error messages, and recovery paths. A training system usually connects four components:

1. A **task template** specifies an objective and variable parameters, such as changing order `{order_id}` to `{status}`.
2. An **environment snapshot** restores a virtual machine to a known initial state.
3. A **demonstration trajectory** stores every screenshot and action for supervised fine-tuning.
4. An **online trajectory** is generated by the current policy, while a program verifier reads application state and returns reward.

```python
class GUIEnv:
    def reset(self, task_id):
        self.vm.restore_snapshot(task_id)
        self.task = self.tasks[task_id]
        return self.screenshot()

    def step(self, action):
        self.vm.execute(action)
        obs = self.screenshot()
        done = self.task.verifier(obs, self.vm.state)
        reward = 1.0 if done else 0.0
        return obs, reward, done, {}
```

::: details Why training usually runs in virtual machines

Online RL produces many failed actions. A virtual machine or containerized desktop isolates user files and accounts and restores a snapshot after every trajectory. [OSWorld](https://arxiv.org/abs/2404.07972) uses real applications in a reproducible computer environment; large-scale training runs many such environments in parallel so that the GPU does not wait on one desktop.

:::

## Moving from One-Step Grounding to Complete Tasks

At this point, the model can turn a screenshot into a legal action. Complete tasks add recovery from early mistakes and sparse terminal feedback. The following sections compare how UI-TARS-2, AutoGLM, MobileRL, ComputerRL, and CogAgent change online training, curriculum, reward distance, and visual representation. Their papers do not disclose one shared recipe, so each mechanism must remain tied to the problem it was designed to solve.

## Step 3: Connect Single-Step Ability into a Training Pipeline

Once grounding works, training becomes dominated by environment interaction. Representative systems share three prerequisites:

1. A pretrained VLM supplies initial perception.
2. Resettable environments such as [AndroidWorld](https://arxiv.org/abs/2405.14573), [OSWorld](https://arxiv.org/abs/2404.07972), and WebArena make trajectories comparable.
3. Parallel environments keep the GPU occupied while applications load.

The papers change different parts of this system. [UI-TARS-2](https://arxiv.org/abs/2509.02544) emphasizes stable multi-turn RL, data flywheels, mixed GUI environments, and a unified sandbox. [AutoGLM](https://arxiv.org/abs/2411.00820) exposes planning and grounding interfaces and a curriculum that evolves with the policy. [MobileRL](https://arxiv.org/abs/2509.18119) changes replay, filtering, and shortest-path reward adjustment for a heavy-tailed mobile-task distribution. [ComputerRL](https://arxiv.org/abs/2508.14040) scales online sampling with hybrid API–GUI actions and parallel virtual desktops and uses Entropulse to counter entropy collapse. [CogAgent](https://arxiv.org/abs/2312.08914) changes the visual representation for high-resolution interfaces.

These works do not share one fixed recipe. Their components should not be presented as an interchangeable list of tricks.

## Step 4: Use UI-TARS-2 to Understand Multi-Turn Online RL

UI-TARS-2 uses one model for perception, reasoning, and actions. The important change is the training object: language-model RL samples one response, whereas GUI RL alternates between the model and an environment, and each new observation depends on the previous action.

### Establish a Policy That Can Enter the Environment

If the initial policy cannot distinguish buttons and fields, most online trajectories fail in the first steps and the terminal verifier rarely returns positive reward. A common teaching pipeline is:

```text
visual-language pretraining
  → demonstration SFT
  → verifier-filtered successful trajectories
  → resettable-environment rollout and online RL
```

This four-stage description explains data flow; it is not a claim that UI-TARS-2 discloses these exact fixed phases or a fixed number of sampled trajectories.

### Record Where the Interface Failed to Change

[UI-TARS](https://arxiv.org/abs/2501.12326) includes reflection as one multi-step reasoning behavior. A trajectory can expose it explicitly:

```xml
<thought>I should submit the form.</thought>
<action>click(450, 320)</action>
<observation>The button changed color, but the page did not advance.</observation>
<reflection>The click may have missed the active region.</reflection>
<action>click(455, 320)</action>
<observation>The success page appeared.</observation>
<action>done</action>
```

The useful training record is not simply “reflection occurred.” It preserves the expected state change, the observed mismatch, and the corrective action.

### A Teaching Reward Decomposition

A minimal decomposition is

$$
r=r_{\text{task}}+\alpha r_{\text{format}}
  +\beta r_{\text{recovery}}-\gamma r_{\text{invalid}}.
$$

The terms check task completion, parseable actions, successful recovery, and illegal actions. Values such as $\alpha=0.1$, $\beta=0.3$, and $\gamma=2.0$ are starting points for a classroom ablation, not UI-TARS-2's disclosed fixed weights. Each environment needs new calibration, especially because a large illegal-action penalty may make the policy overly conservative.

## Step 5: Use AutoGLM to Separate Planning, Grounding, and Device Interfaces

### Why Multiple Devices Need a Unified Action Space

For “search for wireless headphones,” a high-level planner may output “open the search field and enter the query,” while the grounding module locates the field in the current screenshot. A layout change can leave the plan valid while breaking grounding. Recording those errors separately tells us whether to add planning examples or visual-localization examples.

A unified action representation can then map semantic actions to desktop or mobile adapters:

```python
UNIFIED_ACTIONS = {
    "tap": {"x": float, "y": float},
    "long_press": {"x": float, "y": float, "ms": int},
    "swipe": {"start": [x, y], "end": [x, y]},
    "type": {"text": str},
    "key": {"name": str},
    "scroll": {"dy": int},
    "wait": {"ms": int},
    "done": {"summary": str},
}
```

### Run One Observable Device Loop First

The [Open-AutoGLM repository](https://github.com/zai-org/Open-AutoGLM) provides a practical device-control entry point. Begin with a read-only task and preserve every screenshot, raw output, parsed action, and final state:

```bash
python -m open_autoglm.server \
    --model Open-AutoGLM \
    --base-url http://localhost:8000/v1

python -m open_autoglm.run \
    --device emulator-5554 \
    --task "Open the shopping app and search for wireless headphones"
```

Exact commands and model identifiers depend on the repository revision.

## Step 6: Use MobileRL to Adjust Task Difficulty

[MobileRL](https://arxiv.org/abs/2509.18119) studies online reinforcement learning for mobile GUI agents. Mobile environments are more challenging than desktop environments for three reasons:

- **Small screen, dense elements**: A mobile app's home page may have 30 clickable elements densely arranged.
- **Complex gestures**: Long press, swipe, pinch, 3D Touch, and other gestures are far more diverse than mouse clicks.
- **Frequent app switching**: Push notifications, incoming calls, and low battery alerts can interrupt tasks at any time.

### A Simplified Curriculum Constraint

MobileRL adapts replay and filtering from training feedback. To isolate the intuition, sample tasks that are possible but not yet mastered:

$$\text{Curriculum}(\pi_\theta) = \arg\max_{\text{task } \tau} \; \text{Difficulty}(\tau) \quad \text{s.t.} \quad 0.3 \leq P_\theta(\text{success} \mid \tau) \leq 0.7$$

This concentrates sampling on tasks with an intermediate success rate. The 30%–70% interval is an experimental teaching setting, not a threshold prescribed by the MobileRL paper.

### Quantification of Task Difficulty

For a minimal scheduler, task difficulty can be approximated as a weighted sum of four inspectable dimensions:

$$\text{Difficulty}(\tau) = w_1 \cdot \text{Steps}(\tau) + w_2 \cdot \text{Apps}(\tau) + w_3 \cdot \text{GestureComplexity}(\tau) + w_4 \cdot \text{Distraction}(\tau)$$

- $\text{Steps}$: Minimum number of steps to complete the task (5–50)
- $\text{Apps}$: Number of apps to switch between (1–4)
- $\text{GestureComplexity}$: Number of gesture types required (tap=1, swipe=2, long_press=3, multi-touch=5)
- $\text{Distraction}$: Number of simulated distraction events (push notifications, incoming calls)

The weights $w_1=0.4$, $w_2=0.2$, $w_3=0.2$, and $w_4=0.2$ are classroom defaults. A reproduction of MobileRL should use its published replay and filtering algorithm rather than attributing this simplified score to the paper.

### Curriculum Scheduler

```python
class CurriculumSampler:
    def __init__(self, tasks, model):
        self.tasks = tasks
        self.model = model
        self.success_rate = {}  # task_id -> moving average success rate

    def sample(self, batch_size):
        # 1. Evaluate the success rate of each task under the current model
        for tau in self.tasks:
            if tau.id not in self.success_rate:
                self.success_rate[tau.id] = self._estimate(tau)

        # 2. Filter out tasks with success rate between 30% and 70%
        candidates = [t for t in self.tasks
                      if 0.3 <= self.success_rate[t.id] <= 0.7]

        # 3. Sample with difficulty-weighted probability
        weights = [t.difficulty for t in candidates]
        return weighted_sample(candidates, weights, batch_size)

    def _estimate(self, task):
        # Run 10 rollouts to estimate the success rate
        successes = sum(self._rollout(task) for _ in range(10))
        return successes / 10
```

The success rate of each task is re-evaluated at each epoch, allowing the curriculum to dynamically adjust according to the model's current capabilities.

## Step 7: Handle Reward Distance in Long-Horizon Tasks

[ComputerRL](https://arxiv.org/abs/2508.14040) studies scalable end-to-end online Computer Use RL with API–GUI hybrid actions, parallel virtual desktops, and Entropulse, which alternates RL and supervised updates to mitigate entropy collapse. The backward curriculum and intermediate reward below are retained as a separate teaching experiment for the reward-distance problem; they are not a summary of ComputerRL's disclosed method.

### Backward Curriculum

Traditional curriculum learning progresses from easy to hard — first learning a 5-step task, then a 10-step, then a 20-step. Backward curriculum reverses this: **starting from the task endpoint**.

Consider a 50-step task $T = (s_0, a_1, s_1, \ldots, a_{50}, s_{50})$. The training order for backward curriculum is:

```
Round 1: Start from $s_{49}$, execute $a_{50}$ → done (1-step task)
Round 2: Start from $s_{48}$, execute $a_{49}, a_{50}$ → done (2-step task)
Round 3: Start from $s_{47}$, execute $a_{48}, a_{49}, a_{50}$ → done (3-step task)
...
Round 50: Start from $s_0$, complete the full task (50-step task)
```

**Why is this effective?** Backward curriculum ensures that the RL agent is always trained in states that are close to the reward. In forward training, the agent at $s_0$ sees no reward signal; in backward training, the agent at $s_{49}$ receives the reward in one step. This makes credit assignment much simpler — the most recent action receives immediate feedback.

### Intermediate Exploration Rewards

The inverse curriculum addresses the issue of "sparse terminal rewards being too far," but intermediate steps still lack signals. The teaching experiment can add an **intermediate state reward**:

$$r_t = \underbrace{r_{\text{task}}(t=T)}_{\text{Sparse Terminal Reward}} + \lambda \cdot \underbrace{r_{\text{progress}}(s_t, s_{t+1})}_{\text{Dense Progress Reward}}$$

Here, $r_{\text{progress}}$ is generated by an independent "progress evaluator" LLM:

```python
def compute_progress_reward(s_t, s_{t+1}, task):
    prompt = f"""
    Task: {task}
    State before: {describe(s_t)}
    State after: {describe(s_{t+1})}
    Question: did the agent make progress toward the task?
    Answer with a score in [0, 1]:
    - 1.0: significant progress (e.g., filled a required field)
    - 0.5: minor progress (e.g., navigated closer)
    - 0.0: no progress (e.g., clicked irrelevant element)
    - -0.5: regression (e.g., closed important dialog)
    """
    return float(llm_judge(prompt))
```

This LLM-as-judge approach for intermediate rewards is similar to the idea in [Chapter 17: Process Reward Model](../chapter20_prm_search/inference-time-search), where LLMs are used to evaluate the quality of intermediate steps.

### Comparison with Forward Curriculum

The earlier course draft recorded the following illustrative measurements:

| Method                                   | OSLevel-3 Success Rate | Average Steps | Training Cost |
| ---------------------------------------- | ---------------------- | ------------- | ------------- |
| Forward Curriculum + Terminal Reward     | 12.3%                  | 47            | 1×            |
| Forward Curriculum + Progress Reward     | 27.7%                  | 35            | 2.3×          |
| **Reverse Curriculum + Progress Reward** | **51.2%**              | **28**        | 2.8×          |

In this record, reverse curriculum increases success from about 12% to about 51%, while training cost rises to 2.8 times the baseline because the progress evaluator is called repeatedly. These numbers are a teaching record rather than reported ComputerRL results; a reproduction must rerun the comparison in one fixed environment.

## Step 8: Balance High-Resolution Vision Against Latency

[CogAgent](https://arxiv.org/abs/2312.08914) takes a different approach: **using higher-resolution visual encoding to improve recognition of small GUI elements**. The original paper describes an 18B model; a later repository release provides a 9B version. The previously cited arXiv:2408.16500 is CogVLM2, not the CogAgent paper.

### High-Resolution Visual Branch

CogAgent accepts $1120\times1120$ input and combines low- and high-resolution image encoders. The low-resolution branch supplies global context, such as recognizing a shopping page, while the high-resolution branch preserves small labels and toolbar icons.

Higher resolution increases visual encoding and multimodal-fusion cost. An earlier course draft recorded three configurations—$448\times448$ single branch, $1120\times1120$ single branch, and dual-branch fusion—with token, latency, and OSWorld numbers. Those measurements were not tied to a public CogAgent experiment configuration, so they must not be cited as paper results.

Use them only as a measurement template. A reproduction should run all configurations on the same hardware, task set, and step budget and report visual-token count, single-step latency, and task success together.

### Accuracy Versus Latency

Higher resolution increases visual encoding and cross-modal fusion cost. Compare configurations only on the same hardware, task set, maximum step budget, and decoding setup. Report visual-token count, per-step latency, and task success together; otherwise a faster setting may simply be doing less work.

## Step 9: Diagnose What a Failed Trajectory Needs

A falling task-success curve does not locate the defect. Save the task, every screenshot, raw model output, parsed action, environment return, and verifier result, then separate four cases.

**Grounding failure:** the action type is correct but its coordinate lands on an edge or adjacent control. Add localization examples across resolution, scale, and occlusion, and measure whether the click lies inside the target.

**Planning failure:** the coordinate lands on a clickable element that does not serve the current subgoal—for example, submitting a form before entering an amount. This needs complete task-conditioned demonstrations or online trajectories, not more coordinate labels.

**State-check failure:** an action succeeds but the interface is still loading, so the policy repeats it and triggers a duplicate operation. Record expected state changes and train the choice among waiting, retrying, and stopping.

**Verifier failure:** a success message earns reward even though the wrong record changed in the backend. Prefer structured application state and verify parameters, target objects, and side effects instead of matching one piece of text.

[OSWorld](https://arxiv.org/abs/2404.07972) shows why GUI evaluation must combine grounding, software knowledge, and cross-application workflows. Report success rate, mean actions, invalid-action rate, environment-error rate, and per-step latency separately so that environment failures do not become indistinguishable from policy failures.

## Step 10: Move Training Results to Real Desktops

Moving the system from a benchmark to a user's desktop introduces distribution shift, long-tail tasks, and safety boundaries.

### Distributional Shift in Environments

The training environments in the papers are controlled benchmarks such as [OSWorld](https://arxiv.org/abs/2404.07972) and [AndroidWorld](https://arxiv.org/abs/2405.14573). In production, the environment is the real user's computer—each user has a different system version, browser extensions, and font scale.

Evaluation should deliberately vary operating-system versions, display scale, browser extensions, themes, fonts, and resolution, then report each slice separately. Failure trajectories from new configurations can be added to a controlled data flywheel, but deployment data collection must preserve user privacy and cannot silently broaden permissions.

### Long-Tail Tasks

Public benchmarks cover reproducible common tasks. Real requests include rare software, internal workflows, and high-risk system changes. Validated common tasks can run automatically; tasks without test coverage should be limited to read-only exploration or handed back to a person. Pause when the goal is ambiguous, a new application appears, or the next action creates an external side effect.

### Safety Boundaries

GUI Agents can perform destructive actions — delete files, transfer money, send emails. The production environment must have clear safety boundaries.

**Solutions**:

- **Whitelisted Actions**: By default, prohibit `rm -rf`, money transfers over $100, and mass email sending.
- **Double-Confirmation**: Pop-up confirmation is required for high-risk operations.
- **Audit Logs**: All operations are logged and traceable.

See [22.2 Prompt Injection and Instruction-Level](./safety-swarm).

## Summary

A GUI agent learns a trajectory that changes its environment. Supervised fine-tuning first establishes element recognition, coordinate grounding, and basic actions. Online RL then uses resettable environments and task verifiers to improve complete multi-step tasks.

The representative papers fill different gaps. UI-TARS-2 studies stable multi-turn RL and sandboxes; AutoGLM connects planning and grounding; MobileRL adjusts sampling and reward from task difficulty; ComputerRL scales parallel desktop training and mitigates entropy collapse; CogAgent improves high-resolution interface perception. Backward curriculum and progress rewards remain separate teaching proposals for shortening the distance between an action and terminal reward.

Evaluation must return to complete trajectories and inspect grounding, planning, state verification, environment errors, and the final verifier separately. One success rate cannot tell whether the next change belongs in data, reward, environment, or visual resolution.

The next section, [22.2 Prompt Injection and Instruction Hierarchy](./safety-swarm), turns to malicious web content, forged interfaces, and cross-application attacks.
