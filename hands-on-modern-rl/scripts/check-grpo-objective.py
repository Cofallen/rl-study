"""Regression checks for the paper-faithful GRPO teaching objective."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs/chapter18_grpo/snippets/grpo-code-map.py"
SPEC = importlib.util.spec_from_file_location("grpo_code_map", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

AGENTIC_POLICY_PATH = ROOT / "docs/chapter22_agentic/code/policy.py"
AGENTIC_SPEC = importlib.util.spec_from_file_location(
    "agentic_grpo_policy",
    AGENTIC_POLICY_PATH,
)
AGENTIC_MODULE = importlib.util.module_from_spec(AGENTIC_SPEC)
AGENTIC_SPEC.loader.exec_module(AGENTIC_MODULE)

AGENTIC_CODE_DIR = ROOT / "docs/chapter22_agentic/code"
sys.path.insert(0, str(AGENTIC_CODE_DIR))
TRAINER_SPEC = importlib.util.spec_from_file_location(
    "agentic_grpo_trainer",
    AGENTIC_CODE_DIR / "trainer.py",
)
TRAINER_MODULE = importlib.util.module_from_spec(TRAINER_SPEC)
TRAINER_SPEC.loader.exec_module(TRAINER_MODULE)


def assert_close(actual, expected):
    torch.testing.assert_close(
        actual,
        torch.as_tensor(expected, dtype=actual.dtype, device=actual.device),
    )


def paper_grpo_reference(
    new_logprobs,
    old_logprobs,
    ref_logprobs,
    completion_mask,
    advantages,
    clip_eps,
    kl_coef,
):
    """Direct transcription of DeepSeekMath equations (3) and (4)."""
    ratio = torch.exp(new_logprobs - old_logprobs)
    advantage = advantages[:, None]
    clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    policy_objective = torch.minimum(ratio * advantage, clipped_ratio * advantage)

    delta = ref_logprobs - new_logprobs
    kl = torch.exp(delta) - delta - 1.0
    token_objective = policy_objective - kl_coef * kl

    mask = completion_mask.to(token_objective.dtype)
    response_objective = (token_objective * mask).sum(-1) / mask.sum(-1)
    return -response_objective.mean()


def trl_grpo_reduction_reference(
    new_logprobs,
    old_logprobs,
    ref_logprobs,
    completion_mask,
    advantages,
    clip_eps,
    kl_coef,
):
    """Equivalent layout to TRL's per-token loss with loss_type='grpo'."""
    log_ratio = new_logprobs - old_logprobs
    coef_1 = torch.exp(log_ratio)
    coef_2 = torch.clamp(coef_1, 1.0 - clip_eps, 1.0 + clip_eps)
    per_token_loss = -torch.minimum(
        coef_1 * advantages[:, None],
        coef_2 * advantages[:, None],
    )

    delta = ref_logprobs - new_logprobs
    per_token_loss = per_token_loss + kl_coef * (
        torch.exp(delta) - delta - 1.0
    )
    mask = completion_mask.to(per_token_loss.dtype)
    return ((per_token_loss * mask).sum(-1) / mask.sum(-1)).mean()


def check_tokenwise_clip_and_response_reduction():
    ratios = torch.tensor([[2.0, 0.5], [1.0, 100.0]])
    new_logprobs = ratios.log().requires_grad_()
    old_logprobs = torch.zeros_like(new_logprobs)
    ref_logprobs = new_logprobs.detach().clone()
    completion_mask = torch.tensor([[True, True], [True, False]])
    advantages = torch.ones(2)

    loss, metrics = MODULE.grpo_objective_from_logprobs(
        new_logprobs,
        old_logprobs,
        ref_logprobs,
        completion_mask,
        advantages,
        clip_eps=0.2,
        kl_coef=0.0,
    )

    # Positive advantages give token objectives [1.2, 0.5] and [1.0].
    # Original GRPO averages each response first: -mean([0.85, 1.0]).
    assert_close(loss, -0.925)
    assert_close(metrics["policy_loss"], -0.925)

    paper_loss = paper_grpo_reference(
        new_logprobs,
        old_logprobs,
        ref_logprobs,
        completion_mask,
        advantages,
        clip_eps=0.2,
        kl_coef=0.0,
    )
    trl_loss = trl_grpo_reduction_reference(
        new_logprobs,
        old_logprobs,
        ref_logprobs,
        completion_mask,
        advantages,
        clip_eps=0.2,
        kl_coef=0.0,
    )
    assert_close(loss, paper_loss)
    assert_close(loss, trl_loss)

    # A global token mean would be -0.9 and corresponds to BNPO, not GRPO.
    global_token_loss = -(torch.tensor([1.2, 0.5, 1.0]).mean())
    assert not torch.isclose(loss.detach(), global_token_loss)

    loss.backward()
    # The clipped 2.0 token and masked padding position contribute no gradient.
    assert_close(new_logprobs.grad[0, 0], 0.0)
    assert_close(new_logprobs.grad[1, 1], 0.0)
    assert new_logprobs.grad[0, 1].abs() > 0

    print(f"paper_grpo_loss={paper_loss.item():.6f}")
    print(f"trl_loss_type_grpo={trl_loss.item():.6f}")
    print(f"bnpo_global_token_loss={global_token_loss.item():.6f}")


def check_masked_padding_is_inert():
    values = torch.tensor([[1.0, 3.0], [5.0, 9999.0]])
    mask = torch.tensor([[True, True], [True, False]])
    means = MODULE.masked_sequence_mean(values, mask)
    assert_close(means, [2.0, 5.0])

    completion_ids = torch.tensor([[5, 2, 0, 0], [6, 7, 8, 9]])
    completion_mask = MODULE.completion_mask_until_eos(completion_ids, 2)
    assert torch.equal(
        completion_mask,
        torch.tensor([[True, True, False, False], [True, True, True, True]]),
    )


def check_kl_is_tokenwise_nonnegative():
    new_logprobs = torch.zeros((1, 3))
    old_logprobs = torch.zeros_like(new_logprobs)
    ref_logprobs = torch.log(torch.tensor([[0.5, 1.0, 2.0]]))
    mask = torch.ones_like(new_logprobs, dtype=torch.bool)

    _, metrics = MODULE.grpo_objective_from_logprobs(
        new_logprobs,
        old_logprobs,
        ref_logprobs,
        mask,
        torch.zeros(1),
        kl_coef=1.0,
    )
    assert metrics["approx_kl"] >= 0

    _, zero_metrics = MODULE.grpo_objective_from_logprobs(
        new_logprobs,
        old_logprobs,
        new_logprobs,
        mask,
        torch.zeros(1),
        kl_coef=1.0,
    )
    assert_close(zero_metrics["approx_kl"], 0.0)


def check_equal_rewards_have_zero_advantage():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    advantages = MODULE.group_advantages(rewards, group_size=2)
    assert_close(advantages, torch.zeros_like(advantages))


def check_grpo_gspo_and_old_sequence_ratio_are_distinct():
    token_ratios = torch.tensor([1.1, 0.9, 1.5])
    clipped_grpo_objective = token_ratios.clamp(0.8, 1.2).mean()
    old_sequence_ratio = token_ratios.prod()
    old_sequence_objective = old_sequence_ratio.clamp(0.8, 1.2)
    gspo_ratio = old_sequence_ratio.pow(1.0 / token_ratios.numel())

    assert_close(clipped_grpo_objective, 3.2 / 3.0)
    assert_close(old_sequence_ratio, 1.485)
    assert_close(gspo_ratio, 1.485 ** (1.0 / 3.0))
    assert not torch.isclose(clipped_grpo_objective, old_sequence_objective)
    assert not torch.isclose(old_sequence_ratio, gspo_ratio)

    print(f"grpo_token_clip_mean={clipped_grpo_objective.item():.6f}")
    print(f"old_raw_sequence_ratio={old_sequence_ratio.item():.6f}")
    print(f"old_sequence_clip_objective={old_sequence_objective.item():.6f}")
    print(f"gspo_geometric_mean_ratio={gspo_ratio.item():.6f}")


class TinyLogitModel(torch.nn.Module):
    """A trainable logits table that behaves like a tiny causal LM."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    @property
    def device(self):
        return self.logits.device

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        batch_size, sequence_length = input_ids.shape
        return SimpleNamespace(
            logits=self.logits[:batch_size, :sequence_length],
        )


class TinyGenerativeLogitModel(TinyLogitModel):
    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        completions = torch.tensor(
            [[3, 2, 0], [4, 2, 0]],
            device=input_ids.device,
        )
        return torch.cat([input_ids, completions[: input_ids.size(0)]], dim=1)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, texts, padding=True, return_tensors="pt"):
        del padding, return_tensors
        input_ids = torch.tensor([[5, 6] for _ in texts])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def batch_decode(self, completion_ids, skip_special_tokens=True):
        del skip_special_tokens
        answers = []
        for row in completion_ids:
            answers.append(r"\boxed{1}" if row[0].item() == 3 else r"\boxed{0}")
        return answers


class TinyBatch(dict):
    def to(self, device):
        return TinyBatch({key: value.to(device) for key, value in self.items()})


class TinyAgentTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        del return_tensors
        token_ids = [ord(char) % 5 + 2 for char in text]
        if add_special_tokens:
            token_ids.insert(0, 1)
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return TinyBatch(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        )

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token.item()) for token in token_ids)


class TinyAgentModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())
        self.last_generate_kwargs = None

    @property
    def device(self):
        return self.logits.device

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        batch_size, sequence_length = input_ids.shape
        return SimpleNamespace(
            logits=self.logits[:batch_size, :sequence_length],
        )

    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask
        self.last_generate_kwargs = kwargs
        completion = torch.tensor([[3, 4]], device=input_ids.device)
        return torch.cat([input_ids, completion], dim=1)


def check_forward_backward_and_optimizer_step():
    batch_size, sequence_length, vocab_size = 2, 5, 7
    logits = torch.linspace(
        -0.8,
        0.8,
        steps=batch_size * sequence_length * vocab_size,
    ).reshape(batch_size, sequence_length, vocab_size)
    policy = TinyLogitModel(logits)
    reference = TinyLogitModel(logits)
    input_ids = torch.tensor([[0, 1, 2, 3, 4], [0, 2, 1, 5, 6]])
    attention_mask = torch.ones_like(input_ids)
    completion_mask = torch.tensor([[True, True], [True, False]])
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
    }

    actual_logprobs = MODULE.per_token_logprobs(
        policy,
        input_ids,
        attention_mask,
        completion_length=2,
    )
    expected_logprobs = logits[:, :-1].log_softmax(-1).gather(
        -1,
        input_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)[:, -2:]
    assert_close(actual_logprobs, expected_logprobs)

    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    before = policy.logits.detach().clone()
    loss, metrics = MODULE.grpo_loss(
        policy,
        reference,
        batch,
        advantages=torch.tensor([1.0, -0.5]),
        kl_coef=0.04,
    )
    assert_close(loss, -0.25)
    assert_close(metrics["approx_kl"], 0.0)

    optimizer.zero_grad()
    loss.backward()
    assert policy.logits.grad is not None
    assert torch.isfinite(policy.logits.grad).all()
    assert policy.logits.grad.abs().sum() > 0
    optimizer.step()

    parameter_delta = (policy.logits.detach() - before).abs().max()
    assert parameter_delta > 0
    print(f"tiny_model_initial_loss={loss.item():.6f}")
    print(f"tiny_model_max_parameter_delta={parameter_delta.item():.8f}")


def check_complete_train_step():
    batch_size, sequence_length, vocab_size = 2, 5, 7
    logits = torch.linspace(
        -0.5,
        0.5,
        steps=batch_size * sequence_length * vocab_size,
    ).reshape(batch_size, sequence_length, vocab_size)
    policy = TinyGenerativeLogitModel(logits)
    reference = TinyLogitModel(logits)
    tokenizer = TinyTokenizer()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.05)
    before = policy.logits.detach().clone()

    metrics = MODULE.train_step(
        policy,
        reference,
        optimizer,
        tokenizer,
        prompts=["1 + 0 = ?"],
        ground_truths=["1"],
        group_size=2,
    )

    parameter_delta = (policy.logits.detach() - before).abs().max()
    assert torch.isfinite(metrics["loss"])
    assert torch.isfinite(metrics["approx_kl"])
    assert parameter_delta > 0
    print(f"complete_train_step_loss={metrics['loss'].item():.6f}")
    print(f"complete_train_step_kl={metrics['approx_kl'].item():.6f}")
    print(f"complete_train_step_parameter_delta={parameter_delta.item():.8f}")


def check_agentic_grpo_policy_step():
    logits = torch.linspace(-0.4, 0.4, steps=1 * 16 * 8).reshape(1, 16, 8)
    model = TinyAgentModel(logits)
    reference = TinyAgentModel(logits)
    tokenizer = TinyAgentTokenizer()
    policy = AGENTIC_MODULE.Policy(
        model,
        tokenizer,
        lr=0.05,
        clip_eps=0.2,
        kl_coef=0.04,
    )
    policy.set_ref_model(reference)

    generated = policy.generate("ab", max_new_tokens=2)
    assert generated == "3 4"
    assert model.last_generate_kwargs["do_sample"] is True
    assert model.last_generate_kwargs["temperature"] == 1.0
    assert all(not parameter.requires_grad for parameter in reference.parameters())

    logged = policy.get_logprobs("ab", "cd")
    assert not logged.requires_grad
    differentiable = policy._token_logprobs(model, "ab", "cd")
    assert differentiable.requires_grad

    before = model.logits.detach().clone()
    loss = policy.train_step_with_advantage(
        [
            (
                [("ab", "cd"), ("ab observation", "e")],
                1.0,
            )
        ]
    )
    parameter_delta = (model.logits.detach() - before).abs().max()
    assert torch.isfinite(torch.tensor(loss))
    assert parameter_delta > 0
    print(f"agentic_grpo_loss={loss:.6f}")
    print(f"agentic_grpo_parameter_delta={parameter_delta.item():.8f}")


class RecordingPolicy:
    def __init__(self):
        self.train_data = None

    def train_step_with_advantage(self, train_data):
        self.train_data = train_data
        return 0.25


class FixedRolloutWorker:
    def __init__(self):
        self.index = 0

    def rollout(self, prompt, reward_fn):
        del reward_fn
        index = self.index
        self.index += 1
        return {
            "prompt": prompt,
            "interactions": [
                {
                    "context": f"context-{index}",
                    "response": f"response-{index}",
                    "observation": "environment output must not become an action",
                }
            ],
            "reward": float(index),
        }


def check_agentic_trainer_action_boundaries():
    recording_policy = RecordingPolicy()
    trainer = TRAINER_MODULE.GRPOAgentTrainer(
        policy=recording_policy,
        env=object(),
        reward_fn=lambda trajectory: trajectory["reward"],
        group_size=2,
        max_turns=1,
    )
    trainer.worker = FixedRolloutWorker()
    history = trainer.fit(["task"], n_steps=1)

    assert history[0]["loss"] == 0.25
    assert len(recording_policy.train_data) == 2
    first_turns, first_advantage = recording_policy.train_data[0]
    second_turns, second_advantage = recording_policy.train_data[1]
    assert first_turns == [("context-0", "response-0")]
    assert second_turns == [("context-1", "response-1")]
    assert "environment output" not in repr(recording_policy.train_data)
    assert_close(torch.tensor(first_advantage), -(0.5**0.5))
    assert_close(torch.tensor(second_advantage), 0.5**0.5)
    print("agentic_action_boundary_check=passed")


def check_agentic_reward_verifies_output():
    run_path = AGENTIC_CODE_DIR / "run.py"
    parsed = ast.parse(run_path.read_text(encoding="utf-8"), filename=str(run_path))
    selected_nodes = []
    for node in parsed.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TASK_EXPECTED_OUTPUTS"
            for target in node.targets
        ):
            selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "code_reward":
            selected_nodes.append(node)

    namespace = {}
    reward_module = ast.Module(body=selected_nodes, type_ignores=[])
    exec(compile(reward_module, str(run_path), "exec"), namespace)
    prompt = next(iter(namespace["TASK_EXPECTED_OUTPUTS"]))

    correct = {
        "prompt": prompt,
        "interactions": [{"observation": "debug line\n55\n"}],
    }
    wrong_but_executable = {
        "prompt": prompt,
        "interactions": [{"observation": "54\n"}],
    }
    assert namespace["code_reward"](correct) == 1.0
    assert namespace["code_reward"](wrong_but_executable) == 0.0
    print("agentic_verifiable_reward_check=passed")


if __name__ == "__main__":
    check_tokenwise_clip_and_response_reduction()
    check_masked_padding_is_inert()
    check_kl_is_tokenwise_nonnegative()
    check_equal_rewards_have_zero_advantage()
    check_grpo_gspo_and_old_sequence_ratio_are_distinct()
    check_forward_backward_and_optimizer_step()
    check_complete_train_step()
    check_agentic_grpo_policy_step()
    check_agentic_trainer_action_boundaries()
    check_agentic_reward_verifies_output()
    print("GRPO objective checks passed")
