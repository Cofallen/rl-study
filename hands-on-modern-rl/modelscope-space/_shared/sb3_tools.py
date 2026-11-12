from __future__ import annotations

import json
import time
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any, Callable, Iterator

import imageio.v2 as imageio
import numpy as np


def stable_baselines_runtime():
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.evaluation import evaluate_policy

    try:
        from stable_baselines3 import A2C
    except ImportError:
        A2C = None
    return {"DQN": DQN, "PPO": PPO, "A2C": A2C}, BaseCallback, evaluate_policy


def format_metrics(metrics: dict[str, Any], step: int) -> str:
    rows: list[tuple[str, str]] = []
    preferred = (
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "time/fps",
        "train/loss",
        "train/value_loss",
        "train/policy_gradient_loss",
        "train/entropy_loss",
        "train/learning_rate",
        "train/n_updates",
    )
    for key in preferred:
        value = metrics.get(key)
        if value is None:
            continue
        if isinstance(value, (float, np.floating)):
            rendered = f"{float(value):.6g}"
        else:
            rendered = str(value)
        rows.append((key, rendered))
    if not rows:
        return f"PPO/DQN update · total_timesteps={step:,}"
    width = max(len(name) for name, _ in rows)
    bar = "-" * (width + 22)
    body = "\n".join(f"| {name:<{width}} | {value:>15} |" for name, value in rows)
    return f"update · step {step:,}\n{bar}\n{body}\n{bar}"


def train_sb3(
    *,
    root: Path,
    task: Any,
    make_train_env: Callable[[], Any],
    make_eval_env: Callable[[], Any],
    make_record_env: Callable[[], Any],
    budget: int,
    learning_rate: float,
    gamma: float,
    epsilon: float,
    seed: int,
    checkpoints: int | None = None,
    record_episode: Callable[[Any, Any, Path, int], str],
) -> Iterator[dict[str, Any]]:
    algorithms, BaseCallback, evaluate_policy = stable_baselines_runtime()
    algorithm_name = str(getattr(task, "algorithm", task.get("algorithm") if isinstance(task, dict) else "PPO"))
    algorithm_cls = algorithms.get(algorithm_name)
    if algorithm_cls is None:
        raise RuntimeError(f"Unsupported Stable-Baselines3 algorithm: {algorithm_name}")

    checkpoint_count = int(
        checkpoints
        if checkpoints is not None
        else getattr(task, "checkpoints", task.get("checkpoints", 6) if isinstance(task, dict) else 6)
    )
    checkpoint_count = max(1, min(12, int(budget), checkpoint_count))
    checkpoint_targets = [
        max(1, round(int(budget) * index / checkpoint_count))
        for index in range(1, checkpoint_count + 1)
    ]
    checkpoint_targets[-1] = int(budget)
    checkpoint_deltas = [
        target - (checkpoint_targets[index - 1] if index else 0)
        for index, target in enumerate(checkpoint_targets)
    ]

    class MetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.latest: dict[str, Any] = {}

        def _on_step(self) -> bool:
            self.latest = dict(self.logger.name_to_value)
            return True

    train_env = make_train_env()
    eval_env = make_eval_env()
    kwargs = {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "seed": seed,
        "verbose": 0,
        "device": "cpu",
    }
    policy = getattr(task, "policy", task.get("policy", "MlpPolicy") if isinstance(task, dict) else "MlpPolicy")
    if algorithm_name == "DQN":
        kwargs.update(
            buffer_size=max(10_000, min(100_000, budget * 2)),
            learning_starts=max(100, min(2_000, budget // 10)),
            exploration_initial_eps=max(.05, epsilon),
            exploration_final_eps=max(.01, epsilon * .1),
            train_freq=4,
            gradient_steps=1,
            target_update_interval=max(250, min(2_000, budget // 8)),
        )
    elif algorithm_name in {"PPO", "A2C"}:
        # The UI promises one fixed interaction block per epoch.  Choose a
        # rollout length that divides every block, so SB3 does not silently
        # round an epoch upward while the page reports the requested step.
        common_delta = reduce(gcd, checkpoint_deltas)
        upper = min(512, common_delta)
        rollout_steps = next(
            (candidate for candidate in range(upper, 1, -1) if common_delta % candidate == 0),
            2,
        )
        kwargs.update(n_steps=rollout_steps, ent_coef=max(0.0, epsilon * .01))
        if algorithm_name == "PPO":
            # Choose a true divisor of the rollout buffer. This avoids a short
            # final mini-batch and the warning it emits on every learn() call.
            batch_size = next(
                candidate
                for candidate in (64, 50, 40, 32, 25, 20, 16, 10, 8, 5, 4, 2, 1)
                if rollout_steps % candidate == 0
            )
            kwargs.update(batch_size=batch_size, n_epochs=4)

    callback = MetricsCallback()
    model = algorithm_cls(policy, train_env, **kwargs)
    task_key = str(getattr(task, "key", task.get("key") if isinstance(task, dict) else "policy"))
    run_token = f"{int(time.time())}-{seed}"
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    x: list[float] = []
    y: list[float] = []
    yield {"phase": "training", "step": 0, "x": x, "y": y, "log": f"Initialized {algorithm_name} with {policy} on CPU"}
    completed = 0
    try:
        for checkpoint_index, target in enumerate(checkpoint_targets, start=1):
            current = target - completed
            model.learn(total_timesteps=current, reset_num_timesteps=False, callback=callback, progress_bar=False)
            completed = target
            rewards, lengths = evaluate_policy(model, eval_env, n_eval_episodes=3, deterministic=True, return_episode_rewards=True, warn=False)
            score = float(np.mean(rewards))
            spread = float(np.std(rewards))
            x.append(float(completed)); y.append(score)
            details = format_metrics(callback.latest, completed)
            details += f"\nEVAL step={completed:,} mean_reward={score:.2f} std={spread:.2f} mean_length={np.mean(lengths):.1f}"
            checkpoint_dir = artifacts / f"{task_key}-{run_token}-epoch-{checkpoint_index:02d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model_path = checkpoint_dir / "policy"
            model.save(str(model_path))
            model_file = str(model_path.with_suffix(".zip"))
            record_env = make_record_env()
            preview = record_episode(model, record_env, checkpoint_dir, seed + 10_000 + checkpoint_index)
            metadata = checkpoint_dir / "policy.json"
            metadata.write_text(
                json.dumps(
                    {
                        "algorithm": algorithm_name,
                        "policy": policy,
                        "training_step": completed,
                        "total_budget": budget,
                        "seed": seed,
                        "checkpoint_index": checkpoint_index,
                        "checkpoint_count": checkpoint_count,
                        "score": score,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            yield {
                "phase": "training",
                "step": completed,
                "score": score,
                "x": x,
                "y": y,
                "detail": f"{completed:,}/{budget:,} environment steps",
                "metric_detail": f"mean reward ± {spread:.2f}",
                "log": details + f"\nSAVE epoch={checkpoint_index}/{checkpoint_count} model={Path(model_file).name} replay={Path(preview).name}",
                "model": model_file,
                "preview": preview,
                "checkpoint_index": checkpoint_index,
                "checkpoint_count": checkpoint_count,
            }

        yield {
            "phase": "complete",
            "step": completed,
            "score": y[-1] if y else None,
            "x": x,
            "y": y,
            "log": f"Saved {checkpoint_index} independently selectable epoch policies and learned-policy replays",
        }
    finally:
        for env in (train_env, eval_env):
            try:
                env.close()
            except Exception:
                pass


def save_gif(frames: list[np.ndarray], path: Path, fps: int = 20) -> str:
    if not frames:
        raise RuntimeError("The environment returned no RGB frames for the learned-policy replay")
    normalized: list[np.ndarray] = []
    for frame in frames:
        array = np.asarray(frame)
        if array.ndim == 4:
            array = array[0]
        if array.shape[-1] == 4:
            array = array[..., :3]
        normalized.append(array.astype(np.uint8))
    imageio.mimsave(path, normalized, duration=1 / max(1, fps), loop=0)
    return str(path)
