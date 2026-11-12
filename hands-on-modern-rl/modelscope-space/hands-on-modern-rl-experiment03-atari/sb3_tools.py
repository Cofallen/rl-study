from __future__ import annotations

import json
import time
from datetime import datetime, timezone
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
        "rollout/exploration_rate",
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
    checkpoint_count: int | None = None,
    record_episode: Callable[[Any, Any, Path, int], str],
) -> Iterator[dict[str, Any]]:
    algorithms, BaseCallback, evaluate_policy = stable_baselines_runtime()
    algorithm_name = str(getattr(task, "algorithm", task.get("algorithm") if isinstance(task, dict) else "PPO"))
    algorithm_cls = algorithms.get(algorithm_name)
    if algorithm_cls is None:
        raise RuntimeError(f"Unsupported Stable-Baselines3 algorithm: {algorithm_name}")

    class MetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.latest: dict[str, Any] = {}
            self.stop_at: int | None = None

        def _on_step(self) -> bool:
            self.latest = dict(self.logger.name_to_value)
            return self.stop_at is None or self.num_timesteps < self.stop_at

    train_env = make_train_env()
    eval_env = make_eval_env()

    def configured(name: str, default: Any) -> Any:
        if isinstance(task, dict):
            return task.get(name, default)
        return getattr(task, name, default)

    requested_device = str(configured("device", "auto"))
    kwargs = {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "seed": seed,
        "verbose": 0,
        "device": requested_device,
    }
    resolved_config: dict[str, Any] = {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "device": requested_device,
    }

    policy = getattr(task, "policy", task.get("policy", "MlpPolicy") if isinstance(task, dict) else "MlpPolicy")
    if algorithm_name == "DQN":
        configured_learning_starts = int(configured("learning_starts", max(100, budget // 10)))
        learning_starts = max(100, min(configured_learning_starts, max(100, budget // 10)))
        buffer_size = max(10_000, min(int(configured("buffer_size", 100_000)), max(10_000, budget)))
        batch_size = max(8, int(configured("batch_size", 32)))
        configured_target_interval = int(configured("target_update_interval", 10_000))
        target_update_interval = max(250, min(configured_target_interval, max(250, budget // 20)))
        exploration_initial_eps = min(1.0, max(0.05, epsilon))
        exploration_final_eps = min(
            exploration_initial_eps,
            max(0.001, float(configured("exploration_final_eps", 0.01))),
        )
        exploration_fraction = min(1.0, max(0.01, float(configured("exploration_fraction", 0.2))))
        optimize_memory_usage = bool(configured("optimize_memory_usage", True))
        kwargs.update(
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=target_update_interval,
            max_grad_norm=10,
            optimize_memory_usage=optimize_memory_usage,
            replay_buffer_kwargs=(
                {"handle_timeout_termination": False}
                if optimize_memory_usage
                else None
            ),
        )
        resolved_config.update(
            {
                "buffer_size": buffer_size,
                "learning_starts": learning_starts,
                "batch_size": batch_size,
                "train_freq": 4,
                "gradient_steps": 1,
                "target_update_interval": target_update_interval,
                "exploration_fraction": exploration_fraction,
                "exploration_initial_eps": exploration_initial_eps,
                "exploration_final_eps": exploration_final_eps,
                "reward_clipping": True,
                "terminal_on_life_loss": True,
            }
        )
    elif algorithm_name in {"PPO", "A2C"}:
        rollout_steps = max(64, min(512, budget // 8))
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
    resolved_device = str(model.device)
    resolved_config["resolved_device"] = resolved_device
    budget_spec = configured("budget", (1, max(1, budget), budget, 1))
    recommended_budget = int(budget_spec[2]) if isinstance(budget_spec, (list, tuple)) and len(budget_spec) >= 3 else budget
    smoke_test = budget < recommended_budget
    checkpoints = (
        int(checkpoint_count)
        if checkpoint_count is not None
        else (2 if smoke_test else int(configured("checkpoints", 6)))
    )
    checkpoints = max(1, min(12, checkpoints))
    eval_episodes = 1 if smoke_test else max(1, int(configured("eval_episodes", 3)))
    resolved_config.update(
        {
            "recommended_budget": recommended_budget,
            "smoke_test": smoke_test,
            "checkpoints": checkpoints,
            "epochs": checkpoints,
            "steps_per_epoch": budget / checkpoints,
            "eval_episodes": eval_episodes,
        }
    )
    evaluation_steps = sorted({max(1, round(budget * index / checkpoints)) for index in range(1, checkpoints + 1)})
    progress_chunk = min(10_000, max(1_000, budget // 100)) if budget >= 1_000 else max(1, budget // 10)
    resolved_config["progress_chunk"] = progress_chunk
    resolved_config["schedule_total_timesteps"] = budget
    x: list[float] = []
    y: list[float] = []
    initialization_log = f"Initialized {algorithm_name} with {policy} on {resolved_device.upper()}"
    if algorithm_name == "DQN":
        initialization_log += (
            f"\nBASELINE replay_buffer={resolved_config['buffer_size']:,}"
            f" warmup={resolved_config['learning_starts']:,} batch={resolved_config['batch_size']}"
            f" target_update={resolved_config['target_update_interval']:,}"
            f" exploration={resolved_config['exploration_initial_eps']:.2f}→{resolved_config['exploration_final_eps']:.2f}"
            " train_reward=clipped eval_reward=raw"
            f" schedule_horizon={budget:,}"
        )
    if smoke_test:
        initialization_log += f"\nSMOKE_TEST evaluation={checkpoints} epochs × 1 episode"
    initialization_log += (
        f"\nTRAINING_PLAN epochs={checkpoints} steps_per_epoch≈{budget / checkpoints:,.0f}"
        " save_policy=every_epoch"
    )
    yield {"phase": "training", "step": 0, "x": x, "y": y, "log": initialization_log}
    completed = 0
    evaluation_index = 0
    training_started = time.perf_counter()
    best_score = float("-inf")
    best_step = 0
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    task_key = str(getattr(task, "key", task.get("key")))
    run_id = str(time.time_ns())
    checkpoint_records: list[tuple[Path, Path]] = []
    best_model_path: Path | None = None
    try:
        while completed < budget:
            next_evaluation = evaluation_steps[evaluation_index]
            current = min(progress_chunk, next_evaluation - completed, budget - completed)
            target_step = completed + current
            callback.stop_at = target_step
            # Tell SB3 that every streamed chunk belongs to the same full run.
            # The callback pauses at target_step, while SB3 keeps the learning-
            # rate and exploration schedules anchored to the original budget.
            remaining_budget = max(1, budget - int(model.num_timesteps))
            model.learn(total_timesteps=remaining_budget, reset_num_timesteps=False, callback=callback, progress_bar=False)
            completed = min(budget, int(model.num_timesteps))
            if completed < target_step:
                raise RuntimeError(
                    f"Training paused before the requested epoch progress boundary: {completed:,} < {target_step:,}"
                )
            elapsed = max(time.perf_counter() - training_started, 1e-6)
            throughput = completed / elapsed
            eta_seconds = max(0.0, (budget - completed) / max(throughput, 1e-6))
            stage = "replay warm-up" if completed < resolved_config.get("learning_starts", 0) else "gradient updates"
            progress_log = (
                f"PROGRESS step={completed:,}/{budget:,} phase={stage} "
                f"fps={throughput:.1f} eta={eta_seconds / 60:.1f}min"
            )

            if completed < next_evaluation:
                yield {
                    "phase": "training",
                    "step": completed,
                    "x": list(x),
                    "y": list(y),
                    "detail": f"{stage} · {completed:,}/{budget:,} environment steps",
                    "metric_detail": "Awaiting the end-of-epoch evaluation",
                    "log": progress_log,
                }
                continue

            yield {
                "phase": "evaluating",
                "step": completed,
                "x": list(x),
                "y": list(y),
                "detail": f"evaluating epoch {evaluation_index + 1}/{len(evaluation_steps)} · {completed:,}/{budget:,} environment steps",
                "metric_detail": f"Running {eval_episodes} deterministic evaluation episode(s)",
                "log": f"{progress_log}\nEVAL starting epoch {evaluation_index + 1}/{len(evaluation_steps)} at step={completed:,}",
            }
            rewards, lengths = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=eval_episodes,
                deterministic=True,
                return_episode_rewards=True,
                warn=False,
            )
            score = float(np.mean(rewards))
            spread = float(np.std(rewards))
            x.append(float(completed)); y.append(score)
            evaluation_index += 1

            model_stem = f"{task_key}-model-r{run_id}-step-{completed:09d}"
            model_path = artifacts / model_stem
            model.save(str(model_path))
            model_zip = model_path.with_suffix(".zip")
            metadata = artifacts / f"{model_stem}.json"
            metadata.write_text(json.dumps({
                "model_id": model_zip.name,
                "run_id": run_id,
                "task_key": task_key,
                "environment": getattr(task, "environment", task.get("environment")),
                "title": getattr(task, "title", task.get("title")),
                "algorithm": algorithm_name,
                "policy": policy,
                "budget": completed,
                "training_step": completed,
                "total_budget": budget,
                "checkpoint_index": evaluation_index,
                "checkpoint_count": len(evaluation_steps),
                "epoch": evaluation_index,
                "epochs": len(evaluation_steps),
                "steps_per_epoch": budget / len(evaluation_steps),
                "seed": seed,
                "score": score,
                "score_std": spread,
                "run_complete": False,
                "is_best": False,
                "training_config": resolved_config,
                "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }, indent=2), encoding="utf-8")
            checkpoint_records.append((model_zip, metadata))
            if score > best_score:
                best_score = score
                best_step = completed
                best_model_path = model_path
            details = format_metrics(callback.latest, completed)
            details += f"\nEVAL step={completed:,} mean_reward={score:.2f} std={spread:.2f} mean_length={np.mean(lengths):.1f}"
            details += f"\nEPOCH {evaluation_index}/{len(evaluation_steps)} complete"
            details += f"\nCHECKPOINT saved epoch model: {model_zip.name}"
            if completed == best_step:
                details += "\nCHECKPOINT this is the best policy so far and will be used for the final replay"
            yield {
                "phase": "training",
                "step": completed,
                "score": score,
                "x": x,
                "y": y,
                "detail": f"{completed:,}/{budget:,} environment steps",
                "metric_detail": f"mean reward ± {spread:.2f}",
                "log": details,
                "model": str(model_zip),
                "model_id": model_zip.name,
                "checkpoint_index": evaluation_index,
                "checkpoint_count": len(evaluation_steps),
            }

        if best_model_path is None or not best_model_path.with_suffix(".zip").is_file():
            raise RuntimeError("Training finished without a saved epoch model")
        model = algorithm_cls.load(str(best_model_path.with_suffix(".zip")), env=train_env, device=requested_device)
        record_env = make_record_env()
        raw_preview = Path(record_episode(model, record_env, artifacts, seed))
        preview_path = artifacts / f"{best_model_path.name}-preview.gif"
        if raw_preview.resolve() != preview_path.resolve():
            raw_preview.replace(preview_path)
        best_model_zip = best_model_path.with_suffix(".zip")
        for checkpoint_zip, metadata in checkpoint_records:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            payload.update({
                "run_complete": True,
                "is_best": checkpoint_zip == best_model_zip,
                "best_checkpoint_step": best_step,
            })
            if checkpoint_zip == best_model_zip:
                payload["preview"] = str(preview_path)
            metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        yield {
            "phase": "complete",
            "step": completed,
            "score": best_score if np.isfinite(best_score) else (y[-1] if y else None),
            "x": x,
            "y": y,
            "preview": str(preview_path),
            "model": str(best_model_zip),
            "model_id": best_model_zip.name,
            "log": (
                f"Restored best epoch model from step {best_step:,} with mean reward {best_score:.2f}"
                f"\nSaved {len(checkpoint_records)} selectable epoch policy models for run {run_id}"
                f"\nGenerated the best epoch replay: {preview_path.name}"
            ),
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
