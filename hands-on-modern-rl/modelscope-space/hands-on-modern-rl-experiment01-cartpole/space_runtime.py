from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def _task_preview() -> str:
    output = ARTIFACTS / "cartpole-environment.png"
    if output.is_file():
        return str(output)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    try:
        env.reset(seed=42)
        imageio.imwrite(output, env.render())
    finally:
        env.close()
    return str(output)


SPACE = {
    "title": {"en": "CartPole PPO Online Training", "zh": "CartPole PPO 在线训练"},
    "description": {
        "en": "Train CartPole from scratch on CPU, save one evaluated policy per epoch, and replay the exact policy selected below.",
        "zh": "使用 CPU 从零训练 CartPole；每个 epoch 保存并评估一个策略，下方回放始终来自当前选中的真实模型。",
    },
    "badge": "EXPERIMENT 01 · CARTPOLE",
    "training_guide": {
        "success": {
            "en": "A 10-episode mean reward of at least 475 solves CartPole-v1. Compare several epochs instead of treating completion as success.",
            "zh": "CartPole-v1 的 10 回合平均奖励达到 475 分表示任务解决。请比较多个 epoch，不要把“训练完成”当作学会。",
        },
        "preview": {
            "en": "Choose an epoch model below. Its GIF is a fresh deterministic rollout of that exact saved PPO policy.",
            "zh": "在下方选择 epoch 模型；GIF 是该份 PPO 模型重新执行的确定性回放。",
        },
        "time": {
            "en": "The recommended 5,000 steps × 6 epochs normally takes about 30–60 seconds on the shared CPU container.",
            "zh": "推荐的每 epoch 5,000 步 × 6 个 epoch，在共享 CPU 容器上通常需要约 30–60 秒。",
        },
    },
    "device": "CPU",
    "organization_url": "https://modelscope.cn/organization/walkinglab",
    "project_url": "https://github.com/walkinglabs/hands-on-modern-rl",
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/chapter01_cartpole/training",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment01-cartpole/file/view/master/space_runtime.py",
    "notebook_url": "https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment01-cartpole.ipynb",
}


TASKS = [{
    "key": "cartpole-ppo",
    "title": {"en": "CartPole-v1 · PPO", "zh": "CartPole-v1 · PPO"},
    "environment": "CartPole-v1",
    "description": {
        "en": "Move the cart left or right so the pole remains upright for the full 500-step episode.",
        "zh": "控制小车左右移动，使杆在最长 500 步的回合中持续保持竖直。",
    },
    "observation": {"en": "Box(4): position and velocity", "zh": "Box(4)：位置与速度"},
    "action": {"en": "Discrete(2): left / right", "zh": "Discrete(2)：左移 / 右移"},
    "algorithm": "PPO",
    "preview": _task_preview(),
    "budget": (6_000, 120_000, 30_000, 1_000),
    "learning_rate": (1e-5, 0.003, 0.0003, 1e-5),
    "gamma": (0.9, 1.0, 0.99, 0.005),
    "epsilon": (0.0, 0.1, 0.0, 0.005),
    "checkpoints": 6,
    "baseline_name": "CartPole PPO learning baseline",
    "baseline_time": "about 30–60 s on CPU",
    "baseline_outcome": {
        "en": "Mean reward rises from roughly 20 toward 475–500, and the replay keeps the pole upright.",
        "zh": "平均奖励从约 20 分逐渐上升到 475–500 分，回放中杆能够持续保持直立。",
    },
}]


def runtime_status() -> str:
    return "Gymnasium CartPole-v1 · Stable-Baselines3 PPO · CPU READY"


def _evaluate(model: PPO, episodes: int, seed: int) -> tuple[float, float]:
    env = gym.make("CartPole-v1")
    try:
        rewards, _ = evaluate_policy(
            model,
            env,
            n_eval_episodes=episodes,
            deterministic=True,
            return_episode_rewards=True,
            warn=False,
        )
    finally:
        env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def _record(model: PPO, output: Path, seed: int) -> tuple[str, float]:
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames: list[np.ndarray] = []
    score = 0.0
    try:
        observation, _ = env.reset(seed=seed)
        for _ in range(500):
            frames.append(np.asarray(env.render()))
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            score += float(reward)
            if terminated or truncated:
                break
    finally:
        env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, duration=1 / 30, loop=0)
    return str(output), score


def _ppo_rollout_size(steps_per_epoch: int) -> tuple[int, int]:
    upper = min(1_024, max(2, steps_per_epoch))
    n_steps = next((value for value in range(upper, 1, -1) if steps_per_epoch % value == 0), upper)
    batch_size = next(
        (value for value in (64, 50, 40, 32, 25, 20, 16, 10, 8, 5, 4, 2) if value <= n_steps and n_steps % value == 0),
        2,
    )
    return n_steps, batch_size


def run(
    key: str,
    budget: int,
    learning_rate: float,
    gamma: float,
    epsilon: float,
    seed: int,
    checkpoints: int | None = None,
) -> Iterator[dict[str, Any]]:
    task = next(item for item in TASKS if item["key"] == key)
    epoch_count = max(1, min(12, int(checkpoints or task["checkpoints"])))
    steps_per_epoch = max(1, int(budget) // epoch_count)
    n_steps, batch_size = _ppo_rollout_size(steps_per_epoch)
    env = gym.make("CartPole-v1")
    env.reset(seed=int(seed))
    model = PPO(
        "MlpPolicy",
        env,
        seed=int(seed),
        verbose=0,
        device="cpu",
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=float(learning_rate),
        gamma=float(gamma),
        ent_coef=max(0.0, float(epsilon)),
    )
    x: list[float] = [0.0]
    initial_score, initial_std = _evaluate(model, 5, int(seed) + 100)
    y: list[float] = [initial_score]
    yield {
        "phase": "training",
        "step": 0,
        "score": initial_score,
        "x": x,
        "y": y,
        "metric_detail": f"initial mean reward ± {initial_std:.1f}",
        "log": f"PPO initialized · n_steps={n_steps} batch_size={batch_size} initial_reward={initial_score:.1f}",
    }
    run_token = f"{int(time.time())}-{seed}"
    completed = 0
    try:
        for epoch in range(1, epoch_count + 1):
            model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False, progress_bar=False)
            completed = min(int(budget), epoch * steps_per_epoch)
            mean_reward, std_reward = _evaluate(model, 5, int(seed) + 100 + epoch)
            epoch_dir = ARTIFACTS / f"cartpole-{run_token}-epoch-{epoch:02d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(epoch_dir / "policy"))
            model_file = epoch_dir / "policy.zip"
            preview, replay_score = _record(model, epoch_dir / "learned-policy.gif", int(seed) + 10_000 + epoch)
            (epoch_dir / "metadata.json").write_text(json.dumps({
                "environment": "CartPole-v1",
                "algorithm": "PPO",
                "epoch": epoch,
                "epochs": epoch_count,
                "steps_per_epoch": steps_per_epoch,
                "training_step": completed,
                "mean_reward": mean_reward,
                "reward_std": std_reward,
                "replay_reward": replay_score,
                "seed": int(seed),
            }, indent=2), encoding="utf-8")
            x.append(float(completed)); y.append(mean_reward)
            yield {
                "phase": "training",
                "step": completed,
                "score": mean_reward,
                "x": x,
                "y": y,
                "model": str(model_file),
                "preview": preview,
                "checkpoint_index": epoch,
                "checkpoint_count": epoch_count,
                "metric_detail": f"5-episode mean ± {std_reward:.1f}",
                "log": (
                    f"EPOCH {epoch}/{epoch_count} step={completed:,} mean_reward={mean_reward:.1f} std={std_reward:.1f}\n"
                    f"SAVE model={model_file.name} replay_reward={replay_score:.0f}"
                ),
            }
    finally:
        env.close()
    final_score, final_std = _evaluate(model, 10, int(seed) + 20_000)
    yield {
        "phase": "complete",
        "step": completed,
        "score": final_score,
        "x": x,
        "y": y,
        "metric_detail": f"10-episode final mean ± {final_std:.1f}",
        "log": f"Saved {epoch_count} independently selectable PPO policies · final_mean={final_score:.1f}",
    }
