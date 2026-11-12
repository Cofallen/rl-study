from __future__ import annotations

from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent

SPACE = {
    "title": {"en": "JAX MinAtar CPU Game Lab", "zh": "JAX MinAtar CPU 游戏训练场"},
    "description": {
        "en": "Compile an entire pixel-game policy on CPU with JAX, train it in compact MinAtar worlds, and replay the learned semantic game state.",
        "zh": "使用 JAX 在 CPU 上编译像素游戏策略，在紧凑的 MinAtar 世界中训练，并回放学习后的语义游戏状态。",
    },
    "badge": "EXPERIMENT 07 · JAX",
    "training_guide": {
        "success": {"en": "Mean return should improve after compilation, and the semantic replay should show the learned policy taking useful game actions.", "zh": "编译完成后平均回报应提高，语义画面回放中应能看到策略做出有效的游戏动作。"},
        "preview": {"en": "The final Preview is rendered from this run's compact semantic game state, not a decorative screenshot. Compare its behavior with the learning curve.", "zh": "最终 Preview 由本次运行的紧凑语义游戏状态渲染，并非装饰截图；请将回放行为与学习曲线结合判断。"},
        "time": {"en": "The first run usually takes 30 seconds–2 minutes including JAX compilation; warm runs are often 10–60 seconds.", "zh": "首次运行包含 JAX 编译，通常需要 30 秒到 2 分钟；完成预热后一般为 10–60 秒。"},
    },
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/chapter07_dqn/dqn-family",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment07-jax-games/file/view/master/space_runtime.py",
    "notebook_url": "https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/"
    "code/online-experiments/hands-on-modern-rl-experiment07-jax-games.ipynb",
}


def task(key, title, environment, description, action, preview, colors):
    return {
        "key": key,
        "title": {"en": title, "zh": title},
        "environment": environment,
        "description": description,
        "observation": {"en": "10×10 semantic pixel channels", "zh": "10×10 语义像素通道"},
        "action": action,
        "algorithm": "JAX REINFORCE",
        "preview": preview,
        "budget": (20, 5_000, 600, 20),
        "training_unit": {"en": "training episodes", "zh": "训练回合"},
        "learning_rate": (1e-5, 0.01, 0.001, 1e-5),
        "gamma": (0.8, 1.0, 0.99, 0.005),
        "epsilon": (0.0, 0.5, 0.05, 0.01),
        "checkpoints": 8,
        "max_steps": 1_000,
        "colors": colors,
        "baseline_name": "JAX REINFORCE learning baseline",
        "baseline_time": {"en": "about 2–12 minutes on CPU after JIT compilation", "zh": "JIT 编译完成后，CPU 上约 2–12 分钟"},
        "baseline_outcome": {"en": "Mean evaluation return rises above early epochs and the semantic replay scores or survives longer.", "zh": "平均评估回报高于早期 epoch，语义回放中得分提高或存活时间变长。"},
    }


TASKS = [
    task("asterix", "Asterix MinAtar", "Asterix-MinAtar", {"en": "Move through falling enemies and collect treasure in a compact arcade world.", "zh": "在紧凑街机世界中躲避下落的敌人并收集宝物。"}, {"en": "Move left / right / up / down / stay", "zh": "上下左右移动或停留"}, "assets/asterix.png", ["#5b5ce2", "#f59e0b", "#ef4444", "#22c55e"]),
    task("breakout", "Breakout MinAtar", "Breakout-MinAtar", {"en": "Control the paddle, return the ball, and remove every brick.", "zh": "控制球拍反弹小球并清除砖块。"}, {"en": "Left / right / stay", "zh": "左移、右移、停留"}, "assets/breakout.png", ["#f8fafc", "#5b5ce2", "#f97316", "#22c55e"]),
    task("freeway", "Freeway MinAtar", "Freeway-MinAtar", {"en": "Cross traffic lanes repeatedly without colliding with moving cars.", "zh": "反复穿越车流，并避免和移动的汽车相撞。"}, {"en": "Up / down / stay", "zh": "向上、向下、停留"}, "assets/freeway.png", ["#22c55e", "#ef4444", "#f8fafc"]),
    task("space-invaders", "Space Invaders MinAtar", "SpaceInvaders-MinAtar", {"en": "Defend the bottom row by moving, firing, and using bunkers.", "zh": "通过移动、射击和利用掩体防守底部区域。"}, {"en": "Left / right / fire / stay", "zh": "左移、右移、射击、停留"}, "assets/space-invaders.png", ["#22c55e", "#ef4444", "#f8fafc", "#5b5ce2"]),
]


def runtime_status():
    try:
        import gymnax
        import jax

        env, params = gymnax.make("Breakout-MinAtar")
        env.reset(jax.random.PRNGKey(0), params)
        return f"JAX {jax.__version__} · XLA CPU READY"
    except Exception as exc:
        return f"startup check pending · {type(exc).__name__}"


def _flatten_observation(observation) -> np.ndarray:
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def _semantic_frame(observation, colors: list[str]) -> np.ndarray:
    array = np.asarray(observation)
    if array.ndim == 2:
        array = array[..., None]
    height, width, channels = array.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:] = (10, 15, 38)
    for channel in range(min(channels, len(colors))):
        color = colors[channel].lstrip("#")
        value = np.array([int(color[index:index + 2], 16) for index in (0, 2, 4)], dtype=np.uint8)
        mask = array[..., channel] > 0
        rgb[mask] = value
    return np.asarray(Image.fromarray(rgb).resize((480, 480), Image.Resampling.NEAREST))


def _init_network(jax, key, input_size: int, actions: int):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (input_size, 64)) * np.sqrt(2 / input_size),
        "b1": jax.numpy.zeros((64,)),
        "w2": jax.random.normal(k2, (64, actions)) * .05,
        "b2": jax.numpy.zeros((actions,)),
    }


def run(
    key: str,
    budget: int,
    learning_rate: float,
    gamma: float,
    epsilon: float,
    seed: int,
    checkpoints: int | None = None,
):
    import gymnax
    import jax
    import jax.numpy as jnp
    import optax

    task = next(item for item in TASKS if item["key"] == key)
    env, env_params = gymnax.make(task["environment"])
    rng = jax.random.PRNGKey(seed)
    rng, reset_key, network_key = jax.random.split(rng, 3)
    first_observation, _ = env.reset(reset_key, env_params)
    input_size = int(np.prod(first_observation.shape))
    actions = int(env.num_actions)
    network = _init_network(jax, network_key, input_size, actions)
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(network)

    def logits(params, observations):
        hidden = jax.nn.relu(observations @ params["w1"] + params["b1"])
        return hidden @ params["w2"] + params["b2"]

    @jax.jit
    def update(params, state, observations, selected_actions, advantages):
        def loss_fn(current):
            log_probs = jax.nn.log_softmax(logits(current, observations))
            chosen = jnp.take_along_axis(log_probs, selected_actions[:, None], axis=1).squeeze(1)
            entropy = -jnp.mean(jnp.sum(jnp.exp(log_probs) * log_probs, axis=1))
            return -jnp.mean(chosen * advantages) - .01 * entropy
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    def episode(params, episode_key, explore: bool, capture: bool = False):
        episode_key, reset = jax.random.split(episode_key)
        observation, state = env.reset(reset, env_params)
        observations: list[np.ndarray] = []
        selected: list[int] = []
        rewards: list[float] = []
        frames: list[np.ndarray] = []
        total = 0.0
        for step in range(int(task["max_steps"])):
            if capture and (step % 2 == 0 or step < 10):
                frames.append(_semantic_frame(observation, task["colors"]))
            flat = _flatten_observation(observation)
            episode_key, action_key, step_key = jax.random.split(episode_key, 3)
            distribution = jax.nn.softmax(logits(params, jnp.asarray(flat)))
            if explore and float(jax.random.uniform(action_key)) < epsilon:
                action = int(jax.random.randint(action_key, (), 0, actions))
            elif explore:
                action = int(jax.random.choice(action_key, actions, p=distribution))
            else:
                action = int(jnp.argmax(distribution))
            next_observation, state, reward, done, _ = env.step(step_key, state, action, env_params)
            observations.append(flat); selected.append(action); rewards.append(float(reward)); total += float(reward)
            observation = next_observation
            if bool(done):
                break
        return total, observations, selected, rewards, frames, episode_key

    yield {"step": 0, "x": [], "y": [], "log": f"JIT compiling policy update · observation={first_observation.shape} actions={actions}"}
    x: list[float] = []
    y: list[float] = []
    checkpoint_count = max(1, min(int(budget), min(12, int(checkpoints or task["checkpoints"]))))
    checkpoint_targets = [
        max(1, round(int(budget) * index / checkpoint_count))
        for index in range(1, checkpoint_count + 1)
    ]
    checkpoint_targets[-1] = int(budget)
    run_token = f"{int(time.time())}-{seed}"
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    recent_losses: list[float] = []
    checkpoint_index = 0
    for episode_index in range(1, budget + 1):
        rng, episode_key = jax.random.split(rng)
        _, observations, selected, rewards, _, _ = episode(network, episode_key, explore=True)
        returns = []
        value = 0.0
        for reward in reversed(rewards):
            value = reward + gamma * value
            returns.append(value)
        returns.reverse()
        advantages = np.asarray(returns, dtype=np.float32)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        network, optimizer_state, loss = update(
            network, optimizer_state, jnp.asarray(np.asarray(observations)),
            jnp.asarray(np.asarray(selected), dtype=jnp.int32), jnp.asarray(advantages),
        )
        recent_losses.append(float(loss))
        if episode_index == checkpoint_targets[checkpoint_index]:
            evaluations = []
            for offset in range(5):
                rng, eval_key = jax.random.split(rng)
                score, *_ = episode(network, eval_key, explore=False)
                evaluations.append(score)
            score = float(np.mean(evaluations))
            x.append(float(episode_index)); y.append(score)
            rng, record_key = jax.random.split(rng)
            replay_score, _, _, _, frames, _ = episode(network, record_key, explore=False, capture=True)
            if not frames:
                raise RuntimeError("Gymnax returned no semantic frames for replay")
            checkpoint_index += 1
            epoch_dir = artifacts / f"{key}-{run_token}-epoch-{checkpoint_index:02d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model_path = epoch_dir / "policy.npz"
            np.savez(model_path, **{name: np.asarray(value) for name, value in network.items()})
            preview = epoch_dir / "learned-policy.gif"
            imageio.mimsave(preview, frames, duration=.08, loop=0)
            window = max(1, checkpoint_targets[checkpoint_index - 1] - (checkpoint_targets[checkpoint_index - 2] if checkpoint_index > 1 else 0))
            yield {
                "step": episode_index,
                "score": score,
                "x": x,
                "y": y,
                "model": str(model_path),
                "preview": str(preview),
                "checkpoint_index": checkpoint_index,
                "checkpoint_count": checkpoint_count,
                "metric_detail": f"mean return · std={np.std(evaluations):.2f}",
                "log": (
                    f"JAX epoch={checkpoint_index}/{checkpoint_count} episode={episode_index:,} "
                    f"loss={np.mean(recent_losses[-window:]):.6f} eval_return={score:.3f} "
                    f"replay_return={replay_score:.3f}\n"
                    f"SAVE model={model_path.name} replay={preview.name} frames={len(frames)}"
                ),
            }

    yield {
        "phase": "complete",
        "step": budget,
        "score": y[-1] if y else None,
        "x": x,
        "y": y,
        "log": f"Saved {checkpoint_index} independently selectable JAX policies and semantic replays",
    }
