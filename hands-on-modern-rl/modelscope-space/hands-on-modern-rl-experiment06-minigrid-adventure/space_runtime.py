from __future__ import annotations

from pathlib import Path

import numpy as np

from sb3_tools import save_gif, train_sb3


ROOT = Path(__file__).resolve().parent

SPACE = {
    "title": {"en": "MiniGrid Exploration Adventures", "zh": "MiniGrid 探索与冒险训练场"},
    "description": {
        "en": "Train agents in compact mazes with doors, keys, obstacles, and partial observations, then replay what the policy learned to explore.",
        "zh": "在包含门、钥匙、障碍和部分可观测信息的小型迷宫中训练智能体，并回放策略学到的探索行为。",
    },
    "badge": "EXPERIMENT 06 · EXPLORATION",
    "training_guide": {
        "success": {"en": "Success rate or episode return should improve, and the replay should reach the goal or complete the required key, door, and obstacle sequence.", "zh": "成功率或回合回报应提高，回放中应抵达目标，或正确完成钥匙、门和障碍物组成的交互顺序。"},
        "preview": {"en": "After training, Preview becomes this run's MiniGrid replay. Follow the agent's field of view and check whether exploration leads to the task goal.", "zh": "训练后，Preview 会变为本次 MiniGrid 回放；请跟随智能体的局部视野，检查探索是否最终到达任务目标。"},
        "time": {"en": "A smoke run takes under 2 minutes. Recommended PPO learning baselines usually need about 3–30 minutes on CPU; sparse-reward mazes take longest.", "zh": "短流程验证通常不超过 2 分钟；推荐的 PPO 学习配方在 CPU 上约需 3–30 分钟，稀疏奖励迷宫耗时最长。"},
    },
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/chapter14_exploration_marl_hierarchical/intrinsic-motivation-exploration",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment06-minigrid-adventure/file/view/master/space_runtime.py",
    "notebook_url": "https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/"
    "code/online-experiments/hands-on-modern-rl-experiment06-minigrid-adventure.ipynb",
}


def task(key, title, environment, description, preview, budget):
    return {
        "key": key,
        "title": {"en": title, "zh": title},
        "environment": environment,
        "description": description,
        "observation": {"en": "Agent-centered partial RGB view", "zh": "以智能体为中心的局部 RGB 视野"},
        "action": {"en": "Turn / move / interact", "zh": "转向、移动、交互"},
        "algorithm": "PPO",
        "policy": "CnnPolicy",
        "preview": preview,
        "budget": budget,
        "learning_rate": (1e-5, 0.003, 0.0003, 1e-5),
        "gamma": (0.8, 1.0, 0.99, 0.005),
        "epsilon": (0.0, 0.2, 0.01, 0.005),
        "checkpoints": 6,
        "max_steps": 500,
        "baseline_name": "MiniGrid PPO learning baseline",
        "baseline_time": {"en": "about 3–30 minutes on CPU", "zh": "CPU 上约 3–30 分钟"},
        "baseline_outcome": {"en": "Episode reward and success frequency rise, and the replay completes the selected navigation sequence.", "zh": "回合奖励与成功频率上升，回放能够完成所选导航与交互序列。"},
    }


TASKS = [
    task("empty", "Reach the Goal", "MiniGrid-Empty-6x6-v0", {"en": "Learn basic egocentric navigation to a fixed goal.", "zh": "学习以自我视角导航到固定目标。"}, "assets/empty.png", (2_000, 500_000, 100_000, 2_000)),
    task("doorkey", "Door & Key", "MiniGrid-DoorKey-6x6-v0", {"en": "Find the key, carry it to the locked door, open it, and reach the goal.", "zh": "寻找钥匙、携带钥匙打开上锁的门，然后到达目标。"}, "assets/doorkey.png", (5_000, 1_000_000, 400_000, 5_000)),
    task("multiroom", "Multi-Room", "MiniGrid-MultiRoom-N2-S4-v0", {"en": "Explore connected rooms and remember which doorway advances toward the goal.", "zh": "探索相连房间，并记住哪一扇门通向目标。"}, "assets/multiroom.png", (5_000, 1_000_000, 400_000, 5_000)),
    task("unlock", "Unlock the Door", "MiniGrid-Unlock-v0", {"en": "Search for a key and solve the interaction sequence required to unlock a door.", "zh": "寻找钥匙，并完成开门所需的交互顺序。"}, "assets/unlock.png", (5_000, 1_000_000, 400_000, 5_000)),
    task("obstructed", "Obstructed Maze", "MiniGrid-ObstructedMaze-1Q-v1", {"en": "Move objects, uncover a key, and plan around an obstructed doorway.", "zh": "移动物体、找到钥匙，并绕过被阻挡的门口。"}, "assets/obstructed.png", (10_000, 2_000_000, 800_000, 10_000)),
    task("dynamic", "Dynamic Obstacles", "MiniGrid-Dynamic-Obstacles-6x6-v0", {"en": "Reach the goal while reacting to obstacles that move after every action.", "zh": "在障碍物每一步都会移动的情况下到达目标。"}, "assets/dynamic.png", (5_000, 1_500_000, 500_000, 5_000)),
]


def runtime_status():
    try:
        import gymnasium as gym
        import minigrid

        env = gym.make("MiniGrid-Empty-6x6-v0", render_mode="rgb_array")
        env.reset(seed=0)
        env.close()
        return f"MiniGrid {minigrid.__version__} · READY"
    except Exception as exc:
        return f"startup check pending · {type(exc).__name__}"


def _make_env(environment: str, seed: int):
    import gymnasium as gym
    from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

    env = gym.make(environment, render_mode="rgb_array")
    env = RGBImgPartialObsWrapper(env, tile_size=8)
    env = ImgObsWrapper(env)
    env.reset(seed=seed)
    return env


def _record(model, env, artifacts: Path, seed: int, task):
    observation, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    for step in range(int(task["max_steps"])):
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
    return save_gif(frames, artifacts / f"{task['key']}-learned-policy.gif", fps=8)


def run(key: str, budget: int, learning_rate: float, gamma: float, epsilon: float, seed: int, checkpoints: int | None = None):
    task = next(item for item in TASKS if item["key"] == key)
    environment = task["environment"]
    yield from train_sb3(
        root=ROOT,
        task=task,
        make_train_env=lambda: _make_env(environment, seed),
        make_eval_env=lambda: _make_env(environment, seed + 1_000),
        make_record_env=lambda: _make_env(environment, seed + 10_000),
        budget=budget,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        seed=seed,
        checkpoints=checkpoints,
        record_episode=lambda model, env, artifacts, record_seed: _record(model, env, artifacts, record_seed, task),
    )
