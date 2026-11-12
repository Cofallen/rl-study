"""A browser-based collection of small CPU Gymnasium training experiments."""

from __future__ import annotations

import base64
from collections import defaultdict
import ctypes.util
from functools import lru_cache
import html
import importlib
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hands-on-modern-rl-matplotlib")
if sys.platform.startswith("linux") and ctypes.util.find_library("OSMesa"):
    # Prefer Mesa's CPU renderer when the Studio image provides it. Guarding
    # the setting preserves Gymnasium-Robotics registration on base images
    # where the optional system library is unavailable.
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import gradio as gr
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

# Prefer the CPU container's off-screen renderers before MuJoCo is imported.
# EGL is fastest when available; OSMesa remains installed as a fallback.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


ROOT = Path(__file__).parent
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
PREVIEW_DIR = ROOT / "assets" / "previews"
CARD_BACKGROUND_DIR = ROOT / "assets" / "card-backgrounds"
TASK_CARD_DIR = ROOT / "assets" / "task-cards"
LOGO_PATH = ROOT / "assets" / "readmelogo.png"
LOGO_DATA_URI = f"data:image/png;base64,{base64.b64encode(LOGO_PATH.read_bytes()).decode()}"
# These immutable card assets are served directly instead of being copied into
# Gradio's per-component cache on every Gallery update. Their stable URLs let
# the browser reuse decoded images when learners switch paths and goals.
gr.set_static_paths(paths=[CARD_BACKGROUND_DIR, TASK_CARD_DIR])

PROJECT_URL = "https://github.com/walkinglabs/hands-on-modern-rl"
COURSE_URL = "https://walkinglabs.github.io/hands-on-modern-rl/"
MODELSCOPE_NOTEBOOK_URL = (
    "https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/"
    "code/online-experiments/hands-on-modern-rl-experiment-gymnasium.ipynb"
)
BANDIT = "Bandit · ε-greedy"
BLACKJACK = "Blackjack · Monte Carlo"
GRIDWORLD = "GridWorld · Q-Learning"
FROZENLAKE = "FrozenLake · Q-Learning"
CLIFF = "CliffWalking · SARSA"
TAXI = "Taxi · Q-Learning"
CARTPOLE_DQN = "CartPole · DQN"
CARTPOLE_PPO = "CartPole · PPO"
MOUNTAINCAR = "MountainCar · Tabular Q"
ACROBOT = "Acrobot · PPO"
PENDULUM = "Pendulum · PPO"
MOUNTAINCAR_CONTINUOUS = "MountainCarContinuous · SAC"

CURATED_PREVIEWS = {
    BANDIT: "bandit-arm-estimates.png",
    BLACKJACK: "blackjack-policy.png",
    GRIDWORLD: "gridworld-policy.png",
    FROZENLAKE: "frozenlake-policy.png",
    CLIFF: "cliffwalking-trained.gif",
    TAXI: "taxi-trained.gif",
    CARTPOLE_DQN: "cartpole-dqn-trained.gif",
    CARTPOLE_PPO: "cartpole-ppo-trained.gif",
    MOUNTAINCAR: "mountaincar-trained.gif",
    ACROBOT: "acrobot-trained.gif",
    PENDULUM: "pendulum-trained.gif",
    MOUNTAINCAR_CONTINUOUS: "mountaincarcontinuous-trained.gif",
}

CHAPTER_URLS = {
    BANDIT: f"{COURSE_URL}chapter03_mdp/bandit",
    BLACKJACK: f"{COURSE_URL}chapter04_tabular",
    GRIDWORLD: f"{COURSE_URL}chapter03_mdp/value-experiment",
    FROZENLAKE: f"{COURSE_URL}chapter04_tabular",
    CLIFF: f"{COURSE_URL}chapter04_tabular",
    TAXI: f"{COURSE_URL}chapter04_tabular",
    CARTPOLE_DQN: f"{COURSE_URL}chapter07_dqn/from-q-to-dqn",
    CARTPOLE_PPO: f"{COURSE_URL}chapter09_actor_critic",
    MOUNTAINCAR: f"{COURSE_URL}chapter07_dqn/from-q-to-dqn",
    ACROBOT: f"{COURSE_URL}chapter09_actor_critic",
    PENDULUM: f"{COURSE_URL}chapter09_actor_critic/pendulum",
    MOUNTAINCAR_CONTINUOUS: f"{COURSE_URL}chapter11_continuous_control",
}
SCRIPT_URL = (
    "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment-gymnasium/"
    "file/view/master/app.py"
)


EXPERIMENTS = {
    BANDIT: {
        "environment": "4-armed Bernoulli bandit",
        "family": "Bandit",
        "algorithm": "ε-greedy",
        "budget": (200, 10000, 2000, 200),
        "alpha": (0.01, 1.0, 0.1, 0.01),
        "gamma": (0.0, 1.0, 0.0, 0.05),
        "epsilon": (0.0, 1.0, 0.1, 0.01),
        "gamma_visible": False,
    },
    BLACKJACK: {
        "environment": "Blackjack-v1",
        "family": "Toy Text",
        "algorithm": "First-visit Monte Carlo",
        "budget": (5000, 100000, 30000, 5000),
        "alpha": (0.001, 0.2, 0.02, 0.001),
        "gamma": (0.0, 1.0, 1.0, 0.01),
        "epsilon": (0.0, 1.0, 0.2, 0.01),
        "gamma_visible": True,
    },
    GRIDWORLD: {
        "environment": "Custom 4×4 GridWorld",
        "family": "Tabular",
        "algorithm": "Q-Learning",
        "budget": (100, 5000, 1000, 100),
        "alpha": (0.01, 1.0, 0.15, 0.01),
        "gamma": (0.0, 1.0, 0.95, 0.01),
        "epsilon": (0.0, 1.0, 0.2, 0.01),
        "gamma_visible": True,
    },
    FROZENLAKE: {
        "environment": "FrozenLake-v1",
        "family": "Toy Text",
        "algorithm": "Q-Learning",
        "budget": (1000, 30000, 10000, 1000),
        "alpha": (0.01, 1.0, 0.2, 0.01),
        "gamma": (0.0, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 1.0, 1.0, 0.01),
        "gamma_visible": True,
    },
    CLIFF: {
        "environment": "CliffWalking-v1",
        "family": "Toy Text",
        "algorithm": "SARSA",
        "budget": (200, 5000, 1500, 100),
        "alpha": (0.01, 1.0, 0.5, 0.01),
        "gamma": (0.0, 1.0, 1.0, 0.01),
        "epsilon": (0.0, 1.0, 0.1, 0.01),
        "gamma_visible": True,
    },
    TAXI: {
        "environment": "Taxi-v4",
        "family": "Toy Text",
        "algorithm": "Q-Learning",
        "budget": (1000, 30000, 8000, 1000),
        "alpha": (0.01, 1.0, 0.2, 0.01),
        "gamma": (0.0, 1.0, 0.95, 0.01),
        "epsilon": (0.0, 1.0, 1.0, 0.01),
        "gamma_visible": True,
    },
    CARTPOLE_DQN: {
        "environment": "CartPole-v1",
        "family": "Classic Control",
        "algorithm": "DQN",
        "budget": (5000, 100000, 30000, 5000),
        "alpha": (0.00001, 0.001, 0.0001, 0.00001),
        "gamma": (0.8, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 1.0, 1.0, 0.05),
        "gamma_visible": True,
    },
    CARTPOLE_PPO: {
        "environment": "CartPole-v1",
        "family": "Classic Control",
        "algorithm": "PPO",
        "budget": (5000, 100000, 30000, 5000),
        "alpha": (0.00001, 0.003, 0.0003, 0.00001),
        "gamma": (0.8, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 0.05, 0.0, 0.005),
        "gamma_visible": True,
    },
    MOUNTAINCAR: {
        "environment": "MountainCar-v0",
        "family": "Classic Control",
        "algorithm": "Tabular Q-Learning",
        "budget": (1000, 20000, 6000, 1000),
        "alpha": (0.01, 1.0, 0.12, 0.01),
        "gamma": (0.0, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 1.0, 1.0, 0.01),
        "gamma_visible": True,
    },
    ACROBOT: {
        "environment": "Acrobot-v1",
        "family": "Classic Control",
        "algorithm": "PPO",
        "budget": (5000, 100000, 30000, 5000),
        "alpha": (0.00001, 0.003, 0.0003, 0.00001),
        "gamma": (0.8, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 0.05, 0.0, 0.005),
        "gamma_visible": True,
    },
    PENDULUM: {
        "environment": "Pendulum-v1",
        "family": "Classic Control",
        "algorithm": "PPO",
        "budget": (5000, 100000, 30000, 5000),
        "alpha": (0.0001, 0.003, 0.0003, 0.0001),
        "gamma": (0.8, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 0.05, 0.0, 0.005),
        "gamma_visible": True,
    },
    MOUNTAINCAR_CONTINUOUS: {
        "environment": "MountainCarContinuous-v0",
        "family": "Classic Control",
        "algorithm": "SAC",
        "budget": (5000, 100000, 30000, 5000),
        "alpha": (0.00001, 0.003, 0.0003, 0.00001),
        "gamma": (0.8, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 0.05, 0.0, 0.005),
        "gamma_visible": True,
    },
}


def load_optional_registries() -> list[str]:
    """Register optional suites when their packages are installed."""
    loaded = []
    for module_name in ("ale_py", "gymnasium_robotics"):
        try:
            module = importlib.import_module(module_name)
            if hasattr(gym, "register_envs"):
                gym.register_envs(module)
            loaded.append(module_name)
        except Exception:
            continue
    return loaded


OPTIONAL_REGISTRIES = load_optional_registries()
INTERNAL_ENV_PREFIXES = ("GymV21Environment", "GymV26Environment")

RUNTIME_PROBES = {
    "Toy Text": "FrozenLake-v1",
    "Classic Control": "CartPole-v1",
    "Box2D": "LunarLander-v3",
    "Atari / ALE": "ALE/Pong-v5",
    "MuJoCo": "Ant-v5",
    "Robotics": "FetchReach-v4",
    "JAX Phys2D": "phys2d/CartPole-v1",
    "JAX Tabular": "tabular/Blackjack-v0",
}


def registered_runtimes() -> dict[str, str]:
    """Check registry membership without constructing heavyweight environments.

    Native engines and assets are installed while the Studio image is built.
    Constructing Atari, MuJoCo, Robotics, and JAX probes here used to block the
    first page render on every cold CPU start. The selected environment is
    still fully initialized and validated when its training run begins.
    """
    registry = gym.registry
    return {
        family: ("Ready · preinstalled" if env_id in registry else "Unavailable · not registered")
        for family, env_id in RUNTIME_PROBES.items()
    }


RUNTIME_STATUS = registered_runtimes()


def warm_environment_runtimes() -> dict[str, str]:
    """Pay optional-suite import and first-reset costs once at process start.

    Training still creates a fresh environment so runs never share state. The
    warm-up only fills native-library, model-asset, ROM, and JAX compilation
    caches that would otherwise make the first learner wait.
    """
    results: dict[str, str] = {}
    for family, env_id in RUNTIME_PROBES.items():
        if env_id not in gym.registry:
            results[family] = "not registered"
            continue
        env = None
        started = time.perf_counter()
        try:
            env = gym.make(env_id)
            env.reset(seed=0)
            env.step(env.action_space.sample())
            results[family] = f"warm · {time.perf_counter() - started:.2f}s"
        except Exception as exc:
            results[family] = f"unavailable · {type(exc).__name__}: {str(exc)[:90]}"
        finally:
            if env is not None:
                env.close()
    return results


RUNTIME_WARMUP = warm_environment_runtimes()
RUNTIME_READY = sum(value.startswith("warm") for value in RUNTIME_WARMUP.values())


@lru_cache(maxsize=1)
def deep_rl_runtime():
    """Import PyTorch and Stable-Baselines3 only when a deep run starts."""
    from stable_baselines3 import DQN, PPO, SAC
    from stable_baselines3.common.evaluation import evaluate_policy

    return DQN, PPO, SAC, evaluate_policy


@lru_cache(maxsize=1)
def plotting_runtime():
    """Initialize Matplotlib only when a run needs a curve or result image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


class _LazyPlotting:
    def __getattr__(self, name):
        return getattr(plotting_runtime(), name)


plt = _LazyPlotting()


def env_family(env_id: str, entry_point) -> str:
    text = f"{env_id} {entry_point}".lower()
    if env_id.startswith("ALE/"):
        return "Atari / ALE"
    if "robotics" in text or any(name in env_id for name in ("Fetch", "Adroit", "Hand", "Franka")):
        return "Robotics"
    if "mujoco" in text:
        return "MuJoCo"
    if "box2d" in text:
        return "Box2D"
    if "toy_text" in text:
        return "Toy Text"
    if "classic_control" in text:
        return "Classic Control"
    if env_id.startswith("phys2d/"):
        return "JAX Phys2D"
    if env_id.startswith("tabular/"):
        return "JAX Tabular"
    return "Other"


def discover_environment_catalog() -> list[str]:
    choices = []
    tuned_envs = {cfg["environment"] for cfg in EXPERIMENTS.values()}
    for spec in sorted(gym.registry.values(), key=lambda item: item.id.lower()):
        env_id = spec.id
        if env_id.startswith(INTERNAL_ENV_PREFIXES):
            continue
        if env_id in tuned_envs:
            continue
        choices.append(f"{env_family(env_id, spec.entry_point)} · {env_id} · Auto")
    return choices


CATALOG_EXPERIMENTS = discover_environment_catalog()
EXPERIMENT_CHOICES = list(EXPERIMENTS) + CATALOG_EXPERIMENTS
CARD_PAGE_SIZE = 12
LEARNING_PATHS = {
    "Start here": {"Curated"},
    "Decision games": {"Bandit", "Tabular", "Toy Text", "JAX Tabular"},
    "Balance & control": {"Classic Control"},
    "2D physics": {"Box2D", "JAX Phys2D"},
    "Arcade games": {"Atari / ALE"},
    "Robots & movement": {"MuJoCo", "Robotics"},
    "Full catalog": {"All"},
}
PATH_LABELS = {
    "English": {
        "Start here": "Start here · guided",
        "Decision games": "Decision games · cards & grids",
        "Balance & control": "Balance & control · poles & cars",
        "2D physics": "2D physics · land & drive",
        "Arcade games": "Arcade games · pixels",
        "Robots & movement": "Robots & movement · arms & bodies",
        "Full catalog": "Full catalog · advanced",
    },
    "中文": {
        "Start here": "从这里开始 · 精选入门",
        "Decision games": "决策游戏 · 卡牌与网格",
        "Balance & control": "平衡控制 · 杆与小车",
        "2D physics": "二维物理 · 着陆与驾驶",
        "Arcade games": "街机游戏 · 像素画面",
        "Robots & movement": "机器人与运动 · 机械臂和身体",
        "Full catalog": "完整目录 · 进阶探索",
    },
}
ALL_FEATURES = "All tasks"

FEATURE_LABELS = {
    ALL_FEATURES: "Show everything in this path",
    "Tabular decisions": "Small decision tables · Bandit / GridWorld",
    "Card games": "Card games · Blackjack",
    "Navigation": "Find a route · FrozenLake / Taxi",
    "Classic control": "Classic control · CartPole / Acrobot",
    "Balance": "Keep upright · CartPole / InvertedPendulum",
    "Swing-up": "Swing up · Acrobot / Pendulum",
    "Momentum": "Build momentum · MountainCar",
    "Continuous control": "Smooth actions · Pendulum / continuous car",
    "2D physics": "2D physics sandbox",
    "Landing": "Land safely · LunarLander",
    "Driving": "Drive · CarRacing / racing games",
    "Walking": "Walk without falling · BipedalWalker",
    "Paddle & ball": "Paddle & ball · Pong / Breakout",
    "Arcade shooting": "Arcade shooting · Space Invaders",
    "Maze & adventure": "Maze & adventure · Montezuma / Pitfall",
    "Sports": "Sports · Boxing / Tennis",
    "Arcade control": "Other arcade games",
    "Locomotion": "Move a body · Ant / Hopper / Cheetah",
    "Swimming": "Swim · Swimmer",
    "Dexterous hand": "Dexterous hand · Door / Hammer / Pen",
    "Pick & place": "Pick and place an object",
    "Push & slide": "Push or slide an object",
    "Reach": "Reach a target",
    "Robot manipulation": "Other robot manipulation",
    "Physics control": "JAX physics control",
    "Registered tasks": "Other registered environments",
}

FEATURE_LABELS_ZH = {
    ALL_FEATURES: "显示这条路线的全部实验",
    "Tabular decisions": "小型决策表 · 多臂老虎机 / GridWorld",
    "Card games": "卡牌游戏 · Blackjack",
    "Navigation": "寻找路线 · FrozenLake / Taxi",
    "Classic control": "经典控制 · CartPole / Acrobot",
    "Balance": "保持平衡 · CartPole / InvertedPendulum",
    "Swing-up": "摆起目标 · Acrobot / Pendulum",
    "Momentum": "积累动量 · MountainCar",
    "Continuous control": "连续动作 · Pendulum / 连续小车",
    "2D physics": "二维物理沙盒",
    "Landing": "安全着陆 · LunarLander",
    "Driving": "驾驶 · CarRacing / 公路游戏",
    "Walking": "稳定行走 · BipedalWalker",
    "Paddle & ball": "球拍与小球 · Pong / Breakout",
    "Arcade shooting": "街机射击 · Space Invaders",
    "Maze & adventure": "迷宫冒险 · Montezuma / Pitfall",
    "Sports": "体育竞技 · Boxing / Tennis",
    "Arcade control": "其他街机游戏",
    "Locomotion": "身体移动 · Ant / Hopper / Cheetah",
    "Swimming": "游动 · Swimmer",
    "Dexterous hand": "灵巧手 · Door / Hammer / Pen",
    "Pick & place": "抓取并放置物体",
    "Push & slide": "推动或滑动物体",
    "Reach": "到达目标位置",
    "Robot manipulation": "其他机器人操作",
    "Physics control": "JAX 物理控制",
    "Registered tasks": "其他已注册环境",
}


def is_catalog_experiment(experiment: str) -> bool:
    return experiment not in EXPERIMENTS


def catalog_env_id(experiment: str) -> str:
    return experiment.split(" · ", 2)[1]


def catalog_config(experiment: str) -> dict:
    env_id = catalog_env_id(experiment)
    return {
        "environment": env_id,
        "family": experiment.split(" · ", 1)[0],
        "algorithm": "Auto: inspect action space",
        "budget": (200, 100000, 10000, 1000),
        "alpha": (0.00001, 0.01, 0.0003, 0.00001),
        "gamma": (0.8, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 1.0, 1.0, 0.05),
        "gamma_visible": True,
    }


def experiment_config(experiment: str) -> dict:
    return catalog_config(experiment) if is_catalog_experiment(experiment) else EXPERIMENTS[experiment]


FAMILY_VISUALS = {
    "Curated": ("#5b5ce2", "◆", "TUNED"), "Bandit": ("#7c3aed", "▥", "EXPLORE"),
    "Toy Text": ("#0f9f74", "▦", "TABULAR"), "Tabular": ("#0f9f74", "▦", "VALUES"),
    "Classic Control": ("#2563eb", "⚖", "CONTROL"), "Box2D": ("#ea580c", "⌁", "PHYSICS"),
    "Atari / ALE": ("#db2777", "▦", "PIXELS"), "MuJoCo": ("#0891b2", "⌁", "LOCOMOTION"),
    "Robotics": ("#4f46e5", "⌁", "GOAL"), "JAX Phys2D": ("#16a34a", "⚡", "JAX"),
    "JAX Tabular": ("#16a34a", "▦", "JAX"), "Other": ("#64748b", "◇", "ENV"),
}

CARD_BACKGROUNDS = {
    "Bandit": "tabular.webp", "Tabular": "tabular.webp", "Toy Text": "tabular.webp",
    "JAX Tabular": "tabular.webp", "Classic Control": "classic-control.webp",
    "Box2D": "box2d.webp", "Atari / ALE": "atari.webp", "MuJoCo": "mujoco.webp",
    "JAX Phys2D": "box2d.webp", "Robotics": "robotics.webp", "Other": "tabular.webp",
}

CURATED_TASK_CARDS = {
    BANDIT: "bandit.webp",
    BLACKJACK: "blackjack.webp",
    GRIDWORLD: "gridworld.webp",
    FROZENLAKE: "frozenlake.webp",
    CLIFF: "cliffwalking.webp",
    TAXI: "taxi.webp",
    CARTPOLE_DQN: "cartpole.webp",
    CARTPOLE_PPO: "cartpole.webp",
    MOUNTAINCAR: "mountaincar.webp",
    ACROBOT: "acrobot.webp",
    PENDULUM: "pendulum.webp",
    MOUNTAINCAR_CONTINUOUS: "mountaincarcontinuous.webp",
}

FEATURE_TASK_CARDS = {
    "Tabular decisions": "gridworld.webp",
    "Card games": "blackjack.webp",
    "Navigation": "frozenlake.webp",
    "Classic control": "cartpole.webp",
    "Balance": "cartpole.webp",
    "Swing-up": "acrobot.webp",
    "Momentum": "mountaincar.webp",
    "Continuous control": "pendulum.webp",
    "Driving": "taxi.webp",
    "Maze & adventure": "gridworld.webp",
}
def visual_data_uri(experiment: str) -> str:
    payload = Path(gallery_background(experiment)).read_bytes()
    return f"data:image/webp;base64,{base64.b64encode(payload).decode()}"


def gallery_background(experiment: str) -> str:
    """Use real task frames for curated recipes and shared art for the registry."""
    if experiment in CURATED_TASK_CARDS:
        return str(TASK_CARD_DIR / CURATED_TASK_CARDS[experiment])
    family = experiment_config(experiment)["family"]
    feature = experiment_feature(experiment)
    if feature in FEATURE_TASK_CARDS:
        if not (feature == "Driving" and family == "Box2D"):
            return str(TASK_CARD_DIR / FEATURE_TASK_CARDS[feature])
    return str(CARD_BACKGROUND_DIR / CARD_BACKGROUNDS.get(family, "tabular.webp"))


def experiment_goal(experiment: str) -> str:
    env_id = experiment_config(experiment)["environment"]
    goals = {
        "Blackjack-v1": "Beat the dealer without exceeding 21", "FrozenLake-v1": "Reach the goal across slippery ice",
        "CliffWalking-v1": "Cross safely without falling from the cliff", "Taxi-v4": "Pick up and deliver the passenger",
        "CartPole-v1": "Keep the pole balanced", "MountainCar-v0": "Build momentum to climb the hill",
        "MountainCarContinuous-v0": "Climb the hill with continuous force", "Acrobot-v1": "Swing the end link above the target",
        "Pendulum-v1": "Swing up and stabilize the pendulum", "LunarLander-v3": "Land between the flags",
    }
    if env_id in goals: return goals[env_id]
    name = env_id.lower()
    if "pong" in name: return "Move the paddle to return the ball past the opponent"
    if "breakout" in name: return "Keep the ball in play and clear the brick wall"
    if "lunarlander" in name: return "Control the engines and land softly between the flags"
    if "carracing" in name: return "Steer a car around the track as quickly and smoothly as possible"
    if "bipedal" in name or "walker" in name: return "Coordinate the legs to walk forward without falling"
    if "humanoid" in name: return "Coordinate a humanoid body to move forward without falling"
    if "hopper" in name: return "Hop forward while keeping the body upright"
    if "cheetah" in name: return "Coordinate the joints to run forward quickly"
    if "ant" in name: return "Coordinate four legs to travel forward stably"
    if "swimmer" in name: return "Propel the articulated body through fluid"
    if "reach" in name: return "Move the robot end effector to the target position"
    if "push" in name: return "Push an object from its initial position to the target"
    if any(word in name for word in ("pick", "place")): return "Pick up an object and move it to the target"
    if "door" in name: return "Manipulate the robot hand to open the door"
    if "hammer" in name: return "Control the robot hand to drive the nail with a hammer"
    family = experiment_config(experiment)["family"]
    return {
        "Atari / ALE": "Learn control directly from game pixels", "MuJoCo": "Learn continuous physics control",
        "Robotics": "Reach or manipulate a goal", "Box2D": "Learn control in a 2D physics task",
        "Toy Text": "Learn a policy in a compact discrete world", "JAX Phys2D": "Run a JAX physics control task",
        "JAX Tabular": "Run a JAX tabular task", "Bandit": "Balance exploration and exploitation",
        "Tabular": "Propagate values through a small world",
    }.get(family, "Explore a registered Gymnasium task")


def experiment_feature(experiment: str) -> str:
    """Group the full registry by the task capability learners will practice."""
    cfg = experiment_config(experiment); family = cfg["family"]; name = cfg["environment"].lower()
    if experiment in EXPERIMENTS:
        if family in ("Bandit", "Tabular", "Toy Text"): return "Tabular decisions"
        if cfg["algorithm"] == "SAC" or "pendulum" in name or "continuous" in name: return "Continuous control"
        return "Classic control"
    if family == "Robotics":
        if any(word in name for word in ("hand", "adroit", "door", "hammer", "pen")): return "Dexterous hand"
        if any(word in name for word in ("pick", "place", "lift")): return "Pick & place"
        if any(word in name for word in ("push", "slide")): return "Push & slide"
        if "reach" in name: return "Reach"
        return "Robot manipulation"
    if family == "MuJoCo":
        if "swimmer" in name: return "Swimming"
        if any(word in name for word in ("inverted", "pendulum", "standup")): return "Balance"
        return "Locomotion"
    if family == "Atari / ALE":
        if any(word in name for word in ("pong", "breakout", "tennis")): return "Paddle & ball"
        if any(word in name for word in ("race", "driver", "rally", "freeway")): return "Driving"
        if any(word in name for word in ("boxing", "bowling", "ski", "icehockey", "basketball")): return "Sports"
        if any(word in name for word in ("space", "asteroid", "battle", "beam", "galax", "phoenix", "assault")): return "Arcade shooting"
        if any(word in name for word in ("maze", "venture", "montezuma", "hero", "pitfall")): return "Maze & adventure"
        return "Arcade control"
    if family == "Box2D":
        if "lander" in name: return "Landing"
        if "racing" in name: return "Driving"
        if "walker" in name: return "Walking"
        return "2D physics"
    if family == "Classic Control":
        if "cartpole" in name: return "Balance"
        if any(word in name for word in ("pendulum", "acrobot")): return "Swing-up"
        if "mountaincar" in name: return "Momentum"
        return "Classic control"
    if family in ("Toy Text", "JAX Tabular"):
        if any(word in name for word in ("blackjack", "poker")): return "Card games"
        if any(word in name for word in ("lake", "taxi", "cliff", "grid")): return "Navigation"
        return "Tabular decisions"
    if family == "JAX Phys2D":
        return "Locomotion" if any(word in name for word in ("walker", "hopper", "ant", "cheetah")) else "Physics control"
    return "Registered tasks"


def localized_goal(experiment: str, language: str) -> str:
    if language != "中文": return experiment_goal(experiment)
    env_id = experiment_config(experiment)["environment"]
    goals = {
        "4-armed Bernoulli bandit": "在探索未知选项和利用当前最优选项之间取得平衡",
        "Custom 4×4 GridWorld": "学习从起点到终点的高回报路径",
        "Blackjack-v1": "在点数不超过 21 的前提下战胜庄家", "FrozenLake-v1": "穿过湿滑冰面到达终点",
        "CliffWalking-v1": "避开悬崖并安全抵达终点", "Taxi-v4": "接到乘客并送到指定位置",
        "CartPole-v1": "移动小车，使杆子尽可能长时间保持竖直", "MountainCar-v0": "积累动量并冲上山顶",
        "MountainCarContinuous-v0": "用连续推力控制小车爬上山顶", "Acrobot-v1": "摆动双连杆，使末端超过目标高度",
        "Pendulum-v1": "将摆杆甩起并稳定在竖直位置", "LunarLander-v3": "控制推进器，在两面旗帜之间平稳着陆",
    }
    if env_id in goals: return goals[env_id]
    name = env_id.lower()
    if "pong" in name: return "移动球拍，把球回击到对手无法接到的位置"
    if "breakout" in name: return "保持球不落下，并清除上方的砖块"
    if "lunarlander" in name: return "控制推进器，在两面旗帜之间平稳着陆"
    if "carracing" in name: return "控制赛车快速而平稳地沿赛道行驶"
    if "bipedal" in name or "walker" in name or "humanoid" in name: return "协调身体关节向前移动，并避免摔倒"
    if "hopper" in name: return "保持身体直立并连续向前跳跃"
    if "cheetah" in name or "ant" in name: return "协调多个关节，稳定而快速地向前移动"
    if "swimmer" in name: return "控制多节身体，在流体环境中向前游动"
    if "reach" in name: return "把机械臂末端移动到指定目标位置"
    if "push" in name: return "把物体从初始位置推到目标位置"
    family = experiment_config(experiment)["family"]
    return {"Atari / ALE": "直接根据游戏像素学习动作策略", "MuJoCo": "学习连续物理控制策略", "Robotics": "完成机械臂到达或物体操作任务", "Box2D": "在二维物理环境中学习控制策略", "Toy Text": "在小型离散环境中学习策略", "JAX Phys2D": "运行 JAX 二维物理控制任务", "JAX Tabular": "运行 JAX 表格型任务", "Bandit": "平衡探索与利用", "Tabular": "在小型环境中学习状态价值"}.get(family, "探索一个已注册的 Gymnasium 任务")


def card_caption(experiment: str) -> str:
    cfg = experiment_config(experiment)
    algorithm = task_space_summary(experiment)[2] if is_catalog_experiment(experiment) else cfg["algorithm"]
    return f"{cfg['environment']}\n{algorithm}\n{experiment_feature(experiment)}"


@lru_cache(maxsize=256)
def card_items_cached(experiments: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((gallery_background(item), card_caption(item)) for item in experiments)


def card_items(experiments: list[str]) -> list[tuple[str, str]]:
    """Reuse immutable gallery payloads instead of rebuilding file metadata."""
    return list(card_items_cached(tuple(experiments)))


def path_choices(language: str) -> list[tuple[str, str]]:
    labels = PATH_LABELS["English" if language == "English" else "中文"]
    return [(labels[value], value) for value in LEARNING_PATHS]


def feature_choices(query: str, learning_path: str, language: str = "English") -> list[tuple[str, str]]:
    source = filter_choices(query, learning_path, ALL_FEATURES)
    values = [ALL_FEATURES] + sorted({experiment_feature(item) for item in source})
    labels = FEATURE_LABELS if language == "English" else FEATURE_LABELS_ZH
    return [(labels.get(value, value), value) for value in values]


def filter_choices(query: str, learning_path: str, feature: str = ALL_FEATURES) -> list[str]:
    query = (query or "").strip().lower()
    families = LEARNING_PATHS.get(learning_path, {"Curated"})
    if query:
        source = EXPERIMENT_CHOICES
    else:
        source = list(EXPERIMENTS) if "Curated" in families else EXPERIMENT_CHOICES
        if "All" not in families and "Curated" not in families:
            source = [item for item in source if experiment_config(item)["family"] in families]
    if query:
        source = [item for item in source if query in (item + " " + experiment_goal(item) + " " + experiment_config(item)["algorithm"]).lower()]
    if feature and feature != ALL_FEATURES:
        source = [item for item in source if experiment_feature(item) == feature]
    return source


def catalog_page(query: str, learning_path: str, feature: str, page: int, language: str = "English"):
    matches = filter_choices(query, learning_path, feature); pages = max(1, (len(matches) + CARD_PAGE_SIZE - 1) // CARD_PAGE_SIZE); page = max(0, min(int(page), pages - 1))
    visible = matches[page * CARD_PAGE_SIZE:(page + 1) * CARD_PAGE_SIZE]
    copy = copy_for(language)
    meta = f"{len(matches):,} experiments · Page {page + 1}/{pages}" if language == "English" else f"{len(matches):,} 个实验 · 第 {page + 1}/{pages} 页"
    return card_items(visible), visible, page, meta, gr.Button(value=copy["previous"], visible=pages > 1 and page > 0), gr.Button(value=copy["next"], visible=pages > 1 and page < pages - 1)


def reset_catalog(query: str, learning_path: str, feature: str, language: str):
    return *catalog_page(query, learning_path, feature, 0, language), str(time.time_ns())


def reset_family(query: str, learning_path: str, language: str):
    choices = feature_choices(query, learning_path, language)
    return gr.Radio(choices=choices, value=ALL_FEATURES, visible=True), *catalog_page(query, learning_path, ALL_FEATURES, 0, language), str(time.time_ns())


def reset_search(query: str, learning_path: str, language: str):
    choices = feature_choices(query, learning_path, language)
    return gr.Radio(choices=choices, value=ALL_FEATURES, visible=True), *catalog_page(query, learning_path, ALL_FEATURES, 0, language), str(time.time_ns())


def move_catalog(query: str, learning_path: str, feature: str, page: int, language: str, direction: int):
    return *catalog_page(query, learning_path, feature, int(page) + direction, language), str(time.time_ns())


def choose_card(visible: list[str], language: str, event: gr.SelectData):
    """Select a card and return its complete UI update in one round trip."""
    experiment = visible[event.index]
    return experiment, *select_experiment(experiment, language), str(time.time_ns())


def space_text(space) -> str:
    if isinstance(space, gym.spaces.Discrete): return f"Discrete({space.n})"
    if isinstance(space, gym.spaces.Box): return f"Box{space.shape}"
    if isinstance(space, gym.spaces.MultiDiscrete): return f"MultiDiscrete{space.shape}"
    if isinstance(space, gym.spaces.MultiBinary): return f"MultiBinary({space.n})"
    if isinstance(space, gym.spaces.Dict): return "Dict(" + ", ".join(space.spaces.keys()) + ")"
    if isinstance(space, gym.spaces.Tuple): return f"Tuple({len(space.spaces)} parts)"
    return type(space).__name__


def infer_algorithm(action_space, configured: str) -> str:
    if configured != "Auto: inspect action space": return configured
    if isinstance(action_space, gym.spaces.Discrete): return "DQN"
    if isinstance(action_space, gym.spaces.Box): return "SAC"
    if isinstance(action_space, (gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)): return "PPO"
    return "Manual setup"


def task_space_summary(experiment: str) -> tuple[str, str, str, str]:
    """Return useful task metadata without constructing the environment."""
    cfg = experiment_config(experiment); env_id = cfg["environment"]; family = cfg["family"]
    if env_id == "4-armed Bernoulli bandit":
        return "Estimated arm values", "Choose one of 4 arms", cfg["algorithm"], "Ready · built in"
    if env_id == "Custom 4×4 GridWorld":
        return "Grid cell", "Up / Down / Left / Right", cfg["algorithm"], "Ready · built in"

    known = {
        "Blackjack-v1": ("Tuple(3 parts)", "Discrete(2)"),
        "FrozenLake-v1": ("Discrete(16)", "Discrete(4)"),
        "CliffWalking-v1": ("Discrete(48)", "Discrete(4)"),
        "Taxi-v4": ("Discrete(500)", "Discrete(6)"),
        "CartPole-v1": ("Box(4,)", "Discrete(2)"),
        "MountainCar-v0": ("Box(2,)", "Discrete(3)"),
        "MountainCarContinuous-v0": ("Box(2,)", "Box(1,)"),
        "Acrobot-v1": ("Box(6,)", "Discrete(3)"),
        "Pendulum-v1": ("Box(3,)", "Box(1,)"),
    }
    observation, action = known.get(env_id, ("Inspected at run start", "Inspected at run start"))
    algorithm = cfg["algorithm"]
    name = env_id.lower()
    if is_catalog_experiment(experiment):
        if family == "Atari / ALE":
            observation, action, algorithm = "RGB game frames", "Discrete actions", "DQN"
        elif family in {"MuJoCo", "Robotics"}:
            observation = "Continuous state" if family == "MuJoCo" else "Observation + goal"
            action, algorithm = "Continuous actions", "SAC"
        elif family in {"Toy Text", "JAX Tabular"}:
            observation, action, algorithm = "Discrete state", "Discrete actions", "DQN"
        elif family in {"Box2D", "Classic Control", "JAX Phys2D"}:
            continuous = any(word in name for word in ("continuous", "racing", "walker", "pendulum"))
            observation = "Continuous state"
            action = "Continuous actions" if continuous else "Discrete actions"
            algorithm = "SAC" if continuous else "DQN"
        else:
            algorithm = "Auto at run start"
    status = RUNTIME_STATUS.get(family, "Ready · registered")
    return observation, action, algorithm, status


def training_guide_html(experiment: str, language: str) -> str:
    cfg = experiment_config(experiment)
    family = str(cfg.get("family", "Other"))
    if family in {"Toy Text", "JAX Tabular"}:
        duration_en, duration_zh = "Usually 10–60 seconds on CPU.", "CPU 上通常需要 10–60 秒。"
    elif family in {"Classic Control", "Box2D", "JAX Phys2D"}:
        duration_en, duration_zh = "Usually 30 seconds–3 minutes on CPU.", "CPU 上通常需要 30 秒到 3 分钟。"
    elif family == "Atari / ALE":
        duration_en, duration_zh = "Usually 1–5 minutes; harder games need larger budgets.", "通常需要 1–5 分钟；更难的游戏需要更大训练预算。"
    elif family in {"MuJoCo", "Robotics"}:
        duration_en, duration_zh = "Usually 3–15 minutes; first-run physics and rendering setup can add several minutes.", "通常需要 3–15 分钟；首次物理与渲染初始化可能额外增加数分钟。"
    else:
        duration_en, duration_zh = "Usually 1–10 minutes, depending on environment initialization and the selected budget.", "根据环境初始化与所选预算，通常需要 1–10 分钟。"
    if language == "中文":
        title, intro = "怎样判断本次训练结果", "同时检查评估、Preview 和耗时，不要只看“训练完成”。"
        success_title = "怎样算训练成功"
        success = "“训练完成”表示流程正常结束；最终评估高于早期检查点，并且策略行为符合当前任务目标，才说明学到了有效策略。"
        preview_title = "怎样查看 Preview"
        preview = "运行前显示任务或真实示例；运行结束后会替换为本次训练生成的回放 GIF、策略图或结果图。结合曲线检查策略是否真的完成目标。"
        time_title, duration = "大约需要多久", duration_zh
    else:
        title, intro = "How to judge this training run", "Check evaluation, Preview, and elapsed time together—not only the completed status."
        success_title = "What counts as success"
        success = "Training complete confirms that the pipeline ended normally. Learning is demonstrated when final evaluation improves over early checkpoints and behavior matches this task's goal."
        preview_title = "How to read Preview"
        preview = "Before a run it shows the task or a real example. Afterward it is replaced by this run's replay GIF, policy map, or result plot; compare the behavior with the curve."
        time_title, duration = "Typical time", duration_en
    return f'''<section class="training-guide"><div class="training-guide__intro"><span>RESULT CHECKLIST</span><h3>{title}</h3><p>{intro}</p></div><div class="training-guide__grid"><article><b>01</b><h4>{success_title}</h4><p>{success}</p></article><article><b>02</b><h4>{preview_title}</h4><p>{preview}</p></article><article><b>03</b><h4>{time_title}</h4><p>{duration}</p></article></div></section>'''


def task_brief(experiment: str, language: str) -> str:
    cfg = experiment_config(experiment); env_id = cfg["environment"]
    observation, action, algorithm, availability = task_space_summary(experiment)
    if language == "中文":
        return f'''<section class="task-brief"><div class="task-brief__visual"><img src="{visual_data_uri(experiment)}" alt="{html.escape(env_id)} task scene"></div><div class="task-brief__body"><span class="task-kicker">训练前先理解任务</span><h3>{html.escape(env_id)}</h3><p>{html.escape(localized_goal(experiment, language))}</p><div class="task-facts"><span><b>观察</b>{html.escape(observation)}</span><span><b>动作</b>{html.escape(action)}</span><span><b>算法</b>{html.escape(algorithm)}</span><span><b>状态</b>{html.escape(availability)}</span></div><p class="task-hint">调整下方参数后再点击“开始训练”。训练曲线和实时日志会持续更新。</p></div></section>{training_guide_html(experiment, language)}'''
    return f'''<section class="task-brief"><div class="task-brief__visual"><img src="{visual_data_uri(experiment)}" alt="{html.escape(env_id)} task scene"></div><div class="task-brief__body"><span class="task-kicker">UNDERSTAND BEFORE TRAINING</span><h3>{html.escape(env_id)}</h3><p>{html.escape(localized_goal(experiment, language))}</p><div class="task-facts"><span><b>Observation</b>{html.escape(observation)}</span><span><b>Action</b>{html.escape(action)}</span><span><b>Algorithm</b>{html.escape(algorithm)}</span><span><b>Status</b>{html.escape(availability)}</span></div><p class="task-hint">Review the task, adjust the parameters below, then press Start training. The curve and live console will keep updating.</p></div></section>{training_guide_html(experiment, language)}'''


TEXT = {
    "English": {
        "course": "Hands-On Modern RL · CPU experiment collection",
        "title": "Gymnasium Training Playground",
        "description": "Browse every environment registered by Gymnasium and its installed suites. Twelve curated recipes remain ready for quick CPU training.",
        "chapter": "Companion chapter",
        "notebook": "Notebook",
        "script": "Training source",
        "project": "GitHub project",
        "device": "Device",
        "experiments": "Experiments",
        "catalog_title": "Choose an experiment",
        "catalog_copy": "Choose a learning path on the left, then optionally narrow it by goal on the right. Search works across the complete catalog.",
        "catalog_version": "Navigation v2.7 · guarded updates",
        "path": "Learning path",
        "search": "Know a task name? Search the full catalog",
        "search_placeholder": "Optional: try CartPole, Pong, robot...",
        "goal": "Choose a goal (optional)",
        "goal_page": "Page {page}/{pages} · {total} goals",
        "catalog_wait": "Updating experiments…",
        "catalog_wait_detail": "Please wait for the new goals and cards",
        "previous": "← Previous",
        "next": "Next →",
        "settings": "Experiment setup",
        "settings_copy": "Choose steps per epoch and epoch count. Every epoch evaluates and saves one independently selectable policy.",
        "experiment": "Experiment",
        "budget": "Training budget",
        "budget_info": "Episodes for tabular tasks; environment steps for DQN, PPO, and SAC",
        "steps_per_epoch": "Episodes / steps per epoch",
        "steps_per_epoch_info": "One fixed training block before evaluation, model saving, and replay generation",
        "epochs": "Training epochs / saved models",
        "epochs_info": "Every epoch produces one independently selectable policy",
        "alpha": "Learning rate",
        "gamma": "Discount factor γ",
        "epsilon": "Exploration ε",
        "seed": "Random seed",
        "advanced": "Advanced settings",
        "start": "Start training",
        "start_running": "Running…",
        "wait_title": "Run active · please keep this page open",
        "wait_detail": "Initializing the environment and model, training the policy, then rendering the result. This indicator stays active until every stage finishes.",
        "ready": "Ready to train",
        "ready_detail": "Review the task brief, adjust parameters, then start the CPU run",
        "running": "Training in progress",
        "complete": "Training complete",
        "status": "Run status",
        "metric": "Latest evaluation",
        "metric_waiting": "Results appear after training starts",
        "curve": "Learning curve",
        "curve_copy": "The chart updates at each checkpoint. All labels stay in English for readability.",
        "log": "Live training log",
        "log_waiting": "Waiting for a training run...",
        "preview": "Task preview / trained result",
        "preview_copy": "The 12 curated tasks include real trained examples. Other registry tasks show a task illustration until your run produces a replay GIF, policy map, or result plot.",
        "saved_model": "Epoch model",
        "saved_model_info": "Choose the evaluated policy saved at the end of an epoch",
        "saved_model_empty": "No trained epoch models yet. Start training to create the first one.",
        "artifact": "Download run summary",
        "seconds": "s",
    },
    "中文": {
        "course": "《动手学现代强化学习》· CPU 实验合集",
        "title": "Gymnasium 在线训练游乐场",
        "description": "浏览 Gymnasium 及已安装扩展套件注册的全部环境，同时保留 12 个可快速训练的调优配方。",
        "chapter": "阅读配套章节",
        "notebook": "Notebook",
        "script": "训练源码",
        "project": "GitHub 项目",
        "device": "设备",
        "experiments": "实验数量",
        "catalog_title": "选择一个实验",
        "catalog_copy": "在左侧选择学习路线，再在右侧按训练目标细分。搜索会覆盖完整实验目录。",
        "catalog_version": "导航版本 v2.7 · 防重复操作",
        "path": "学习路线",
        "search": "知道任务名称？搜索完整目录",
        "search_placeholder": "可选：输入 CartPole、Pong、robot…",
        "goal": "选择训练目标（可选）",
        "goal_page": "第 {page}/{pages} 页 · 共 {total} 个目标",
        "catalog_wait": "正在更新实验…",
        "catalog_wait_detail": "请等待新的目标与实验卡片返回",
        "previous": "← 上一页",
        "next": "下一页 →",
        "settings": "实验设置",
        "settings_copy": "设置每个 epoch 的训练步数与 epoch 数；每个 epoch 都会评估并保存一个可独立选择的策略。",
        "experiment": "实验",
        "budget": "训练预算",
        "budget_info": "表格任务使用回合数；DQN、PPO 与 SAC 使用环境步数",
        "steps_per_epoch": "每个 epoch 的回合数 / 步数",
        "steps_per_epoch_info": "一段固定训练量结束后执行评估、保存模型并生成回放",
        "epochs": "训练 epochs / 保存模型数",
        "epochs_info": "每个 epoch 都会生成一个可独立选择的策略",
        "alpha": "学习率",
        "gamma": "折扣因子 γ",
        "epsilon": "探索率 ε",
        "seed": "随机种子",
        "advanced": "高级参数",
        "start": "开始训练",
        "start_running": "运行中…",
        "wait_title": "任务正在运行 · 请保持页面打开",
        "wait_detail": "正在初始化环境与模型、训练策略并生成结果回放。所有阶段完成前，这里会一直显示等待状态。",
        "ready": "等待训练",
        "ready_detail": "先阅读任务说明，调整参数，再启动 CPU 训练",
        "running": "训练进行中",
        "complete": "训练完成",
        "status": "训练状态",
        "metric": "最新评估",
        "metric_waiting": "训练开始后显示结果",
        "curve": "学习曲线",
        "curve_copy": "每个检查点更新一次曲线。图表标记统一保留英文。",
        "log": "实时训练日志",
        "log_waiting": "等待训练任务...",
        "preview": "任务预览 / 训练结果",
        "preview_copy": "12 个精选任务附带真实训练示例；其他注册表任务在运行前显示任务示意图，本次训练完成后替换为回放 GIF、策略图或结果曲线。",
        "saved_model": "Epoch 模型",
        "saved_model_info": "选择每个 epoch 结束时保存并评估的策略",
        "saved_model_empty": "尚未训练出 epoch 模型；开始训练后会生成第一份。",
        "artifact": "下载运行摘要",
        "seconds": "秒",
    },
}


def copy_for(language: str) -> dict[str, str]:
    return TEXT["English" if language == "English" else "中文"]


def elapsed_line(started: float, level: str, message: str) -> str:
    return f"{time.perf_counter() - started:7.1f}s  {level:<7} {message}"


def console_panel(logs: str, language: str) -> str:
    return f"""
    <section class="console-panel" aria-live="polite" aria-atomic="true">
      <div class="console-head"><span class="console-dot"></span>{copy_for(language)['log']}</div>
      <pre class="console-text">{html.escape(logs)}</pre>
    </section>
    """


def status_card(state: str, title: str, detail: str, language: str) -> str:
    return f"""
    <div class="run-state run-state--{state}">
      <span class="run-state__dot"></span>
      <div><span class="summary-label">{copy_for(language)['status']}</span><strong>{title}</strong><small>{detail}</small></div>
    </div>
    """


def metric_card(value: str, detail: str, language: str) -> str:
    return f"""
    <div class="live-metric">
      <span class="summary-label">{copy_for(language)['metric']}</span>
      <div class="metric-reading"><strong>{value}</strong><small>{detail}</small></div>
    </div>
    """


def waiting_panel(language: str) -> str:
    copy = copy_for(language)
    elapsed_label = "elapsed" if language == "English" else "已等待"
    return f"""
    <section class="run-wait" role="status" aria-live="polite">
      <span class="run-wait__spinner" aria-hidden="true"></span>
      <div class="run-wait__copy">
        <strong>{copy['wait_title']}</strong>
        <small>{copy['wait_detail']}</small>
        <em class="run-wait__elapsed" data-start-ms="{int(time.time() * 1000)}" data-label="{elapsed_label}">0s {elapsed_label}</em>
      </div>
      <span class="run-wait__pulse" aria-hidden="true"><i></i></span>
    </section>
    """


def panel_html(title: str, text: str, cls: str = "panel-copy") -> str:
    return f'<h2 class="panel-title">{title}</h2><p class="{cls}">{text}</p>'


def catalog_header_html(language: str) -> str:
    copy = copy_for(language)
    return f'<div class="catalog-heading"><div><h2 class="panel-title">{copy["catalog_title"]}</h2><p class="panel-copy">{copy["catalog_copy"]}</p></div><span class="ui-version">{copy["catalog_version"]}</span></div>'


def goal_pager_html(language: str) -> str:
    copy = copy_for(language)
    return (
        '<div class="goal-local-pager" hidden data-page-size="6" '
        f'data-template="{html.escape(copy["goal_page"])}" '
        f'data-previous="{html.escape(copy["previous"])}" data-next="{html.escape(copy["next"])}">'
        '<button type="button" data-goal-page="previous"></button>'
        '<span class="goal-local-pager__meta"></span>'
        '<button type="button" data-goal-page="next"></button>'
        '</div>'
    )


def catalog_wait_html(language: str) -> str:
    copy = copy_for(language)
    return (
        '<div class="catalog-wait" role="status" aria-live="polite">'
        '<span class="catalog-wait__spinner" aria-hidden="true"></span>'
        f'<span><strong>{copy["catalog_wait"]}</strong><small>{copy["catalog_wait_detail"]}</small></span>'
        '</div>'
    )


def card_preload_html() -> str:
    """Keep one decoded copy of every shared card image in browser memory."""
    filenames = sorted(set(CURATED_TASK_CARDS.values()) | set(FEATURE_TASK_CARDS.values()) | set(CARD_BACKGROUNDS.values()))
    paths = []
    for filename in filenames:
        path = TASK_CARD_DIR / filename
        if not path.exists():
            path = CARD_BACKGROUND_DIR / filename
        if path.exists():
            paths.append(path)
    images = "".join(
        f'<img src="/gradio_api/file={html.escape(str(path))}" alt="" loading="eager" decoding="async">'
        for path in paths
    )
    return f'<div class="card-image-preload" aria-hidden="true">{images}</div>'


def hero_html(language: str, experiment: str = BANDIT) -> str:
    copy = copy_for(language)
    cfg = experiment_config(experiment)
    chapter_url = CHAPTER_URLS.get(experiment, COURSE_URL)
    return f"""
    <main class="app-shell">
      <section class="hero">
        <div class="brand-lockup"><a href="https://modelscope.cn/organization/walkinglab" target="_blank" rel="noreferrer">WALKINGLAB</a><span>×</span><a href="{PROJECT_URL}" target="_blank" rel="noreferrer">HANDS-ON MODERN RL</a></div>
        <img class="project-mark" src="{LOGO_DATA_URI}" alt="Hands-On Modern RL" />
        <div class="hero-topline"><span class="experiment-badge">CPU PLAYGROUND</span><span class="hero-course">{copy['course']}</span></div>
        <h1>{copy['title']}</h1>
        <p class="hero-copy">{copy['description']}</p>
        <nav class="hero-links">
          <a class="hero-link primary" href="{PROJECT_URL}" target="_blank" rel="noreferrer">GitHub · walkinglabs/hands-on-modern-rl</a>
          <a class="hero-link" href="https://modelscope.cn/organization/walkinglab" target="_blank" rel="noreferrer">WalkingLab</a>
          <a class="hero-link" href="{chapter_url}" target="_blank" rel="noreferrer">{copy['chapter']}</a>
          <a class="hero-link" href="{MODELSCOPE_NOTEBOOK_URL}" target="_blank" rel="noreferrer">{copy['notebook']}</a>
          <a class="hero-link" href="{SCRIPT_URL}" target="_blank" rel="noreferrer">{copy['script']}</a>
        </nav>
      </section>
      <section class="lab-strip">
        <span>{copy['experiments']} <strong>{len(EXPERIMENT_CHOICES)}</strong></span>
        <span>Runtimes <strong>{RUNTIME_READY}/{len(RUNTIME_PROBES)} READY</strong></span>
        <span>Environment <strong>{cfg['environment']}</strong></span>
        <span>Algorithm <strong>{cfg['algorithm']}</strong></span>
      </section>
    </main>
    """


def footer_html() -> str:
    return f'<div class="footer-note">Gymnasium CPU Playground · <a href="{COURSE_URL}" target="_blank">Hands-On Modern RL</a> · WalkingLab</div>'


def learning_figure(x: list[float], y: list[float], title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(x, y, color="#5b5ce2", linewidth=2.2)
    if x:
        ax.scatter([x[-1]], [y[-1]], color="#15a873", s=34, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Training progress")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def result_preview_image(
    experiment: str,
    status: str,
    metric_value: str,
    metric_label: str,
    filename: str | None = None,
    x: list[float] | None = None,
    y: list[float] | None = None,
    note: str = "",
    algorithm: str | None = None,
) -> str:
    """Create a durable result preview whenever an environment has no replay."""
    cfg = experiment_config(experiment)
    slug = re.sub(r"[^a-z0-9]+", "-", experiment.lower()).strip("-")
    path = ARTIFACT_DIR / (filename or f"{slug}-result.png")
    fig = plt.figure(figsize=(9.6, 5.0), facecolor="#f7f8fc")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.72], wspace=.34)
    info = fig.add_subplot(grid[0, 0]); plot = fig.add_subplot(grid[0, 1])
    info.set_facecolor("#20245b"); info.set_xticks([]); info.set_yticks([])
    for spine in info.spines.values(): spine.set_visible(False)
    info.text(.09, .88, status.upper(), color="#a5b4fc", fontsize=10, fontweight="bold", transform=info.transAxes)
    info.text(.09, .71, metric_value, color="white", fontsize=27, fontweight="bold", transform=info.transAxes, wrap=True)
    info.text(.09, .61, metric_label, color="#cbd5e1", fontsize=10, transform=info.transAxes, wrap=True)
    environment_label = textwrap.fill(cfg["environment"], width=21, break_long_words=False)
    algorithm_label = textwrap.fill(f"{algorithm or cfg['algorithm']} · CPU", width=24, break_long_words=False)
    note_label = textwrap.fill(note[:130], width=32, break_long_words=False)
    info.text(.09, .43, environment_label, color="white", fontsize=12, fontweight="bold", linespacing=1.25, transform=info.transAxes)
    info.text(.09, .27, algorithm_label, color="#cbd5e1", fontsize=9.5, linespacing=1.25, transform=info.transAxes)
    info.text(.09, .07, note_label, color="#aeb7ca", fontsize=8.5, linespacing=1.2, transform=info.transAxes)
    if x and y:
        plot.plot(x, y, color="#5b5ce2", linewidth=2.4)
        plot.scatter([x[-1]], [y[-1]], color="#13a36f", s=45, zorder=3)
        plot.set_xlabel("Training progress"); plot.set_ylabel(metric_label); plot.grid(alpha=.2)
        plot.set_title("Training result", loc="left", fontweight="bold", color="#172033")
    else:
        plot.axis("off")
        plot.text(.5, .59, "RESULT PREVIEW", ha="center", va="center", color="#5b5ce2", fontsize=19, fontweight="bold")
        plot.text(.5, .43, experiment, ha="center", va="center", color="#172033", fontsize=13, wrap=True)
        plot.text(.5, .30, "A policy map, replay GIF, or result image\nwill replace this panel after training.", ha="center", va="center", color="#68748a", fontsize=10, linespacing=1.5)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def example_preview(experiment: str, _run_state: str | None = None) -> str:
    """Show a real trained artifact before a new run replaces it."""
    filename = CURATED_PREVIEWS.get(experiment)
    if filename:
        path = PREVIEW_DIR / filename
        if path.exists():
            return str(path)
    # The full registry cannot ship a trained replay for every optional or
    # legacy environment. Use its task illustration without presenting it as
    # a learned result; a successful run replaces it with the generated GIF.
    return gallery_background(experiment)


def policy_grid_image(
    grid: list[str], policy: dict[tuple[int, int], str], title: str, filename: str,
    values: dict[tuple[int, int], float] | None = None,
):
    rows, cols = len(grid), len(grid[0])
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    colors = {"S": "#dbeafe", "G": "#bbf7d0", "H": "#fecaca", "T": "#fecaca", "F": "#f8fafc", ".": "#f8fafc"}
    for row in range(rows):
        for col in range(cols):
            cell = grid[row][col]
            rect = plt.Rectangle((col, row), 1, 1, facecolor=colors.get(cell, "#f8fafc"), edgecolor="#cbd5e1", linewidth=1.5)
            ax.add_patch(rect)
            label = {"S": "START", "G": "GOAL", "H": "HOLE", "T": "TRAP"}.get(cell, policy.get((row, col), "·"))
            ax.text(col + 0.5, row + 0.46, label, ha="center", va="center", fontsize=12, fontweight="bold", color="#27324a")
            if values and (row, col) in values:
                ax.text(col + 0.5, row + 0.78, f"{values[(row, col)]:.2f}", ha="center", fontsize=8, color="#64748b")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    path = ARTIFACT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def save_summary(experiment: str, payload: dict) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", experiment.lower()).strip("-")
    path = ARTIFACT_DIR / f"{slug}-run-summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


MODEL_INDEX = ARTIFACT_DIR / "trained-models.json"


def epoch_specs(experiment: str) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    minimum, maximum, default, step = experiment_config(experiment)["budget"]
    epochs = 6

    def aligned(value: float) -> float:
        return max(float(step), round(max(float(step), value) / float(step)) * float(step))

    steps_default = aligned(float(default) / epochs)
    return (
        (min(steps_default, aligned(float(minimum) / epochs)), max(steps_default, aligned(float(maximum))), steps_default, float(step)),
        (1.0, 12.0, float(epochs), 1.0),
    )


def epoch_targets(total: int, epochs: int) -> dict[int, int]:
    count = max(1, min(int(total), min(12, int(epochs))))
    targets = [max(1, round(int(total) * index / count)) for index in range(1, count + 1)]
    targets[-1] = int(total)
    return {step: index for index, step in enumerate(targets, start=1)}


def model_slug(experiment: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", experiment.lower()).strip("-")


def model_epoch_dir(experiment: str, run_id: str, epoch: int) -> Path:
    path = ARTIFACT_DIR / f"{model_slug(experiment)}-{run_id}-epoch-{epoch:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_models(experiment: str | None = None) -> list[dict]:
    try:
        payload = json.loads(MODEL_INDEX.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        payload = {"models": []}
    records = [record for record in payload.get("models", []) if isinstance(record, dict) and record.get("model_id")]
    if experiment is not None:
        records = [record for record in records if record.get("experiment") == experiment]
    return sorted(records, key=lambda record: str(record.get("created_at", "")), reverse=True)


def write_models(records: list[dict]) -> None:
    temporary = MODEL_INDEX.with_suffix(".tmp")
    temporary.write_text(json.dumps({"models": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(MODEL_INDEX)


def register_model(
    experiment: str,
    run_id: str,
    epoch: int,
    epochs: int,
    step: int,
    total: int,
    score: float,
    model: str,
    preview: str,
) -> str:
    model_id = f"{run_id}-epoch-{epoch:02d}"
    record = {
        "model_id": model_id,
        "experiment": experiment,
        "environment": experiment_config(experiment)["environment"],
        "algorithm": experiment_config(experiment)["algorithm"],
        "run_id": run_id,
        "epoch": int(epoch),
        "epochs": int(epochs),
        "step": int(step),
        "total": int(total),
        "score": float(score),
        "model": str(model),
        "preview": str(preview),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    }
    records = [existing for existing in load_models() if existing.get("model_id") != model_id]
    records.insert(0, record)
    write_models(records)
    return model_id


def model_selector(experiment: str, language: str, preferred: str | None = None, interactive: bool = True):
    records = load_models(experiment)
    choices = []
    for record in records:
        label = (
            f"Epoch {record['epoch']}/{record['epochs']} · {record['step']:,}/{record['total']:,} · score {record['score']:.2f} · {record['run_id'][-8:]}"
            if language == "English"
            else f"Epoch {record['epoch']}/{record['epochs']} · {record['step']:,}/{record['total']:,} · 得分 {record['score']:.2f} · {record['run_id'][-8:]}"
        )
        choices.append((label, record["model_id"]))
    values = {value for _, value in choices}
    selected = preferred if preferred in values else (choices[0][1] if choices else None)
    copy = copy_for(language)
    return gr.Dropdown(
        choices=choices,
        value=selected,
        label=copy["saved_model"],
        info=copy["saved_model_info"] if choices else copy["saved_model_empty"],
        interactive=interactive and bool(choices),
        visible=True,
    )


def preview_provenance(experiment: str, model_id: str | None, language: str) -> str:
    if not model_id:
        message = copy_for(language)["saved_model_empty"]
        return f'<div class="preview-provenance"><span></span>{html.escape(message)}</div>'
    record = next((item for item in load_models(experiment) if item["model_id"] == model_id), None)
    if record is None:
        message = copy_for(language)["saved_model_empty"]
        return f'<div class="preview-provenance"><span></span>{html.escape(message)}</div>'
    message = (
        f"Selected exact saved policy · epoch {record['epoch']}/{record['epochs']} · {record['step']:,} steps · score {record['score']:.2f}"
        if language == "English"
        else f"当前为真实保存策略 · epoch {record['epoch']}/{record['epochs']} · {record['step']:,} 步 · 得分 {record['score']:.2f}"
    )
    return f'<div class="preview-provenance preview-provenance--ready"><span></span>{html.escape(message)}</div>'


def select_saved_model(experiment: str, model_id: str | None, language: str):
    record = next((item for item in load_models(experiment) if item["model_id"] == model_id), None)
    if record is None or not Path(str(record.get("preview", ""))).is_file():
        return gr.skip(), preview_provenance(experiment, None, language)
    return str(record["preview"]), preview_provenance(experiment, model_id, language)


def run_bandit(budget: int, alpha: float, epsilon: float, seed: int, language: str):
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    probabilities = np.array([0.35, 0.50, 0.72, 0.58])
    q = np.zeros(4)
    counts = np.zeros(4, dtype=int)
    rewards: list[float] = []
    logs = ["Multi-Armed Bandit training console", "=" * 72, elapsed_line(started, "CONFIG", f"arms=4  probabilities={probabilities.tolist()}"), elapsed_line(started, "CONFIG", f"steps={budget}  alpha={alpha:g}  epsilon={epsilon:g}  seed={seed}")]
    chunk = max(20, budget // 20)
    for step in range(1, budget + 1):
        action = int(rng.integers(4)) if rng.random() < epsilon else int(np.argmax(q))
        reward = float(rng.random() < probabilities[action])
        counts[action] += 1
        q[action] += alpha * (reward - q[action])
        rewards.append(reward)
        if step % chunk == 0 or step == budget:
            avg = float(np.mean(rewards))
            logs.append(elapsed_line(started, "TRAIN", f"step={step}/{budget}  avg_reward={avg:.3f}  best_estimate=arm-{int(np.argmax(q)) + 1}"))
            yield status_card("running", copy_for(language)["running"], f"{step:,}/{budget:,} steps", language), metric_card(f"{avg:.3f}", f"estimated best arm: {int(np.argmax(q)) + 1}", language), learning_figure(list(range(1, step + 1)), (np.cumsum(rewards) / np.arange(1, step + 1)).tolist(), "Bandit cumulative average reward", "Average reward"), gr.skip(), None, console_panel("\n".join(logs), language)
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(1, 5)
    ax.bar(positions - 0.16, probabilities, 0.32, label="True probability", color="#93c5fd")
    ax.bar(positions + 0.16, q, 0.32, label="Learned estimate", color="#5b5ce2")
    ax.set(xticks=positions, xlabel="Arm", ylabel="Reward probability", ylim=(0, 1), title="True vs learned arm values")
    ax.legend(); ax.grid(axis="y", alpha=0.2); fig.tight_layout()
    preview = ARTIFACT_DIR / "bandit-arm-estimates.png"
    fig.savefig(preview, dpi=150, bbox_inches="tight")
    plt.close(fig)
    summary = save_summary("bandit", {"experiment": "Bandit", "q_values": q.tolist(), "counts": counts.tolist(), "average_reward": float(np.mean(rewards)), "parameters": {"budget": budget, "alpha": alpha, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"best_arm={int(np.argmax(q)) + 1}  artifact={summary}"))
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} steps · {time.perf_counter() - started:.1f}s", language), metric_card(f"{np.mean(rewards):.3f}", f"best arm: {int(np.argmax(q)) + 1} · selected {counts[int(np.argmax(q))]} times", language), learning_figure(list(range(1, budget + 1)), (np.cumsum(rewards) / np.arange(1, budget + 1)).tolist(), "Bandit cumulative average reward", "Average reward"), str(preview), summary, console_panel("\n".join(logs), language)


GRID_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARROWS = ["↑", "↓", "←", "→"]


def grid_step(state: tuple[int, int], action: int):
    dr, dc = GRID_ACTIONS[action]
    nxt = (min(3, max(0, state[0] + dr)), min(3, max(0, state[1] + dc)))
    if nxt == (1, 1):
        return nxt, -1.0, True
    if nxt == (3, 3):
        return nxt, 1.0, True
    return nxt, -0.01, False


def run_gridworld(budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str):
    started = time.perf_counter(); rng = np.random.default_rng(seed); q = np.zeros((4, 4, 4)); rewards = []
    logs = ["GridWorld Q-Learning console", "=" * 72, elapsed_line(started, "CONFIG", f"episodes={budget} alpha={alpha:g} gamma={gamma:g} epsilon={epsilon:g} seed={seed}")]
    chunk = max(10, budget // 20)
    for episode in range(1, budget + 1):
        state = (0, 0); total = 0.0
        for _ in range(100):
            action = int(rng.integers(4)) if rng.random() < epsilon else int(rng.choice(np.flatnonzero(q[state] == q[state].max())))
            nxt, reward, done = grid_step(state, action)
            q[state][action] += alpha * (reward + (0 if done else gamma * q[nxt].max()) - q[state][action])
            total += reward; state = nxt
            if done: break
        rewards.append(total)
        if episode % chunk == 0 or episode == budget:
            recent = float(np.mean(rewards[-min(50, len(rewards)):]))
            logs.append(elapsed_line(started, "TRAIN", f"episode={episode}/{budget} recent_reward={recent:.3f}"))
            yield status_card("running", copy_for(language)["running"], f"{episode:,}/{budget:,} episodes", language), metric_card(f"{recent:.3f}", "mean reward over recent episodes", language), learning_figure(list(range(1, episode + 1)), rewards, "GridWorld episode reward", "Episode reward"), gr.skip(), None, console_panel("\n".join(logs), language)
    policy = {(r, c): ARROWS[int(np.argmax(q[r, c]))] for r in range(4) for c in range(4) if (r, c) not in {(1, 1), (3, 3)}}
    values = {(r, c): float(q[r, c].max()) for r in range(4) for c in range(4)}
    preview = policy_grid_image(["S...", ".T..", "....", "...G"], policy, "Learned GridWorld policy", "gridworld-policy.png", values)
    summary = save_summary("gridworld", {"experiment": "GridWorld", "q_values": q.tolist(), "policy": {f"{r},{c}": arrow for (r, c), arrow in policy.items()}, "parameters": {"budget": budget, "alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"artifact={summary}"))
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} episodes · {time.perf_counter() - started:.1f}s", language), metric_card(f"{np.mean(rewards[-50:]):.3f}", "final 50-episode mean reward", language), learning_figure(list(range(1, budget + 1)), rewards, "GridWorld episode reward", "Episode reward"), preview, summary, console_panel("\n".join(logs), language)


def run_frozenlake(budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str):
    started = time.perf_counter(); rng = np.random.default_rng(seed); env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True); q = np.zeros((16, 4)); successes = []
    logs = ["FrozenLake Q-Learning console", "=" * 72, elapsed_line(started, "CONFIG", f"episodes={budget} slippery=true alpha={alpha:g} gamma={gamma:g} epsilon_start={epsilon:g}")]
    chunk = max(50, budget // 20)
    for episode in range(1, budget + 1):
        state, _ = env.reset(seed=seed + episode); done = False; won = 0.0
        current_eps = max(0.02, epsilon * (1 - episode / budget))
        while not done:
            action = int(rng.integers(4)) if rng.random() < current_eps else int(rng.choice(np.flatnonzero(q[state] == q[state].max())))
            nxt, reward, terminated, truncated, _ = env.step(action); done = terminated or truncated
            q[state, action] += alpha * (reward + (0 if done else gamma * q[nxt].max()) - q[state, action]); state = nxt; won = max(won, float(reward))
        successes.append(won)
        if episode % chunk == 0 or episode == budget:
            rate = float(np.mean(successes[-min(500, len(successes)):]))
            logs.append(elapsed_line(started, "TRAIN", f"episode={episode}/{budget} epsilon={current_eps:.3f} recent_success={rate:.1%}"))
            curve = (np.cumsum(successes) / np.arange(1, len(successes) + 1)).tolist()
            yield status_card("running", copy_for(language)["running"], f"{episode:,}/{budget:,} episodes", language), metric_card(f"{rate:.1%}", "recent success rate", language), learning_figure(list(range(1, episode + 1)), curve, "FrozenLake cumulative success rate", "Success rate"), gr.skip(), None, console_panel("\n".join(logs), language)
    env.close(); desc = ["SFFF", "FHFH", "FFFH", "HFFG"]; policy = {(s // 4, s % 4): ARROWS[int(np.argmax(q[s]))] for s in range(16) if desc[s // 4][s % 4] not in "HG"}
    preview = policy_grid_image(desc, policy, "Learned policy on slippery FrozenLake", "frozenlake-policy.png")
    summary = save_summary("frozenlake", {"experiment": "FrozenLake", "q_values": q.tolist(), "success_rate": float(np.mean(successes[-500:])), "parameters": {"budget": budget, "alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"final_success={np.mean(successes[-500:]):.1%} artifact={summary}"))
    curve = (np.cumsum(successes) / np.arange(1, len(successes) + 1)).tolist()
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} episodes · {time.perf_counter() - started:.1f}s", language), metric_card(f"{np.mean(successes[-500:]):.1%}", "final 500-episode success rate", language), learning_figure(list(range(1, budget + 1)), curve, "FrozenLake cumulative success rate", "Success rate"), preview, summary, console_panel("\n".join(logs), language)


def blackjack_policy_image(q: dict, filename: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), sharex=True, sharey=True)
    for usable_ace, ax in enumerate(axes):
        policy = np.zeros((10, 10))
        for player_sum in range(12, 22):
            for dealer_card in range(1, 11):
                policy[player_sum - 12, dealer_card - 1] = int(np.argmax(q[(player_sum, dealer_card, bool(usable_ace))]))
        image = ax.imshow(policy, origin="lower", cmap="coolwarm", vmin=0, vmax=1, aspect="auto")
        ax.set_title("Usable ace" if usable_ace else "No usable ace")
        ax.set_xlabel("Dealer showing")
        ax.set_xticks(range(10), range(1, 11))
        ax.set_yticks(range(10), range(12, 22))
    axes[0].set_ylabel("Player sum")
    colorbar = fig.colorbar(image, ax=axes, ticks=[0, 1], shrink=.82)
    colorbar.ax.set_yticklabels(["Stick", "Hit"])
    fig.suptitle("Blackjack Monte Carlo policy", fontweight="bold")
    path = ARTIFACT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_blackjack(budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str):
    started = time.perf_counter(); rng = np.random.default_rng(seed); env = gym.make("Blackjack-v1", sab=True)
    q = defaultdict(lambda: np.zeros(2, dtype=float)); rewards = []
    logs = ["Blackjack first-visit Monte Carlo console", "=" * 72, elapsed_line(started, "CONFIG", f"episodes={budget} alpha={alpha:g} gamma={gamma:g} epsilon_start={epsilon:g}")]
    chunk = max(500, budget // 20)
    for episode in range(1, budget + 1):
        state, _ = env.reset(seed=seed + episode); trajectory = []; done = False
        current_eps = max(0.02, epsilon * (1 - episode / budget))
        while not done:
            action = int(rng.integers(2)) if rng.random() < current_eps else int(rng.choice(np.flatnonzero(q[state] == q[state].max())))
            nxt, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, action, float(reward))); state = nxt; done = terminated or truncated
        rewards.append(float(reward)); returns = 0.0; visited = set()
        for old_state, action, reward_step in reversed(trajectory):
            returns = reward_step + gamma * returns
            key = (old_state, action)
            if key not in visited:
                q[old_state][action] += alpha * (returns - q[old_state][action]); visited.add(key)
        if episode % chunk == 0 or episode == budget:
            win_rate = float(np.mean(np.asarray(rewards[-min(5000, len(rewards)):]) > 0))
            logs.append(elapsed_line(started, "TRAIN", f"episode={episode}/{budget} epsilon={current_eps:.3f} recent_win_rate={win_rate:.1%}"))
            cumulative = (np.cumsum(rewards) / np.arange(1, len(rewards) + 1)).tolist()
            yield status_card("running", copy_for(language)["running"], f"{episode:,}/{budget:,} episodes", language), metric_card(f"{win_rate:.1%}", "recent win rate", language), learning_figure(list(range(1, episode + 1)), cumulative, "Blackjack cumulative mean return", "Mean return"), gr.skip(), None, console_panel("\n".join(logs), language)
    env.close(); preview = blackjack_policy_image(q, "blackjack-policy.png")
    serialized_q = {str(state): values.tolist() for state, values in q.items()}
    summary = save_summary("blackjack", {"experiment": BLACKJACK, "q_values": serialized_q, "win_rate": float(np.mean(np.asarray(rewards[-5000:]) > 0)), "parameters": {"budget": budget, "alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"states={len(q)} artifact={summary}")); cumulative = (np.cumsum(rewards) / np.arange(1, len(rewards) + 1)).tolist()
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} episodes · {time.perf_counter() - started:.1f}s", language), metric_card(f"{np.mean(np.asarray(rewards[-5000:]) > 0):.1%}", "final 5,000-episode win rate", language), learning_figure(list(range(1, budget + 1)), cumulative, "Blackjack cumulative mean return", "Mean return"), preview, summary, console_panel("\n".join(logs), language)


def record_discrete_policy(env_id: str, q: np.ndarray, seed: int, filename: str, max_steps: int) -> str:
    env = gym.make(env_id, render_mode="rgb_array"); frames = []
    try:
        state, _ = env.reset(seed=seed)
        for step in range(max_steps):
            if step % 2 == 0:
                frame = env.render()
                if frame is not None: frames.append(frame)
            state, _, terminated, truncated, _ = env.step(int(np.argmax(q[int(state)])))
            if terminated or truncated:
                frame = env.render()
                if frame is not None: frames.append(frame)
                break
    finally:
        env.close()
    if not frames: raise RuntimeError("Environment returned no RGB frames")
    path = ARTIFACT_DIR / filename; imageio.mimsave(path, frames, duration=1 / 15, loop=0); return str(path)


def run_discrete_control(experiment: str, env_id: str, method: str, budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str):
    started = time.perf_counter(); rng = np.random.default_rng(seed); env = gym.make(env_id)
    q = np.zeros((env.observation_space.n, env.action_space.n)); rewards = []
    logs = [f"{experiment} training console", "=" * 72, elapsed_line(started, "CONFIG", f"method={method} episodes={budget} alpha={alpha:g} gamma={gamma:g} epsilon_start={epsilon:g}")]
    chunk = max(50, budget // 20); max_steps = 1000 if env_id.startswith("Cliff") else 200
    for episode in range(1, budget + 1):
        state, _ = env.reset(seed=seed + episode); current_eps = max(0.02, epsilon * (1 - episode / budget))
        action = int(rng.integers(env.action_space.n)) if rng.random() < current_eps else int(rng.choice(np.flatnonzero(q[state] == q[state].max())))
        total = 0.0
        for _ in range(max_steps):
            nxt, reward, terminated, truncated, _ = env.step(action); done = terminated or truncated
            nxt_action = int(rng.integers(env.action_space.n)) if rng.random() < current_eps else int(rng.choice(np.flatnonzero(q[nxt] == q[nxt].max())))
            target = reward if done else reward + gamma * (q[nxt, nxt_action] if method == "SARSA" else q[nxt].max())
            q[state, action] += alpha * (target - q[state, action]); total += float(reward); state, action = nxt, nxt_action
            if done: break
        rewards.append(total)
        if episode % chunk == 0 or episode == budget:
            recent = float(np.mean(rewards[-min(100, len(rewards)):]))
            logs.append(elapsed_line(started, "TRAIN", f"episode={episode}/{budget} epsilon={current_eps:.3f} recent_reward={recent:.1f}"))
            yield status_card("running", copy_for(language)["running"], f"{episode:,}/{budget:,} episodes", language), metric_card(f"{recent:.1f}", "recent mean episode reward", language), learning_figure(list(range(1, episode + 1)), rewards, f"{experiment} episode reward", "Episode reward"), gr.skip(), None, console_panel("\n".join(logs), language)
    env.close(); slug = "cliffwalking" if env_id.startswith("Cliff") else "taxi"
    try:
        gif = record_discrete_policy(env_id, q, seed + 10000, f"{slug}-trained.gif", max_steps)
        preview_kind = "replay GIF"
    except Exception as exc:
        gif = result_preview_image(experiment, "Training complete", f"{np.mean(rewards[-100:]):.1f}", "Final mean reward", x=list(range(1, budget + 1)), y=rewards, note=f"Replay unavailable: {type(exc).__name__}")
        preview_kind = "result image"
        logs.append(elapsed_line(started, "WARN", f"replay_unavailable={type(exc).__name__}: {exc}"))
    summary = save_summary(slug, {"experiment": experiment, "q_values": q.tolist(), "parameters": {"budget": budget, "alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"preview={preview_kind} path={gif} artifact={summary}"))
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} episodes · {time.perf_counter() - started:.1f}s", language), metric_card(f"{np.mean(rewards[-100:]):.1f}", "final 100-episode mean reward", language), learning_figure(list(range(1, budget + 1)), rewards, f"{experiment} episode reward", "Episode reward"), gif, summary, console_panel("\n".join(logs), language)


def mountain_state(obs: np.ndarray, bins=(24, 20)) -> tuple[int, int]:
    low = np.array([-1.2, -0.07]); high = np.array([0.6, 0.07]); scaled = (np.asarray(obs) - low) / (high - low)
    indices = np.floor(scaled * np.array(bins)).astype(int)
    return tuple(np.clip(indices, 0, np.array(bins) - 1))


def record_tabular_control(env_id: str, policy, seed: int, filename: str, max_steps: int = 500) -> str:
    env = gym.make(env_id, render_mode="rgb_array"); frames = []
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(max_steps):
            frame = env.render()
            if frame is not None: frames.append(frame)
            action = policy(obs); obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated: break
    finally:
        env.close()
    if not frames: raise RuntimeError("Environment returned no RGB frames")
    path = ARTIFACT_DIR / filename; imageio.mimsave(path, frames, duration=1 / 30, loop=0); return str(path)


def run_mountaincar(budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str):
    started = time.perf_counter(); rng = np.random.default_rng(seed); env = gym.make("MountainCar-v0"); q = np.zeros((24, 20, 3)); rewards = []
    logs = ["MountainCar tabular Q-Learning console", "=" * 72, elapsed_line(started, "CONFIG", f"episodes={budget} bins=24x20 alpha={alpha:g} gamma={gamma:g} epsilon_start={epsilon:g}")]
    chunk = max(25, budget // 20)
    for episode in range(1, budget + 1):
        obs, _ = env.reset(seed=seed + episode); state = mountain_state(obs); total = 0.0
        current_eps = max(0.02, epsilon * (1 - episode / budget))
        for _ in range(200):
            action = int(rng.integers(3)) if rng.random() < current_eps else int(np.argmax(q[state]))
            nxt_obs, reward, terminated, truncated, _ = env.step(action); nxt = mountain_state(nxt_obs)
            shaped = reward + 25.0 * max(0.0, float(nxt_obs[0] - obs[0])) + (100.0 if terminated else 0.0)
            q[state][action] += alpha * (shaped + (0 if terminated else gamma * q[nxt].max()) - q[state][action])
            total += reward; obs = nxt_obs; state = nxt
            if terminated or truncated: break
        rewards.append(total)
        if episode % chunk == 0 or episode == budget:
            recent = float(np.mean(rewards[-min(100, len(rewards)):]))
            logs.append(elapsed_line(started, "TRAIN", f"episode={episode}/{budget} epsilon={current_eps:.3f} recent_reward={recent:.1f}"))
            yield status_card("running", copy_for(language)["running"], f"{episode:,}/{budget:,} episodes", language), metric_card(f"{recent:.1f}", "recent mean episode reward", language), learning_figure(list(range(1, episode + 1)), rewards, "MountainCar episode reward", "Episode reward"), gr.skip(), None, console_panel("\n".join(logs), language)
    env.close()
    try:
        gif = record_tabular_control("MountainCar-v0", lambda obs: int(np.argmax(q[mountain_state(obs)])), seed + 10000, "mountaincar-trained.gif", 200)
        preview_kind = "replay GIF"
    except Exception as exc:
        gif = result_preview_image(MOUNTAINCAR, "Training complete", f"{np.mean(rewards[-100:]):.1f}", "Final mean reward", x=list(range(1, budget + 1)), y=rewards, note=f"Replay unavailable: {type(exc).__name__}")
        preview_kind = "result image"
        logs.append(elapsed_line(started, "WARN", f"replay_unavailable={type(exc).__name__}: {exc}"))
    summary = save_summary("mountaincar", {"experiment": "MountainCar", "q_values": q.tolist(), "parameters": {"budget": budget, "alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"preview={preview_kind} path={gif} artifact={summary}"))
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} episodes · {time.perf_counter() - started:.1f}s", language), metric_card(f"{np.mean(rewards[-100:]):.1f}", "final 100-episode mean reward", language), learning_figure(list(range(1, budget + 1)), rewards, "MountainCar episode reward", "Episode reward"), gif, summary, console_panel("\n".join(logs), language)


def record_model(model, env_id: str, seed: int, filename: str, max_steps: int) -> str:
    """Record a deterministic policy using the configured headless renderer."""
    env = gym.make(env_id, render_mode="rgb_array")
    frames = []
    try:
        obs, _ = env.reset(seed=seed)
        for step in range(max_steps):
            if step % 2 == 0:
                frame = env.render()
                if isinstance(frame, np.ndarray) and frame.ndim == 3:
                    frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()
    if not frames:
        raise RuntimeError("Environment returned no RGB frames")
    path = ARTIFACT_DIR / filename
    imageio.mimsave(path, frames, duration=1 / 15, loop=0)
    return str(path)


def run_deep_control(
    experiment: str,
    env_id: str,
    algorithm: str,
    budget: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    seed: int,
    language: str,
    epochs: int = 6,
    run_id: str | None = None,
):
    started = time.perf_counter()
    DQN, PPO, SAC, evaluate_policy = deep_rl_runtime()
    env = gym.make(env_id)
    if isinstance(env.observation_space, gym.spaces.Dict):
        policy = "MultiInputPolicy"
    elif isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) == 3:
        policy = "CnnPolicy"
    else:
        policy = "MlpPolicy"
    if algorithm == "DQN":
        model = DQN(policy, env, learning_rate=alpha, gamma=gamma, learning_starts=min(1000, max(100, budget // 10)), buffer_size=max(10000, budget), exploration_initial_eps=epsilon, exploration_final_eps=0.05, seed=seed, device="cpu", verbose=0)
    elif algorithm == "SAC":
        model = SAC(policy, env, learning_rate=alpha, gamma=gamma, learning_starts=min(1000, max(100, budget // 10)), buffer_size=max(10000, budget), batch_size=64, seed=seed, device="cpu", verbose=0)
    else:
        model = PPO(policy, env, learning_rate=alpha, gamma=gamma, n_steps=min(1024, max(128, budget)), batch_size=64, seed=seed, device="cpu", verbose=0)
    run_id = run_id or f"{int(time.time())}-{seed}"
    targets = epoch_targets(int(budget), int(epochs))
    epoch_count = len(targets)
    logs = [
        f"{experiment} training console",
        "=" * 72,
        elapsed_line(started, "CONFIG", f"environment={env_id} algorithm={algorithm} policy={policy} total_steps={budget} epochs={epoch_count} learning_rate={alpha:g} gamma={gamma:g} seed={seed} device=cpu"),
    ]
    xs: list[float] = []
    rewards: list[float] = []
    trained = 0
    last_preview = example_preview(experiment)
    last_model = ""
    try:
        for target, epoch in targets.items():
            model.learn(total_timesteps=max(1, target - trained), reset_num_timesteps=False, progress_bar=False)
            trained = target
            eval_env = gym.make(env_id)
            try:
                values, _ = evaluate_policy(model, eval_env, n_eval_episodes=3, deterministic=True, return_episode_rewards=True, warn=False)
            finally:
                eval_env.close()
            mean = float(np.mean(values))
            xs.append(float(trained)); rewards.append(mean)
            epoch_dir = model_epoch_dir(experiment, run_id, epoch)
            model_path = epoch_dir / "policy"
            model.save(model_path)
            last_model = str(model_path.with_suffix(".zip"))
            try:
                last_preview = record_model(
                    model,
                    env_id,
                    seed + 10_000 + epoch,
                    f"{epoch_dir.name}/learned-policy.gif",
                    500 if env_id in {"CartPole-v1", "Acrobot-v1"} else 999,
                )
                preview_kind = "replay GIF"
            except Exception as exc:
                last_preview = result_preview_image(
                    experiment,
                    "Training complete",
                    f"{mean:.1f}",
                    "Evaluation reward",
                    filename=f"{epoch_dir.name}/learned-policy.png",
                    x=xs,
                    y=rewards,
                    note=f"Training succeeded; replay unavailable: {type(exc).__name__}",
                    algorithm=algorithm,
                )
                preview_kind = "result image"
                logs.append(elapsed_line(started, "WARN", f"epoch={epoch} replay_unavailable={type(exc).__name__}: {exc}"))
            model_id = register_model(experiment, run_id, epoch, epoch_count, trained, budget, mean, last_model, last_preview)
            logs.append(elapsed_line(started, "EVAL", f"epoch={epoch}/{epoch_count} step={trained}/{budget} mean_reward={mean:.1f}"))
            logs.append(elapsed_line(started, "SAVE", f"model_id={model_id} model={Path(last_model).name} preview={preview_kind}"))
            yield (
                status_card("running", copy_for(language)["running"], f"Epoch {epoch}/{epoch_count} · {trained:,}/{budget:,} steps", language),
                metric_card(f"{mean:.1f}", "3-episode evaluation reward", language),
                learning_figure(xs, rewards, f"{experiment} evaluation reward", "Mean reward"),
                last_preview,
                None,
                console_panel("\n".join(logs), language),
            )
    finally:
        env.close()
    summary = save_summary(model_slug(experiment), {
        "experiment": experiment,
        "evaluation_steps": xs,
        "evaluation_rewards": rewards,
        "model": last_model,
        "models": [record["model_id"] for record in load_models(experiment) if record["run_id"] == run_id],
        "parameters": {"budget": budget, "epochs": epoch_count, "learning_rate": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed},
    })
    logs.append(elapsed_line(started, "DONE", f"saved_models={epoch_count} artifact={summary}"))
    yield status_card("complete", copy_for(language)["complete"], f"{epoch_count} epoch models · {time.perf_counter() - started:.1f}s", language), metric_card(f"{rewards[-1]:.1f}", "final evaluation reward", language), learning_figure(xs, rewards, f"{experiment} evaluation reward", "Mean reward"), last_preview, summary, console_panel("\n".join(logs), language)


def error_figure(title: str, message: str, heading: str = "Run stopped"):
    fig, ax = plt.subplots(figsize=(8.2, 4.0)); ax.axis("off")
    ax.text(.5, .62, heading, ha="center", va="center", fontsize=20, fontweight="bold", color="#27324a")
    ax.text(.5, .43, title, ha="center", va="center", fontsize=13, color="#5b5ce2")
    ax.text(.5, .25, message[:180], ha="center", va="center", fontsize=10, color="#68748a", wrap=True)
    fig.tight_layout(); return fig


def run_catalog_experiment(experiment: str, budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str, epochs: int = 6, run_id: str | None = None):
    env_id = catalog_env_id(experiment); started = time.perf_counter()
    logs = [f"{env_id} automatic training console", "=" * 72, elapsed_line(started, "REGISTER", f"environment={env_id} family={experiment.split(' · ', 1)[0]}"), elapsed_line(started, "CONFIG", f"budget={budget} learning_rate={alpha:g} gamma={gamma:g} epsilon={epsilon:g} seed={seed}")]
    yield status_card("running", copy_for(language)["running"], "Inspecting environment and action space", language), metric_card("AUTO", "selecting a compatible baseline", language), error_figure(env_id, "Inspecting environment and action space...", "Preparing environment"), gr.skip(), None, console_panel("\n".join(logs), language)
    env = None
    try:
        env = gym.make(env_id)
        action_space = env.action_space; observation_space = env.observation_space
        logs.append(elapsed_line(started, "SPACE", f"observation={observation_space} action={action_space}"))
        if isinstance(action_space, gym.spaces.Discrete):
            algorithm = "DQN"
        elif isinstance(action_space, gym.spaces.Box):
            algorithm = "SAC"
        elif isinstance(action_space, (gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)):
            algorithm = "PPO"
        else:
            raise ValueError(f"Unsupported action space for the automatic baseline: {action_space}")
        logs.append(elapsed_line(started, "AUTO", f"selected_algorithm={algorithm}")); env.close(); env = None
        yield status_card("running", copy_for(language)["running"], f"Auto selected {algorithm}", language), metric_card(algorithm, f"action space: {action_space}", language), error_figure(env_id, f"Initializing the {algorithm} model...", "Starting training"), gr.skip(), None, console_panel("\n".join(logs), language)
        for status, metric, curve, preview, artifact, console in run_deep_control(experiment, env_id, algorithm, budget, alpha, gamma, epsilon, seed, language, epochs, run_id):
            deep_text = re.search(r'<pre class="console-text">(.*?)</pre>', console, re.DOTALL)
            combined = "\n".join(logs) + ("\n\n" + html.unescape(deep_text.group(1)) if deep_text else "")
            yield status, metric, curve, preview, artifact, console_panel(combined, language)
    except Exception as exc:
        if env is not None:
            env.close()
        message = f"{type(exc).__name__}: {exc}"; logs.append(elapsed_line(started, "ERROR", message)); logs.append(elapsed_line(started, "HINT", "All maintained runtimes are preinstalled. This registered ID may require a retired legacy engine; choose its current environment version."))
        summary = save_summary(env_id, {"experiment": experiment, "environment": env_id, "status": "registered-but-unavailable", "error": message, "parameters": {"budget": budget, "learning_rate": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
        diagnostic = result_preview_image(experiment, "Diagnostic", "LEGACY", "environment status", note=message)
        yield status_card("idle", "Legacy environment", "Choose the current maintained version", language), metric_card("LEGACY", "see the latest log lines", language), error_figure(env_id, message), diagnostic, summary, console_panel("\n".join(logs), language)


def _bandit_preview(probabilities: np.ndarray, q: np.ndarray, output: Path) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(1, len(q) + 1)
    ax.bar(positions - 0.16, probabilities, 0.32, label="True probability", color="#93c5fd")
    ax.bar(positions + 0.16, q, 0.32, label="Learned estimate", color="#5b5ce2")
    ax.set(xticks=positions, xlabel="Arm", ylabel="Reward probability", ylim=(0, 1), title="True vs learned arm values")
    ax.legend(); ax.grid(axis="y", alpha=0.2); fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(output)


def run_bandit_epochs(budget: int, alpha: float, epsilon: float, seed: int, language: str, epochs: int, run_id: str):
    started = time.perf_counter(); rng = np.random.default_rng(seed)
    probabilities = np.array([0.35, 0.50, 0.72, 0.58]); q = np.zeros(4); counts = np.zeros(4, dtype=int); rewards: list[float] = []
    targets = epoch_targets(budget, epochs); logs = ["Multi-Armed Bandit training console", "=" * 72, elapsed_line(started, "CONFIG", f"steps={budget} epochs={len(targets)} alpha={alpha:g} epsilon={epsilon:g} seed={seed}")]
    completed = 0; last_preview = example_preview(BANDIT); last_model = ""
    for target, epoch in targets.items():
        for _ in range(completed, target):
            action = int(rng.integers(4)) if rng.random() < epsilon else int(np.argmax(q)); reward = float(rng.random() < probabilities[action])
            counts[action] += 1; q[action] += alpha * (reward - q[action]); rewards.append(reward)
        completed = target; score = float(np.mean(rewards[-min(len(rewards), target - (list(targets)[epoch - 2] if epoch > 1 else 0)):]))
        epoch_dir = model_epoch_dir(BANDIT, run_id, epoch); model_path = epoch_dir / "policy.npz"; np.savez(model_path, q=q, counts=counts, probabilities=probabilities)
        last_model = str(model_path); last_preview = _bandit_preview(probabilities, q, epoch_dir / "learned-policy.png")
        model_id = register_model(BANDIT, run_id, epoch, len(targets), completed, budget, score, last_model, last_preview)
        logs.append(elapsed_line(started, "EPOCH", f"{epoch}/{len(targets)} step={completed}/{budget} average_reward={score:.3f} best_arm={int(np.argmax(q))+1}")); logs.append(elapsed_line(started, "SAVE", f"model_id={model_id}"))
        curve = (np.cumsum(rewards) / np.arange(1, len(rewards) + 1)).tolist()
        yield status_card("running", copy_for(language)["running"], f"Epoch {epoch}/{len(targets)} · {completed:,}/{budget:,} steps", language), metric_card(f"{score:.3f}", f"estimated best arm: {int(np.argmax(q))+1}", language), learning_figure(list(range(1, completed + 1)), curve, "Bandit cumulative average reward", "Average reward"), last_preview, None, console_panel("\n".join(logs), language)
    summary = save_summary("bandit", {"experiment": BANDIT, "q_values": q.tolist(), "counts": counts.tolist(), "models": [r["model_id"] for r in load_models(BANDIT) if r["run_id"] == run_id], "parameters": {"budget": budget, "epochs": len(targets), "alpha": alpha, "epsilon": epsilon, "seed": seed}})
    yield status_card("complete", copy_for(language)["complete"], f"{len(targets)} epoch models · {time.perf_counter()-started:.1f}s", language), metric_card(f"{np.mean(rewards):.3f}", f"best arm: {int(np.argmax(q))+1}", language), learning_figure(list(range(1, budget + 1)), (np.cumsum(rewards)/np.arange(1,budget+1)).tolist(), "Bandit cumulative average reward", "Average reward"), last_preview, summary, console_panel("\n".join(logs), language)


def run_gridworld_epochs(budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str, epochs: int, run_id: str):
    started = time.perf_counter(); rng = np.random.default_rng(seed); q = np.zeros((4,4,4)); rewards: list[float] = []; targets = epoch_targets(budget, epochs); completed = 0
    logs = ["GridWorld Q-Learning console", "="*72, elapsed_line(started,"CONFIG",f"episodes={budget} epochs={len(targets)} alpha={alpha:g} gamma={gamma:g} epsilon={epsilon:g}")]; last_preview = example_preview(GRIDWORLD)
    for target, epoch in targets.items():
        for _ in range(completed, target):
            state=(0,0); total=0.0
            for _ in range(100):
                action=int(rng.integers(4)) if rng.random()<epsilon else int(rng.choice(np.flatnonzero(q[state]==q[state].max()))); nxt,reward,done=grid_step(state,action)
                q[state][action]+=alpha*(reward+(0 if done else gamma*q[nxt].max())-q[state][action]); total+=reward; state=nxt
                if done: break
            rewards.append(total)
        completed=target; score=float(np.mean(rewards[-min(50,len(rewards)):])); epoch_dir=model_epoch_dir(GRIDWORLD,run_id,epoch); model_path=epoch_dir/"q-table.npz"; np.savez(model_path,q=q)
        policy={(r,c):ARROWS[int(np.argmax(q[r,c]))] for r in range(4) for c in range(4) if (r,c) not in {(1,1),(3,3)}}; values={(r,c):float(q[r,c].max()) for r in range(4) for c in range(4)}
        last_preview=policy_grid_image(["S...",".T..","....","...G"],policy,"Learned GridWorld policy",f"{epoch_dir.name}/learned-policy.png",values); model_id=register_model(GRIDWORLD,run_id,epoch,len(targets),completed,budget,score,str(model_path),last_preview)
        logs.append(elapsed_line(started,"EPOCH",f"{epoch}/{len(targets)} episode={completed}/{budget} recent_reward={score:.3f}")); logs.append(elapsed_line(started,"SAVE",f"model_id={model_id}"))
        yield status_card("running",copy_for(language)["running"],f"Epoch {epoch}/{len(targets)} · {completed:,}/{budget:,} episodes",language),metric_card(f"{score:.3f}","recent mean reward",language),learning_figure(list(range(1,completed+1)),rewards,"GridWorld episode reward","Episode reward"),last_preview,None,console_panel("\n".join(logs),language)
    summary=save_summary("gridworld",{"experiment":GRIDWORLD,"q_values":q.tolist(),"parameters":{"budget":budget,"epochs":len(targets),"alpha":alpha,"gamma":gamma,"epsilon":epsilon,"seed":seed}})
    yield status_card("complete",copy_for(language)["complete"],f"{len(targets)} epoch models · {time.perf_counter()-started:.1f}s",language),metric_card(f"{score:.3f}","final recent mean reward",language),learning_figure(list(range(1,budget+1)),rewards,"GridWorld episode reward","Episode reward"),last_preview,summary,console_panel("\n".join(logs),language)


def run_frozenlake_epochs(budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str, epochs: int, run_id: str):
    started=time.perf_counter(); rng=np.random.default_rng(seed); env=gym.make("FrozenLake-v1",map_name="4x4",is_slippery=True); q=np.zeros((16,4)); successes: list[float]=[]; targets=epoch_targets(budget,epochs); completed=0; desc=["SFFF","FHFH","FFFH","HFFG"]
    logs=["FrozenLake Q-Learning console","="*72,elapsed_line(started,"CONFIG",f"episodes={budget} epochs={len(targets)} slippery=true")]; last_preview=example_preview(FROZENLAKE)
    try:
        for target,epoch in targets.items():
            for episode in range(completed+1,target+1):
                state,_=env.reset(seed=seed+episode); done=False; won=0.0; current_eps=max(.02,epsilon*(1-episode/budget))
                while not done:
                    action=int(rng.integers(4)) if rng.random()<current_eps else int(rng.choice(np.flatnonzero(q[state]==q[state].max()))); nxt,reward,terminated,truncated,_=env.step(action); done=terminated or truncated
                    q[state,action]+=alpha*(reward+(0 if done else gamma*q[nxt].max())-q[state,action]); state=nxt; won=max(won,float(reward))
                successes.append(won)
            completed=target; score=float(np.mean(successes[-min(500,len(successes)):])); epoch_dir=model_epoch_dir(FROZENLAKE,run_id,epoch); model_path=epoch_dir/"q-table.npz"; np.savez(model_path,q=q)
            policy={(s//4,s%4):ARROWS[int(np.argmax(q[s]))] for s in range(16) if desc[s//4][s%4] not in "HG"}; last_preview=policy_grid_image(desc,policy,"Learned policy on slippery FrozenLake",f"{epoch_dir.name}/learned-policy.png")
            model_id=register_model(FROZENLAKE,run_id,epoch,len(targets),completed,budget,score,str(model_path),last_preview); logs.append(elapsed_line(started,"EPOCH",f"{epoch}/{len(targets)} episode={completed}/{budget} success={score:.1%}")); logs.append(elapsed_line(started,"SAVE",f"model_id={model_id}"))
            curve=(np.cumsum(successes)/np.arange(1,len(successes)+1)).tolist(); yield status_card("running",copy_for(language)["running"],f"Epoch {epoch}/{len(targets)} · {completed:,}/{budget:,} episodes",language),metric_card(f"{score:.1%}","recent success rate",language),learning_figure(list(range(1,completed+1)),curve,"FrozenLake cumulative success rate","Success rate"),last_preview,None,console_panel("\n".join(logs),language)
    finally: env.close()
    summary=save_summary("frozenlake",{"experiment":FROZENLAKE,"q_values":q.tolist(),"success_rate":score,"parameters":{"budget":budget,"epochs":len(targets),"alpha":alpha,"gamma":gamma,"epsilon":epsilon,"seed":seed}}); curve=(np.cumsum(successes)/np.arange(1,len(successes)+1)).tolist()
    yield status_card("complete",copy_for(language)["complete"],f"{len(targets)} epoch models · {time.perf_counter()-started:.1f}s",language),metric_card(f"{score:.1%}","final success rate",language),learning_figure(list(range(1,budget+1)),curve,"FrozenLake cumulative success rate","Success rate"),last_preview,summary,console_panel("\n".join(logs),language)


def run_blackjack_epochs(budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str, epochs: int, run_id: str):
    started=time.perf_counter(); rng=np.random.default_rng(seed); env=gym.make("Blackjack-v1",sab=True); q=defaultdict(lambda:np.zeros(2,dtype=float)); rewards: list[float]=[]; targets=epoch_targets(budget,epochs); completed=0; logs=["Blackjack first-visit Monte Carlo console","="*72,elapsed_line(started,"CONFIG",f"episodes={budget} epochs={len(targets)}")]; last_preview=example_preview(BLACKJACK)
    try:
        for target,epoch in targets.items():
            for episode in range(completed+1,target+1):
                state,_=env.reset(seed=seed+episode); trajectory=[]; done=False; current_eps=max(.02,epsilon*(1-episode/budget))
                while not done:
                    action=int(rng.integers(2)) if rng.random()<current_eps else int(rng.choice(np.flatnonzero(q[state]==q[state].max()))); nxt,reward,terminated,truncated,_=env.step(action); trajectory.append((state,action,float(reward))); state=nxt; done=terminated or truncated
                rewards.append(float(reward)); returns=0.0; visited=set()
                for old_state,action,reward_step in reversed(trajectory):
                    returns=reward_step+gamma*returns; pair=(old_state,action)
                    if pair not in visited: q[old_state][action]+=alpha*(returns-q[old_state][action]); visited.add(pair)
            completed=target; score=float(np.mean(np.asarray(rewards[-min(5000,len(rewards)):])>0)); epoch_dir=model_epoch_dir(BLACKJACK,run_id,epoch); model_path=epoch_dir/"policy.json"; serialized={str(state):values.tolist() for state,values in q.items()}; model_path.write_text(json.dumps({"q_values":serialized},ensure_ascii=False),encoding="utf-8")
            last_preview=blackjack_policy_image(q,f"{epoch_dir.name}/learned-policy.png"); model_id=register_model(BLACKJACK,run_id,epoch,len(targets),completed,budget,score,str(model_path),last_preview); logs.append(elapsed_line(started,"EPOCH",f"{epoch}/{len(targets)} episode={completed}/{budget} win_rate={score:.1%}")); logs.append(elapsed_line(started,"SAVE",f"model_id={model_id}"))
            curve=(np.cumsum(rewards)/np.arange(1,len(rewards)+1)).tolist(); yield status_card("running",copy_for(language)["running"],f"Epoch {epoch}/{len(targets)} · {completed:,}/{budget:,} episodes",language),metric_card(f"{score:.1%}","recent win rate",language),learning_figure(list(range(1,completed+1)),curve,"Blackjack cumulative mean return","Mean return"),last_preview,None,console_panel("\n".join(logs),language)
    finally: env.close()
    summary=save_summary("blackjack",{"experiment":BLACKJACK,"states":len(q),"win_rate":score,"parameters":{"budget":budget,"epochs":len(targets),"alpha":alpha,"gamma":gamma,"epsilon":epsilon,"seed":seed}}); curve=(np.cumsum(rewards)/np.arange(1,len(rewards)+1)).tolist()
    yield status_card("complete",copy_for(language)["complete"],f"{len(targets)} epoch models · {time.perf_counter()-started:.1f}s",language),metric_card(f"{score:.1%}","final win rate",language),learning_figure(list(range(1,budget+1)),curve,"Blackjack cumulative mean return","Mean return"),last_preview,summary,console_panel("\n".join(logs),language)


def run_discrete_epochs(experiment: str, env_id: str, method: str, budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str, epochs: int, run_id: str):
    started=time.perf_counter(); rng=np.random.default_rng(seed); env=gym.make(env_id); q=np.zeros((env.observation_space.n,env.action_space.n)); rewards: list[float]=[]; targets=epoch_targets(budget,epochs); completed=0; max_steps=1000 if env_id.startswith("Cliff") else 200; logs=[f"{experiment} training console","="*72,elapsed_line(started,"CONFIG",f"method={method} episodes={budget} epochs={len(targets)}")]; last_preview=example_preview(experiment)
    try:
        for target,epoch in targets.items():
            for episode in range(completed+1,target+1):
                state,_=env.reset(seed=seed+episode); current_eps=max(.02,epsilon*(1-episode/budget)); action=int(rng.integers(env.action_space.n)) if rng.random()<current_eps else int(rng.choice(np.flatnonzero(q[state]==q[state].max()))); total=0.0
                for _ in range(max_steps):
                    nxt,reward,terminated,truncated,_=env.step(action); done=terminated or truncated; nxt_action=int(rng.integers(env.action_space.n)) if rng.random()<current_eps else int(rng.choice(np.flatnonzero(q[nxt]==q[nxt].max()))); target_value=reward if done else reward+gamma*(q[nxt,nxt_action] if method=="SARSA" else q[nxt].max()); q[state,action]+=alpha*(target_value-q[state,action]); total+=float(reward); state,action=nxt,nxt_action
                    if done: break
                rewards.append(total)
            completed=target; score=float(np.mean(rewards[-min(100,len(rewards)):])); epoch_dir=model_epoch_dir(experiment,run_id,epoch); model_path=epoch_dir/"q-table.npz"; np.savez(model_path,q=q)
            try: last_preview=record_discrete_policy(env_id,q,seed+10000+epoch,f"{epoch_dir.name}/learned-policy.gif",max_steps)
            except Exception as exc: last_preview=result_preview_image(experiment,"Training complete",f"{score:.1f}","Recent mean reward",filename=f"{epoch_dir.name}/learned-policy.png",x=list(range(1,completed+1)),y=rewards,note=f"Replay unavailable: {type(exc).__name__}")
            model_id=register_model(experiment,run_id,epoch,len(targets),completed,budget,score,str(model_path),last_preview); logs.append(elapsed_line(started,"EPOCH",f"{epoch}/{len(targets)} episode={completed}/{budget} reward={score:.1f}")); logs.append(elapsed_line(started,"SAVE",f"model_id={model_id}"))
            yield status_card("running",copy_for(language)["running"],f"Epoch {epoch}/{len(targets)} · {completed:,}/{budget:,} episodes",language),metric_card(f"{score:.1f}","recent mean reward",language),learning_figure(list(range(1,completed+1)),rewards,f"{experiment} episode reward","Episode reward"),last_preview,None,console_panel("\n".join(logs),language)
    finally: env.close()
    summary=save_summary(model_slug(experiment),{"experiment":experiment,"q_values":q.tolist(),"parameters":{"budget":budget,"epochs":len(targets),"alpha":alpha,"gamma":gamma,"epsilon":epsilon,"seed":seed}})
    yield status_card("complete",copy_for(language)["complete"],f"{len(targets)} epoch models · {time.perf_counter()-started:.1f}s",language),metric_card(f"{score:.1f}","final recent mean reward",language),learning_figure(list(range(1,budget+1)),rewards,f"{experiment} episode reward","Episode reward"),last_preview,summary,console_panel("\n".join(logs),language)


def run_mountaincar_epochs(budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str, epochs: int, run_id: str):
    started=time.perf_counter(); rng=np.random.default_rng(seed); env=gym.make("MountainCar-v0"); q=np.zeros((24,20,3)); rewards: list[float]=[]; targets=epoch_targets(budget,epochs); completed=0; logs=["MountainCar Q-Learning console","="*72,elapsed_line(started,"CONFIG",f"episodes={budget} epochs={len(targets)} bins=24x20")]; last_preview=example_preview(MOUNTAINCAR)
    try:
        for target,epoch in targets.items():
            for episode in range(completed+1,target+1):
                obs,_=env.reset(seed=seed+episode); state=mountain_state(obs); total=0.0; current_eps=max(.02,epsilon*(1-episode/budget))
                for _ in range(200):
                    action=int(rng.integers(3)) if rng.random()<current_eps else int(np.argmax(q[state])); nxt_obs,reward,terminated,truncated,_=env.step(action); nxt=mountain_state(nxt_obs); shaped=reward+25.0*max(0.0,float(nxt_obs[0]-obs[0]))+(100.0 if terminated else 0.0); q[state][action]+=alpha*(shaped+(0 if terminated else gamma*q[nxt].max())-q[state][action]); total+=reward; obs=nxt_obs; state=nxt
                    if terminated or truncated: break
                rewards.append(total)
            completed=target; score=float(np.mean(rewards[-min(100,len(rewards)):])); epoch_dir=model_epoch_dir(MOUNTAINCAR,run_id,epoch); model_path=epoch_dir/"q-table.npz"; np.savez(model_path,q=q)
            try: last_preview=record_tabular_control("MountainCar-v0",lambda obs:int(np.argmax(q[mountain_state(obs)])),seed+10000+epoch,f"{epoch_dir.name}/learned-policy.gif",200)
            except Exception as exc: last_preview=result_preview_image(MOUNTAINCAR,"Training complete",f"{score:.1f}","Recent mean reward",filename=f"{epoch_dir.name}/learned-policy.png",x=list(range(1,completed+1)),y=rewards,note=f"Replay unavailable: {type(exc).__name__}")
            model_id=register_model(MOUNTAINCAR,run_id,epoch,len(targets),completed,budget,score,str(model_path),last_preview); logs.append(elapsed_line(started,"EPOCH",f"{epoch}/{len(targets)} episode={completed}/{budget} reward={score:.1f}")); logs.append(elapsed_line(started,"SAVE",f"model_id={model_id}"))
            yield status_card("running",copy_for(language)["running"],f"Epoch {epoch}/{len(targets)} · {completed:,}/{budget:,} episodes",language),metric_card(f"{score:.1f}","recent mean reward",language),learning_figure(list(range(1,completed+1)),rewards,"MountainCar episode reward","Episode reward"),last_preview,None,console_panel("\n".join(logs),language)
    finally: env.close()
    summary=save_summary("mountaincar",{"experiment":MOUNTAINCAR,"q_values":q.tolist(),"parameters":{"budget":budget,"epochs":len(targets),"alpha":alpha,"gamma":gamma,"epsilon":epsilon,"seed":seed}})
    yield status_card("complete",copy_for(language)["complete"],f"{len(targets)} epoch models · {time.perf_counter()-started:.1f}s",language),metric_card(f"{score:.1f}","final recent mean reward",language),learning_figure(list(range(1,budget+1)),rewards,"MountainCar episode reward","Episode reward"),last_preview,summary,console_panel("\n".join(logs),language)


def train(experiment: str, steps_per_epoch: float, epochs: float, alpha: float, gamma: float, epsilon: float, seed: float, language: str):
    steps_per_epoch, epochs, seed = int(steps_per_epoch), max(1, min(12, int(epochs))), int(seed)
    budget = steps_per_epoch * epochs
    run_id = f"{int(time.time())}-{time.time_ns() % 1_000_000:06d}"
    try:
        if experiment not in EXPERIMENT_CHOICES:
            raise ValueError("This environment is not registered in the current runtime. Refresh the page and choose an available task.")
        if is_catalog_experiment(experiment):
            yield from run_catalog_experiment(experiment, budget, alpha, gamma, epsilon, seed, language, epochs, run_id)
        elif experiment == BANDIT:
            yield from run_bandit_epochs(budget, alpha, epsilon, seed, language, epochs, run_id)
        elif experiment == BLACKJACK:
            yield from run_blackjack_epochs(budget, alpha, gamma, epsilon, seed, language, epochs, run_id)
        elif experiment == GRIDWORLD:
            yield from run_gridworld_epochs(budget, alpha, gamma, epsilon, seed, language, epochs, run_id)
        elif experiment == FROZENLAKE:
            yield from run_frozenlake_epochs(budget, alpha, gamma, epsilon, seed, language, epochs, run_id)
        elif experiment == CLIFF:
            yield from run_discrete_epochs(CLIFF, "CliffWalking-v1", "SARSA", budget, alpha, gamma, epsilon, seed, language, epochs, run_id)
        elif experiment == TAXI:
            yield from run_discrete_epochs(TAXI, "Taxi-v4", "Q-Learning", budget, alpha, gamma, epsilon, seed, language, epochs, run_id)
        elif experiment == MOUNTAINCAR:
            yield from run_mountaincar_epochs(budget, alpha, gamma, epsilon, seed, language, epochs, run_id)
        else:
            env_id = EXPERIMENTS[experiment]["environment"]
            yield from run_deep_control(experiment, env_id, EXPERIMENTS[experiment]["algorithm"], budget, alpha, gamma, epsilon, seed, language, epochs, run_id)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        diagnostic = result_preview_image(experiment, "Run stopped", "ERROR", "training result", note=message)
        summary = save_summary(experiment, {"experiment": experiment, "status": "failed", "error": message, "parameters": {"steps_per_epoch": steps_per_epoch, "epochs": epochs, "budget": budget, "learning_rate": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
        logs = [f"{experiment} training console", "=" * 72, elapsed_line(time.perf_counter(), "ERROR", message), "RESULT  A diagnostic preview and JSON summary were produced."]
        yield status_card("idle", "Training stopped", "Diagnostic result produced", language), metric_card("ERROR", "see the latest log lines", language), error_figure(experiment, message), diagnostic, summary, console_panel("\n".join(logs), language)


def train_with_ui(experiment: str, steps_per_epoch: float, epochs: float, alpha: float, gamma: float, epsilon: float, seed: float, language: str):
    """Stream waiting, training, and completion states through one request."""
    copy = copy_for(language)
    unchanged = (gr.skip(),) * 8
    yield *unchanged, gr.HTML(value=waiting_panel(language), visible=True), gr.Button(value=copy["start_running"], interactive=False)
    preferred = None
    for result in train(experiment, steps_per_epoch, epochs, alpha, gamma, epsilon, seed, language):
        values = list(result)
        artifact_value = values[4]
        values[4] = gr.File(value=None, visible=False) if artifact_value is None else gr.File(value=artifact_value, label=copy["artifact"], visible=True)
        records = load_models(experiment)
        if records:
            preferred = records[0]["model_id"]
        values.extend([model_selector(experiment, language, preferred, interactive=False), preview_provenance(experiment, preferred, language)])
        yield *values, gr.skip(), gr.skip()
    final_selector = model_selector(experiment, language, preferred, interactive=True)
    yield *(gr.skip(),) * 6, final_selector, preview_provenance(experiment, preferred, language), gr.HTML(value="", visible=False), gr.Button(value=copy["start"], interactive=True)


def slider_update(label: str, spec: tuple[float, float, float, float], visible: bool = True):
    minimum, maximum, value, step = spec
    return gr.Slider(minimum=minimum, maximum=maximum, value=value, step=step, label=label, visible=visible)


def select_experiment(experiment: str, language: str):
    copy = copy_for(language); cfg = experiment_config(experiment)
    step_spec, epoch_spec = epoch_specs(experiment)
    selector = model_selector(experiment, language)
    return (
        hero_html(language, experiment),
        task_brief(experiment, language),
        slider_update(copy["steps_per_epoch"], step_spec),
        slider_update(copy["epochs"], epoch_spec),
        slider_update(copy["alpha"], cfg["alpha"]),
        slider_update(copy["gamma"], cfg["gamma"], cfg["gamma_visible"]),
        slider_update(copy["epsilon"], cfg["epsilon"], cfg["algorithm"] not in {"PPO", "SAC"}),
        status_card("idle", copy["ready"], copy["ready_detail"], language),
        metric_card("—", copy["metric_waiting"], language),
        console_panel(copy["log_waiting"], language),
        example_preview(experiment),
        gr.File(value=None, label=copy["artifact"], visible=False),
        selector,
        preview_provenance(experiment, selector.value, language),
    )


def switch_language(language: str, experiment: str, seed: float, learning_path: str, query: str, feature_value: str, page: float):
    copy = copy_for(language); cfg = experiment_config(experiment)
    step_spec, epoch_spec = epoch_specs(experiment)
    feature_options = feature_choices(query, learning_path, language)
    valid_features = {value for _, value in feature_options}
    selected_feature = feature_value if feature_value in valid_features else ALL_FEATURES
    gallery_values = catalog_page(query, learning_path, selected_feature, int(page), language)
    selector = model_selector(experiment, language)
    return (
        hero_html(language, experiment), catalog_header_html(language),
        gr.Radio(choices=path_choices(language), value=learning_path, label=copy["path"]),
        gr.Textbox(value=query, label=copy["search"], placeholder=copy["search_placeholder"]),
        gr.Radio(choices=feature_options, value=selected_feature, label=copy["goal"]),
        goal_pager_html(language),
        catalog_wait_html(language),
        *gallery_values,
        panel_html(copy["settings"], copy["settings_copy"]), gr.Accordion(label=copy["advanced"], open=False), task_brief(experiment, language),
        slider_update(copy["steps_per_epoch"], step_spec), slider_update(copy["epochs"], epoch_spec), slider_update(copy["alpha"], cfg["alpha"]), slider_update(copy["gamma"], cfg["gamma"], cfg["gamma_visible"]), slider_update(copy["epsilon"], cfg["epsilon"], cfg["algorithm"] not in {"PPO", "SAC"}),
        gr.Number(value=seed, precision=0, label=copy["seed"]), gr.Button(value=copy["start"]), status_card("idle", copy["ready"], copy["ready_detail"], language),
        metric_card("—", copy["metric_waiting"], language), panel_html(copy["curve"], copy["curve_copy"]), console_panel(copy["log_waiting"], language),
        panel_html(copy["preview"], copy["preview_copy"], "artifact-note"), gr.File(label=copy["artifact"], visible=False), selector, preview_provenance(experiment, selector.value, language),
    )


CSS = """
:root { --ink:#172033; --muted:#68748a; --line:#e4e8f0; --canvas:#f4f6fa; --brand:#5b5ce2; --green:#13a36f; }
.gradio-container { width:100%!important; min-width:0!important; max-width:1180px!important; margin:0 auto!important; padding:28px clamp(10px,2vw,22px) 52px!important; box-sizing:border-box!important; background:var(--canvas); }
.gradio-container>.main { width:100%!important; min-width:0!important; padding:clamp(8px,1.5vw,24px)!important; box-sizing:border-box!important; }
.gradio-container .contain { width:100%!important; min-width:0!important; }
.hero-stack { position:relative!important; margin:0!important; padding:0!important; border:0!important; background:transparent!important; }
.language-bar { position:absolute!important; z-index:5!important; top:18px!important; right:20px!important; width:auto!important; min-width:0!important; margin:0!important; padding:0!important; border:0!important; background:transparent!important; }
.language-switch { width:216px!important; min-width:216px!important; margin:0!important; padding:3px!important; border:1px solid rgba(255,255,255,.18)!important; border-radius:10px!important; background:rgba(14,20,46,.58)!important; box-shadow:0 7px 20px rgba(5,8,24,.22)!important; backdrop-filter:blur(12px)!important; }
.language-switch>div { display:grid!important; grid-template-columns:1fr 1fr!important; gap:3px!important; }
.language-switch label { display:flex!important; cursor:pointer!important; }.language-switch input{display:none!important}
.language-switch label span { width:100%!important; min-height:34px!important; justify-content:center!important; padding:7px 13px!important; border-radius:7px!important; border:0!important; color:rgba(255,255,255,.72)!important; background:transparent!important; font-size:13px!important; font-weight:700!important; }
.language-switch label:has(input:checked) span,.language-switch input:checked+span { color:#fff!important; background:linear-gradient(135deg,#6667e8,#7778f2)!important; box-shadow:0 3px 9px rgba(13,15,55,.28)!important; }
.hero { position:relative; overflow:hidden; padding:38px 42px 34px; border:1px solid rgba(129,140,248,.2); border-radius:26px; color:#f8fafc; background:radial-gradient(circle at 88% 8%,rgba(125,127,255,.42),transparent 31%),radial-gradient(circle at 92% 92%,rgba(61,207,170,.18),transparent 30%),linear-gradient(132deg,#11182c 0%,#25265d 58%,#4546a4 100%); box-shadow:0 22px 54px rgba(25,32,56,.16); }
.project-mark { display:block; width:290px; max-width:55%; height:auto; margin:0 0 22px; padding:9px 13px; border-radius:11px; background:#fff; box-shadow:0 8px 24px rgba(8,15,35,.2); }
.brand-lockup{display:flex;align-items:center;gap:9px;width:max-content;max-width:calc(100% - 230px);margin:0 0 14px;padding:8px 12px;border:1px solid rgba(255,255,255,.28);border-radius:10px;background:rgba(12,17,59,.28);font-size:12px;font-weight:900;letter-spacing:.075em}.brand-lockup a{color:#fff!important;text-decoration:none!important}.brand-lockup a:last-child{color:#cfd5ff!important}.brand-lockup span{color:#aeb7ff}
.hero-topline{display:flex;align-items:center;gap:11px;margin-bottom:22px}.experiment-badge{padding:6px 11px;border:1px solid #fff;border-radius:999px;color:#25265d;background:#fff;box-shadow:0 4px 12px rgba(8,15,35,.16);font-size:12px;font-weight:800;letter-spacing:.06em}.hero-course{color:#b9c0d4;font-size:13px;font-weight:650}
.hero h1{max-width:760px;margin:0 0 12px;color:#fff;font-size:clamp(32px,5vw,48px);line-height:1.1;letter-spacing:-.035em}.hero-copy{max-width:760px;margin:0;color:#cdd3e2;font-size:15px;line-height:1.7}.hero-links{display:flex;flex-wrap:wrap;gap:9px;margin-top:25px}.hero-link{display:inline-flex;align-items:center;min-height:38px;padding:0 14px;border:1px solid rgba(255,255,255,.18);border-radius:9px;color:#eef2ff!important;background:rgba(255,255,255,.08);font-size:13px;font-weight:650;text-decoration:none!important}.hero-link.primary{color:#172554!important;background:#fff;border-color:#fff}
.lab-strip{display:flex;flex-wrap:wrap;gap:8px 22px;margin:17px 0 22px;padding:13px 18px;border:1px solid var(--line);border-radius:13px;background:#fff;color:var(--muted);font-size:13px;box-shadow:0 6px 20px rgba(18,25,43,.035)}.lab-strip strong{margin-left:5px;color:var(--ink)}
.catalog-card{margin:0 0 18px!important;padding:22px!important;border:1px solid var(--line)!important;border-radius:17px!important;background:#fff!important;box-shadow:0 10px 30px rgba(18,25,43,.045)!important}.catalog-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:18px}.ui-version{flex:none;padding:6px 10px;border:1px solid #dbe2f2;border-radius:999px;color:#536178;background:#f7f9fd;font-size:10px;font-weight:750}.catalog-family{min-width:0!important}.catalog-family>div{display:grid!important;grid-template-columns:repeat(auto-fit,minmax(210px,1fr))!important;gap:8px!important}.catalog-family label,.catalog-feature label{position:relative!important;display:flex!important;cursor:pointer!important}.catalog-family input,.catalog-feature input{position:absolute!important;width:1px!important;height:1px!important;opacity:0!important;pointer-events:none!important}.catalog-family label span{width:100%!important;min-height:42px!important;justify-content:flex-start!important;padding:10px 13px!important;border:1px solid #dfe3ef!important;border-radius:10px!important;background:#fff!important;font-size:12px!important;font-weight:750!important}.catalog-family label:has(input:checked) span,.catalog-family input:checked+span{color:#fff!important;border-color:#5b5ce2!important;background:#5b5ce2!important;box-shadow:0 5px 14px rgba(91,92,226,.18)!important}.catalog-search{min-width:0!important}.catalog-feature{margin:2px 0 14px!important;padding:11px 13px!important;border:1px solid #e6e8f4!important;border-radius:12px!important;background:#f8f9fd!important}.catalog-feature>div{display:grid!important;grid-template-columns:repeat(auto-fit,minmax(220px,1fr))!important;gap:7px!important}.catalog-feature label span{width:100%!important;min-height:38px!important;justify-content:flex-start!important;padding:8px 11px!important;border:1px solid transparent!important;border-radius:9px!important;font-size:12px!important;font-weight:700!important}.catalog-feature label:has(input:checked) span,.catalog-feature input:checked+span{color:#fff!important;border-color:#5b5ce2!important;background:#5b5ce2!important;box-shadow:none!important}.catalog-meta{margin-right:auto!important;color:var(--muted);font-size:12px;font-weight:700}.catalog-pager{align-items:center!important;justify-content:flex-end!important;gap:8px!important}.catalog-pager button{max-width:110px!important;border-radius:9px!important}.experiment-gallery{max-height:660px;overflow:auto;padding:4px!important}.experiment-gallery .grid-wrap{display:grid!important;grid-template-columns:repeat(auto-fill,minmax(230px,270px))!important;justify-content:start!important;gap:12px!important}.experiment-gallery button,.experiment-gallery .thumbnail-item{position:relative!important;width:100%!important;min-width:0!important;max-width:270px!important;overflow:hidden!important;border:1px solid var(--line)!important;border-radius:14px!important;background:#0b1230!important;box-shadow:0 7px 18px rgba(18,25,43,.045)!important;transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease!important}.experiment-gallery button::after,.experiment-gallery .thumbnail-item::after{content:attr(data-feature)!important;position:absolute!important;right:12px!important;bottom:11px!important;z-index:3!important;max-width:72%!important;padding:6px 10px!important;border:1px solid rgba(255,255,255,.28)!important;border-radius:999px!important;color:#fff!important;background:rgba(16,24,56,.78)!important;backdrop-filter:blur(8px)!important;font-size:10px!important;font-weight:800!important;line-height:1.2!important;text-align:right!important;white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important}.experiment-gallery button:hover,.experiment-gallery .thumbnail-item:hover{transform:translateY(-2px);border-color:#a5b4fc!important;box-shadow:0 12px 25px rgba(50,55,120,.12)!important}.experiment-gallery .image-container,.experiment-gallery [data-testid="image"]{width:100%!important;height:auto!important;aspect-ratio:2/1!important;background:#0b1230!important;overflow:hidden!important}.experiment-gallery img{display:block!important;width:100%!important;height:100%!important;aspect-ratio:2/1!important;object-fit:cover!important;object-position:center!important}.experiment-gallery .caption,.experiment-gallery .label{position:absolute!important;inset:0 0 auto 0!important;z-index:2!important;display:block!important;min-height:72px!important;padding:14px 16px 18px!important;overflow:visible!important;text-overflow:clip!important;white-space:pre-line!important;overflow-wrap:anywhere!important;background:linear-gradient(180deg,rgba(5,9,30,.94),rgba(5,9,30,.76) 70%,transparent)!important;color:#fff!important;font-size:clamp(12px,1.25vw,17px)!important;font-weight:800!important;line-height:1.28!important;text-align:left!important;text-shadow:0 1px 2px rgba(0,0,0,.4)!important;pointer-events:none!important}.selected-experiment input{font-weight:750!important;color:var(--brand)!important;background:#f5f5ff!important}
.task-brief{display:grid;grid-template-columns:minmax(210px,34%) 1fr;gap:20px;margin:0 0 18px;padding:14px;border:1px solid #dfe3f5;border-radius:15px;background:linear-gradient(135deg,#fafaff,#f6fbff)}.task-brief__visual{display:flex;align-items:center;overflow:hidden;border-radius:11px;background:#171b3f}.task-brief__visual img{display:block;width:100%;height:auto;max-height:250px;min-height:190px;object-fit:contain;border-radius:11px}.task-brief__body{padding:9px 9px 7px}.task-kicker{color:var(--brand);font-size:10px;font-weight:850;letter-spacing:.12em}.task-brief h3{margin:6px 0;color:var(--ink);font-size:23px}.task-brief p{margin:0 0 13px;color:var(--muted);font-size:13px;line-height:1.6}.task-facts{display:grid;grid-template-columns:1fr 1fr;gap:8px}.task-facts span{padding:9px 11px;border:1px solid var(--line);border-radius:9px;background:#fff;color:var(--ink);font-size:11px;overflow-wrap:anywhere}.task-facts b{display:block;margin-bottom:3px;color:#8a94a8;font-size:9px;letter-spacing:.09em;text-transform:uppercase}.task-hint{margin-top:12px!important;margin-bottom:0!important;font-weight:650;color:#4b5563!important}
.training-guide{display:grid;grid-template-columns:minmax(210px,.62fr) minmax(0,1.8fr);gap:22px;margin:-4px 0 18px;padding:20px 22px;border:1px solid #dfe4f4;border-radius:17px;background:linear-gradient(135deg,#f8f9ff,#fff);box-shadow:0 10px 30px rgba(18,25,43,.04)}.training-guide__intro{padding:5px 2px}.training-guide__intro>span{display:block;margin-bottom:7px;color:var(--brand);font-size:10px;font-weight:850;letter-spacing:.13em}.training-guide__intro h3{margin:0 0 5px;color:var(--ink);font-size:18px}.training-guide__intro p,.training-guide article p{margin:0;color:var(--muted);font-size:12px;line-height:1.55}.training-guide__grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.training-guide article{padding:14px;border:1px solid #e0e5f0;border-radius:12px;background:#fff}.training-guide article>b{display:block;margin-bottom:8px;color:var(--brand);font-size:10px;letter-spacing:.12em}.training-guide article h4{margin:0 0 6px;color:var(--ink);font-size:13px}.training-guide article p{font-size:11px}
.control-card,.chart-card,.preview-card{border:1px solid var(--line)!important;border-radius:17px!important;background:#fff!important;box-shadow:0 10px 30px rgba(18,25,43,.045)!important}.training-layout{align-items:flex-start!important}.training-layout>.control-card,.training-layout>.results-stack{align-self:flex-start!important}.results-stack{min-width:0!important;gap:18px!important;padding:0!important;border:0!important;background:transparent!important}.control-card,.chart-card,.preview-card{padding:22px!important}.advanced-settings{margin:2px 0 12px!important;overflow:hidden!important;border:1px solid #dfe4ee!important;border-radius:11px!important;background:#fafbfe!important}.advanced-settings>button,.advanced-settings summary{min-height:44px!important;color:#3f485c!important;font-size:12px!important;font-weight:800!important}.result-summary{display:grid!important;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr)!important;gap:10px!important;margin:0 0 10px!important}.result-summary>div{min-width:0!important}.result-summary .run-state,.result-summary .live-metric{height:100%;margin-top:0}.panel-title{margin:0 0 5px;color:var(--ink);font-size:19px}.panel-copy,.artifact-note{margin:0 0 17px;color:var(--muted);font-size:13px;line-height:1.6}.policy-preview{min-height:360px!important;border:1px solid var(--line)!important;border-radius:13px!important;background:#f8f9fc!important;overflow:hidden!important}.policy-preview .image-container,.policy-preview [data-testid="image"]{min-height:360px!important;background:#f8f9fc!important}.policy-preview img{display:block!important;width:100%!important;height:100%!important;min-height:360px!important;max-height:560px!important;object-fit:contain!important;background:#f8f9fc!important}.artifact-download{height:76px!important;min-height:76px!important;margin-top:8px!important;overflow:hidden!important}.artifact-download [data-testid="status-tracker"]{height:76px!important}.artifact-download .empty{height:50px!important;min-height:50px!important}
.preview-provenance{display:flex;gap:9px;align-items:flex-start;margin:0 0 12px;padding:10px 12px;border:1px solid #e0e5f0;border-radius:10px;color:#59657a;background:#f8f9fc;font-size:11px;line-height:1.5}.preview-provenance span{flex:0 0 auto;width:8px;height:8px;margin-top:4px;border-radius:50%;background:#9ba5b5}.preview-provenance--ready{color:#16664d;border-color:#cfeadf;background:#f2fbf7}.preview-provenance--ready span{background:#13a36f}
.primary-btn{min-height:46px!important;border:0!important;border-radius:11px!important;background:linear-gradient(135deg,#5153d6,#6969ec)!important;font-weight:750!important}.primary-btn:disabled{opacity:.8!important;cursor:wait!important}.run-wait{position:relative;display:grid;grid-template-columns:auto 1fr;gap:12px;align-items:center;overflow:hidden;margin:0 0 16px;padding:14px 16px 17px;border:1px solid #c7d2fe;border-radius:13px;background:linear-gradient(135deg,#f5f5ff,#f0f7ff);box-shadow:0 8px 24px rgba(79,70,229,.08)}.run-wait__spinner{width:24px;height:24px;border:3px solid #d9ddff;border-top-color:#5b5ce2;border-radius:50%;animation:run-spin .8s linear infinite}.run-wait__copy strong,.run-wait__copy small{display:block}.run-wait__copy strong{color:#292d65;font-size:13px}.run-wait__copy small{margin-top:4px;color:#68748a;font-size:11px;line-height:1.5}.run-wait__elapsed{display:inline-block;margin-top:7px;color:#5b5ce2;font-size:11px;font-style:normal;font-weight:750}.run-wait__pulse{position:absolute;right:0;bottom:0;left:0;height:3px;background:#e0e7ff}.run-wait__pulse i{display:block;width:38%;height:100%;border-radius:999px;background:linear-gradient(90deg,transparent,#6366f1,#22c55e,transparent);animation:run-pulse 1.4s ease-in-out infinite}@keyframes run-spin{to{transform:rotate(360deg)}}@keyframes run-pulse{0%{transform:translateX(-110%)}100%{transform:translateX(285%)}}.run-state,.live-metric{display:flex;gap:12px;margin-top:14px;padding:14px 15px;border-radius:13px;background:#f8f9fc}.run-state__dot{width:9px;height:9px;margin-top:6px;border-radius:50%;background:#94a3b8}.run-state--running .run-state__dot{background:#5b5ce2;box-shadow:0 0 0 5px rgba(91,92,226,.13);animation:run-dot 1.2s ease-in-out infinite}@keyframes run-dot{50%{box-shadow:0 0 0 9px rgba(91,92,226,.04)}}.run-state--complete .run-state__dot{background:#13a36f}.run-state strong,.run-state small,.summary-label{display:block}.summary-label{color:#8a94a8;font-size:10px;font-weight:800;letter-spacing:.1em;text-transform:uppercase}.run-state strong{margin-top:3px;color:var(--ink);font-size:14px}.run-state small,.live-metric small{margin-top:3px;color:var(--muted);font-size:12px}.metric-reading{display:flex;align-items:baseline;gap:9px;margin-top:4px}.metric-reading strong{color:var(--ink);font-size:24px}
.console-panel{overflow:hidden;margin-top:18px;border:1px solid #202b3d;border-radius:13px;background:#0f1623}.console-head{display:flex;align-items:center;gap:9px;padding:11px 15px;border-bottom:1px solid #263244;color:#e2e8f0;font-size:12px;font-weight:750}.console-dot{width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 0 4px rgba(34,197,94,.12)}.console-text{box-sizing:border-box;height:300px;margin:0;padding:17px 18px;overflow:auto;white-space:pre;color:#cbd5e1!important;background:#0f1623!important;font:12px/1.58 "SFMono-Regular",Consolas,monospace!important;scrollbar-gutter:stable}.footer-note{margin-top:18px;text-align:center;color:#94a3b8;font-size:12px}.footer-note a{color:var(--brand)!important;text-decoration:none!important;font-weight:650}
@media(max-width:900px){.training-layout{flex-direction:column!important}.training-layout>.control-card,.training-layout>.results-stack{width:100%!important;min-width:0!important;flex:1 1 auto!important}}
@media(max-width:760px){.language-bar{top:14px!important;right:14px!important}.language-switch{width:196px!important;min-width:196px!important}.hero{padding:70px 22px 25px;border-radius:19px}.brand-lockup{max-width:100%;flex-wrap:wrap;font-size:10px;letter-spacing:.045em}.hero-topline{align-items:flex-start;flex-direction:column}.project-mark{max-width:70%}.catalog-card{padding:16px!important}.catalog-heading{display:block}.ui-version{display:inline-flex;margin:0 0 14px}.catalog-family>div,.catalog-feature>div{grid-template-columns:1fr!important}.experiment-gallery{max-height:580px}.experiment-gallery .grid-wrap{grid-template-columns:1fr!important}.experiment-gallery button,.experiment-gallery .thumbnail-item{max-width:none!important}.task-brief,.training-guide,.training-guide__grid{grid-template-columns:1fr}.task-brief__visual img{min-height:160px}.task-facts,.result-summary{grid-template-columns:1fr!important}.policy-preview,.policy-preview .image-container,.policy-preview [data-testid="image"],.policy-preview img{min-height:230px!important}.policy-preview img{max-height:420px!important}}
.catalog-family input,.catalog-feature input{position:absolute!important;inset:0!important;width:100%!important;height:100%!important;margin:0!important;opacity:0!important;cursor:pointer!important;pointer-events:auto!important}
.catalog-filter-row{display:grid!important;grid-template-columns:minmax(0,1fr) minmax(0,1fr)!important;align-items:stretch!important;gap:14px!important;margin:4px 0 16px!important}.catalog-filter-pane{min-width:0!important;margin:0!important;padding:14px!important;border:1px solid #e4e8f2!important;border-radius:13px!important;background:#fafbfe!important}.catalog-filter-pane--path{background:linear-gradient(145deg,#fbfbff,#f7f8ff)!important}.catalog-filter-pane--goal{background:linear-gradient(145deg,#fbfdfd,#f6fbfa)!important}.catalog-filter-pane .catalog-family,.catalog-filter-pane .catalog-feature{margin:0!important;padding:0!important;border:0!important;background:transparent!important}.catalog-filter-pane .catalog-family>div,.catalog-filter-pane .catalog-feature>div{grid-template-columns:1fr!important}.catalog-filter-pane .catalog-family label span,.catalog-filter-pane .catalog-feature label span{min-height:40px!important}.catalog-filter-pane .catalog-feature label span{border-color:#e7eaf1!important;background:#fff!important}.catalog-filter-pane .catalog-feature label:has(input:checked) span,.catalog-filter-pane .catalog-feature input:checked+span{border-color:#5b5ce2!important;background:#5b5ce2!important}.catalog-search{margin-bottom:10px!important}
.card-image-preload{position:absolute!important;width:1px!important;height:1px!important;overflow:hidden!important;opacity:.001!important;pointer-events:none!important}.card-image-preload img{position:absolute!important;width:1px!important;height:1px!important}
.goal-local-pager{display:grid!important;grid-template-columns:88px 1fr 88px!important;align-items:center!important;gap:7px!important;margin-top:10px!important}.goal-local-pager[hidden]{display:none!important}.goal-local-pager button{min-width:0!important;min-height:34px!important;padding:6px 8px!important;border:1px solid #dfe3ef!important;border-radius:8px!important;color:#465166!important;background:#fff!important;font-size:11px!important;font-weight:750!important}.goal-local-pager button:disabled{opacity:.38!important}.goal-local-pager__meta{text-align:center;color:#68748a;font-size:11px;font-weight:700}
.catalog-feature label[hidden]{display:none!important}
.catalog-interaction{position:relative!important}.catalog-wait-host{display:none!important;position:fixed!important;z-index:9999!important;inset:0!important;margin:0!important;padding:0!important;border:0!important;background:rgba(241,244,250,.74)!important;backdrop-filter:blur(3px)!important;pointer-events:auto!important}.catalog-interaction[data-catalog-busy="true"] .catalog-wait-host{display:flex!important;align-items:center!important;justify-content:center!important}.catalog-interaction[data-catalog-busy="true"] .catalog-controls,.catalog-interaction[data-catalog-busy="true"] .experiment-gallery{opacity:.48!important;filter:saturate(.65)!important}.catalog-wait-host>div{width:auto!important}.catalog-wait{display:flex!important;align-items:center!important;justify-content:center!important;gap:11px!important;width:max-content!important;max-width:calc(100vw - 36px)!important;padding:14px 18px!important;border:1px solid #c7d2fe!important;border-radius:13px!important;color:#293064!important;background:rgba(255,255,255,.98)!important;box-shadow:0 18px 44px rgba(35,43,92,.2)!important}.catalog-wait strong,.catalog-wait small{display:block!important}.catalog-wait strong{font-size:13px!important}.catalog-wait small{margin-top:3px;color:#68748a;font-size:11px!important}.catalog-wait__spinner{flex:none;width:21px;height:21px;border:2px solid #d9ddff;border-top-color:#5b5ce2;border-radius:50%;animation:run-spin .75s linear infinite}
.catalog-done{display:none!important}
.selection-done{display:none!important}#selected-task-detail{scroll-margin-top:18px!important}#selected-task-detail.task-detail-arrived{animation:task-arrived 1.15s ease-out}@keyframes task-arrived{0%{box-shadow:0 0 0 4px rgba(91,92,226,.2),0 16px 38px rgba(91,92,226,.16)}100%{box-shadow:none}}
.experiment-gallery .grid-wrap{display:block!important;width:100%!important;height:auto!important;min-height:0!important}.experiment-gallery .grid-container{display:grid!important;width:100%!important;grid-template-columns:repeat(auto-fill,minmax(230px,270px))!important;justify-content:start!important;gap:12px!important}.experiment-gallery .gallery-item{width:100%!important;min-width:0!important}
.experiment-gallery button,.experiment-gallery .thumbnail-item{height:auto!important;aspect-ratio:2/1!important}.experiment-gallery .caption-label{position:absolute!important;inset:0 0 auto 0!important;z-index:2!important;display:block!important;width:100%!important;min-height:72px!important;padding:14px 16px 18px!important;overflow:visible!important;text-overflow:clip!important;white-space:pre-line!important;overflow-wrap:anywhere!important;background:linear-gradient(180deg,rgba(5,9,30,.94),rgba(5,9,30,.76) 70%,transparent)!important;color:#fff!important;font-size:clamp(12px,1.25vw,17px)!important;font-weight:800!important;line-height:1.28!important;text-align:left!important;text-shadow:0 1px 2px rgba(0,0,0,.4)!important;pointer-events:none!important}
@media(max-width:760px){.experiment-gallery .grid-container{grid-template-columns:1fr!important}}
@media(max-width:900px){.catalog-filter-row{grid-template-columns:1fr!important}.catalog-filter-pane{padding:12px!important}}
"""


AUTO_SCROLL_JS = """
function initializeGymPlaygroundUi() {
  if (window.__gymPlaygroundUiReady) return;
  window.__gymPlaygroundUiReady = true;
  const selector = "#live-training-console .console-text";
  let active = null, follow = true, saved = 0, internal = false, scheduled = false;
  let goalPage = 0, goalSignature = "", goalPagerBound = null;
  let catalogBefore = "", catalogBusyTimer = null, catalogBusyStarted = 0;
  let selectionPending = false, selectionBefore = "";
  const catalogResult = () => {
    const goals = [...document.querySelectorAll(".catalog-feature label")].map(label => label.textContent.trim()).join("|");
    const cards = [...document.querySelectorAll(".experiment-gallery img")].map(img => img.src).join("|");
    const meta = document.querySelector(".catalog-meta")?.textContent.trim() || "";
    const done = document.querySelector(".catalog-done")?.textContent.trim() || "";
    return `${goals}::${cards}::${meta}::${done}`;
  };
  const setCatalogBusy = busy => {
    const catalog = document.querySelector(".catalog-interaction");
    if (!catalog) return;
    catalog.dataset.catalogBusy = String(busy);
    catalog.setAttribute("aria-busy", String(busy));
    catalogBusyStarted = busy ? performance.now() : 0;
    clearTimeout(catalogBusyTimer);
    if (busy) catalogBusyTimer = setTimeout(() => setCatalogBusy(false), 30000);
  };
  document.addEventListener("pointerdown", event => {
    const label = event.target.closest(".catalog-family label,.catalog-feature label");
    if (label) label.dataset.wasChecked = String(Boolean(label.querySelector("input:checked")));
  }, true);
  document.addEventListener("click", event => {
    const label = event.target.closest(".catalog-family label,.catalog-feature label");
    const pageButton = event.target.closest(".catalog-pager button");
    if (label?.dataset.wasChecked === "true" || pageButton?.disabled) return;
    if (label || pageButton) {
      catalogBefore = catalogResult();
      setTimeout(() => setCatalogBusy(true), 0);
    }
  }, true);
  document.addEventListener("change", event => {
    if (event.target.closest(".catalog-search")) setCatalogBusy(true);
  }, true);
  document.addEventListener("click", event => {
    if (!event.target.closest(".experiment-gallery button,.experiment-gallery .thumbnail-item")) return;
    selectionPending = true;
    selectionBefore = document.querySelector(".selection-done")?.textContent.trim() || "";
  }, true);
  const updateGoalPager = () => {
    const goal = document.querySelector(".catalog-feature");
    const pager = document.querySelector(".goal-local-pager");
    if (!goal || !pager) return;
    const labels = [...goal.querySelectorAll("label")];
    const selected = labels.findIndex(label => label.querySelector("input:checked"));
    const signature = labels.map(label => label.querySelector("input")?.value || label.textContent.trim()).join("|");
    const pageSize = Number(pager.dataset.pageSize || 6);
    const pages = Math.max(1, Math.ceil(labels.length / pageSize));
    if (signature !== goalSignature) {
      goalSignature = signature;
      goalPage = selected >= 0 ? Math.floor(selected / pageSize) : 0;
    }
    goalPage = Math.max(0, Math.min(goalPage, pages - 1));
    labels.forEach((label, index) => { label.hidden = Math.floor(index / pageSize) !== goalPage; });
    const previous = pager.querySelector('[data-goal-page="previous"]');
    const next = pager.querySelector('[data-goal-page="next"]');
    previous.textContent = pager.dataset.previous;
    next.textContent = pager.dataset.next;
    previous.disabled = goalPage === 0;
    next.disabled = goalPage >= pages - 1;
    const template = pager.dataset.template || "Page {page}/{pages} · {total} goals";
    pager.querySelector(".goal-local-pager__meta").textContent = template
      .replace("{page}", String(goalPage + 1)).replace("{pages}", String(pages)).replace("{total}", String(labels.length));
    pager.hidden = pages <= 1;
    if (goalPagerBound !== pager) {
      goalPagerBound = pager;
      pager.addEventListener("click", event => {
        const button = event.target.closest("button[data-goal-page]");
        if (!button || button.disabled) return;
        goalPage += button.dataset.goalPage === "next" ? 1 : -1;
        updateGoalPager();
      });
    }
  };
  const update = () => {
    scheduled = false;
    const catalog = document.querySelector(".catalog-interaction");
    if (catalog?.dataset.catalogBusy === "true" && performance.now() - catalogBusyStarted > 120 && catalogResult() !== catalogBefore) {
      const remaining = Math.max(0, 420 - (performance.now() - catalogBusyStarted));
      setTimeout(() => setCatalogBusy(false), remaining);
    }
    const selectionDone = document.querySelector(".selection-done")?.textContent.trim() || "";
    if (selectionPending && selectionDone && selectionDone !== selectionBefore) {
      selectionPending = false;
      const detail = document.querySelector("#selected-task-detail");
      if (detail) {
        detail.classList.remove("task-detail-arrived");
        requestAnimationFrame(() => {
          detail.classList.add("task-detail-arrived");
          detail.scrollIntoView({behavior:"smooth", block:"start"});
        });
      }
    }
    updateGoalPager();
    const element = document.querySelector(selector);
    document.querySelectorAll(".experiment-gallery .caption:not([data-feature-ready]),.experiment-gallery .label:not([data-feature-ready]),.experiment-gallery .caption-label:not([data-feature-ready])").forEach(caption => {
      const lines = caption.textContent.split(String.fromCharCode(10)).map(line => line.trim()).filter(Boolean);
      if (lines.length >= 3) {
        const card = caption.closest("button,.thumbnail-item");
        if (card) card.dataset.feature = lines.slice(2).join(" · ");
        caption.textContent = lines.slice(0, 2).join(String.fromCharCode(10));
      }
      caption.dataset.featureReady = "true";
    });
    if (element && element !== active) {
      active = element;
      active.addEventListener("scroll", () => {
        if (internal) return;
        follow = active.scrollHeight - active.clientHeight - active.scrollTop <= 24;
        saved = active.scrollTop;
      }, {passive:true});
    }
    if (active) {
      internal = true;
      if (follow) active.scrollTop = active.scrollHeight;
      else active.scrollTop = Math.min(saved, Math.max(0, active.scrollHeight - active.clientHeight));
    }
    const timer = document.querySelector(".run-wait__elapsed");
    if (timer) {
      const elapsed = Math.max(0, Math.floor((Date.now() - Number(timer.dataset.startMs)) / 1000));
      timer.textContent = `${elapsed}s ${timer.dataset.label}`;
    }
    if (active) requestAnimationFrame(() => { internal = false; });
  };
  const schedule = () => { if (!scheduled) { scheduled = true; requestAnimationFrame(update); } };
  new MutationObserver(schedule).observe(document.body, {childList:true, subtree:true, characterData:true});
  setInterval(schedule, 1000);
  schedule();
}
initializeGymPlaygroundUi();
"""


DEFAULT_LANGUAGE = "English"
DEFAULT_EXPERIMENT = CARTPOLE_PPO
copy = copy_for(DEFAULT_LANGUAGE)
cfg = EXPERIMENTS[DEFAULT_EXPERIMENT]
initial_step_spec, initial_epoch_spec = epoch_specs(DEFAULT_EXPERIMENT)
initial_feature_choices = feature_choices("", "Start here", DEFAULT_LANGUAGE)
initial_cards, initial_visible, initial_page, initial_meta, _, _ = catalog_page("", "Start here", ALL_FEATURES, 0, DEFAULT_LANGUAGE)

with gr.Blocks(title="Hands-On Modern RL · Gymnasium CPU Playground") as demo:
    with gr.Column(elem_classes="hero-stack"):
        hero = gr.HTML(hero_html(DEFAULT_LANGUAGE, DEFAULT_EXPERIMENT))
        with gr.Row(elem_classes="language-bar"):
            language = gr.Radio(choices=[("English", "English"), ("中文", "中文")], value=DEFAULT_LANGUAGE, show_label=False, elem_classes="language-switch")

    with gr.Column(elem_classes="catalog-card"):
        catalog_header = gr.HTML(catalog_header_html(DEFAULT_LANGUAGE))
        gr.HTML(card_preload_html())
        with gr.Column(elem_classes="catalog-interaction"):
            catalog_wait = gr.HTML(value=catalog_wait_html(DEFAULT_LANGUAGE), elem_classes="catalog-wait-host")
            with gr.Column(elem_classes="catalog-controls"):
                search = gr.Textbox(label=copy["search"], placeholder=copy["search_placeholder"], elem_classes="catalog-search")
                with gr.Row(elem_classes="catalog-filter-row"):
                    with gr.Column(elem_classes="catalog-filter-pane catalog-filter-pane--path"):
                        family = gr.Radio(choices=path_choices(DEFAULT_LANGUAGE), value="Start here", label=copy["path"], elem_classes="catalog-family")
                    with gr.Column(elem_classes="catalog-filter-pane catalog-filter-pane--goal"):
                        feature = gr.Radio(choices=initial_feature_choices, value=ALL_FEATURES, label=copy["goal"], visible=True, elem_classes="catalog-feature")
                        goal_pager = gr.HTML(goal_pager_html(DEFAULT_LANGUAGE), elem_classes="goal-pager-host")
            gallery = gr.Gallery(value=initial_cards, label=None, show_label=False, columns=4, object_fit="cover", height="auto", allow_preview=False, buttons=[], elem_classes="experiment-gallery")
            visible_experiments = gr.State(initial_visible)
            catalog_page_state = gr.State(initial_page)
            with gr.Row(elem_classes="catalog-pager"):
                catalog_meta = gr.Markdown(initial_meta, elem_classes="catalog-meta")
                previous_page = gr.Button(copy["previous"], size="sm", visible=False)
                next_page = gr.Button(copy["next"], size="sm", visible=False)
            catalog_done = gr.HTML(value="0", elem_classes="catalog-done")

    task_info = gr.HTML(task_brief(DEFAULT_EXPERIMENT, DEFAULT_LANGUAGE), elem_id="selected-task-detail")
    selection_done = gr.HTML(value="0", elem_classes="selection-done")

    with gr.Row(elem_classes="training-layout"):
        with gr.Column(scale=1, min_width=310, elem_classes="control-card"):
            settings_header = gr.HTML(panel_html(copy["settings"], copy["settings_copy"]))
            experiment = gr.Textbox(value=DEFAULT_EXPERIMENT, label="Selected experiment", interactive=False, elem_classes="selected-experiment")
            steps_per_epoch = gr.Slider(minimum=initial_step_spec[0], maximum=initial_step_spec[1], value=initial_step_spec[2], step=initial_step_spec[3], label=copy["steps_per_epoch"], info=copy["steps_per_epoch_info"])
            epochs = gr.Slider(minimum=initial_epoch_spec[0], maximum=initial_epoch_spec[1], value=initial_epoch_spec[2], step=initial_epoch_spec[3], label=copy["epochs"], info=copy["epochs_info"])
            with gr.Accordion(copy["advanced"], open=False, elem_classes="advanced-settings") as advanced:
                alpha = gr.Slider(minimum=.00001, maximum=1, value=cfg["alpha"][2], step=.00001, label=copy["alpha"])
                gamma = gr.Slider(minimum=0, maximum=1, value=0, step=.05, label=copy["gamma"], visible=False)
                epsilon = gr.Slider(minimum=0, maximum=1, value=.1, step=.01, label=copy["epsilon"])
                seed = gr.Number(value=42, precision=0, label=copy["seed"])
            start = gr.Button(copy["start"], variant="primary", elem_classes="primary-btn")
        with gr.Column(scale=2, elem_classes="results-stack"):
            with gr.Column(elem_classes="chart-card"):
                chart_header = gr.HTML(panel_html(copy["curve"], copy["curve_copy"]))
                with gr.Row(elem_classes="result-summary"):
                    status = gr.HTML(status_card("idle", copy["ready"], copy["ready_detail"], DEFAULT_LANGUAGE))
                    metric = gr.HTML(metric_card("—", copy["metric_waiting"], DEFAULT_LANGUAGE))
                wait_state = gr.HTML(value="", visible=False)
                curve = gr.Plot(show_label=False)
                console = gr.HTML(console_panel(copy["log_waiting"], DEFAULT_LANGUAGE), elem_id="live-training-console")

            with gr.Column(elem_classes="preview-card"):
                preview_header = gr.HTML(panel_html(copy["preview"], copy["preview_copy"], "artifact-note"))
                trained_model = model_selector(DEFAULT_EXPERIMENT, DEFAULT_LANGUAGE)
                preview_status = gr.HTML(preview_provenance(DEFAULT_EXPERIMENT, trained_model.value, DEFAULT_LANGUAGE))
                preview = gr.Image(value=example_preview(DEFAULT_EXPERIMENT), show_label=False, interactive=False, elem_classes="policy-preview")
                artifact = gr.File(
                    label=copy["artifact"], interactive=False, visible=False,
                    height=76, elem_classes="artifact-download",
                )

    gr.HTML(footer_html())

    catalog_outputs = [feature, gallery, visible_experiments, catalog_page_state, catalog_meta, previous_page, next_page, catalog_done]
    page_outputs = [gallery, visible_experiments, catalog_page_state, catalog_meta, previous_page, next_page, catalog_done]
    search.change(reset_search, inputs=[search, family, language], outputs=catalog_outputs, queue=False, show_progress="hidden", trigger_mode="always_last")
    family.change(reset_family, inputs=[search, family, language], outputs=catalog_outputs, queue=False, show_progress="hidden", trigger_mode="always_last")
    feature.input(reset_catalog, inputs=[search, family, feature, language], outputs=page_outputs, queue=False, show_progress="hidden", trigger_mode="always_last")
    previous_page.click(lambda q, f, t, p, lang: move_catalog(q, f, t, p, lang, -1), inputs=[search, family, feature, catalog_page_state, language], outputs=page_outputs, queue=False, show_progress="hidden")
    next_page.click(lambda q, f, t, p, lang: move_catalog(q, f, t, p, lang, 1), inputs=[search, family, feature, catalog_page_state, language], outputs=page_outputs, queue=False, show_progress="hidden")
    gallery.select(choose_card, inputs=[visible_experiments, language], outputs=[experiment, hero, task_info, steps_per_epoch, epochs, alpha, gamma, epsilon, status, metric, console, preview, artifact, trained_model, preview_status, selection_done], queue=False, show_progress="hidden")
    language.change(switch_language, inputs=[language, experiment, seed, family, search, feature, catalog_page_state], outputs=[hero, catalog_header, family, search, feature, goal_pager, catalog_wait, gallery, visible_experiments, catalog_page_state, catalog_meta, previous_page, next_page, settings_header, advanced, task_info, steps_per_epoch, epochs, alpha, gamma, epsilon, seed, start, status, metric, chart_header, console, preview_header, artifact, trained_model, preview_status], queue=False)
    start.click(train_with_ui, inputs=[experiment, steps_per_epoch, epochs, alpha, gamma, epsilon, seed, language], outputs=[status, metric, curve, preview, artifact, console, trained_model, preview_status, wait_state, start], concurrency_limit=1)
    trained_model.change(select_saved_model, inputs=[experiment, trained_model, language], outputs=[preview, preview_status], queue=False, show_progress="hidden")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(css=CSS, js=AUTO_SCROLL_JS, footer_links=[])
