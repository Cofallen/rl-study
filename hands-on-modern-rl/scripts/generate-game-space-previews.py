#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPACES = ROOT / "modelscope-space"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def save(frame, path: Path):
    array = np.asarray(frame)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path, optimize=True)
    print(path.relative_to(ROOT), array.shape)


def load_runtime(dirname: str, module_name: str):
    directory = SPACES / dirname
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(module_name, directory / "space_runtime.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def vizdoom_preview():
    runtime = load_runtime("hands-on-modern-rl-experiment02-vizdoom", "preview_vizdoom")
    task = next(item for item in runtime.TASKS if item["key"] == "deathmatch")
    env = runtime._make_env(task["environment"], 42)
    env.reset(seed=42)
    save(env.render(), runtime.ROOT / task["preview"])
    env.close()


def atari_preview():
    runtime = load_runtime("hands-on-modern-rl-experiment03-atari", "preview_atari")
    task = next(item for item in runtime.TASKS if item["key"] == "freeway")
    env = runtime._make_vec_env(task["environment"], 42)
    env.reset()
    save(env.render(mode="rgb_array"), runtime.ROOT / task["preview"])
    env.close()


def board_previews():
    import pyspiel

    runtime = load_runtime("hands-on-modern-rl-experiment04-board-selfplay", "preview_board")
    for task in runtime.TASKS:
        game = pyspiel.load_game(task["environment"])
        state = game.new_initial_state()
        save(runtime._frame(task["title"]["en"], str(state), "Initial game state"), runtime.ROOT / task["preview"])


def multiagent_previews():
    runtime = load_runtime("hands-on-modern-rl-experiment05-multiagent-games", "preview_multiagent")
    for task in runtime.TASKS:
        env = runtime._raw_env(task["key"], max_cycles=20)
        env.reset(seed=42)
        frame = env.render()
        if frame is None:
            raise RuntimeError(f"{task['key']} returned no RGB preview")
        save(frame, runtime.ROOT / task["preview"])
        env.close()


def minigrid_previews():
    runtime = load_runtime("hands-on-modern-rl-experiment06-minigrid-adventure", "preview_minigrid")
    for task in runtime.TASKS:
        env = runtime._make_env(task["environment"], 42)
        env.reset(seed=42)
        save(env.render(), runtime.ROOT / task["preview"])
        env.close()


def jax_previews():
    import gymnax
    import jax

    runtime = load_runtime("hands-on-modern-rl-experiment07-jax-games", "preview_jax")
    for index, task in enumerate(runtime.TASKS):
        env, params = gymnax.make(task["environment"])
        observation, _ = env.reset(jax.random.PRNGKey(42 + index), params)
        save(runtime._semantic_frame(observation, task["colors"]), runtime.ROOT / task["preview"])


if __name__ == "__main__":
    vizdoom_preview()
    atari_preview()
    board_previews()
    multiagent_previews()
    minigrid_previews()
    jax_previews()
