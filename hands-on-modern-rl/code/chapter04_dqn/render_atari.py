"""Render an Atari DQN model as a GIF.

The training script uses SB3's Atari wrapper stack plus 4-frame stacking. This
renderer mirrors that observation pipeline for action selection, while saving
the raw RGB game frames so the generated GIF is easy to read in the chapter.

Examples:
    python code/chapter04_dqn/render_atari.py \
      --model output/dqn_atari_runs/ALE_Pong-v5_dqn_seed0/final_model.zip \
      --output docs/chapter04_dqn/images/dqn-atari-pong-smoke.gif

    python code/chapter04_dqn/render_atari.py \
      --policy random \
      --output docs/chapter04_dqn/images/dqn-atari-pong-random.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ale_py
import gymnasium as gym
import imageio
import numpy as np
from PIL import Image
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper


gym.register_envs(ale_py)


def make_render_env(args: argparse.Namespace):
    env_kwargs = {}
    if args.env_id.startswith("ALE/"):
        env_kwargs = {
            "frameskip": 1,
            "repeat_action_probability": 0.0,
        }

    env = gym.make(args.env_id, render_mode="rgb_array", **env_kwargs)
    env = AtariWrapper(
        env,
        noop_max=args.noop_max,
        frame_skip=args.frame_skip,
        screen_size=args.screen_size,
        terminal_on_life_loss=not args.no_terminal_on_life_loss,
        clip_reward=not args.no_clip_reward,
        action_repeat_probability=args.repeat_action_probability,
    )
    return gym.wrappers.FrameStackObservation(env, stack_size=args.frame_stack)


def model_observation(obs: np.ndarray) -> np.ndarray:
    """Convert Gymnasium frame stack output to the SB3 CnnPolicy shape."""
    arr = np.asarray(obs)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return arr


def scaled_frame(frame: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return frame
    image = Image.fromarray(frame)
    width, height = image.size
    return np.asarray(
        image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    )


def render(args: argparse.Namespace) -> None:
    if args.policy == "model":
        if args.model is None:
            raise SystemExit("--model is required when --policy model is used.")
        model = DQN.load(args.model)
    else:
        model = None

    env = make_render_env(args)

    obs, _ = env.reset(seed=args.seed)
    frames = []
    total_reward = 0.0

    for step in range(args.max_steps):
        if step % args.render_every == 0:
            frames.append(scaled_frame(env.render(), args.scale))

        if args.policy == "model":
            action, _ = model.predict(model_observation(obs), deterministic=True)
        elif args.policy == "random":
            action = env.action_space.sample()
        else:
            action = 0

        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += float(reward)

        if terminated or truncated:
            break

    env.close()

    if not frames:
        raise SystemExit("No frames rendered.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, duration=1000 / args.fps, loop=0)
    print(
        f"Saved {output} | policy={args.policy}, reward={total_reward:.1f}, "
        f"env_steps={step + 1}, gif_frames={len(frames)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an Atari DQN model.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--policy", choices=["model", "random", "noop"], default="model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--render-every", type=int, default=4)
    parser.add_argument("--scale", type=int, default=2)

    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--screen-size", type=int, default=84)
    parser.add_argument("--repeat-action-probability", type=float, default=0.0)
    parser.add_argument("--no-terminal-on-life-loss", action="store_true")
    parser.add_argument("--no-clip-reward", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
