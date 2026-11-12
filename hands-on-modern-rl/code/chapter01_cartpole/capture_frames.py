"""Capture real rendered frames from Gymnasium CartPole-v1 using a trained policy."""

import argparse
import os

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from importlib.util import module_from_spec, spec_from_file_location


def load_actor_critic(script_path):
    spec = spec_from_file_location("pytorch_ppo", script_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ActorCritic


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Capture CartPole evaluation episode frames")
    parser.add_argument(
        "--model",
        default=os.path.join(here, "output", "pytorch_ppo_cartpole.pth"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(here, "output", "cartpole_frames_seed42.png"),
    )
    parser.add_argument("--seed", type=int, default=10042)
    return parser.parse_args()


def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    actor_critic = load_actor_critic(os.path.join(here, "2-pytorch_ppo.py"))
    model = actor_critic()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)
    frames = []
    score = 0
    terminated = truncated = False

    while not (terminated or truncated):
        if score in {0, 125, 250, 375}:
            frames.append((score, env.render(), np.array(obs, copy=True)))
        with torch.no_grad():
            action, _, _ = model.get_action(
                torch.as_tensor(obs, dtype=torch.float32),
                deterministic=True,
            )
        obs, reward, terminated, truncated, _ = env.step(action.item())
        score += int(reward)

    frames.append((score, env.render(), np.array(obs, copy=True)))
    env.close()

    fig, axes = plt.subplots(1, len(frames), figsize=(15, 3.3))
    for ax, (step, frame, state) in zip(axes, frames):
        ax.imshow(frame)
        ax.set_title(f"step {step}\nangle={np.degrees(state[2]):+.2f}°")
        ax.axis("off")
    fig.suptitle(
        f"Deterministic evaluation in Gymnasium CartPole-v1 — score {score}",
        fontsize=14,
    )
    fig.tight_layout()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    print(f"Saved: {args.output}")
    print(f"Evaluation score: {score}")


if __name__ == "__main__":
    main()
