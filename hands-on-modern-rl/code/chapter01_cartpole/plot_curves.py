"""Generate reviewable CartPole training curves from the CSV exported by the training script."""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Plot measured CartPole PPO curves")
    parser.add_argument(
        "--input",
        default=os.path.join(here, "output", "training_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(here, "output"),
    )
    return parser.parse_args()


def load_metrics(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No data to plot: {path}")
    return {
        key: np.array([float(row[key]) for row in rows])
        for key in rows[0]
    }


def save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    args = parse_args()
    metrics = load_metrics(args.input)
    steps = metrics["total_timesteps"]
    seed = int(metrics["seed"][0])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(11, 5.5))
    reward = metrics["mean_episode_reward"]
    ax.plot(steps, reward, "-o", color="#0F766E", linewidth=2.3, markersize=4)
    ax.axhline(
        500,
        color="#64748B",
        linestyle="--",
        linewidth=1,
        label="Episode limit (500)",
    )
    ax.set(
        xlabel="Environment steps",
        ylabel="Mean reward of completed episodes",
        title=f"CartPole-v1 PyTorch PPO — measured run (seed={seed})",
    )
    ax.set_ylim(0, 525)
    ax.legend()
    save(fig, args.output_dir, "cartpole_reward_seed42.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [
        ("value_loss", "Value loss", "#C2410C", 1.0),
        ("entropy", "Policy entropy", "#2563EB", 1.0),
        ("approx_kl", "Approximate KL", "#7C3AED", 1.0),
        ("clip_fraction", "Clip fraction", "#0F766E", 100.0),
    ]
    for ax, (key, title, color, scale) in zip(axes.flat, panels):
        ax.plot(
            steps,
            metrics[key] * scale,
            "-o",
            color=color,
            linewidth=1.8,
            markersize=3,
        )
        ax.set_title(title)
        ax.set_ylabel("percent" if key == "clip_fraction" else "value")
    for ax in axes[-1]:
        ax.set_xlabel("Environment steps")
    fig.suptitle(
        f"Training diagnostics from the same measured run (seed={seed})",
        fontsize=15,
    )
    fig.tight_layout()
    save(fig, args.output_dir, "cartpole_diagnostics_seed42.png")


if __name__ == "__main__":
    main()
