"""4 x 4 GridWorld: a reproducible value-iteration and Q-Learning experiment.

The script uses only the Python standard library. Running it produces:

- ``results.json``: environment, value table, optimal policy, and multi-seed statistics;
- ``learning_curves.csv``: cross-seed mean and standard deviation of per-episode reward;
- ``gridworld-environment.svg``: the experiment environment, actions, and reward conventions;
- ``gridworld-value-iteration.svg``: value propagation and the optimal policy;
- ``gridworld-q-learning.svg``: Q-Learning curves and the exploration-rate comparison.

How to run:

    python3 gridworld_q_learning.py --output-dir output/value-experiment

Textbook figures can be exported to an additional directory with ``--assets-dir``.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Callable


State = tuple[int, int]

GRID_SIZE = 4
START: State = (0, 0)
TRAP: State = (1, 1)
GOAL: State = (3, 3)
TERMINALS = {TRAP, GOAL}

# Action indices and display order are always kept consistent.
ACTIONS: tuple[tuple[str, State], ...] = (
    ("up", (-1, 0)),
    ("down", (1, 0)),
    ("left", (0, -1)),
    ("right", (0, 1)),
)
ARROWS = ("↑", "↓", "←", "→")

STEP_REWARD = -0.01
TRAP_REWARD = -1.0
GOAL_REWARD = 1.0
GAMMA = 0.95
ALPHA = 0.15
EPISODES = 500
N_SEEDS = 30
MAX_STEPS = 100


def all_states() -> list[State]:
    return [(row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE)]


def transition(state: State, action: int) -> tuple[State, float, bool]:
    """Take one deterministic transition step.

    The reward is attached to the transition *into* the next state. Entering the goal or the
    trap ends the episode, so terminal states have no successor value; this avoids counting
    the terminal reward twice.
    """

    if state in TERMINALS:
        return state, 0.0, True

    dr, dc = ACTIONS[action][1]
    next_state = (state[0] + dr, state[1] + dc)
    if not (0 <= next_state[0] < GRID_SIZE and 0 <= next_state[1] < GRID_SIZE):
        next_state = state

    if next_state == GOAL:
        return next_state, GOAL_REWARD, True
    if next_state == TRAP:
        return next_state, TRAP_REWARD, True
    return next_state, STEP_REWARD, False


def action_value(values: dict[State, float], state: State, action: int) -> float:
    next_state, reward, done = transition(state, action)
    return reward if done else reward + GAMMA * values[next_state]


def value_iteration(tolerance: float = 1e-12) -> tuple[dict[State, float], list[dict[State, float]]]:
    """Synchronous value iteration; history[0] is the all-zero initialization."""

    values = {state: 0.0 for state in all_states()}
    history = [values.copy()]

    for _ in range(1_000):
        updated = values.copy()
        for state in all_states():
            if state not in TERMINALS:
                updated[state] = max(
                    action_value(values, state, action)
                    for action in range(len(ACTIONS))
                )
        delta = max(abs(updated[state] - values[state]) for state in all_states())
        values = updated
        history.append(values.copy())
        if delta < tolerance:
            break
    else:
        raise RuntimeError("Value iteration did not converge within 1000 sweeps")

    return values, history


def optimal_actions(values: dict[State, float], state: State) -> list[int]:
    if state in TERMINALS:
        return []
    candidates = [action_value(values, state, action) for action in range(len(ACTIONS))]
    best = max(candidates)
    return [index for index, value in enumerate(candidates) if abs(value - best) < 1e-10]


def linear_epsilon(episode: int, start: float = 1.0, end: float = 0.05) -> float:
    progress = min(episode / (EPISODES - 1), 1.0)
    return start + progress * (end - start)


def constant_epsilon(value: float) -> Callable[[int], float]:
    return lambda _episode: value


@dataclass
class TrainingRun:
    rewards: list[float]
    steps: list[int]
    q_values: list[list[list[float]]]


def zero_q_table() -> list[list[list[float]]]:
    return [
        [[0.0 for _ in ACTIONS] for _ in range(GRID_SIZE)]
        for _ in range(GRID_SIZE)
    ]


def best_action_indices(q_values: list[list[list[float]]], state: State) -> list[int]:
    row = q_values[state[0]][state[1]]
    best = max(row)
    return [index for index, value in enumerate(row) if abs(value - best) < 1e-12]


def train_q_learning(seed: int, epsilon_schedule: Callable[[int], float]) -> TrainingRun:
    rng = random.Random(seed)
    q_values = zero_q_table()
    rewards: list[float] = []
    steps_per_episode: list[int] = []

    for episode in range(EPISODES):
        state = START
        episode_reward = 0.0
        epsilon = epsilon_schedule(episode)

        for step in range(1, MAX_STEPS + 1):
            if rng.random() < epsilon:
                action = rng.randrange(len(ACTIONS))
            else:
                action = rng.choice(best_action_indices(q_values, state))

            next_state, reward, done = transition(state, action)
            next_best = 0.0 if done else max(q_values[next_state[0]][next_state[1]])
            old_value = q_values[state[0]][state[1]][action]
            td_target = reward + GAMMA * next_best
            q_values[state[0]][state[1]][action] += ALPHA * (td_target - old_value)

            episode_reward += reward
            state = next_state
            if done:
                break

        rewards.append(episode_reward)
        steps_per_episode.append(step)

    return TrainingRun(rewards, steps_per_episode, q_values)


def greedy_evaluation(q_values: list[list[list[float]]]) -> dict[str, float]:
    """One greedy evaluation in the deterministic environment; ties broken stably by action index."""

    state = START
    total_reward = 0.0
    path = [state]
    for step in range(1, MAX_STEPS + 1):
        action = best_action_indices(q_values, state)[0]
        next_state, reward, done = transition(state, action)
        total_reward += reward
        state = next_state
        path.append(state)
        if done:
            return {
                "success": float(state == GOAL),
                "steps": float(step),
                "reward": total_reward,
                "path": path,
            }
    return {"success": 0.0, "steps": float(MAX_STEPS), "reward": total_reward, "path": path}


def aggregate_runs(runs: list[TrainingRun]) -> dict[str, list[float]]:
    reward_mean = [fmean(run.rewards[i] for run in runs) for i in range(EPISODES)]
    reward_std = [pstdev(run.rewards[i] for run in runs) for i in range(EPISODES)]
    step_mean = [fmean(run.steps[i] for run in runs) for i in range(EPISODES)]
    return {"reward_mean": reward_mean, "reward_std": reward_std, "step_mean": step_mean}


def moving_average(values: list[float], window: int = 20) -> list[float]:
    result = []
    for index in range(len(values)):
        left = max(0, index - window + 1)
        result.append(fmean(values[left:index + 1]))
    return result


def run_schedule(name: str, schedule: Callable[[int], float]) -> dict[str, object]:
    runs = [train_q_learning(seed, schedule) for seed in range(N_SEEDS)]
    aggregate = aggregate_runs(runs)
    evaluations = [greedy_evaluation(run.q_values) for run in runs]
    return {
        "name": name,
        "runs": runs,
        "aggregate": aggregate,
        "last_100_reward": fmean(
            reward for run in runs for reward in run.rewards[-100:]
        ),
        "success_rate": fmean(result["success"] for result in evaluations),
        "evaluation_steps": fmean(result["steps"] for result in evaluations),
        "evaluation_reward": fmean(result["reward"] for result in evaluations),
    }


def svg_text(x: float, y: float, text: str, size: int = 16, **attrs: object) -> str:
    attributes = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<text x="{x}" y="{y}" font-size="{size}" {attributes}>{html.escape(text)}</text>'


def value_color(value: float) -> str:
    normalized = max(0.0, min(1.0, (value + 0.05) / 1.05))
    red = round(244 - 105 * normalized)
    green = round(247 - 54 * normalized)
    blue = round(250 - 2 * normalized)
    return f"rgb({red},{green},{blue})"


def draw_grid(
    values: dict[State, float], x: int, y: int, title: str,
    policy: dict[State, list[int]] | None = None,
) -> str:
    cell = 58
    parts = [svg_text(x + 2 * cell, y - 18, title, 18, text_anchor="middle", font_weight="600")]
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            state = (row, col)
            px, py = x + col * cell, y + row * cell
            fill = "#fee2e2" if state == TRAP else "#dcfce7" if state == GOAL else value_color(values[state])
            parts.append(
                f'<rect x="{px}" y="{py}" width="{cell}" height="{cell}" '
                f'fill="{fill}" stroke="#64748b" stroke-width="1"/>'
            )
            if state == START:
                parts.append(svg_text(px + 7, py + 16, "S", 12, fill="#475569", font_weight="700"))
            if state == TRAP:
                label = "X"
            elif state == GOAL:
                label = "G"
            elif policy is not None:
                label = "".join(ARROWS[action] for action in policy[state])
            else:
                label = f"{values[state]:.3f}"
            parts.append(svg_text(px + cell / 2, py + 36, label, 16, text_anchor="middle", fill="#0f172a", font_weight="600"))
    return "".join(parts)


def render_environment_svg(path: Path) -> None:
    """Draw the plain environment figure used to open the chapter."""

    width, height = 1120, 430
    grid_x, grid_y, cell = 330, 70, 72
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs>',
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="120%"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.10"/></filter>',
        '</defs>',
        '<rect width="100%" height="100%" rx="24" fill="#f8fafc"/>',
        '<g font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Songti SC,Noto Sans CJK SC,sans-serif">',
        f'<rect x="{grid_x - 18}" y="{grid_y - 18}" width="{4 * cell + 36}" height="{4 * cell + 36}" rx="18" fill="#ffffff" filter="url(#shadow)"/>',
    ]

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            state = (row, col)
            x, y = grid_x + col * cell, grid_y + row * cell
            if state == START:
                fill, stroke = "#dbeafe", "#2563eb"
            elif state == TRAP:
                fill, stroke = "#fee2e2", "#dc2626"
            elif state == GOAL:
                fill, stroke = "#dcfce7", "#16a34a"
            else:
                fill, stroke = "#ffffff", "#94a3b8"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="{stroke}" stroke-width="{2 if state in {START, TRAP, GOAL} else 1}"/>')
            parts.append(svg_text(x + 9, y + 17, f"({row},{col})", 11, fill="#64748b"))
            if state == START:
                parts.append(svg_text(x + cell / 2, y + 45, "S", 24, text_anchor="middle", font_weight="800", fill="#1d4ed8"))
            elif state == TRAP:
                parts.append(svg_text(x + cell / 2, y + 45, "X", 24, text_anchor="middle", font_weight="800", fill="#b91c1c"))
            elif state == GOAL:
                parts.append(svg_text(x + cell / 2, y + 45, "G", 24, text_anchor="middle", font_weight="800", fill="#15803d"))

    # The figure defines only the state and action spaces. Rewards and discounting are covered in the prose.
    parts.append(svg_text(grid_x + 4 * cell + 92, grid_y + 152, "↑", 34, text_anchor="middle", font_weight="600", fill="#334155"))
    parts.append(svg_text(grid_x + 4 * cell + 92, grid_y + 230, "↓", 34, text_anchor="middle", font_weight="600", fill="#334155"))
    parts.append(svg_text(grid_x + 4 * cell + 52, grid_y + 191, "←", 34, text_anchor="middle", font_weight="600", fill="#334155"))
    parts.append(svg_text(grid_x + 4 * cell + 132, grid_y + 191, "→", 34, text_anchor="middle", font_weight="600", fill="#334155"))

    parts.append('</g></svg>')
    path.write_text("".join(parts), encoding="utf-8")


def render_value_svg(history: list[dict[State, float]], values: dict[State, float], path: Path) -> None:
    snapshots = [0, 1, 3, min(6, len(history) - 1)]
    width, height = 1120, 610
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Songti SC,Noto Sans CJK SC,sans-serif">',
        svg_text(560, 38, f"γ = {GAMMA}", 18, text_anchor="middle", font_weight="600", fill="#475569"),
    ]
    for index, sweep in enumerate(snapshots):
        parts.append(draw_grid(history[sweep], 40 + index * 270, 95, f"V_{sweep}"))

    policy = {state: optimal_actions(values, state) for state in all_states() if state not in TERMINALS}
    parts.append(draw_grid(values, 175, 390, "V*"))
    parts.append(draw_grid(values, 710, 390, "π*", policy))
    parts.append('</g></svg>')
    path.write_text("".join(parts), encoding="utf-8")


def line_points(values: list[float], x: float, y: float, width: float, height: float, y_min: float, y_max: float) -> str:
    points = []
    for index, value in enumerate(values):
        px = x + index / (len(values) - 1) * width
        py = y + (y_max - value) / (y_max - y_min) * height
        points.append(f"{px:.1f},{py:.1f}")
    return " ".join(points)


def render_learning_svg(schedule_results: list[dict[str, object]], path: Path) -> None:
    width, height = 1120, 520
    chart_x, chart_y, chart_w, chart_h = 85, 55, 930, 380
    y_min, y_max = -1.25, 1.05
    colors = ["#2563eb", "#d97706", "#7c3aed"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Songti SC,Noto Sans CJK SC,sans-serif">',
    ]

    for tick in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        py = chart_y + (y_max - tick) / (y_max - y_min) * chart_h
        parts.append(f'<line x1="{chart_x}" y1="{py}" x2="{chart_x + chart_w}" y2="{py}" stroke="#e2e8f0"/>')
        parts.append(svg_text(chart_x - 12, py + 5, f"{tick:.1f}", 13, text_anchor="end", fill="#64748b"))
    for tick in [0, 100, 200, 300, 400, 500]:
        px = chart_x + tick / EPISODES * chart_w
        parts.append(f'<line x1="{px}" y1="{chart_y + chart_h}" x2="{px}" y2="{chart_y + chart_h + 6}" stroke="#64748b"/>')
        parts.append(svg_text(px, chart_y + chart_h + 25, str(tick), 13, text_anchor="middle", fill="#64748b"))
    parts.append(f'<line x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{chart_y + chart_h}" stroke="#334155"/>')
    parts.append(f'<line x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" stroke="#334155"/>')
    parts.append(svg_text(chart_x + chart_w / 2, chart_y + chart_h + 52, "k", 18, text_anchor="middle", font_style="italic", fill="#334155"))
    parts.append(svg_text(28, chart_y + chart_h / 2, "R̄", 19, text_anchor="middle", font_style="italic", fill="#334155", transform=f"rotate(-90 28 {chart_y + chart_h / 2})"))

    for result, color in zip(schedule_results, colors):
        aggregate = result["aggregate"]
        smooth = moving_average(aggregate["reward_mean"], 20)
        parts.append(
            f'<polyline points="{line_points(smooth, chart_x, chart_y, chart_w, chart_h, y_min, y_max)}" '
            f'fill="none" stroke="{color}" stroke-width="3"/>'
        )

    card_x = 710
    legend_labels = ("ε: 1.00 → 0.05", "ε = 0.05", "ε = 0.30")
    for index, (result, color) in enumerate(zip(schedule_results, colors)):
        y = 78 + index * 34
        parts.append(f'<line x1="{card_x}" y1="{y}" x2="{card_x + 32}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        parts.append(svg_text(card_x + 43, y + 5, legend_labels[index], 15, font_weight="600", fill="#0f172a"))
    parts.append('</g></svg>')
    path.write_text("".join(parts), encoding="utf-8")


def serializable_values(values: dict[State, float]) -> list[list[float]]:
    return [[values[(row, col)] for col in range(GRID_SIZE)] for row in range(GRID_SIZE)]


def copy_assets(source_dir: Path, assets_dir: Path | None) -> None:
    if assets_dir is None:
        return
    assets_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "gridworld-environment.svg",
        "gridworld-value-iteration.svg",
        "gridworld-q-learning.svg",
    ):
        (assets_dir / filename).write_bytes((source_dir / filename).read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/value-experiment"))
    parser.add_argument("--assets-dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    values, history = value_iteration()
    assert abs(values[START] - 0.728537125) < 1e-9
    assert len(history) - 1 == 7
    schedules = [
        ("ε: 1.00 → 0.05", linear_epsilon),
        ("固定 ε = 0.05", constant_epsilon(0.05)),
        ("固定 ε = 0.30", constant_epsilon(0.30)),
    ]
    schedule_results = [run_schedule(name, schedule) for name, schedule in schedules]
    assert all(result["success_rate"] == 1.0 for result in schedule_results)
    assert all(result["evaluation_steps"] == 6.0 for result in schedule_results)

    render_environment_svg(args.output_dir / "gridworld-environment.svg")
    render_value_svg(history, values, args.output_dir / "gridworld-value-iteration.svg")
    render_learning_svg(schedule_results, args.output_dir / "gridworld-q-learning.svg")

    baseline = schedule_results[0]
    with (args.output_dir / "learning_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "reward_mean", "reward_std", "steps_mean"])
        aggregate = baseline["aggregate"]
        for episode in range(EPISODES):
            writer.writerow([
                episode + 1,
                f"{aggregate['reward_mean'][episode]:.6f}",
                f"{aggregate['reward_std'][episode]:.6f}",
                f"{aggregate['step_mean'][episode]:.6f}",
            ])

    results = {
        "environment": {
            "grid_size": GRID_SIZE,
            "start": START,
            "trap": TRAP,
            "goal": GOAL,
            "step_reward": STEP_REWARD,
            "trap_reward": TRAP_REWARD,
            "goal_reward": GOAL_REWARD,
            "gamma": GAMMA,
        },
        "value_iteration": {
            "converged_sweeps": len(history) - 1,
            "values": serializable_values(values),
            "snapshots": {
                str(sweep): serializable_values(history[sweep])
                for sweep in [0, 1, 3, min(6, len(history) - 1)]
            },
            "optimal_policy": {
                f"{state[0]},{state[1]}": [ARROWS[action] for action in optimal_actions(values, state)]
                for state in all_states() if state not in TERMINALS
            },
        },
        "q_learning": {
            "episodes": EPISODES,
            "seeds": N_SEEDS,
            "alpha": ALPHA,
            "schedules": [
                {
                    "name": result["name"],
                    "last_100_reward": result["last_100_reward"],
                    "success_rate": result["success_rate"],
                    "evaluation_steps": result["evaluation_steps"],
                    "evaluation_reward": result["evaluation_reward"],
                }
                for result in schedule_results
            ],
        },
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    copy_assets(args.output_dir, args.assets_dir)

    print(f"Value iteration converged after {len(history) - 1} sweeps")
    print(f"V*(0,0) = {values[START]:.6f}")
    for result in schedule_results:
        print(
            f"{result['name']}: last-100-episode reward={result['last_100_reward']:.3f}, "
            f"greedy success rate={result['success_rate'] * 100:.0f}%, "
            f"greedy steps={result['evaluation_steps']:.1f}"
        )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
