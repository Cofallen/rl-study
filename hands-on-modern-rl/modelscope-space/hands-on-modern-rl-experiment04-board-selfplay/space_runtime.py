from __future__ import annotations

import math
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent

SPACE = {
    "title": {"en": "Board Games & Self-Play Lab", "zh": "棋盘游戏与自博弈训练场"},
    "description": {
        "en": "Train tabular self-play and counterfactual-regret agents in real OpenSpiel games, then replay the policy's decisions move by move.",
        "zh": "在真实 OpenSpiel 游戏中训练表格自博弈和反事实遗憾最小化策略，并逐步回放策略决策。",
    },
    "badge": "EXPERIMENT 04 · SELF-PLAY",
    "training_guide": {
        "success": {"en": "Look for lower exploitability, lower regret, or a stronger evaluation win rate, depending on the game. The final board or policy visualization should show legal, coherent decisions.", "zh": "根据游戏类型观察可利用度或遗憾值下降，或者评估胜率提高；最终棋盘或策略图还应表现出合法且连贯的决策。"},
        "preview": {"en": "Preview starts with the board game. After training it shows this run's policy map or a move-by-move trajectory, so inspect decisions as well as the curve.", "zh": "Preview 起初展示棋盘；训练后会显示本次策略图或逐步对局轨迹，需要同时检查决策过程和曲线。"},
        "time": {"en": "Default tabular and CFR recipes usually finish in 10–90 seconds on CPU.", "zh": "默认表格算法和 CFR 配方通常可在 CPU 上用 10–90 秒完成。"},
    },
    "course_url": "https://walkinglabs.github.io/hands-on-modern-rl/chapter32_selfplay/",
    "source_url": "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment04-board-selfplay/file/view/master/space_runtime.py",
    "notebook_url": "https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/"
    "code/online-experiments/hands-on-modern-rl-experiment04-board-selfplay.ipynb",
}


def task(key, title, environment, description, observation, action, algorithm, preview, budget):
    unit = {"en": "CFR iterations", "zh": "CFR 迭代"} if algorithm == "CFR+" else {"en": "self-play games", "zh": "自博弈对局"}
    return {
        "key": key,
        "title": {"en": title, "zh": title},
        "environment": environment,
        "description": description,
        "observation": observation,
        "action": action,
        "algorithm": algorithm,
        "preview": preview,
        "budget": budget,
        "training_unit": unit,
        "learning_rate": (0.01, 1.0, 0.25, 0.01),
        "gamma": (0.8, 1.0, 1.0, 0.01),
        "epsilon": (0.0, 1.0, 0.15, 0.01),
        "checkpoints": 8,
        "baseline_name": f"{algorithm} learning baseline",
        "baseline_time": {"en": "about 15 seconds–3 minutes on CPU", "zh": "CPU 上约 15 秒–3 分钟"},
        "baseline_outcome": {"en": "Exploitability or regret falls for CFR, while self-play policies produce stronger legal play and stable evaluation returns.", "zh": "CFR 的可利用度或遗憾值下降；自博弈策略产生更强的合法决策和更稳定的评估回报。"},
    }


TASKS = [
    task("kuhn-poker", "Kuhn Poker", "kuhn_poker", {"en": "Learn an equilibrium strategy in a tiny imperfect-information poker game.", "zh": "在小型不完全信息扑克游戏中学习均衡策略。"}, {"en": "Private card and betting history", "zh": "私有牌和下注历史"}, {"en": "Pass or bet", "zh": "过牌或下注"}, "CFR+", "assets/kuhn-poker.png", (50, 20_000, 2_000, 50)),
    task("leduc-poker", "Leduc Poker", "leduc_poker", {"en": "Balance betting, folding, and hidden information across two betting rounds.", "zh": "在两轮下注中权衡下注、弃牌和隐藏信息。"}, {"en": "Private/public cards and betting history", "zh": "私有牌、公共牌和下注历史"}, {"en": "Fold / call / raise", "zh": "弃牌、跟注、加注"}, "CFR+", "assets/leduc-poker.png", (50, 50_000, 5_000, 50)),
    task("tic-tac-toe", "Tic-Tac-Toe", "tic_tac_toe", {"en": "Discover blocking, forks, and winning lines through tabular self-play.", "zh": "通过表格自博弈发现阻挡、双威胁和获胜连线。"}, {"en": "3×3 board", "zh": "3×3 棋盘"}, {"en": "Place a mark in a legal cell", "zh": "在合法空格落子"}, "Self-play Q-Learning", "assets/tic-tac-toe.png", (100, 200_000, 50_000, 100)),
    task("connect-four", "Connect Four", "connect_four", {"en": "Learn vertical, horizontal, and diagonal threats through self-play.", "zh": "通过自博弈学习纵向、横向和斜向威胁。"}, {"en": "7×6 board", "zh": "7×6 棋盘"}, {"en": "Drop a piece into a legal column", "zh": "在合法列中投入棋子"}, "Self-play Q-Learning", "assets/connect-four.png", (500, 500_000, 100_000, 500)),
]


def runtime_status():
    try:
        import pyspiel

        game = pyspiel.load_game("tic_tac_toe")
        game.new_initial_state()
        return "OpenSpiel · READY"
    except Exception as exc:
        return f"startup check pending · {type(exc).__name__}"


def _font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _frame(title: str, state_text: str, footer: str, size=(760, 460)) -> np.ndarray:
    image = Image.new("RGB", size, "#f4f6fa")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, size[0] - 18, size[1] - 18), radius=22, fill="#ffffff", outline="#dfe4ef", width=2)
    draw.rounded_rectangle((34, 34, 235, size[1] - 34), radius=16, fill="#20245b")
    draw.text((55, 62), "LEARNED POLICY", font=_font(15), fill="#a5b4fc")
    draw.multiline_text((55, 105), title, font=_font(25), fill="white", spacing=7)
    draw.multiline_text((55, 310), footer, font=_font(14), fill="#cbd5e1", spacing=6)
    draw.text((275, 50), "Game state", font=_font(20), fill="#172033")
    draw.multiline_text((275, 94), state_text, font=_font(22), fill="#27324a", spacing=9)
    return np.asarray(image)


def _save_replay(key: str, title: str, states: list[tuple[str, str]], artifacts: Path | None = None) -> str:
    artifacts = artifacts or ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    frames = [_frame(title, state, footer) for state, footer in states]
    path = artifacts / f"{key}-learned-policy.gif"
    imageio.mimsave(path, frames, duration=.75, loop=0)
    return str(path)


def _sample(probabilities: dict[int, float], rng: random.Random) -> int:
    actions = list(probabilities)
    weights = np.asarray([max(0.0, probabilities[action]) for action in actions], dtype=float)
    if weights.sum() <= 0:
        return rng.choice(actions)
    weights /= weights.sum()
    return actions[int(np.searchsorted(np.cumsum(weights), rng.random(), side="right").clip(0, len(actions) - 1))]


def _cfr_replay(task, game, policy, seed: int, artifacts: Path) -> str:
    rng = random.Random(seed)
    state = game.new_initial_state()
    replay: list[tuple[str, str]] = [(str(state), "Initial state")]
    while not state.is_terminal():
        if state.is_chance_node():
            action = _sample(dict(state.chance_outcomes()), rng)
            label = f"Chance → {action}"
        else:
            player = state.current_player()
            probabilities = policy.action_probabilities(state, player)
            action = _sample(probabilities, rng)
            label = f"Player {player} → {state.action_to_string(player, action)}"
        state.apply_action(action)
        replay.append((str(state), label))
    replay.append((str(state), f"Returns: {state.returns()}"))
    return _save_replay(task["key"], task["title"]["en"], replay, artifacts)


def _save_cfr_policy(policy, path: Path) -> str:
    tabular = policy.to_tabular() if hasattr(policy, "to_tabular") else policy
    probabilities = np.asarray(tabular.action_probability_array, dtype=np.float32)
    lookup = dict(tabular.state_lookup)
    np.savez_compressed(path, action_probabilities=probabilities, state_lookup=json.dumps(lookup, ensure_ascii=False))
    return str(path)


def _cfr_run(task, budget: int, seed: int, checkpoints: int):
    import pyspiel
    from open_spiel.python.algorithms import cfr, exploitability

    game = pyspiel.load_game(task["environment"])
    solver = cfr.CFRPlusSolver(game)
    checkpoint_count = max(1, min(int(checkpoints), int(budget)))
    targets = {max(1, round(budget * index / checkpoint_count)): index for index in range(1, checkpoint_count + 1)}
    run_token = f"{int(time.time())}-{seed}"
    x: list[float] = []
    y: list[float] = []
    for iteration in range(1, budget + 1):
        solver.evaluate_and_update_policy()
        if iteration in targets:
            gap = float(exploitability.exploitability(game, solver.average_policy()))
            score = -gap
            x.append(float(iteration)); y.append(score)
            checkpoint_index = targets[iteration]
            artifacts = ROOT / "artifacts" / f"{task['key']}-{run_token}-epoch-{checkpoint_index:02d}"
            artifacts.mkdir(parents=True, exist_ok=True)
            policy = solver.average_policy()
            model = _save_cfr_policy(policy, artifacts / "policy.npz")
            preview = _cfr_replay(task, game, policy, seed + 1000 + checkpoint_index, artifacts)
            yield {"step": iteration, "score": score, "x": x, "y": y, "model": model, "preview": preview, "checkpoint_index": checkpoint_index, "checkpoint_count": checkpoint_count, "metric_detail": f"negative exploitability · gap={gap:.6f}", "log": f"CFR+ iteration={iteration:,} exploitability={gap:.8f}\nSAVE epoch={checkpoint_index}/{checkpoint_count} policy={Path(model).name}"}

    yield {"phase": "complete", "step": budget, "score": y[-1], "x": x, "y": y, "log": f"Saved {checkpoint_count} CFR policies and sampled one terminal game per epoch"}


def _state_key(state, player: int) -> str:
    try:
        return state.information_state_string(player)
    except Exception:
        try:
            return state.observation_string(player)
        except Exception:
            return str(state)


def _choose_q(q, state, player: int, epsilon: float, rng: random.Random) -> int:
    legal = state.legal_actions(player)
    if rng.random() < epsilon:
        return rng.choice(legal)
    key = _state_key(state, player)
    values = [q[(player, key, action)] for action in legal]
    best = max(values)
    return rng.choice([action for action, value in zip(legal, values) if value == best])


def _evaluate_q(game, q, seed: int, episodes: int = 100) -> float:
    rng = random.Random(seed)
    wins = 0.0
    for episode in range(episodes):
        learner = episode % 2
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                action = _sample(dict(state.chance_outcomes()), rng)
            else:
                player = state.current_player()
                action = _choose_q(q, state, player, 0.0, rng) if player == learner else rng.choice(state.legal_actions(player))
            state.apply_action(action)
        result = state.returns()[learner]
        wins += 1.0 if result > 0 else .5 if result == 0 else 0.0
    return wins / episodes


def _q_replay(task, game, q, rng: random.Random, artifacts: Path) -> str:
    state = game.new_initial_state()
    replay: list[tuple[str, str]] = [(str(state), "Initial board")]
    while not state.is_terminal():
        player = state.current_player()
        action = _choose_q(q, state, player, 0.0, rng)
        label = f"Player {player} → {state.action_to_string(player, action)}"
        state.apply_action(action)
        replay.append((str(state), label))
    replay.append((str(state), f"Returns: {state.returns()}"))
    return _save_replay(task["key"], task["title"]["en"], replay, artifacts)


def _save_q_table(q, path: Path) -> str:
    entries = [
        {"player": int(player), "state": state, "action": int(action), "value": float(value)}
        for (player, state, action), value in q.items()
    ]
    path.write_text(json.dumps({"q_values": entries}, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _q_run(task, budget: int, alpha: float, gamma: float, epsilon: float, seed: int, checkpoints: int):
    import pyspiel

    game = pyspiel.load_game(task["environment"])
    q = defaultdict(float)
    rng = random.Random(seed)
    checkpoint_count = max(1, min(int(checkpoints), int(budget)))
    targets = {max(1, round(budget * index / checkpoint_count)): index for index in range(1, checkpoint_count + 1)}
    run_token = f"{int(time.time())}-{seed}"
    x: list[float] = []
    y: list[float] = []
    for episode in range(1, budget + 1):
        state = game.new_initial_state()
        last: dict[int, tuple[str, int]] = {}
        while not state.is_terminal():
            if state.is_chance_node():
                state.apply_action(_sample(dict(state.chance_outcomes()), rng))
                continue
            player = state.current_player()
            key = _state_key(state, player)
            action = _choose_q(q, state, player, epsilon, rng)
            previous = last.get(player)
            if previous is not None:
                legal = state.legal_actions(player)
                bootstrap = max((q[(player, key, candidate)] for candidate in legal), default=0.0)
                q[(player, previous[0], previous[1])] += alpha * (gamma * bootstrap - q[(player, previous[0], previous[1])])
            last[player] = (key, action)
            state.apply_action(action)
        returns = state.returns()
        for player, (key, action) in last.items():
            q[(player, key, action)] += alpha * (returns[player] - q[(player, key, action)])
        if episode in targets:
            win_rate = _evaluate_q(game, q, seed + episode, episodes=80)
            x.append(float(episode)); y.append(win_rate)
            checkpoint_index = targets[episode]
            artifacts = ROOT / "artifacts" / f"{task['key']}-{run_token}-epoch-{checkpoint_index:02d}"
            artifacts.mkdir(parents=True, exist_ok=True)
            model = _save_q_table(q, artifacts / "q-table.json")
            preview = _q_replay(task, game, q, random.Random(seed + 1000 + checkpoint_index), artifacts)
            yield {"step": episode, "score": win_rate, "x": x, "y": y, "model": model, "preview": preview, "checkpoint_index": checkpoint_index, "checkpoint_count": checkpoint_count, "metric_detail": "win/draw rate versus random", "log": f"SELFPLAY episode={episode:,} states={len(q):,} win_or_draw_rate={win_rate:.3f}\nSAVE epoch={checkpoint_index}/{checkpoint_count} policy={Path(model).name}"}

    yield {"phase": "complete", "step": budget, "score": y[-1], "x": x, "y": y, "log": f"Saved {checkpoint_count} Q-table policies and greedy self-play replays"}


def run(key: str, budget: int, learning_rate: float, gamma: float, epsilon: float, seed: int, checkpoints: int | None = None):
    task = next(item for item in TASKS if item["key"] == key)
    checkpoint_count = int(checkpoints or task["checkpoints"])
    if task["algorithm"] == "CFR+":
        yield from _cfr_run(task, budget, seed, checkpoint_count)
    else:
        yield from _q_run(task, budget, learning_rate, gamma, epsilon, seed, checkpoint_count)
