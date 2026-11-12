"""
Train a DQN baseline on an early Pokemon Red exploration task with PyBoy.

This script is intentionally scoped to a small, real training target: make a
neural Q-network learn short-horizon movement/exploration signals in the first
area of Pokemon Red. It does not ship a ROM, emulator state, or pretrained
weights. You must provide a legally obtained ROM, and optionally a PyBoy state
file that starts the game after the intro/menu.

Example:
    python code/chapter04_dqn/dqn_pokemon_red_pyboy.py \
        --rom /path/to/PokemonRed.gb \
        --state /path/to/start.state \
        --total-timesteps 500000
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


EXPECTED_ROM_SHA1 = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"

# Pokemon Red WRAM addresses used for a light-weight DQN reward signal.
# They are stable for the US Pokemon Red ROM used by many public experiments.
W_Y_COORD = 0xD361
W_X_COORD = 0xD362
W_CUR_MAP = 0xD35E
W_BADGES = 0xD356
W_PARTY_COUNT = 0xD163

ACTION_NAMES = ["noop", "up", "down", "left", "right", "a", "b", "start"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN baseline for early Pokemon Red exploration with PyBoy."
    )
    parser.add_argument("--rom", type=Path, required=True, help="Path to a legal Pokemon Red ROM.")
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Optional PyBoy state file. Use one that starts after the intro/menu.",
    )
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=2_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--max-episode-steps", type=int, default=2_000)
    parser.add_argument("--action-frames", type=int, default=8)
    parser.add_argument("--sticky-action-prob", type=float, default=0.08)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("output/dqn_pokemon_red"))
    parser.add_argument(
        "--skip-rom-check",
        action="store_true",
        help="Skip SHA1 validation when using a compatible ROM variant.",
    )
    return parser.parse_args()


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(rom: Path, state: Path | None, skip_rom_check: bool) -> None:
    if not rom.exists():
        raise SystemExit(f"ROM not found: {rom}")
    if state is not None and not state.exists():
        raise SystemExit(f"PyBoy state file not found: {state}")
    if not skip_rom_check:
        rom_hash = sha1(rom)
        if rom_hash != EXPECTED_ROM_SHA1:
            raise SystemExit(
                "ROM SHA1 does not match the expected Pokemon Red ROM.\n"
                f"Expected: {EXPECTED_ROM_SHA1}\n"
                f"Actual:   {rom_hash}\n"
                "Use a compatible legally obtained ROM, or pass --skip-rom-check "
                "if you know this variant uses the same RAM layout."
            )


class PokemonRedDQNEnv(gym.Env):
    """
    Minimal PyBoy environment for DQN.

    Observation:
        84x84 grayscale screen image. VecFrameStack adds temporal context.

    Reward:
        Dense shaping for early exploration:
        - reward for visiting a new (map, x, y) coordinate
        - small reward for changing coordinates
        - larger reward for reaching a new map, gaining a badge, or changing party size
        - small per-step penalty and loop penalty

    This is a DQN baseline, not a full Pokemon-playing benchmark. For serious
    experiments, start from a saved state after the intro and evaluate specific
    subtasks such as leaving the house or reaching a new map.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        rom_path: Path,
        state_path: Path | None,
        max_episode_steps: int,
        action_frames: int,
        sticky_action_prob: float,
        seed: int,
    ) -> None:
        super().__init__()
        try:
            from pyboy import PyBoy
        except ImportError as exc:
            raise RuntimeError(
                "PyBoy is required for Pokemon Red experiments. Install it with:\n"
                "  pip install pyboy"
            ) from exc

        self.rom_path = rom_path
        self.state_path = state_path
        self.max_episode_steps = max_episode_steps
        self.action_frames = action_frames
        self.sticky_action_prob = sticky_action_prob
        self.rng = np.random.default_rng(seed)

        self.pyboy = PyBoy(str(rom_path), window="null")
        self.pyboy.set_emulation_speed(0)

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        self.step_count = 0
        self.visited: set[tuple[int, int, int]] = set()
        self.recent_positions: deque[tuple[int, int, int]] = deque(maxlen=80)
        self.last_position: tuple[int, int, int] | None = None
        self.last_badges = 0
        self.last_party_count = 0
        self.last_action = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.state_path is not None:
            with self.state_path.open("rb") as state_file:
                self.pyboy.load_state(state_file)
        else:
            self.pyboy.stop(save=False)
            from pyboy import PyBoy

            self.pyboy = PyBoy(str(self.rom_path), window="null")
            self.pyboy.set_emulation_speed(0)

        self.step_count = 0
        self.visited.clear()
        self.recent_positions.clear()
        self.last_position = self._position()
        self.last_badges = self._read(W_BADGES)
        self.last_party_count = self._read(W_PARTY_COUNT)
        self.visited.add(self.last_position)
        self.recent_positions.append(self.last_position)
        self.last_action = 0

        return self._observation(), self._info()

    def step(self, action: int):
        self.step_count += 1
        if self.rng.random() < self.sticky_action_prob:
            action = self.last_action
        self.last_action = action

        self._press(action)
        for _ in range(self.action_frames):
            self.pyboy.tick()
        self._release(action)

        position = self._position()
        badges = self._read(W_BADGES)
        party_count = self._read(W_PARTY_COUNT)

        reward = -0.002
        if position != self.last_position:
            reward += 0.01
        if position not in self.visited:
            reward += 0.08
            self.visited.add(position)
        if self.last_position is not None and position[0] != self.last_position[0]:
            reward += 1.0
        if badges != self.last_badges:
            reward += 5.0
        if party_count != self.last_party_count:
            reward += 0.5

        self.recent_positions.append(position)
        if len(self.recent_positions) == self.recent_positions.maxlen:
            unique_recent = len(set(self.recent_positions))
            if unique_recent < 6:
                reward -= 0.05

        self.last_position = position
        self.last_badges = badges
        self.last_party_count = party_count

        terminated = False
        truncated = self.step_count >= self.max_episode_steps
        return self._observation(), reward, terminated, truncated, self._info()

    def render(self):
        return np.asarray(self.pyboy.screen.ndarray)

    def close(self) -> None:
        if hasattr(self, "pyboy"):
            self.pyboy.stop(save=False)

    def _read(self, address: int) -> int:
        return int(self.pyboy.memory[address])

    def _position(self) -> tuple[int, int, int]:
        return (self._read(W_CUR_MAP), self._read(W_X_COORD), self._read(W_Y_COORD))

    def _info(self) -> dict[str, int]:
        map_id, x_coord, y_coord = self._position()
        return {
            "map_id": map_id,
            "x": x_coord,
            "y": y_coord,
            "unique_positions": len(self.visited),
            "badges": self._read(W_BADGES),
            "party_count": self._read(W_PARTY_COUNT),
        }

    def _observation(self) -> np.ndarray:
        frame = np.asarray(self.pyboy.screen.ndarray)
        gray = self._to_gray(frame)
        small = self._resize_nearest(gray, 84, 84)
        return small[:, :, None].astype(np.uint8)

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.astype(np.uint8)
        rgb = frame[:, :, :3].astype(np.float32)
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        return gray.astype(np.uint8)

    def _resize_nearest(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        y_idx = (np.linspace(0, image.shape[0] - 1, height)).astype(np.int32)
        x_idx = (np.linspace(0, image.shape[1] - 1, width)).astype(np.int32)
        return image[np.ix_(y_idx, x_idx)]

    def _press(self, action: int) -> None:
        name = ACTION_NAMES[action]
        if name != "noop":
            self.pyboy.button(name, delay=self.action_frames)

    def _release(self, action: int) -> None:
        # PyBoy's button(..., delay=n) schedules the release automatically.
        # This hook keeps the action application explicit for readers.
        return None


def make_env(args: argparse.Namespace, seed: int, monitor_dir: Path):
    def _init() -> gym.Env:
        env = PokemonRedDQNEnv(
            rom_path=args.rom.resolve(),
            state_path=args.state.resolve() if args.state else None,
            max_episode_steps=args.max_episode_steps,
            action_frames=args.action_frames,
            sticky_action_prob=args.sticky_action_prob,
            seed=seed,
        )
        return Monitor(env, filename=str(monitor_dir / f"monitor-{seed}.csv"))

    return _init


def main() -> None:
    args = parse_args()
    validate_inputs(args.rom, args.state, args.skip_rom_check)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = args.log_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env(args, args.seed, monitor_dir)])
    env = VecFrameStack(env, n_stack=4, channels_order="last")

    eval_env = DummyVecEnv([make_env(args, args.seed + 10_000, monitor_dir)])
    eval_env = VecFrameStack(eval_env, n_stack=4, channels_order="last")

    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=0.99,
        train_freq=args.train_freq,
        gradient_steps=1,
        target_update_interval=args.target_update_interval,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        tensorboard_log=str(args.log_dir / "tb"),
        verbose=1,
        seed=args.seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(args.log_dir / "checkpoints"),
        name_prefix="dqn_pokemon_red",
        save_replay_buffer=True,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.log_dir / "best"),
        log_path=str(args.log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=3,
        deterministic=True,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=args.progress_bar,
    )
    model.save(args.log_dir / "final_model")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
