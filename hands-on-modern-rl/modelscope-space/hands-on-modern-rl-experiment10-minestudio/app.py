from __future__ import annotations

import importlib.util
import subprocess
import sys


def ensure_minestudio_simulator() -> None:
    """Install MineStudio without its unrelated offline/distributed extras."""
    if importlib.util.find_spec("minestudio") is not None:
        return
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-deps",
            "minestudio==1.1.6",
        ],
        check=True,
        timeout=300,
    )


ensure_minestudio_simulator()

import space_runtime  # noqa: E402
from game_ui import launch  # noqa: E402


if __name__ == "__main__":
    launch(space_runtime)
