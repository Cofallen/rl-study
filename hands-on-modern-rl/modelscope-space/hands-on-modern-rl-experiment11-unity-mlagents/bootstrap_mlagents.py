from __future__ import annotations

import importlib.util
import subprocess
import sys


def ensure_mlagents() -> None:
    """Install the official trainer on ModelScope's Python 3.10 patch release."""
    if importlib.util.find_spec("mlagents") and importlib.util.find_spec("mlagents_envs"):
        return
    print("Preparing official Unity ML-Agents 1.1.0 for ModelScope Python 3.10", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--ignore-requires-python",
            "mlagents==1.1.0",
            "mlagents-envs==1.1.0",
        ],
        check=True,
    )
