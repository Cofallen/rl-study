from bootstrap_native_runtime import ensure_native_runtime
from bootstrap_mlagents import ensure_mlagents

ensure_native_runtime()
ensure_mlagents()

import space_runtime
from game_ui import launch


if __name__ == "__main__":
    launch(space_runtime)
