import space_runtime
from game_ui import launch


if __name__ == "__main__":
    space_runtime.start_runtime_warmup()
    launch(space_runtime)
