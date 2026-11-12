from __future__ import annotations

import os

from game_ui import AUTO_SCROLL_JS, CSS, build_demo
import space_runtime


demo = build_demo(space_runtime)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        css=CSS,
        js=AUTO_SCROLL_JS,
        footer_links=[],
    )
