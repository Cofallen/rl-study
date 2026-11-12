from __future__ import annotations

import ctypes.util
import os
import shutil
import subprocess


SYSTEM_PACKAGES = (
    "ca-certificates",
    "ffmpeg",
    "libasound2",
    "libgl1",
    "libgl1-mesa-dri",
    "libglib2.0-0",
    "libglu1-mesa",
    "libnss3",
    "libvulkan1",
    "libx11-6",
    "libx11-xcb1",
    "libxcomposite1",
    "libxcursor1",
    "libxdamage1",
    "libxext6",
    "libxfixes3",
    "libxi6",
    "libxinerama1",
    "libxrandr2",
    "libxrender1",
    "libxtst6",
    "mesa-utils",
    "xvfb",
)


def _runtime_is_ready() -> bool:
    return bool(
        shutil.which("Xvfb")
        and shutil.which("ffmpeg")
        and ctypes.util.find_library("GL")
    )


def ensure_native_runtime() -> None:
    """Prepare the headless Unity renderer inside ModelScope's xGPU SDK image.

    ModelScope currently exposes xGPU only to the Gradio SDK. Its SDK image does
    not process packages.txt, but the application runs as root, so install the
    small native rendering layer once when a fresh container starts.
    """
    if _runtime_is_ready():
        print("Unity renderer ready: Xvfb, ffmpeg, and Mesa are installed", flush=True)
        return

    apt_get = shutil.which("apt-get")
    if not apt_get:
        raise RuntimeError(
            "ModelScope xGPU image is missing apt-get; Xvfb and Mesa cannot be prepared"
        )
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        raise RuntimeError(
            "ModelScope xGPU image did not start the app as root; native Unity packages cannot be installed"
        )

    print(
        "Preparing native Unity renderer for the ModelScope xGPU container "
        "(Xvfb + Mesa + ffmpeg)",
        flush=True,
    )
    environment = os.environ.copy()
    environment["DEBIAN_FRONTEND"] = "noninteractive"
    subprocess.run(
        [apt_get, "update"],
        check=True,
        env=environment,
        timeout=300,
    )
    subprocess.run(
        [apt_get, "install", "-y", "--no-install-recommends", *SYSTEM_PACKAGES],
        check=True,
        env=environment,
        timeout=600,
    )
    if not _runtime_is_ready():
        raise RuntimeError(
            "Native package installation completed, but Xvfb, ffmpeg, or Mesa is still unavailable"
        )
    print("Native Unity renderer installation complete", flush=True)
