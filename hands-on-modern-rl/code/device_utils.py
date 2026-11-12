"""Shared PyTorch device helpers for course experiments."""

from __future__ import annotations

import torch

VALID_DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def is_mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps and mps.is_available())


def resolve_torch_device(choice: str = "auto") -> torch.device:
    """Pick a torch.device: cuda, then mps, then cpu when choice is auto."""
    choice = choice.lower()
    if choice not in VALID_DEVICE_CHOICES:
        raise ValueError(f"device must be one of {VALID_DEVICE_CHOICES}, got {choice!r}")

    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if choice == "mps" and not is_mps_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")

    return torch.device(choice)


def resolve_sb3_device(choice: str = "auto") -> str:
    """Return a Stable-Baselines3 device string (cuda, mps, or cpu)."""
    return str(resolve_torch_device(choice))


def describe_device(device: torch.device | str) -> str:
    """Human-readable label for logs."""
    name = str(device)
    if name.startswith("cuda") and torch.cuda.is_available():
        index = 0 if name == "cuda" else int(name.split(":")[1])
        props = torch.cuda.get_device_properties(index)
        return f"CUDA ({props.name}, {props.total_memory / 1e9:.1f} GB)"
    if name == "mps" and is_mps_available():
        return "Apple MPS (Metal)"
    if name == "cpu":
        return "CPU"
    return name


def print_device_report(device: torch.device | str | None = None) -> torch.device:
    """Print PyTorch/CUDA/MPS availability and return the resolved device."""
    print("=" * 50)
    print("PyTorch device report")
    print("=" * 50)
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  CUDA:     {torch.cuda.is_available()}")
    print(f"  MPS:      {is_mps_available()}")
    resolved = resolve_torch_device("auto") if device is None else resolve_torch_device(str(device))
    print(f"  Selected: {describe_device(resolved)} ({resolved})")
    print("=" * 50)
    return resolved
