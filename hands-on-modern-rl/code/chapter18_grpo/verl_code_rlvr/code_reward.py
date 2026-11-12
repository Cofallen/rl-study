# code_reward.py
# Reward function for veRL code-generation RLVR: run the generated code as a standalone program against stdin/stdout tests.
#
# Background (issue #53):
#   The original doc assumed Eurus-2-RL-Data carried a `tests` field (Python assert statements) that could be exec'd directly.
#   In practice the dataset's code samples have no tests / entry_point; reward_model.ground_truth is a
#   JSON string {"inputs": [...], "outputs": [...]} (stdin/stdout test pairs).
#   So this instead: extracts the code -> writes it to a temp file -> executes it in a real subprocess,
#   feeding each input on stdin, comparing stdout against the expected output, and returning the pass rate.
#
# veRL interface (verl/workers/reward_manager/naive.py):
#   score = self.compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)
#   ground_truth comes from the dataset's reward_model["ground_truth"].
#   When returning a dict, "score" is the main reward; the other keys are attached as logging info.
#
# Usage:
#   python code_reward.py            # sanity check: runs a few constructed cases to verify reward logic
#
# Wired into training via the verl config:
#   custom_reward_function.path=.../code_reward.py
#   custom_reward_function.name=compute_score

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_TIMEOUT_S = 10.0
_MAX_OUTPUT_BYTES = 100_000
_MAX_MEMORY_BYTES = 2 * 1024**3
_SANDBOX_OPT_IN = "HOMRL_ALLOW_UNSAFE_CODE_EXECUTION"
_RUNNER_SOURCE = r"""
import math
import resource
import runpy
import sys


def set_soft_limit(kind, requested):
    _, current_hard = resource.getrlimit(kind)
    limit = requested if current_hard == resource.RLIM_INFINITY else min(
        requested, current_hard
    )
    resource.setrlimit(kind, (limit, current_hard))


solution_path = sys.argv[1]
timeout_s = float(sys.argv[2])
max_output_bytes = int(sys.argv[3])
max_memory_bytes = int(sys.argv[4])
set_soft_limit(resource.RLIMIT_CPU, max(1, math.ceil(timeout_s)))
set_soft_limit(resource.RLIMIT_FSIZE, max_output_bytes)
set_soft_limit(resource.RLIMIT_CORE, 0)
set_soft_limit(resource.RLIMIT_NOFILE, 64)
if sys.platform.startswith("linux") and hasattr(resource, "RLIMIT_AS"):
    set_soft_limit(resource.RLIMIT_AS, max_memory_bytes)

sys.argv = [solution_path]
runpy.run_path(solution_path, run_name="__main__")
"""


def extract_code(response: str) -> str:
    """Extracts the Python code block from the model's output.

    The model typically outputs something like:
        "```python\ndef solve():\n    ...```"
    We take only the part between ```python and ```.
    If the model did not use code-block formatting, the whole answer is treated as code (fallback; this usually causes a syntax error and reward=0).
    """
    match = _CODE_BLOCK_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()


def _normalize(text: str) -> str:
    """Strip trailing whitespace before comparing, so \r or extra blank lines don't cause false mismatches."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def _require_execution_opt_in() -> None:
    """Stop users from mistaking a local subprocess for a secure sandbox."""
    if os.environ.get(_SANDBOX_OPT_IN) != "1":
        raise RuntimeError(
            "Refusing to directly execute model-generated code. subprocess is not a secure sandbox; "
            f"first isolate network, credentials, and training files in a container/VM, then set "
            f"{_SANDBOX_OPT_IN}=1."
        )


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Terminate the test process and any child processes in its group."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def run_io_tests(code: str, ground_truth_json: str, timeout_s: float = _TIMEOUT_S):
    """Runs code as a standalone program, testing it against the inputs/outputs in ground_truth.

    Returns (pass_rate, details for the first few tests). Any exception (syntax error, runtime crash,
    timeout, output mismatch) affects only the corresponding case and never aborts the whole scoring run.
    """
    try:
        tests = json.loads(ground_truth_json)
    except (TypeError, json.JSONDecodeError) as exc:
        return 0.0, f"failed to parse ground_truth: {exc!r}"

    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])
    if not inputs or len(inputs) != len(outputs):
        return 0.0, f"inputs/outputs count mismatch: {len(inputs)} vs {len(outputs)}"

    _require_execution_opt_in()

    # A subprocess isolates interpreter state only — not the filesystem, network, credentials, or resources.
    # The caller must first place training inside a least-privilege container or VM.
    tmp_dir = tempfile.mkdtemp(prefix="homrl-code-reward-")
    tmp_path = Path(tmp_dir) / "solution.py"
    runner_path = Path(tmp_dir) / "runner.py"
    tmp_path.write_text(code, encoding="utf-8")
    runner_path.write_text(_RUNNER_SOURCE, encoding="utf-8")

    try:
        passed = 0
        details = []
        for inp, expected in zip(inputs, outputs):
            try:
                stdout_path = Path(tmp_dir) / "stdout.txt"
                with stdout_path.open("wb") as stdout_file:
                    proc = subprocess.Popen(
                        [
                            sys.executable,
                            "-I",
                            str(runner_path),
                            str(tmp_path),
                            str(timeout_s),
                            str(_MAX_OUTPUT_BYTES),
                            str(_MAX_MEMORY_BYTES),
                        ],
                        stdin=subprocess.PIPE,
                        stdout=stdout_file,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        cwd=tmp_dir,
                        start_new_session=True,
                    )
                    try:
                        proc.communicate(input=inp, timeout=timeout_s)
                    except subprocess.TimeoutExpired:
                        _terminate_process_group(proc)
                        proc.communicate()
                        details.append("FAIL(timeout)")
                        continue
                if proc.returncode != 0:
                    details.append("FAIL(nonzero exit)")
                    continue
                stdout = stdout_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                got = _normalize(stdout)
                want = _normalize(expected)
                if got == want:
                    passed += 1
                    details.append("PASS")
                else:
                    details.append("FAIL(output mismatch)")
            except Exception as exc:  # noqa: BLE001
                details.append(f"FAIL({exc!r})")
        return passed / len(inputs), "; ".join(details[:5])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """veRL reward entry point.

    Args:
        data_source: dataset source (in this experiment, one of codecontests/taco/apps/codeforces)
        solution_str: the model's full generated response (markdown text)
        ground_truth: dataset's reward_model["ground_truth"]; for code samples this is a JSON string of I/O tests
        extra_info: dataset's extra_info column (this dataset only has index/split, unused)

    Returns:
        dict: {"score": pass_rate, "pass_rate": pass_rate, "format": whether code was extracted}
        veRL uses "score" as the main PPO reward.
    """
    # `format` only indicates whether the code was emitted in a ```python block.
    # When the format is not followed, extract_code falls back to running the whole answer as code (usually a syntax error, score=0),
    # but the `format` metric should faithfully reflect whether the model learned to emit code blocks.
    match = _CODE_BLOCK_RE.search(solution_str)
    format_ok = 1.0 if match else 0.0
    code = extract_code(solution_str)
    if not code:
        return {"score": 0.0, "pass_rate": 0.0, "format": 0.0}

    pass_rate, detail = run_io_tests(code, ground_truth)
    return {"score": pass_rate, "pass_rate": pass_rate, "format": format_ok}


# ---------------------------------------------------------------------------
# Sanity check: verifies the reward logic directly, no training environment needed
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _require_execution_opt_in()
    # Construct ground_truth (inputs/outputs) using a simple A+B problem
    ab_gt = json.dumps({"inputs": ["1 2", "10 20", "-3 5"], "outputs": ["3", "30", "2"]})

    correct = "```python\nimport sys\n\n\nfor line in sys.stdin:\n    a, b = map(int, line.split())\n    print(a + b)\n```"
    wrong = "```python\nimport sys\n\n\nfor line in sys.stdin:\n    a, b = map(int, line.split())\n    print(a - b)\n```"
    no_code = "I don't know how to solve this."

    for name, resp in [("correct code", correct), ("wrong code", wrong), ("no code", no_code)]:
        result = compute_score("synthetic", resp, ab_gt, None)
        print(f"{name:8s} -> score={result['score']:.2f} pass_rate={result['pass_rate']:.2f} "
              f"format={result['format']:.0f}")
