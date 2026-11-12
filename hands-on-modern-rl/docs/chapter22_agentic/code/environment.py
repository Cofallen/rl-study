# environment.py
import os
import subprocess
import sys
import tempfile


class SandboxEnv:
    """最小代码执行环境：subprocess + timeout，不构成安全边界。"""

    def __init__(self, timeout=10):
        self.timeout = timeout

    def step(self, action_type: str, action_args: dict) -> dict:
        """执行一步动作，返回观测和终止状态。"""
        if action_type == "execute_code":
            return self._exec_code(action_args["code"])
        elif action_type == "finish":
            return {"observation": "", "done": True}
        else:
            return {"observation": f"Unknown action: {action_type}", "done": False}

    def _exec_code(self, code: str) -> dict:
        """在当前 Python 的子进程中执行代码，并限制等待时间。"""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.flush()
                temp_path = f.name
                result = subprocess.run(
                    [sys.executable, temp_path],
                    timeout=self.timeout,
                    capture_output=True,
                    text=True,
                )
                return {
                    "observation": (result.stdout + result.stderr)[-500:],  # 截断
                    "done": False,
                }
        except subprocess.TimeoutExpired:
            return {"observation": "TIMEOUT", "done": True}
        except Exception as e:
            return {"observation": f"ERROR: {e}", "done": False}
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def reset(self):
        """重置环境状态（新 episode 开始时调用）。"""
        pass
