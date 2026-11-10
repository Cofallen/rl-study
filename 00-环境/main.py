"""
课程环境一键验证脚本
运行: python verify_env.py
"""
import sys

checks = []

# 1. Python 版本
ver = sys.version_info
checks.append(("Python >= 3.10", ver >= (3, 10)))

# 2. 核心包
for pkg in ["numpy", "matplotlib", "scipy", "tqdm"]:
    try:
        __import__(pkg)
        checks.append((pkg, True))
    except ImportError:
        checks.append((pkg, False))

# 3. PyTorch
try:
    import torch
    checks.append(("PyTorch", True))
    checks.append(("CUDA GPU", torch.cuda.is_available()))
except ImportError:
    checks.append(("PyTorch", False))

# 4. RL 环境
try:
    import gymnasium
    checks.append(("Gymnasium", True))
except ImportError:
    checks.append(("Gymnasium", False))

# 5. LLM 框架
for pkg in ["transformers", "datasets", "peft", "accelerate", "trl"]:
    try:
        __import__(pkg)
        checks.append((pkg, True))
    except ImportError:
        checks.append((pkg, False))

# 输出报告
print("\n" + "=" * 45)
print("  环境验证报告")
print("=" * 45)
for name, ok in checks:
    status = "OK" if ok else "MISSING"
    print(f"  {name:<25} {status}")
print("=" * 45)
passed = sum(1 for _, ok in checks if ok)
print(f"  通过: {passed}/{len(checks)}")
print("=" * 45 + "\n")