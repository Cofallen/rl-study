---
title: JAX MinAtar CPU 游戏训练场
emoji: ⚡
colorFrom: purple
colorTo: orange
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
---

# WalkingLab × Hands-On Modern RL · JAX MinAtar CPU Game Lab

**WalkingLab** 与开源课程 **Hands-On Modern RL（《动手学现代强化学习》）** 的 JAX 游戏实验。在 ModelScope CPU 容器中使用 JAX/Optax 训练 MinAtar 策略。首次运行会编译更新函数，随后持续报告策略梯度损失、评估回报并生成语义像素回放。

课程项目：<https://github.com/walkinglabs/hands-on-modern-rl>

WalkingLab：<https://modelscope.cn/organization/walkinglab>

## 配套实验 Notebook

[直接在 ModelScope Notebook 中运行 JAX MinAtar 实验](https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment07-jax-games.ipynb)。Notebook 与当前创空间复用同一份 JAX/Optax 训练运行时，并显示编译日志、评估曲线和语义像素回放。
