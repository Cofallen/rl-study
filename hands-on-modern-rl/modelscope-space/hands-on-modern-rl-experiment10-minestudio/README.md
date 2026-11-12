---
title: MineStudio xGPU Minecraft Agent Lab
emoji: 🧪
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
---

# WalkingLab × Hands-On Modern RL · MineStudio xGPU Minecraft Agent Lab

**WalkingLab** 与开源课程 **Hands-On Modern RL（《动手学现代强化学习》）** 的实验 10。它启动 MineStudio 的真实 Minecraft 模拟器，用 xGPU 训练视觉 CNN PPO，并从本次策略生成第一人称 GIF 回放。

- Project: <https://github.com/walkinglabs/hands-on-modern-rl>
- WalkingLab: <https://modelscope.cn/organization/walkinglab>
- Companion chapter: <https://walkinglabs.github.io/hands-on-modern-rl>
- Live Studio: <https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment10-minestudio>
- Training source: <https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment10-minestudio/file/view/master/space_runtime.py>
- Companion Notebook: <https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment10-minestudio.ipynb> (requires a scheduled xGPU Notebook)

The live Studio bundles the official Eclipse Temurin JRE 8 archive so Minecraft startup does not depend on a slow first-run GitHub redirect. Its SHA-256 is `f1a7bea0804bfa5627dac412fe7a0d751c4228592e356d6a32a30da54a48ed7a`; the unmodified archive contains the upstream `NOTICE`, `LICENSE`, `ASSEMBLY_EXCEPTION`, and third-party license files (GPL-2.0 with Classpath Exception).

MineStudio's separate 458 MB simulator engine remains sourced from the upstream `CraftJarvis/SimulatorEngine` repository. The container uses the `hf-mirror.com` endpoint because direct `huggingface.co` egress is unavailable on ModelScope xGPU, and keeps the downloaded engine under `/mnt/workspace` for later runs.
