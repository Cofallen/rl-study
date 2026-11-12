---
title: ManiSkill xGPU Robot Lab
emoji: 🧪
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
---

# WalkingLab × Hands-On Modern RL · ManiSkill xGPU Robot Lab

**WalkingLab** 与开源课程 **Hands-On Modern RL（《动手学现代强化学习》）** 的实验 08。它使用 ManiSkill 3 的 PhysX 环境与 CUDA PPO 训练机器人策略，并从本次学习到的策略生成 GIF 回放。

- Project: <https://github.com/walkinglabs/hands-on-modern-rl>
- WalkingLab: <https://modelscope.cn/organization/walkinglab>
- Companion chapter: <https://walkinglabs.github.io/hands-on-modern-rl>
- Live Studio: <https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment08-maniskill>
- Training source: <https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment08-maniskill/file/view/master/space_runtime.py>
- Companion Notebook: <https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment08-maniskill.ipynb> (requires a scheduled xGPU Notebook)

The live Studio may bundle the official SAPIEN `linux-so.zip` GPU PhysX runtime to avoid a slow first-run GitHub download. The binary remains under NVIDIA's BSD-3-Clause terms; the required notice is included at `assets/physx-BSD-3-Clause.txt`.

The Studio selects Mesa Lavapipe before SAPIEN starts because ModelScope xGPU exposes CUDA compute without the host NVIDIA Vulkan graphics capability. It first attempts parallel GPU PhysX and automatically uses the official CPU PhysX state backend when GPU simulation is unavailable; PPO remains on CUDA in both cases. The run log and downloaded metadata record the backend actually used. The final replay uses Mesa's CPU Vulkan driver when available and otherwise generates a camera-free task-space GIF from the learned policy's real TCP, object, and goal poses.
