---
title: Unity ML-Agents xGPU Arena
emoji: 🧪
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
---

# WalkingLab × Hands-On Modern RL · Unity ML-Agents xGPU Arena

**WalkingLab** 与开源课程 **Hands-On Modern RL（《动手学现代强化学习》）** 的实验 11。页面直接使用 Unity ML-Agents Linux 场景训练 PPO，并录制本次运行产生的 Unity 画面。默认任务是 **Huggy · 小狗捡树枝**，它基于 Unity 官方 Puppo the Corgi 演示；页面也保留 Basic、3D Ball、Food Collector 与 Walker。

- Project: <https://github.com/walkinglabs/hands-on-modern-rl>
- WalkingLab: <https://modelscope.cn/organization/walkinglab>
- Unity scene Dataset: <https://modelscope.cn/datasets/walkinglab/hands-on-modern-rl-unity-environments>
- Companion chapter: <https://walkinglabs.github.io/hands-on-modern-rl>
- Live Studio: <https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment11-unity-mlagents>
- Training source: <https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment11-unity-mlagents/file/view/master/space_runtime.py>
- Companion Notebook: <https://modelscope.cn/notebook/share/github/walkinglabs/hands-on-modern-rl/blob/main/code/online-experiments/hands-on-modern-rl-experiment11-unity-mlagents.ipynb> (requires a scheduled xGPU Notebook)

Scene archives come from the versioned WalkingLab Dataset above, use persistent resumable storage, and automatically fall back from `aria2c` to `curl`. Huggy's 39 MB Linux build and the official 18-scene Startup build are cached after their first Dataset download. ModelScope currently exposes xGPU only through its Gradio SDK, so the app prepares Xvfb, Mesa, and ffmpeg when a fresh xGPU container starts. Every run must produce visible Unity frames and a real animated replay; if rendering fails, the run stops with an explicit error instead of substituting a reward-curve GIF.

Every gallery card and initial task preview uses a locally stored, compressed capture of the actual Unity environment—no generated illustration and no external hotlink. Huggy's capture comes from the [official Huggy README](https://github.com/huggingface/Huggy); Basic, 3D Ball, Food Collector, and Walker come from the [Unity ML-Agents Release 20 documentation assets](https://github.com/Unity-Technologies/ml-agents/tree/release_20/docs/images). Training then replaces the still capture with live Unity frames and the final replay GIF.
