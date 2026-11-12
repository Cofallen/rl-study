from __future__ import annotations

import base64
import html
import inspect
import io
import json
import math
import os
import time
import traceback
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hands-on-modern-rl-matplotlib")

from typing import Any, Iterable

import gradio as gr
import matplotlib
import numpy as np
from PIL import Image, ImageStat

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_URL = "https://github.com/walkinglabs/hands-on-modern-rl"
COURSE_URL = "https://walkinglabs.github.io/hands-on-modern-rl/"
ORGANIZATION_URL = "https://modelscope.cn/organization/walkinglab"
DEFAULT_LANGUAGE = "English"


TEXT = {
    "English": {
        "course": "Open-source reinforcement learning experiments",
        "brand": "WALKINGLAB × HANDS-ON MODERN RL",
        "brand_note": "Learn the idea, change the parameters, and train it online.",
        "organization": "WalkingLab",
        "dataset": "Environment Dataset",
        "chapter": "Companion chapter",
        "notebook": "Notebook",
        "source": "Training source",
        "project": "GitHub project",
        "device": "Device",
        "tasks": "Playable tasks",
        "runtime": "Runtime",
        "choose": "Choose a game",
        "choose_copy": "Every card opens a real training recipe. Select a task, review its goal, then start a run.",
        "understand": "UNDERSTAND BEFORE TRAINING",
        "observation": "Observation",
        "action": "Action",
        "algorithm": "Algorithm",
        "status": "Status",
        "ready_runtime": "Ready · preinstalled",
        "hint": "Adjust the parameters below, then start training. The curve and console update throughout the run.",
        "setup": "Experiment setup",
        "setup_copy": "Choose steps per epoch and epoch count. Every epoch ends with evaluation and one saved policy.",
        "selected": "Selected game",
        "steps_per_epoch": "Environment steps per epoch",
        "steps_per_epoch_info": "How many environment interactions to train before evaluation and model saving",
        "epochs": "Training epochs / saved models",
        "epochs_info": "One epoch means one fixed step block; each epoch saves one evaluated policy",
        "advanced": "Advanced settings",
        "lr": "Learning rate",
        "gamma": "Discount factor γ",
        "epsilon": "Exploration ε",
        "seed": "Random seed",
        "baseline_title": "Recommended learning baseline",
        "baseline_badge": "DEFAULT PRESET",
        "baseline_budget": "Total environment steps",
        "baseline_warmup": "Replay warm-up",
        "baseline_exploration": "Exploration",
        "baseline_time": "Typical runtime",
        "baseline_checkpoints": "Epoch schedule",
        "baseline_expected": "Expected signal",
        "baseline_restore": "Restore recommended baseline",
        "saved_model": "Epoch model",
        "saved_model_info": "The evaluated policy saved at the end of every epoch appears here",
        "saved_model_empty": "No epoch models yet. Start training to save the first policy.",
        "preview_waiting": "No epoch model is selected. Start training, then choose a saved epoch here.",
        "start": "Start training",
        "running_button": "Training…",
        "run_status": "Run status",
        "ready": "Ready to train",
        "ready_detail": "Select a game, review its goal, and start the run",
        "running": "Training in progress",
        "complete": "Training complete",
        "failed": "Run stopped",
        "metric": "Latest evaluation",
        "metric_waiting": "Results appear after training starts",
        "curve": "Learning curve",
        "curve_copy": "The chart updates at the end of every epoch. Labels remain in English for readability.",
        "log": "Live training log",
        "log_waiting": "Waiting for a training run…",
        "preview": "Task preview / learned policy",
        "preview_copy": "Every epoch ends with evaluation and one saved policy. Choose an epoch below to compare its exact learned behavior; non-best replays are generated once when selected.",
        "download": "Download run summary",
        "wait_title": "Training is active",
        "wait_detail": "Preparing the environment, updating the policy, evaluating it, and rendering the learned result.",
        "guide_title": "How to judge this training run",
        "guide_copy": "Read these three signals before increasing steps per epoch or the epoch count.",
        "guide_success": "What counts as success",
        "guide_preview": "How to read Preview",
        "guide_time": "Typical time",
        "guide_success_default": "Training complete confirms that the pipeline finished. Learning is demonstrated when later epoch models improve over early ones and the learned behavior matches the task goal.",
        "guide_preview_default": "Before training, Preview shows the task. During or after training it changes to this run's live frames, replay GIF, policy map, or result visualization.",
        "guide_time_default": "Most default recipes take 30 seconds to 5 minutes. First-run downloads, compilation, or 3D rendering can add several minutes.",
        "reference": "Environment reference",
    },
    "中文": {
        "course": "开源强化学习在线实验",
        "brand": "WALKINGLAB × 动手学现代强化学习",
        "brand_note": "理解任务、调整参数，并在网页中完成真实训练。",
        "organization": "WalkingLab 主页",
        "dataset": "环境场景 Dataset",
        "chapter": "阅读配套章节",
        "notebook": "Notebook",
        "source": "训练源码",
        "project": "GitHub 项目",
        "device": "设备",
        "tasks": "可训练任务",
        "runtime": "运行环境",
        "choose": "选择一个游戏",
        "choose_copy": "每张卡片都对应一个真实训练配方。选择任务、阅读目标，然后启动训练。",
        "understand": "训练前先理解任务",
        "observation": "观察",
        "action": "动作",
        "algorithm": "算法",
        "status": "状态",
        "ready_runtime": "就绪 · 已预装",
        "hint": "调整下方参数后启动训练。学习曲线和训练日志会持续更新。",
        "setup": "实验设置",
        "setup_copy": "设置每个 epoch 的环境步数和 epoch 数量。每个 epoch 结束时评估一次并保存一个策略。",
        "selected": "已选游戏",
        "steps_per_epoch": "每个 epoch 的环境步数",
        "steps_per_epoch_info": "每训练多少个环境交互步后评估并保存模型",
        "epochs": "训练 epochs / 保存模型数",
        "epochs_info": "此处一个 epoch 表示一段固定步数；每个 epoch 保存一个已评估策略",
        "advanced": "高级参数",
        "lr": "学习率",
        "gamma": "折扣因子 γ",
        "epsilon": "探索率 ε",
        "seed": "随机种子",
        "baseline_title": "推荐学习 Baseline",
        "baseline_badge": "默认配置",
        "baseline_budget": "总环境步数",
        "baseline_warmup": "经验回放预热",
        "baseline_exploration": "探索调度",
        "baseline_time": "典型耗时",
        "baseline_checkpoints": "Epoch 计划",
        "baseline_expected": "预期学习信号",
        "baseline_restore": "恢复推荐 Baseline",
        "saved_model": "Epoch 模型",
        "saved_model_info": "每个 epoch 结束时保存的已评估策略都会显示在这里",
        "saved_model_empty": "还没有 epoch 模型。开始训练后会保存第一个策略。",
        "preview_waiting": "尚未选择 epoch 模型。开始训练后，可在这里选择已保存的 epoch。",
        "start": "开始训练",
        "running_button": "训练中…",
        "run_status": "训练状态",
        "ready": "等待训练",
        "ready_detail": "选择游戏、阅读目标，然后启动训练",
        "running": "训练进行中",
        "complete": "训练完成",
        "failed": "训练停止",
        "metric": "最新评估",
        "metric_waiting": "训练开始后显示结果",
        "curve": "学习曲线",
        "curve_copy": "每个 epoch 结束时更新一次曲线；图表标记统一使用英文。",
        "log": "实时训练日志",
        "log_waiting": "等待训练任务…",
        "preview": "任务预览 / 学习后的策略",
        "preview_copy": "每个 epoch 结束时评估并保存一个策略。可在下方选择不同 epoch 比较真实行为；非最佳策略的回放会在第一次选择时生成并缓存。",
        "download": "下载运行摘要",
        "wait_title": "训练正在运行",
        "wait_detail": "正在准备环境、更新策略、执行评估并渲染学习结果。",
        "guide_title": "怎样判断本次训练结果",
        "guide_copy": "增加每个 epoch 的步数或 epoch 数量前，先看下面三个信号。",
        "guide_success": "怎样算训练成功",
        "guide_preview": "怎样查看 Preview",
        "guide_time": "大约需要多久",
        "guide_success_default": "“训练完成”表示流程已经正常结束；后期 epoch 模型的评估高于早期模型，并且学习后的行为符合任务目标，才说明策略确实学到了东西。",
        "guide_preview_default": "训练前，Preview 显示任务画面；训练中或训练后，它会切换为本次运行的实时帧、回放 GIF、策略图或结果图。",
        "guide_time_default": "默认配方通常需要 30 秒到 5 分钟；首次下载、编译或三维渲染可能额外增加数分钟。",
        "reference": "环境资料",
    },
}


def copy_for(language: str) -> dict[str, str]:
    return TEXT["中文" if language == "中文" else "English"]


def task_value(task: Any, key: str, default: Any = None) -> Any:
    if isinstance(task, dict):
        return task.get(key, default)
    return getattr(task, key, default)


def local_value(value: Any, language: str) -> str:
    if isinstance(value, dict):
        return str(value.get("zh" if language == "中文" else "en") or value.get("en") or next(iter(value.values())))
    return str(value)


def normalize_tasks(tasks: Iterable[Any]) -> list[Any]:
    result = list(tasks)
    if not result:
        raise ValueError("At least one game task is required")
    keys = [task_value(task, "key") for task in result]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("Every game task needs a unique non-empty key")
    return result


def get_task(tasks: list[Any], key: str) -> Any:
    return next((task for task in tasks if task_value(task, "key") == key), tasks[0])


def preview_path(root: Path, task: Any) -> str:
    path = Path(str(task_value(task, "preview")))
    if not path.is_absolute():
        path = root / path
    return str(path)


@lru_cache(maxsize=128)
def embedded_image(path_value: str) -> str:
    """Return a compact, proxy-independent task-detail image data URL."""
    path = Path(path_value)
    if path.suffix.lower() == ".svg":
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/svg+xml;base64,{encoded}"
    with Image.open(path) as source:
        frame_count = int(getattr(source, "n_frames", 1))
        if frame_count > 1:
            sample_count = min(frame_count, 18)
            sample_indices = sorted({
                round(index * (frame_count - 1) / (sample_count - 1))
                for index in range(sample_count)
            })
            candidates: list[tuple[tuple[float, float], Image.Image]] = []
            for frame_index in sample_indices:
                source.seek(frame_index)
                candidate = source.convert("RGB")
                grayscale = candidate.convert("L")
                score = (grayscale.entropy(), float(ImageStat.Stat(grayscale).var[0]))
                candidates.append((score, candidate))
            frame = max(candidates, key=lambda item: item[0])[1]
        else:
            source.seek(0)
            has_alpha = "A" in source.getbands() or "transparency" in source.info
            frame = source.convert("RGBA" if has_alpha else "RGB")
        frame.thumbnail((960, 640), Image.Resampling.LANCZOS)
    lossy = io.BytesIO()
    frame.save(lossy, format="WEBP", quality=88, method=4)
    lossless = io.BytesIO()
    frame.save(lossless, format="WEBP", lossless=True, method=4)
    payload = min(lossy.getvalue(), lossless.getvalue(), key=len)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


def hero_html(space: dict[str, Any], tasks: list[Any], task: Any, language: str, runtime_status: str) -> str:
    copy = copy_for(language)
    title = local_value(space["title"], language)
    description = local_value(space["description"], language)
    course = space.get("course_url", COURSE_URL)
    notebook = space.get("notebook_url")
    source = space.get("source_url", PROJECT_URL)
    project = space.get("project_url", PROJECT_URL)
    organization = space.get("organization_url", ORGANIZATION_URL)
    dataset = space.get("dataset_url")
    device = html.escape(str(space.get("device", "CPU")))
    badge = html.escape(str(space.get("badge", "CPU GAME LAB")))
    return f"""
    <main class="app-shell">
      <section class="hero">
        <div class="brand-lockup">
          <a href="{html.escape(organization)}" target="_blank" rel="noreferrer">WALKINGLAB</a>
          <span aria-hidden="true">×</span>
          <a href="{html.escape(project)}" target="_blank" rel="noreferrer">HANDS-ON MODERN RL</a>
        </div>
        <p class="brand-note">{html.escape(copy['brand_note'])}</p>
        <div class="hero-topline"><span class="experiment-badge">{badge}</span><span class="hero-course">{copy['course']}</span></div>
        <h1>{html.escape(title)}</h1>
        <p class="hero-copy">{html.escape(description)}</p>
        <nav class="hero-links">
          <a class="hero-link primary" href="{html.escape(project)}" target="_blank" rel="noreferrer">GitHub · walkinglabs/hands-on-modern-rl</a>
          <a class="hero-link" href="{html.escape(organization)}" target="_blank" rel="noreferrer">{copy['organization']}</a>
          {f'<a class="hero-link" href="{html.escape(str(dataset))}" target="_blank" rel="noreferrer">{copy["dataset"]}</a>' if dataset else ''}
          <a class="hero-link" href="{html.escape(course)}" target="_blank" rel="noreferrer">{copy['chapter']}</a>
          {f'<a class="hero-link" href="{html.escape(str(notebook))}" target="_blank" rel="noreferrer">{copy["notebook"]}</a>' if notebook else ''}
          <a class="hero-link" href="{html.escape(source)}" target="_blank" rel="noreferrer">{copy['source']}</a>
        </nav>
      </section>
      <section class="lab-strip">
        <span>{copy['tasks']} <strong>{len(tasks)}</strong></span>
        <span>{copy['device']} <strong>{device}</strong></span>
        <span>{copy['runtime']} <strong>{html.escape(runtime_status)}</strong></span>
        <span>Environment <strong>{html.escape(str(task_value(task, 'environment')))}</strong></span>
      </section>
    </main>
    """


def training_guide(space: dict[str, Any], language: str) -> str:
    copy = copy_for(language)
    guide = space.get("training_guide", {})
    success = local_value(guide.get("success", copy["guide_success_default"]), language)
    preview = local_value(guide.get("preview", copy["guide_preview_default"]), language)
    duration = local_value(guide.get("time", copy["guide_time_default"]), language)
    return f"""
    <section class="training-guide">
      <div class="training-guide__intro">
        <span class="task-kicker">RESULT CHECKLIST</span>
        <h3>{copy['guide_title']}</h3>
        <p>{copy['guide_copy']}</p>
      </div>
      <div class="training-guide__grid">
        <article><b>01</b><h4>{copy['guide_success']}</h4><p>{html.escape(success)}</p></article>
        <article><b>02</b><h4>{copy['guide_preview']}</h4><p>{html.escape(preview)}</p></article>
        <article><b>03</b><h4>{copy['guide_time']}</h4><p>{html.escape(duration)}</p></article>
      </div>
    </section>
    """


def task_brief(root: Path, task: Any, language: str, space: dict[str, Any]) -> str:
    copy = copy_for(language)
    preview = embedded_image(preview_path(root, task))
    reference_url = task_value(task, "reference_url")
    reference = (
        f'<a class="task-reference" href="{html.escape(str(reference_url))}" target="_blank" '
        f'rel="noreferrer">{html.escape(copy["reference"])} ↗</a>'
        if reference_url else ""
    )
    return f"""
    <section class="task-brief">
      <div class="task-brief__visual"><img src="{html.escape(preview)}" alt="{html.escape(str(task_value(task, 'environment')))}"></div>
      <div class="task-brief__body">
        <span class="task-kicker">{copy['understand']}</span>
        <h3>{html.escape(local_value(task_value(task, 'title'), language))}</h3>
        <p>{html.escape(local_value(task_value(task, 'description'), language))}</p>
        <div class="task-facts">
          <span><b>{copy['observation']}</b>{html.escape(local_value(task_value(task, 'observation'), language))}</span>
          <span><b>{copy['action']}</b>{html.escape(local_value(task_value(task, 'action'), language))}</span>
          <span><b>{copy['algorithm']}</b>{html.escape(str(task_value(task, 'algorithm')))}</span>
          <span><b>{copy['status']}</b>{copy['ready_runtime']}</span>
        </div>
        <p class="task-hint">{copy['hint']}</p>{reference}
      </div>
    </section>{training_guide(space, language)}
    """


def panel_html(title: str, text: str, cls: str = "panel-copy") -> str:
    return f'<h2 class="panel-title">{html.escape(title)}</h2><p class="{cls}">{html.escape(text)}</p>'


def epoch_specs(task: Any) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Return explicit epoch controls or derive them from a legacy total budget."""
    explicit_steps = task_value(task, "steps_per_epoch")
    explicit_epochs = task_value(task, "epochs")
    if explicit_steps is not None and explicit_epochs is not None:
        return (
            tuple(float(value) for value in explicit_steps),
            tuple(float(value) for value in explicit_epochs),
        )
    budget_min, budget_max, budget_default, budget_step = slider_spec(
        task, "budget", (100, 10_000, 1_000, 100)
    )
    epoch_default = max(2, min(12, int(task_value(task, "checkpoints", 6))))
    step = max(1.0, budget_step)

    def aligned(value: float) -> float:
        return max(step, round(max(step, value) / step) * step)

    # The restored preset must never be shorter than the declared learning
    # baseline; rounding down made the default display its own smoke warning.
    steps_default = max(step, math.ceil(max(step, budget_default / epoch_default) / step) * step)
    steps_min = min(steps_default, aligned(budget_min / epoch_default))
    steps_max = max(steps_default, aligned(budget_max))
    return (
        (steps_min, steps_max, steps_default, step),
        (1.0, 12.0, float(epoch_default), 1.0),
    )


def training_unit(task: Any, language: str) -> str:
    value = task_value(
        task,
        "training_unit",
        {"en": "environment steps", "zh": "环境步"},
    )
    return local_value(value, language)


def epoch_control_copy(task: Any, language: str) -> tuple[str, str]:
    unit = training_unit(task, language)
    if language == "中文":
        return f"每个 epoch 的{unit}", f"每完成固定数量的{unit}，执行评估、保存模型并生成回放"
    return f"{unit[:1].upper() + unit[1:]} per epoch", f"One fixed block of {unit} before evaluation, model saving, and replay generation"


def baseline_card(task: Any, language: str) -> str:
    copy = copy_for(language)
    step_spec, epoch_spec = epoch_specs(task)
    steps_per_epoch = int(step_spec[2])
    epochs = int(epoch_spec[2])
    budget = steps_per_epoch * epochs
    unit = training_unit(task, language)
    checkpoint_schedule = (
        f"{epochs} epochs × {steps_per_epoch:,} {unit} · one saved model per epoch"
        if language != "中文"
        else f"{epochs} 个 epochs × 每个 {steps_per_epoch:,} {unit} · 每个 epoch 保存一个模型"
    )
    duration = local_value(task_value(task, "baseline_time", {"en": "runtime depends on the selected environment", "zh": "运行时间取决于所选环境"}), language)
    outcome = local_value(task_value(task, "baseline_outcome", {"en": "Evaluation should improve over early checkpoints.", "zh": "评估结果应高于早期检查点。"}), language)
    algorithm = str(task_value(task, "algorithm", "RL"))
    learning_rate = slider_spec(task, "learning_rate", (1e-5, .1, 3e-4, 1e-5))[2]
    name = str(task_value(task, "baseline_name", f"{algorithm} recommended baseline"))
    return f"""
    <section class="baseline-preset">
      <div class="baseline-preset__head">
        <div><span>{html.escape(copy['baseline_badge'])}</span><h3>{html.escape(copy['baseline_title'])}</h3></div>
        <strong>{html.escape(name)}</strong>
      </div>
      <div class="baseline-preset__facts">
        <span><b>{html.escape(copy['baseline_budget'])}</b>{budget:,} {html.escape(unit)}</span>
        <span><b>{html.escape(copy['algorithm'])}</b>{html.escape(algorithm)}</span>
        <span><b>{html.escape(copy['lr'])}</b>{learning_rate:g}</span>
        <span><b>{html.escape(copy['baseline_time'])}</b>{html.escape(duration)}</span>
      </div>
      <p><b>{html.escape(copy['baseline_checkpoints'])}</b>{html.escape(checkpoint_schedule)}</p>
      <p><b>{html.escape(copy['baseline_expected'])}</b>{html.escape(outcome)}</p>
    </section>
    """


def checkpoint_plan(task: Any, language: str, steps_per_epoch: float, epochs: float) -> str:
    epoch_steps = max(1, int(steps_per_epoch))
    epoch_count = max(1, min(12, int(epochs)))
    total_steps = epoch_steps * epoch_count
    unit = training_unit(task, language)
    steps = [epoch_steps * index for index in range(1, epoch_count + 1)]
    rendered_steps = " · ".join(f"{step:,}" for step in steps)
    recommended = int(slider_spec(task, "budget", (100, 10_000, 1_000, 100))[2])
    mode = (
        "Baseline-length run" if total_steps >= recommended else "Short diagnostic run"
    ) if language != "中文" else (
        "Baseline 长度训练" if total_steps >= recommended else "短流程验证"
    )
    summary = (
        f"{epoch_steps:,} {unit}/epoch × {epoch_count} epochs = {total_steps:,} total {unit} = {epoch_count} saved models"
        if language != "中文"
        else f"每个 epoch {epoch_steps:,} {unit} × {epoch_count} 个 epochs = 总计 {total_steps:,} {unit} = {epoch_count} 个保存模型"
    )
    label = "Models saved at" if language != "中文" else "模型保存位置"
    definition = (
        f"Here, an epoch is one fixed block of {unit} followed by evaluation and model saving."
        if language != "中文"
        else f"此处的 epoch 是固定数量的{unit}，结束后执行评估并保存模型。"
    )
    return f"""
    <section class="checkpoint-plan">
      <span>{html.escape(mode)}</span>
      <strong>{html.escape(summary)}</strong>
      <p><b>{html.escape(label)}</b>{html.escape(rendered_steps)}</p>
      <p>{html.escape(definition)}</p>
    </section>
    """


def preview_provenance(task: Any, language: str, model: str | None = None, detail: str | None = None) -> str:
    copy = copy_for(language)
    title = local_value(task_value(task, "title"), language)
    algorithm = str(task_value(task, "algorithm"))
    if model:
        message = detail or (
            f"Selected policy: {title} · {algorithm} · {model}"
            if language != "中文"
            else f"已选策略：{title} · {algorithm} · {model}"
        )
        state = "ready"
    else:
        message = detail or copy["preview_waiting"]
        state = "waiting"
    return (
        f'<div class="preview-provenance preview-provenance--{state}">'
        f'<span class="preview-provenance__dot"></span><span>{html.escape(message)}</span></div>'
    )


def saved_model_choices(records: list[dict], language: str) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    total = len(records)
    for index, record in enumerate(records):
        run_id = str(record.get("run_id") or "")
        checkpoint_index = int(record.get("checkpoint_index") or 0)
        checkpoint_count = int(record.get("checkpoint_count") or 0)
        if run_id and checkpoint_index:
            prefix = (
                f"Run …{run_id[-6:]} · Epoch {checkpoint_index}/{checkpoint_count}"
                if language != "中文"
                else f"运行 …{run_id[-6:]} · Epoch {checkpoint_index}/{checkpoint_count}"
            )
        else:
            ordinal = total - index
            prefix = f"Legacy model {ordinal}" if language != "中文" else f"旧模型 {ordinal}"
        title = local_value(record.get("title", record.get("task_key", "Atari")), language)
        details: list[str] = [title]
        trained_steps = int(record.get("training_step") or record.get("budget") or 0)
        total_budget = int(record.get("total_budget") or 0)
        unit = local_value(record.get("training_unit", {"en": "steps", "zh": "步"}), language)
        if trained_steps:
            details.append(f"{trained_steps:,}/{total_budget:,} {unit}" if total_budget else f"{trained_steps:,} {unit}")
        score = record.get("score")
        if score is not None:
            details.append(f"score {float(score):.2f}")
        if record.get("is_best"):
            details.append("BEST" if language != "中文" else "最佳")
        elif record.get("run_complete") is False:
            details.append("PARTIAL RUN" if language != "中文" else "未完成运行")
        created = str(record.get("created_at") or "").replace("T", " ").replace("+00:00", " UTC")
        if created:
            details.append(created[:16] + (" UTC" if "UTC" in created else ""))
        choices.append((f"{prefix} · " + " · ".join(details), str(record["model_id"])))
    return choices


def status_card(state: str, title: str, detail: str, language: str) -> str:
    return f"""
    <div class="run-state run-state--{state}">
      <span class="run-state__dot"></span>
      <div><span class="summary-label">{copy_for(language)['run_status']}</span><strong>{html.escape(title)}</strong><small>{html.escape(detail)}</small></div>
    </div>
    """


def metric_card(value: str, detail: str, language: str) -> str:
    return f"""
    <div class="live-metric">
      <span class="summary-label">{copy_for(language)['metric']}</span>
      <div class="metric-reading"><strong>{html.escape(value)}</strong><small>{html.escape(detail)}</small></div>
    </div>
    """


def console_panel(logs: str, language: str) -> str:
    return f"""
    <section class="console-panel" aria-live="polite">
      <div class="console-head"><span class="console-dot"></span>{copy_for(language)['log']}</div>
      <pre class="console-text">{html.escape(logs)}</pre>
    </section>
    """


def waiting_panel(language: str) -> str:
    copy = copy_for(language)
    elapsed_label = "elapsed" if language == "English" else "已等待"
    return f"""
    <section class="run-wait" role="status" aria-live="polite">
      <span class="run-wait__spinner" aria-hidden="true"></span>
      <div><strong>{copy['wait_title']}</strong><small>{copy['wait_detail']}</small>
      <em class="run-wait__elapsed" data-start-ms="{int(time.time() * 1000)}" data-label="{elapsed_label}">0s {elapsed_label}</em></div>
    </section>
    """


def learning_figure(x: list[float], y: list[float], title: str, ylabel: str = "Evaluation score"):
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(x, y, color="#5b5ce2", linewidth=2.2)
    if x:
        ax.scatter([x[-1]], [y[-1]], color="#15a873", s=36, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Training progress")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    # Gradio serializes the Figure object after this function returns.  Closing
    # it only unregisters the pyplot manager, so repeated checkpoints do not
    # retain every historical chart in the long-running Studio process.
    plt.close(fig)
    return fig


def result_image(root: Path, task: Any, status: str, score: float | None, x: list[float], y: list[float], note: str) -> str:
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    key = str(task_value(task, "key"))
    path = artifact_dir / f"{key}-result.png"
    fig = plt.figure(figsize=(9.6, 5.0), facecolor="#f7f8fc")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.8], wspace=.3)
    info = fig.add_subplot(grid[0, 0])
    plot = fig.add_subplot(grid[0, 1])
    info.set_facecolor("#20245b")
    info.set_xticks([]); info.set_yticks([])
    for spine in info.spines.values():
        spine.set_visible(False)
    info.text(.09, .86, status.upper(), color="#a5b4fc", fontsize=10, fontweight="bold", transform=info.transAxes)
    info.text(.09, .67, "—" if score is None else f"{score:.2f}", color="white", fontsize=28, fontweight="bold", transform=info.transAxes)
    info.text(.09, .57, "Final evaluation score", color="#cbd5e1", fontsize=9, transform=info.transAxes)
    info.text(.09, .39, str(task_value(task, "environment")), color="white", fontsize=12, fontweight="bold", wrap=True, transform=info.transAxes)
    info.text(.09, .24, str(task_value(task, "algorithm")), color="#cbd5e1", fontsize=10, transform=info.transAxes)
    info.text(.09, .07, note[:180], color="#aeb7ca", fontsize=8.5, wrap=True, transform=info.transAxes)
    if x and y:
        plot.plot(x, y, color="#5b5ce2", linewidth=2.3)
        plot.scatter([x[-1]], [y[-1]], color="#13a36f", s=44, zorder=3)
        plot.set_xlabel("Training progress")
        plot.set_ylabel("Evaluation score")
        plot.grid(alpha=.2)
        plot.set_title("Learned policy result", loc="left", fontweight="bold")
    else:
        plot.axis("off")
        plot.text(.5, .55, "RESULT", ha="center", color="#5b5ce2", fontsize=22, fontweight="bold")
        plot.text(.5, .4, note[:220], ha="center", color="#68748a", fontsize=10, wrap=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def save_summary(root: Path, task: Any, payload: dict[str, Any]) -> str:
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{task_value(task, 'key')}-run-summary.json"
    if is_dataclass(task):
        task_payload = asdict(task)
    elif isinstance(task, dict):
        task_payload = task
    else:
        task_payload = {name: getattr(task, name) for name in dir(task) if not name.startswith("_") and not callable(getattr(task, name))}
    data = {"task": task_payload, **payload}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return str(path)


def local_model_index(root: Path) -> Path:
    return root / "artifacts" / "saved-models.json"


def load_local_models(root: Path, task_key: str | None = None) -> list[dict[str, Any]]:
    path = local_model_index(root)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("models", [])
    except (OSError, ValueError, TypeError):
        return []
    valid = [record for record in records if isinstance(record, dict) and record.get("model_id")]
    if task_key is not None:
        valid = [record for record in valid if str(record.get("task_key")) == task_key]
    return sorted(valid, key=lambda record: str(record.get("created_at") or ""), reverse=True)


def write_local_models(root: Path, records: list[dict[str, Any]]) -> None:
    path = local_model_index(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"models": records}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def register_local_model(
    root: Path,
    task: Any,
    event: Any,
    run_id: str,
    total_budget: int,
    checkpoint_count: int,
) -> dict[str, Any] | None:
    model = event_value(event, "model")
    if not model:
        return None
    checkpoint_index = int(event_value(event, "checkpoint_index", 0) or 0)
    training_step = int(event_value(event, "step", 0) or 0)
    model_id = str(event_value(event, "model_id") or f"{run_id}-epoch-{max(1, checkpoint_index):02d}")
    preview = event_value(event, "preview")
    record = {
        "model_id": model_id,
        "model": str(model),
        "preview": str(preview) if preview else None,
        "task_key": str(task_value(task, "key")),
        "title": task_value(task, "title"),
        "algorithm": str(task_value(task, "algorithm")),
        "training_unit": task_value(task, "training_unit", {"en": "steps", "zh": "步"}),
        "run_id": run_id,
        "checkpoint_index": checkpoint_index,
        "checkpoint_count": checkpoint_count,
        "training_step": training_step,
        "total_budget": int(total_budget),
        "score": event_value(event, "score"),
        "is_best": False,
        "run_complete": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    }
    records = load_local_models(root)
    records = [existing for existing in records if str(existing.get("model_id")) != model_id]
    records.insert(0, record)
    write_local_models(root, records)
    return record


def finish_local_run(root: Path, run_id: str, complete: bool) -> None:
    records = load_local_models(root)
    current = [record for record in records if str(record.get("run_id")) == run_id]
    best_id = None
    scored = [record for record in current if record.get("score") is not None]
    if scored:
        best_id = str(max(scored, key=lambda record: float(record["score"]))["model_id"])
    for record in records:
        if str(record.get("run_id")) == run_id:
            record["run_complete"] = bool(complete)
            record["is_best"] = str(record.get("model_id")) == best_id
    write_local_models(root, records)


def local_model_details(root: Path, model_id: str) -> dict[str, Any]:
    record = next((item for item in load_local_models(root) if str(item.get("model_id")) == model_id), None)
    if record is None:
        raise ValueError("Unknown saved-model identifier")
    return record


def event_value(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def slider_spec(task: Any, name: str, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    raw = task_value(task, name, fallback)
    return tuple(float(value) for value in raw)


def slider_update(label: str, spec: tuple[float, float, float, float], info: str | None = None):
    return gr.Slider(minimum=spec[0], maximum=spec[1], value=spec[2], step=spec[3], label=label, info=info)


def build_demo(space_module: Any):
    root = Path(space_module.__file__).resolve().parent
    tasks = normalize_tasks(space_module.TASKS)
    space = dict(space_module.SPACE)
    runtime_status = str(space_module.runtime_status())
    default_language = DEFAULT_LANGUAGE
    default_task = tasks[0]

    def gallery_items(language: str):
        return [
            (preview_path(root, task), f"{local_value(task_value(task, 'title'), language)}\n{task_value(task, 'algorithm')}")
            for task in tasks
        ]

    def model_state(key: str, language: str, preferred: str | None = None):
        records = (
            list(space_module.list_trained_models(key))
            if hasattr(space_module, "list_trained_models")
            else load_local_models(root, key)
        )
        choices = saved_model_choices(records, language)
        available = {str(record["model_id"]): record for record in records}
        default_record = records[0] if records else None
        if default_record and default_record.get("run_id"):
            latest_run_id = str(default_record["run_id"])
            default_record = next(
                (
                    record
                    for record in records
                    if str(record.get("run_id")) == latest_run_id and record.get("is_best")
                ),
                default_record,
            )
        selected_model = preferred if preferred in available else (str(default_record["model_id"]) if default_record else None)
        selected_record = available.get(str(selected_model)) if selected_model else None
        return records, choices, selected_model, selected_record

    def model_dropdown(key: str, language: str, preferred: str | None = None, interactive: bool = True):
        _, choices, selected_model, selected_record = model_state(key, language, preferred)
        copy = copy_for(language)
        return (
            gr.Dropdown(
                choices=choices,
                value=selected_model,
                label=copy["saved_model"],
                info=copy["saved_model_info"] if choices else copy["saved_model_empty"],
                interactive=interactive and bool(choices),
                allow_custom_value=True,
                elem_classes="model-selector",
            ),
            selected_record,
        )

    def choose_task(language: str, seed: float, event: gr.SelectData):
        index = max(0, min(int(event.index), len(tasks) - 1))
        task = tasks[index]
        copy = copy_for(language)
        epoch_steps, epoch_count = epoch_specs(task)
        epoch_label, epoch_info = epoch_control_copy(task, language)
        lr = slider_spec(task, "learning_rate", (1e-5, .1, 3e-4, 1e-5))
        gamma = slider_spec(task, "gamma", (0, 1, .99, .01))
        epsilon = slider_spec(task, "epsilon", (0, 1, .1, .01))
        selector, selected_record = model_dropdown(str(task_value(task, "key")), language)
        selected_preview = str(selected_record.get("preview")) if selected_record and selected_record.get("preview") else preview_path(root, task)
        selected_model = str(selected_record["model_id"]) if selected_record else None
        return (
            task_value(task, "key"),
            hero_html(space, tasks, task, language, runtime_status),
            task_brief(root, task, language, space),
            baseline_card(task, language),
            gr.Slider(minimum=epoch_steps[0], maximum=epoch_steps[1], value=epoch_steps[2], step=epoch_steps[3], label=epoch_label, info=epoch_info),
            gr.Slider(minimum=epoch_count[0], maximum=epoch_count[1], value=epoch_count[2], step=epoch_count[3], label=copy["epochs"], info=copy["epochs_info"]),
            checkpoint_plan(task, language, epoch_steps[2], epoch_count[2]),
            gr.Slider(minimum=lr[0], maximum=lr[1], value=lr[2], step=lr[3], label=copy["lr"]),
            gr.Slider(minimum=gamma[0], maximum=gamma[1], value=gamma[2], step=gamma[3], label=copy["gamma"]),
            gr.Slider(minimum=epsilon[0], maximum=epsilon[1], value=epsilon[2], step=epsilon[3], label=copy["epsilon"]),
            status_card("idle", copy["ready"], copy["ready_detail"], language),
            metric_card("—", copy["metric_waiting"], language),
            console_panel(copy["log_waiting"], language),
            selected_preview,
            gr.File(value=None, label=copy["download"], visible=False),
            selector,
            preview_provenance(task, language, selected_model),
        )

    def switch_language(language: str, key: str, seed: float, selected_model: str | None, steps_per_epoch: float, epochs: float):
        task = get_task(tasks, key)
        copy = copy_for(language)
        epoch_steps_spec, epoch_count_spec = epoch_specs(task)
        epoch_label, epoch_info = epoch_control_copy(task, language)
        selector, selected_record = model_dropdown(key, language, selected_model)
        selected_model = str(selected_record["model_id"]) if selected_record else None
        return (
            hero_html(space, tasks, task, language, runtime_status),
            panel_html(copy["choose"], copy["choose_copy"]),
            gr.Gallery(value=gallery_items(language)),
            task_brief(root, task, language, space),
            panel_html(copy["setup"], copy["setup_copy"]),
            baseline_card(task, language),
            gr.Slider(minimum=epoch_steps_spec[0], maximum=epoch_steps_spec[1], value=steps_per_epoch, step=epoch_steps_spec[3], label=epoch_label, info=epoch_info),
            gr.Slider(minimum=epoch_count_spec[0], maximum=epoch_count_spec[1], value=epochs, step=epoch_count_spec[3], label=copy["epochs"], info=copy["epochs_info"]),
            checkpoint_plan(task, language, steps_per_epoch, epochs),
            gr.Accordion(label=copy["advanced"], open=False),
            gr.Textbox(value=key, label=copy["selected"]),
            gr.Number(value=seed, label=copy["seed"], precision=0),
            gr.Button(value=copy["baseline_restore"]),
            gr.Button(value=copy["start"]),
            status_card("idle", copy["ready"], copy["ready_detail"], language),
            metric_card("—", copy["metric_waiting"], language),
            panel_html(copy["curve"], copy["curve_copy"]),
            console_panel(copy["log_waiting"], language),
            panel_html(copy["preview"], copy["preview_copy"], "artifact-note"),
            gr.File(label=copy["download"], visible=False),
            selector,
            preview_provenance(task, language, selected_model),
        )

    def restore_baseline(key: str, language: str):
        task = get_task(tasks, key)
        copy = copy_for(language)
        epoch_steps, epoch_count = epoch_specs(task)
        epoch_label, epoch_info = epoch_control_copy(task, language)
        lr = slider_spec(task, "learning_rate", (1e-5, .1, 3e-4, 1e-5))
        gamma = slider_spec(task, "gamma", (0, 1, .99, .01))
        epsilon = slider_spec(task, "epsilon", (0, 1, .1, .01))
        return (
            gr.Slider(minimum=epoch_steps[0], maximum=epoch_steps[1], value=epoch_steps[2], step=epoch_steps[3], label=epoch_label, info=epoch_info),
            gr.Slider(minimum=epoch_count[0], maximum=epoch_count[1], value=epoch_count[2], step=epoch_count[3], label=copy["epochs"], info=copy["epochs_info"]),
            checkpoint_plan(task, language, epoch_steps[2], epoch_count[2]),
            slider_update(copy["lr"], lr),
            slider_update(copy["gamma"], gamma),
            slider_update(copy["epsilon"], epsilon),
            baseline_card(task, language),
        )

    def update_checkpoint_plan(key: str, language: str, steps_per_epoch: float, epochs: float):
        return checkpoint_plan(get_task(tasks, key), language, steps_per_epoch, epochs)

    def train_with_ui(key: str, steps_per_epoch: float, epochs: float, learning_rate: float, gamma: float, epsilon: float, seed: float, selected_model: str | None, language: str):
        task = get_task(tasks, key)
        copy = copy_for(language)
        resolved_epoch_steps = max(1, int(steps_per_epoch))
        resolved_epochs = max(1, min(12, int(epochs)))
        params = {
            "budget": resolved_epoch_steps * resolved_epochs,
            "checkpoints": resolved_epochs,
            "learning_rate": float(learning_rate),
            "gamma": float(gamma),
            "epsilon": float(epsilon),
            "seed": int(seed),
        }
        device = str(space.get("device", "CPU"))
        logs = [f"0.0s  CONFIG  environment={task_value(task, 'environment')} algorithm={task_value(task, 'algorithm')} device={device}"]
        logs.append(f"0.0s  CONFIG  total_budget={params['budget']} seed={params['seed']}")
        logs.append(f"0.0s  CONFIG  epochs={resolved_epochs} steps_per_epoch={resolved_epoch_steps:,} save_policy=every_epoch")
        recommended_budget = int(slider_spec(task, "budget", (100, 10_000, 1_000, 100))[2])
        if params["budget"] < recommended_budget:
            logs.append(
                f"0.0s  WARNING budget is below the recommended {recommended_budget:,}-step learning baseline; "
                "this run is a smoke test and may not produce visible learned behavior"
            )
        else:
            logs.append(f"0.0s  BASELINE {task_value(task, 'baseline_name', 'recommended')} selected")
        started = time.perf_counter()
        run_id = f"{int(time.time())}-{time.time_ns() % 1_000_000:06d}"
        runtime_owns_models = hasattr(space_module, "list_trained_models")
        run_signature = inspect.signature(space_module.run)
        accepts_extra = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in run_signature.parameters.values())
        run_params = {
            name: value
            for name, value in params.items()
            if accepts_extra or name in run_signature.parameters
        }
        last_x: list[float] = []
        last_y: list[float] = []
        last_score: float | None = None
        last_model: str | None = None
        saved_models: list[str] = []
        preview = preview_path(root, task)
        wait = waiting_panel(language)
        selector, selected_record = model_dropdown(key, language, selected_model, interactive=False)
        selected_model = str(selected_record["model_id"]) if selected_record else None
        yield (
            status_card("running", copy["running"], "Environment initialization", language),
            metric_card("—", "Preparing runtime", language),
            None,
            preview,
            gr.File(value=None, label=copy["download"], visible=False),
            console_panel("\n".join(logs), language),
            gr.HTML(value=wait, visible=True),
            gr.Button(value=copy["running_button"], interactive=False),
            selector,
            preview_provenance(task, language, detail=("Training a new policy…" if language != "中文" else "正在训练一个新策略……")),
        )
        try:
            for event in space_module.run(key, **run_params):
                message = str(event_value(event, "log", "")).strip()
                if message:
                    if not message.lstrip().startswith(tuple("0123456789")):
                        message = f"{time.perf_counter() - started:7.1f}s  TRAIN   {message}"
                    logs.extend(message.splitlines())
                    if len(logs) > 1_200:
                        logs = logs[:2] + ["... older log lines omitted from the live view ..."] + logs[-1_197:]
                x = event_value(event, "x", last_x)
                y = event_value(event, "y", last_y)
                if x is not None and y is not None:
                    last_x = [float(value) for value in x]
                    last_y = [float(value) for value in y]
                score = event_value(event, "score", last_score)
                if score is not None and math.isfinite(float(score)):
                    last_score = float(score)
                event_preview = event_value(event, "preview")
                if event_preview is not None:
                    preview = event_preview if isinstance(event_preview, (np.ndarray, Image.Image)) else str(event_preview)
                if bool(event_value(event, "preview_only", False)):
                    # High-frequency live frames should update only the image;
                    # rebuilding plots, logs, selectors, and status cards causes
                    # visible stutter in remote 3D environments.
                    yield (
                        gr.skip(), gr.skip(), gr.skip(), preview, gr.skip(),
                        gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                    )
                    continue
                checkpoint_index = int(event_value(event, "checkpoint_index", 0) or 0)
                checkpoint_count = int(event_value(event, "checkpoint_count", resolved_epochs) or resolved_epochs)
                event_model = event_value(event, "model")
                if event_model:
                    if runtime_owns_models:
                        last_model = str(event_value(event, "model_id") or Path(str(event_model)).name)
                    else:
                        record = register_local_model(root, task, event, run_id, params["budget"], checkpoint_count)
                        last_model = str(record["model_id"]) if record else None
                    if last_model not in saved_models:
                        saved_models.append(last_model)
                    selector, selected_record = model_dropdown(key, language, last_model, interactive=False)
                    selected_model = str(selected_record["model_id"]) if selected_record else last_model
                checkpoint_note = (
                    (
                        f"Epoch {checkpoint_index}/{checkpoint_count} model saved · available after this run finishes"
                        if language != "中文"
                        else f"Epoch {checkpoint_index}/{checkpoint_count} 模型已保存 · 本次训练结束后可选择回放"
                    )
                    if checkpoint_index
                    else ("Training a new policy…" if language != "中文" else "正在训练一个新策略……")
                )
                phase = str(event_value(event, "phase", "training"))
                detail = str(event_value(event, "detail", f"{int(event_value(event, 'step', 0)):,}/{params['budget']:,}"))
                metric_detail = str(event_value(event, "metric_detail", "Mean evaluation score"))
                curve = learning_figure(last_x, last_y, f"{task_value(task, 'environment')} · {task_value(task, 'algorithm')}") if last_x else None
                yield (
                    status_card("running", copy["running"], detail, language),
                    metric_card("—" if last_score is None else f"{last_score:.2f}", metric_detail, language),
                    curve,
                    preview,
                    gr.File(value=None, label=copy["download"], visible=False),
                    console_panel("\n".join(logs), language),
                    gr.HTML(value=wait, visible=True),
                    gr.Button(value=copy["running_button"], interactive=False),
                    selector,
                    preview_provenance(task, language, selected_model, detail=checkpoint_note),
                )
            if isinstance(preview, (str, Path)) and str(preview) == preview_path(root, task):
                preview = result_image(root, task, "training complete", last_score, last_x, last_y, "The environment did not expose replay frames; this plot records the learned result.")
            summary = save_summary(root, task, {
                "status": "complete",
                "parameters": {**params, "steps_per_epoch": resolved_epoch_steps, "epochs": resolved_epochs},
                "score": last_score,
                "curve": {"x": last_x, "y": last_y},
                "preview": preview,
                "model": last_model,
                "models": saved_models,
                "logs": logs,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            })
            logs.append(f"{time.perf_counter() - started:7.1f}s  DONE    training, evaluation, and visualization complete")
            curve = learning_figure(last_x, last_y, f"{task_value(task, 'environment')} · {task_value(task, 'algorithm')}") if last_x else None
            if not runtime_owns_models:
                finish_local_run(root, run_id, True)
            selector, selected_record = model_dropdown(key, language, last_model)
            selected_model = str(selected_record["model_id"]) if selected_record else last_model
            yield (
                status_card("complete", copy["complete"], f"{time.perf_counter() - started:.1f}s elapsed", language),
                metric_card("—" if last_score is None else f"{last_score:.2f}", "Final evaluation score", language),
                curve,
                preview,
                gr.File(value=summary, label=copy["download"], visible=True),
                console_panel("\n".join(logs), language),
                gr.HTML(value="", visible=False),
                gr.Button(value=copy["start"], interactive=True),
                selector,
                preview_provenance(task, language, selected_model),
            )
        except Exception as exc:
            logs.append(f"{time.perf_counter() - started:7.1f}s  ERROR   {type(exc).__name__}: {exc}")
            logs.extend(traceback.format_exc(limit=8).splitlines())
            preview = result_image(root, task, "run stopped", last_score, last_x, last_y, f"{type(exc).__name__}: {exc}")
            summary = save_summary(root, task, {
                "status": "failed",
                "parameters": {**params, "steps_per_epoch": resolved_epoch_steps, "epochs": resolved_epochs},
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "logs": logs,
            })
            if not runtime_owns_models:
                finish_local_run(root, run_id, False)
            selector, selected_record = model_dropdown(key, language, selected_model)
            selected_model = str(selected_record["model_id"]) if selected_record else None
            yield (
                status_card("error", copy["failed"], str(exc), language),
                metric_card("—" if last_score is None else f"{last_score:.2f}", "Last valid evaluation", language),
                learning_figure(last_x, last_y, f"{task_value(task, 'environment')} · stopped") if last_x else None,
                preview,
                gr.File(value=summary, label=copy["download"], visible=True),
                console_panel("\n".join(logs), language),
                gr.HTML(value="", visible=False),
                gr.Button(value=copy["start"], interactive=True),
                selector,
                preview_provenance(task, language, selected_model, detail=(f"Training stopped: {type(exc).__name__}: {exc}" if language != "中文" else f"训练停止：{type(exc).__name__}: {exc}")),
            )

    def select_saved_model(key: str, model_id: str | None, language: str):
        task = get_task(tasks, key)
        if not model_id:
            return preview_path(root, task), preview_provenance(task, language)
        try:
            record = (
                space_module.model_details(str(model_id))
                if hasattr(space_module, "model_details")
                else local_model_details(root, str(model_id))
            )
            if str(record.get("task_key")) != key:
                raise ValueError("The selected model belongs to a different task")
            selected_preview = record.get("preview")
            if not selected_preview or not Path(str(selected_preview)).is_file():
                if hasattr(space_module, "render_preview"):
                    result = space_module.render_preview(str(model_id))
                    selected_preview = event_value(result, "preview")
                elif hasattr(space_module, "render_saved_model"):
                    result = space_module.render_saved_model(key, str(record.get("model")), str(model_id))
                    selected_preview = event_value(result, "preview", result)
                else:
                    raise RuntimeError("this runtime did not provide a replay for the saved epoch")
            budget = int(record.get("training_step") or record.get("budget") or 0)
            total_budget = int(record.get("total_budget") or 0)
            score = record.get("score")
            facts = [str(model_id)]
            unit = training_unit(task, language)
            if budget:
                facts.append(f"{budget:,}/{total_budget:,} {unit}" if total_budget else f"{budget:,} {unit}")
            if score is not None:
                facts.append(f"score {float(score):.2f}")
            detail = (
                "Selected saved policy: " + " · ".join(facts)
                if language != "中文"
                else "已选择保存的策略：" + " · ".join(facts)
            )
            return str(selected_preview), preview_provenance(task, language, str(model_id), detail)
        except Exception as exc:
            message = (
                f"Saved-model preview unavailable: {exc}"
                if language != "中文"
                else f"暂时无法显示该模型的回放：{exc}"
            )
            return gr.skip(), preview_provenance(task, language, detail=message)

    copy = copy_for(default_language)
    initial_epoch_steps, initial_epochs = epoch_specs(default_task)
    initial_epoch_label, initial_epoch_info = epoch_control_copy(default_task, default_language)
    initial_lr = slider_spec(default_task, "learning_rate", (1e-5, .1, 3e-4, 1e-5))
    initial_gamma = slider_spec(default_task, "gamma", (0, 1, .99, .01))
    initial_epsilon = slider_spec(default_task, "epsilon", (0, 1, .1, .01))
    _, initial_model_choices, initial_model, initial_record = model_state(str(task_value(default_task, "key")), default_language)
    initial_preview = str(initial_record.get("preview")) if initial_record and initial_record.get("preview") else preview_path(root, default_task)

    with gr.Blocks(title=f"Hands-On Modern RL · {local_value(space['title'], 'English')}") as demo:
        with gr.Column(elem_classes="hero-stack"):
            hero = gr.HTML(hero_html(space, tasks, default_task, default_language, runtime_status))
            with gr.Row(elem_classes="language-bar"):
                language = gr.Radio(choices=["English", "中文"], value=default_language, show_label=False, elem_classes="language-switch")

        with gr.Column(elem_classes="catalog-card"):
            catalog_header = gr.HTML(panel_html(copy["choose"], copy["choose_copy"]))
            gallery = gr.Gallery(value=gallery_items(default_language), show_label=False, columns=3, object_fit="cover", height="auto", allow_preview=False, buttons=[], elem_classes="experiment-gallery")

        task_info = gr.HTML(task_brief(root, default_task, default_language, space), elem_id="selected-task-detail")

        with gr.Row(elem_classes="training-layout"):
            with gr.Column(scale=1, min_width=310, elem_classes="control-card"):
                settings_header = gr.HTML(panel_html(copy["setup"], copy["setup_copy"]))
                baseline_info = gr.HTML(baseline_card(default_task, default_language))
                selected = gr.Textbox(value=task_value(default_task, "key"), label=copy["selected"], interactive=False, elem_classes="selected-experiment")
                steps_per_epoch = gr.Slider(minimum=initial_epoch_steps[0], maximum=initial_epoch_steps[1], value=initial_epoch_steps[2], step=initial_epoch_steps[3], label=initial_epoch_label, info=initial_epoch_info)
                epochs = gr.Slider(minimum=initial_epochs[0], maximum=initial_epochs[1], value=initial_epochs[2], step=initial_epochs[3], label=copy["epochs"], info=copy["epochs_info"])
                checkpoint_info = gr.HTML(checkpoint_plan(default_task, default_language, initial_epoch_steps[2], initial_epochs[2]))
                with gr.Accordion(copy["advanced"], open=False, elem_classes="advanced-settings") as advanced:
                    learning_rate = slider_update(copy["lr"], initial_lr)
                    gamma = slider_update(copy["gamma"], initial_gamma)
                    epsilon = slider_update(copy["epsilon"], initial_epsilon)
                    seed = gr.Number(value=42, precision=0, label=copy["seed"])
                restore = gr.Button(copy["baseline_restore"], elem_classes="baseline-restore")
                start = gr.Button(copy["start"], variant="primary", elem_classes="primary-btn")
            with gr.Column(scale=2, elem_classes="results-stack"):
                with gr.Column(elem_classes="chart-card"):
                    chart_header = gr.HTML(panel_html(copy["curve"], copy["curve_copy"]))
                    with gr.Row(elem_classes="result-summary"):
                        status = gr.HTML(status_card("idle", copy["ready"], copy["ready_detail"], default_language))
                        metric = gr.HTML(metric_card("—", copy["metric_waiting"], default_language))
                    wait_state = gr.HTML(value="", visible=False)
                    curve = gr.Plot(show_label=False)
                    console = gr.HTML(console_panel(copy["log_waiting"], default_language), elem_id="live-training-console")

                with gr.Column(elem_classes="preview-card"):
                    preview_header = gr.HTML(panel_html(copy["preview"], copy["preview_copy"], "artifact-note"))
                    model_selector = gr.Dropdown(
                        choices=initial_model_choices,
                        value=initial_model,
                        label=copy["saved_model"],
                        info=copy["saved_model_info"] if initial_model_choices else copy["saved_model_empty"],
                        interactive=bool(initial_model_choices),
                        allow_custom_value=True,
                        elem_classes="model-selector",
                    )
                    preview_status = gr.HTML(preview_provenance(default_task, default_language, initial_model))
                    preview = gr.Image(value=initial_preview, show_label=False, interactive=False, elem_classes="policy-preview")
                    artifact = gr.File(label=copy["download"], interactive=False, visible=False, height=76, elem_classes="artifact-download")

        gr.HTML(f'<div class="footer-note">{html.escape(local_value(space["title"], "English"))} · <a href="{COURSE_URL}" target="_blank">Hands-On Modern RL</a> · WalkingLab</div>')

        gallery.select(choose_task, inputs=[language, seed], outputs=[selected, hero, task_info, baseline_info, steps_per_epoch, epochs, checkpoint_info, learning_rate, gamma, epsilon, status, metric, console, preview, artifact, model_selector, preview_status], queue=False, show_progress="hidden")
        language.change(switch_language, inputs=[language, selected, seed, model_selector, steps_per_epoch, epochs], outputs=[hero, catalog_header, gallery, task_info, settings_header, baseline_info, steps_per_epoch, epochs, checkpoint_info, advanced, selected, seed, restore, start, status, metric, chart_header, console, preview_header, artifact, model_selector, preview_status], queue=False, show_progress="hidden")
        restore.click(restore_baseline, inputs=[selected, language], outputs=[steps_per_epoch, epochs, checkpoint_info, learning_rate, gamma, epsilon, baseline_info], queue=False, show_progress="hidden")
        steps_per_epoch.change(update_checkpoint_plan, inputs=[selected, language, steps_per_epoch, epochs], outputs=[checkpoint_info], queue=False, show_progress="hidden")
        epochs.change(update_checkpoint_plan, inputs=[selected, language, steps_per_epoch, epochs], outputs=[checkpoint_info], queue=False, show_progress="hidden")
        start.click(train_with_ui, inputs=[selected, steps_per_epoch, epochs, learning_rate, gamma, epsilon, seed, model_selector, language], outputs=[status, metric, curve, preview, artifact, console, wait_state, start, model_selector, preview_status], concurrency_limit=1)
        model_selector.change(select_saved_model, inputs=[selected, model_selector, language], outputs=[preview, preview_status], queue=False, show_progress="hidden")

    return demo


CSS = r"""
:root{--ink:#172033;--muted:#68748a;--line:#e4e8f0;--canvas:#f4f6fa;--brand:#5b5ce2;--green:#13a36f}
.gradio-container{width:100%!important;min-width:0!important;max-width:1180px!important;margin:0 auto!important;padding:28px clamp(10px,2vw,22px) 52px!important;background:var(--canvas);box-sizing:border-box!important}
.gradio-container>.main{width:100%!important;min-width:0!important;padding:clamp(8px,1.5vw,24px)!important;box-sizing:border-box!important}.gradio-container .contain{width:100%!important;min-width:0!important}
.hero-stack{position:relative!important;margin:0!important;padding:0!important;border:0!important;background:transparent!important}
.language-bar{position:absolute!important;z-index:5!important;top:18px!important;right:20px!important;width:auto!important;min-width:0!important;margin:0!important;padding:0!important;border:0!important;background:transparent!important}
.language-switch{width:216px!important;min-width:216px!important;margin:0!important;padding:5px!important;border:1px solid rgba(255,255,255,.82)!important;border-radius:12px!important;background:rgba(255,255,255,.96)!important;box-shadow:0 10px 28px rgba(20,24,74,.2)!important;backdrop-filter:blur(12px)}
.language-switch>.wrap:not([data-testid]){display:grid!important;grid-template-columns:1fr 1fr!important;gap:5px!important}.language-switch label{display:flex!important;align-items:center!important;justify-content:center!important;min-height:38px!important;margin:0!important;padding:0 12px!important;border:0!important;border-radius:8px!important;background:transparent!important}.language-switch label input{position:absolute!important;opacity:0!important;pointer-events:none!important}.language-switch label span{color:#343b68!important;font-weight:800!important}.language-switch label.selected,.language-switch label:has(input:checked){background:#554ee6!important;box-shadow:0 4px 12px rgba(72,65,201,.28)!important}.language-switch label.selected span,.language-switch label:has(input:checked) span{color:#fff!important}
.app-shell{margin-bottom:18px}.hero{position:relative;overflow:hidden;padding:28px 34px 30px;border-radius:24px 24px 0 0;color:white;background:radial-gradient(circle at 84% 12%,rgba(139,92,246,.45),transparent 28%),linear-gradient(135deg,#1f255f,#3731a8 56%,#5b3fc5);box-shadow:0 24px 55px rgba(30,37,95,.18)}
.hero:after{content:"";position:absolute;inset:auto -80px -120px auto;width:280px;height:280px;border:44px solid rgba(255,255,255,.07);border-radius:50%}.brand-lockup{position:relative;z-index:2;display:flex;align-items:center;gap:9px;width:max-content;max-width:calc(100% - 230px);padding:8px 12px;border:1px solid rgba(255,255,255,.28);border-radius:10px;background:rgba(12,17,59,.28);box-shadow:0 8px 24px rgba(10,13,50,.14);font-size:12px;font-weight:900;letter-spacing:.075em}.brand-lockup a{color:#fff!important;text-decoration:none!important}.brand-lockup a:last-child{color:#cfd5ff!important}.brand-lockup span{color:#aeb7ff}.brand-note{position:relative;z-index:1;margin:9px 0 18px!important;color:#dfe3ff!important;font-size:12px!important}.hero-topline{display:flex;gap:12px;align-items:center;margin-bottom:15px}.experiment-badge{display:inline-flex;padding:7px 11px;border:1px solid rgba(255,255,255,.3);border-radius:999px;color:#fff!important;background:rgba(255,255,255,.12);font-size:11px;font-weight:850;letter-spacing:.13em}.hero-course{font-size:13px;color:#d9ddff!important}.hero h1{position:relative;z-index:1;max-width:780px;margin:0 0 12px;color:#fff!important;font-size:clamp(30px,5vw,56px);line-height:1.04;letter-spacing:-.035em}.hero-copy{position:relative;z-index:1;max-width:760px;margin:0;color:#e1e4ff!important;font-size:16px;line-height:1.65}.hero-links{position:relative;z-index:1;display:flex;flex-wrap:wrap;gap:9px;margin-top:22px}.hero-link{display:inline-flex;padding:10px 14px;border:1px solid rgba(255,255,255,.22);border-radius:10px;color:white!important;text-decoration:none!important;background:rgba(255,255,255,.08);font-size:13px;font-weight:750}.hero-link.primary{color:#272b72!important;background:white}.lab-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;overflow:hidden;border:1px solid #dde2ec;border-top:0;border-radius:0 0 18px 18px;background:#dde2ec}.lab-strip span{padding:13px 16px;color:var(--muted);background:#fff;font-size:11px;text-transform:uppercase;letter-spacing:.07em}.lab-strip strong{display:block;margin-top:3px;color:var(--ink);font-size:13px;text-transform:none;letter-spacing:0}
.catalog-card,.control-card,.chart-card,.preview-card,.task-brief{border:1px solid #e1e6ef!important;border-radius:17px!important;background:#fff!important;box-shadow:0 14px 34px rgba(31,42,77,.055)!important}.catalog-card{padding:24px!important;margin-bottom:18px!important}.panel-title{margin:0 0 5px!important;color:var(--ink);font-size:18px!important}.panel-copy,.artifact-note{margin:0 0 18px!important;color:var(--muted)!important;font-size:13px!important;line-height:1.55!important}
.experiment-gallery{height:auto!important;min-height:0!important;margin-top:6px!important;border:0!important;background:transparent!important}.experiment-gallery .grid-wrap{flex:none!important;height:auto!important;min-height:0!important}.experiment-gallery .grid-container{display:grid!important;flex:none!important;height:auto!important;min-height:0!important;align-content:start!important;grid-template-columns:repeat(3,minmax(0,1fr))!important;grid-auto-rows:auto!important;gap:13px!important}.experiment-gallery button,.experiment-gallery .thumbnail-item{position:relative!important;height:auto!important;aspect-ratio:16/9!important;overflow:hidden!important;border:2px solid transparent!important;border-radius:14px!important;background:#101532!important;transition:transform .16s ease,border-color .16s ease!important}.experiment-gallery button:hover{transform:translateY(-2px);border-color:#7778eb!important}.experiment-gallery img{width:100%!important;height:100%!important;object-fit:cover!important}.experiment-gallery .caption-label{position:absolute!important;z-index:2!important;inset:auto 0 0!important;padding:34px 13px 12px!important;color:#fff!important;background:linear-gradient(transparent,rgba(4,8,28,.92))!important;font-size:12px!important;font-weight:800!important;line-height:1.35!important;white-space:pre-line!important;text-align:left!important}
.task-brief{display:grid;grid-template-columns:minmax(270px,.9fr) minmax(0,1.7fr);gap:26px;margin:18px 0!important;padding:13px!important;background:linear-gradient(135deg,#fff,#f7f9ff)!important}.task-brief__visual{overflow:hidden;border-radius:12px;background:#101532}.task-brief__visual img{display:block;width:100%;height:100%;min-height:205px;object-fit:cover}.task-brief__body{padding:12px 12px 8px 0}.task-kicker{display:block;margin-bottom:7px;color:#5b5ce2;font-size:10px;font-weight:900;letter-spacing:.13em}.task-brief h3{margin:0 0 5px;color:var(--ink);font-size:23px}.task-brief p{margin:0;color:var(--muted);font-size:13px;line-height:1.55}.task-facts{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:14px 0}.task-facts span{padding:9px 11px;border:1px solid #e1e6ef;border-radius:9px;background:rgba(255,255,255,.88);color:var(--ink);font-size:11px}.task-facts b{display:block;margin-bottom:3px;color:#7a879d;font-size:8px;letter-spacing:.12em;text-transform:uppercase}.task-hint{font-weight:650;color:#465166!important}.task-reference{display:inline-flex;margin-top:10px;padding:7px 10px;border:1px solid #d9def4;border-radius:8px;color:#4f51ce!important;background:#fff;text-decoration:none!important;font-size:11px;font-weight:800}
.training-guide{display:grid;grid-template-columns:minmax(210px,.62fr) minmax(0,1.8fr);gap:22px;margin:-4px 0 18px;padding:20px 22px;border:1px solid #dfe4f4;border-radius:17px;background:linear-gradient(135deg,#f8f9ff,#fff);box-shadow:0 12px 28px rgba(31,42,77,.045)}.training-guide__intro{padding:5px 2px}.training-guide__intro h3{margin:0 0 5px;color:var(--ink);font-size:18px}.training-guide__intro p,.training-guide article p{margin:0;color:var(--muted);font-size:12px;line-height:1.55}.training-guide__grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.training-guide article{position:relative;padding:14px 14px 13px;border:1px solid #e0e5f0;border-radius:12px;background:#fff}.training-guide article>b{display:block;margin-bottom:8px;color:#5b5ce2;font-size:10px;letter-spacing:.12em}.training-guide article h4{margin:0 0 6px;color:var(--ink);font-size:13px}.training-guide article p{font-size:11px}
.baseline-preset{margin:0 0 16px;padding:15px;border:1px solid #cfd5ff;border-radius:13px;background:linear-gradient(145deg,#f4f5ff,#fff);box-shadow:0 8px 20px rgba(71,68,190,.07)}.baseline-preset__head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:11px}.baseline-preset__head span{display:block;margin-bottom:3px;color:#5b5ce2;font-size:8px;font-weight:900;letter-spacing:.12em}.baseline-preset__head h3{margin:0;color:var(--ink);font-size:14px}.baseline-preset__head strong{max-width:45%;padding:5px 7px;border-radius:7px;color:#403fa7;background:#e9eaff;font-size:9px;text-align:right}.baseline-preset__facts{display:grid;grid-template-columns:1fr 1fr;gap:7px}.baseline-preset__facts span{padding:8px;border:1px solid #e1e4f2;border-radius:8px;color:#222a40;background:rgba(255,255,255,.9);font-size:10px}.baseline-preset__facts b,.baseline-preset>p b{display:block;margin-bottom:2px;color:#7a8399;font-size:7px;letter-spacing:.09em;text-transform:uppercase}.baseline-preset>p{margin:9px 1px 0!important;color:#566178!important;font-size:10px!important;line-height:1.45!important}.baseline-restore{margin:2px 0 8px!important;border-color:#cfd5ff!important;color:#403fa7!important;background:#f7f7ff!important;font-weight:750!important}
.checkpoint-plan{margin:-2px 0 13px;padding:11px 12px;border:1px solid #dfe4ee;border-radius:11px;background:#f8f9fc}.checkpoint-plan>span{display:block;margin-bottom:3px;color:#5b5ce2;font-size:8px;font-weight:900;letter-spacing:.09em;text-transform:uppercase}.checkpoint-plan>strong{display:block;color:#2a3348;font-size:11px;line-height:1.45}.checkpoint-plan>p{margin:6px 0 0!important;color:#68748a!important;font-size:9px!important;line-height:1.45!important;overflow-wrap:anywhere}.checkpoint-plan>p b{margin-right:6px;color:#3f485c}
.training-layout{align-items:flex-start!important}.training-layout>.control-card,.training-layout>.results-stack{align-self:flex-start!important}.results-stack{min-width:0!important;gap:18px!important;padding:0!important;border:0!important;background:transparent!important}.control-card,.chart-card,.preview-card{padding:25px!important}.advanced-settings{margin:2px 0 12px!important;overflow:hidden!important;border:1px solid #dfe4ee!important;border-radius:11px!important;background:#fafbfe!important}.advanced-settings>button,.advanced-settings summary{min-height:44px!important;color:#3f485c!important;font-size:12px!important;font-weight:800!important}.result-summary{display:grid!important;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr)!important;gap:10px!important;margin:0 0 10px!important}.result-summary>div{min-width:0!important}.primary-btn{min-height:46px!important;border:0!important;border-radius:10px!important;background:linear-gradient(135deg,#5b5ce2,#7c4dff)!important;font-weight:850!important}.run-state,.live-metric{display:flex;gap:11px;align-items:center;margin-top:10px;padding:14px 15px;border:1px solid #e3e7ef;border-radius:12px;background:#fafbfe}.result-summary .run-state,.result-summary .live-metric{height:100%;margin-top:0}.run-state__dot{width:9px;height:9px;border-radius:50%;background:#8b95a8;box-shadow:0 0 0 5px rgba(139,149,168,.12)}.run-state--running .run-state__dot{background:#8b5cf6;animation:pulse 1.2s infinite}.run-state--complete .run-state__dot{background:#13a36f}.run-state--error .run-state__dot{background:#e05252}.summary-label{display:block;margin-bottom:2px;color:#7b879c;font-size:8px;font-weight:850;letter-spacing:.12em;text-transform:uppercase}.run-state strong,.live-metric strong{display:block;color:var(--ink);font-size:14px}.run-state small,.live-metric small{display:block;margin-top:2px;color:var(--muted);font-size:11px}.metric-reading{display:flex;gap:9px;align-items:baseline}.metric-reading strong{font-size:20px}
.model-selector{margin:0 0 10px!important}.model-selector input,.model-selector [role="combobox"]{min-height:46px!important;border-radius:10px!important;background:#fff!important}.preview-provenance{display:flex;gap:9px;align-items:flex-start;margin:0 0 12px;padding:10px 12px;border:1px solid #e0e5f0;border-radius:10px;color:#59657a;background:#f8f9fc;font-size:11px;line-height:1.5}.preview-provenance__dot{flex:0 0 auto;width:8px;height:8px;margin-top:4px;border-radius:50%;background:#9ba5b5}.preview-provenance--ready{color:#16664d;border-color:#cfeadf;background:#f2fbf7}.preview-provenance--ready .preview-provenance__dot{background:#13a36f}
.console-panel{overflow:hidden;margin-top:14px;border:1px solid #29315e;border-radius:12px;background:#11162d}.console-head{padding:9px 13px;border-bottom:1px solid #28305b;color:#dbe1ff;font-size:11px;font-weight:800}.console-dot{display:inline-block;width:7px;height:7px;margin-right:8px;border-radius:50%;background:#31d39b}.console-text{height:300px!important;margin:0!important;padding:13px!important;overflow:auto!important;color:#d5dcf4!important;background:#11162d!important;font:11px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace!important;white-space:pre-wrap!important}.run-wait{display:flex;gap:12px;align-items:center;margin:0 0 13px;padding:13px 15px;border:1px solid #d6d8ff;border-radius:11px;background:#f8f7ff;color:#313774}.run-wait strong,.run-wait small,.run-wait em{display:block}.run-wait small{margin-top:2px;color:#68748a;font-size:11px}.run-wait em{margin-top:4px;color:#5b5ce2;font-size:10px;font-style:normal;font-weight:800}.run-wait__spinner{width:20px;height:20px;border:2px solid #d7d9ff;border-top-color:#5b5ce2;border-radius:50%;animation:spin .75s linear infinite}
.preview-card{margin-top:0!important}.policy-preview,.policy-preview .image-container,.policy-preview img{min-height:320px!important}.policy-preview img{max-height:520px!important;object-fit:contain!important;background:#0f1430}.artifact-download{height:76px!important;min-height:76px!important;margin-top:8px!important;overflow:hidden!important}.artifact-download [data-testid="status-tracker"]{height:76px!important}.artifact-download .empty{height:50px!important;min-height:50px!important}.footer-note{padding:27px 0 0;text-align:center;color:#8390a6;font-size:11px}.footer-note a{color:#5b5ce2!important;font-weight:750}
@keyframes spin{to{transform:rotate(360deg)}}@keyframes pulse{50%{box-shadow:0 0 0 8px rgba(139,92,246,.08)}}
@media(max-width:900px){.lab-strip{grid-template-columns:1fr 1fr}.experiment-gallery .grid-container{grid-template-columns:repeat(2,minmax(0,1fr))!important}.task-brief,.training-guide{grid-template-columns:1fr}.training-guide__grid{grid-template-columns:1fr}.task-brief__body{padding:7px}.training-layout{flex-direction:column!important}.training-layout>.control-card,.training-layout>.results-stack{width:100%!important;min-width:0!important;flex:1 1 auto!important}.language-bar{position:static!important;margin:12px 12px 16px auto!important}.brand-lockup{max-width:calc(100% - 220px)}.hero{padding-top:24px}}
@media(max-width:620px){.hero{padding:22px 20px 24px;border-radius:18px 18px 0 0}.brand-lockup{max-width:100%;font-size:10px;letter-spacing:.045em}.experiment-gallery .grid-container{grid-template-columns:1fr!important}.task-facts,.result-summary{grid-template-columns:1fr!important}.lab-strip{grid-template-columns:1fr}.language-switch{width:190px!important;min-width:190px!important}}
"""


AUTO_SCROLL_JS = r"""
function initializeGameLabUi(){
  if(window.__gameLabUiReady)return;window.__gameLabUiReady=true;
  let active=null,follow=true,saved=0,internal=false;
  const update=()=>{
    const element=document.querySelector("#live-training-console .console-text");
    if(element&&element!==active){active=element;follow=true;saved=0;active.addEventListener("scroll",()=>{if(internal)return;follow=active.scrollHeight-active.clientHeight-active.scrollTop<=24;saved=active.scrollTop},{passive:true})}
    if(active){internal=true;if(follow)active.scrollTop=active.scrollHeight;else active.scrollTop=Math.min(saved,Math.max(0,active.scrollHeight-active.clientHeight));requestAnimationFrame(()=>{internal=false})}
    const timer=document.querySelector(".run-wait__elapsed");if(timer){const elapsed=Math.max(0,Math.floor((Date.now()-Number(timer.dataset.startMs))/1000));timer.textContent=`${elapsed}s ${timer.dataset.label}`}
  };
  new MutationObserver(()=>requestAnimationFrame(update)).observe(document.body,{childList:true,subtree:true,characterData:true});setInterval(update,1000);update();
}
initializeGameLabUi();
"""


def launch(space_module: Any) -> None:
    demo = build_demo(space_module)
    demo.queue(default_concurrency_limit=1, max_size=12).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        css=CSS,
        js=AUTO_SCROLL_JS,
        footer_links=[],
    )
