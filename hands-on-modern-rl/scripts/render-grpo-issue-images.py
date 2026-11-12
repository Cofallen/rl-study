"""Render GitHub Issue #75 formula comparisons as high-resolution PNG files."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hands-on-modern-rl-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "chapter18_grpo" / "images"
CHINESE_FONT = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")

INK = "#172033"
MUTED = "#5b6474"
RED = "#c62828"
RED_BG = "#fff0f0"
GREEN = "#19733b"
GREEN_BG = "#edf9f0"
BLUE = "#1756a9"
BLUE_BG = "#edf4ff"
GRAY_BG = "#f5f7fa"
BORDER = "#d7dce4"


def setup_canvas(height: float):
    fig = plt.figure(figsize=(12, height), dpi=160, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, width, height, facecolor, edgecolor=BORDER):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)


def zh(ax, x, y, text, size=18, color=INK, weight="normal", ha="left"):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontproperties=CHINESE_FONT,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va="top",
        linespacing=1.45,
    )


def math(ax, x, y, text, size=20, color=INK, ha="left"):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=size,
        color=color,
        ha=ha,
        va="center",
        math_fontfamily="dejavuserif",
    )


def footer(ax, source):
    zh(ax, 0.05, 0.045, f"来源：{source}", size=11, color=MUTED)


def save(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(path)


def render_original_formula():
    fig, ax = setup_canvas(9.4)
    zh(ax, 0.05, 0.965, "DeepSeekMath 原始 GRPO：先逐 token，再按回答长度归一化", 25, weight="bold")
    zh(
        ax,
        0.05,
        0.91,
        "式（3）中的 ratio、clip 和 KL 都带有 token 下标 t。",
        16,
        color=MUTED,
    )

    box(ax, 0.04, 0.57, 0.92, 0.29, GRAY_BG)
    zh(ax, 0.07, 0.835, "策略目标", 17, color=BLUE, weight="bold")
    math(
        ax,
        0.5,
        0.755,
        r"$\mathcal{J}_{\mathrm{GRPO}}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left(\mathcal{C}_{i,t}-\beta\widehat{D}_{i,t}\right)\right]$",
        19,
        ha="center",
    )
    math(
        ax,
        0.5,
        0.655,
        r"$\mathcal{C}_{i,t}=\min\left(\rho_{i,t}\widehat{A}_i,\ \mathrm{clip}\left(\rho_{i,t},1-\varepsilon,1+\varepsilon\right)\widehat{A}_i\right)$",
        18,
        ha="center",
    )
    zh(
        ax,
        0.07,
        0.605,
        "关键顺序：逐 token ratio / clip / KL → 每段回答除以有效 token 数 → 回答组平均",
        14,
        color=MUTED,
    )

    box(ax, 0.04, 0.34, 0.44, 0.17, GREEN_BG, GREEN)
    zh(ax, 0.07, 0.485, "逐 token ratio", 17, color=GREEN, weight="bold")
    math(
        ax,
        0.26,
        0.405,
        r"$\rho_{i,t}(\theta)=\frac{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})}$",
        17,
        ha="center",
    )

    box(ax, 0.52, 0.34, 0.44, 0.17, BLUE_BG, BLUE)
    zh(ax, 0.55, 0.485, "逐 token KL（式 4）", 17, color=BLUE, weight="bold")
    math(
        ax,
        0.74,
        0.42,
        r"$\Delta_{i,t}=\ell^{\mathrm{ref}}_{i,t}-\ell^{\theta}_{i,t}$",
        17,
        ha="center",
    )
    math(
        ax,
        0.74,
        0.37,
        r"$\widehat{D}_{i,t}=\exp(\Delta_{i,t})-\Delta_{i,t}-1$",
        17,
        ha="center",
    )

    box(ax, 0.04, 0.12, 0.92, 0.16, GREEN_BG, GREEN)
    zh(ax, 0.07, 0.255, "结果监督下的组内优势", 17, color=GREEN, weight="bold")
    math(
        ax,
        0.5,
        0.19,
        r"$\widehat{A}_i=\frac{r_i-\mathrm{mean}(r_1,\ldots,r_G)}{\mathrm{std}(r_1,\ldots,r_G)}$",
        19,
        ha="center",
    )
    zh(ax, 0.07, 0.14, "同一回答中的 token 共享一个优势，但 ratio、clip 和 KL 仍逐 token 计算。", 14, color=MUTED)

    footer(ax, "DeepSeekMath, Equations (3) and (4), arXiv:2402.03300")
    save(fig, "issue-75-deepseekmath-grpo-formulas.png")


def render_algorithm_comparison():
    fig, ax = setup_canvas(10.8)
    zh(ax, 0.05, 0.97, "原实现、原始 GRPO 与正确 GSPO", 26, weight="bold")
    zh(ax, 0.05, 0.925, "三者的关键区别在于 ratio 的粒度和长度归一化发生在哪里。", 16, color=MUTED)

    box(ax, 0.04, 0.665, 0.92, 0.2, RED_BG, RED)
    zh(ax, 0.07, 0.84, "原实现：回答级连乘", 19, color=RED, weight="bold")
    math(
        ax,
        0.5,
        0.755,
        r"$\rho_i=\exp\left(\sum_t\Delta_{i,t}\right)=\prod_t\rho_{i,t}$",
        24,
        color=RED,
        ha="center",
    )
    zh(ax, 0.07, 0.705, "一段回答只有一个 ratio；整段只 clip 一次；没有长度归一化。", 15, color=INK)

    box(ax, 0.04, 0.39, 0.92, 0.22, GREEN_BG, GREEN)
    zh(ax, 0.07, 0.585, "原始 GRPO：逐 token ratio 和 clip", 19, color=GREEN, weight="bold")
    math(
        ax,
        0.5,
        0.515,
        r"$\rho_{i,t}=\exp(\Delta_{i,t})$",
        23,
        color=GREEN,
        ha="center",
    )
    math(
        ax,
        0.5,
        0.445,
        r"$\frac{1}{|o_i|}\sum_t\min\left(\rho_{i,t}\widehat{A}_i,\ \mathrm{clip}(\rho_{i,t})\widehat{A}_i\right)$",
        19,
        color=GREEN,
        ha="center",
    )
    zh(ax, 0.07, 0.42, "每个 token 分别 clip，最后每段回答按自己的有效 token 数求平均。", 15, color=INK)

    box(ax, 0.04, 0.115, 0.92, 0.22, BLUE_BG, BLUE)
    zh(ax, 0.07, 0.31, "正确 GSPO：回答级几何平均", 19, color=BLUE, weight="bold")
    math(
        ax,
        0.5,
        0.23,
        r"$s_i(\theta)=\exp\left(\frac{1}{|o_i|}\sum_t\log\rho_{i,t}\right)=\left(\prod_t\rho_{i,t}\right)^{1/|o_i|}$",
        20,
        color=BLUE,
        ha="center",
    )
    zh(ax, 0.07, 0.16, "整段回答使用一个 ratio，但 1 / |o_i| 不能省略；原实现因此也不是 GSPO。", 15, color=INK)

    footer(ax, "DeepSeekMath Eq. (3); GSPO Eq. (7), arXiv:2507.18071")
    save(fig, "issue-75-grpo-gspo-comparison.png")


def render_numeric_check():
    fig, ax = setup_canvas(7.0)
    zh(ax, 0.05, 0.955, "同一组 token ratio，三种算法得到不同结果", 25, weight="bold")
    math(
        ax,
        0.5,
        0.86,
        r"$[\rho_1,\rho_2,\rho_3]=[1.1,\ 0.9,\ 1.5],\qquad \varepsilon=0.2,\qquad \widehat{A}=1$",
        21,
        ha="center",
    )

    box(ax, 0.04, 0.51, 0.28, 0.27, GREEN_BG, GREEN)
    zh(ax, 0.18, 0.75, "原始 GRPO", 18, color=GREEN, weight="bold", ha="center")
    zh(ax, 0.18, 0.69, "分别 clip", 15, color=MUTED, ha="center")
    math(ax, 0.18, 0.62, r"$\frac{1.1+0.9+1.2}{3}$", 21, color=GREEN, ha="center")
    zh(ax, 0.18, 0.56, "1.066667", 22, color=GREEN, weight="bold", ha="center")

    box(ax, 0.36, 0.51, 0.28, 0.27, RED_BG, RED)
    zh(ax, 0.5, 0.75, "原实现", 18, color=RED, weight="bold", ha="center")
    zh(ax, 0.5, 0.69, "先连乘，再整段 clip", 15, color=MUTED, ha="center")
    math(ax, 0.5, 0.62, r"$\mathrm{clip}(1.1\times0.9\times1.5)$", 18, color=RED, ha="center")
    zh(ax, 0.5, 0.56, "1.200000", 22, color=RED, weight="bold", ha="center")

    box(ax, 0.68, 0.51, 0.28, 0.27, BLUE_BG, BLUE)
    zh(ax, 0.82, 0.75, "正确 GSPO", 18, color=BLUE, weight="bold", ha="center")
    zh(ax, 0.82, 0.69, "几何平均", 15, color=MUTED, ha="center")
    math(ax, 0.82, 0.62, r"$(1.1\times0.9\times1.5)^{1/3}$", 19, color=BLUE, ha="center")
    zh(ax, 0.82, 0.56, "1.140886", 22, color=BLUE, weight="bold", ha="center")

    box(ax, 0.04, 0.12, 0.92, 0.3, GRAY_BG)
    zh(ax, 0.07, 0.395, "数值回归对拍", 18, color=INK, weight="bold")
    zh(ax, 0.09, 0.33, "DeepSeekMath 原公式直译", 15)
    zh(ax, 0.75, 0.33, "-0.925000", 16, color=GREEN, weight="bold")
    zh(ax, 0.09, 0.275, 'TRL loss_type="grpo"', 15)
    zh(ax, 0.75, 0.275, "-0.925000", 16, color=GREEN, weight="bold")
    zh(ax, 0.09, 0.22, "修复后的仓库实现", 15)
    zh(ax, 0.75, 0.22, "-0.925000", 16, color=GREEN, weight="bold")
    zh(ax, 0.09, 0.165, "BNPO 全局 token 平均", 15)
    zh(ax, 0.75, 0.165, "-0.900000", 16, color=RED, weight="bold")

    footer(ax, "scripts/check-grpo-objective.py")
    save(fig, "issue-75-grpo-numeric-check.png")


if __name__ == "__main__":
    render_original_formula()
    render_algorithm_comparison()
    render_numeric_check()
