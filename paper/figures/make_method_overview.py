"""Render the numerical D2NN architecture and training protocol schematic."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUTPUT_DIR = Path(__file__).resolve().parent
COLORS = {
    "baseline": "#4D4D4D",
    "robust": "#3775BA",
    "hybrid": "#B64342",
    "electronic": "#42949E",
    "neutral": "#767676",
    "light": "#F4F4F4",
    "line": "#B8B8B8",
    "text": "#272727",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def add_box(ax, xy, width, height, text, *, edge, face="white", fontsize=6.8, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=COLORS["text"],
        linespacing=1.2,
        zorder=3,
    )
    return patch


def add_arrow(ax, start, end, *, color=None, style="-"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=1.0,
        linestyle=style,
        color=color or COLORS["neutral"],
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=1,
    )
    ax.add_patch(arrow)


def add_curve_arrow(ax, start, end, *, color, radius, style="--"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        connectionstyle=f"arc3,rad={radius}",
        mutation_scale=8,
        linewidth=1.0,
        linestyle=style,
        color=color,
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=1,
    )
    ax.add_patch(arrow)


def draw_phase_stack(ax, x, y, width=0.12, height=0.25):
    for offset, alpha in ((0.024, 0.35), (0.012, 0.62), (0.0, 1.0)):
        ax.add_patch(
            Rectangle(
                (x + offset, y + offset),
                width,
                height,
                facecolor="#DCE8F6",
                edgecolor=COLORS["robust"],
                linewidth=0.9,
                alpha=alpha,
                zorder=2,
            )
        )
    ax.text(x + width / 2, y + height / 2, "3 个相位层", ha="center", va="center", fontsize=6.6)


def draw_inference_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.98, "a", fontsize=9, fontweight="bold", va="top")
    ax.text(0.045, 0.98, "数值推理路径", fontsize=8.3, fontweight="bold", va="top")
    ax.text(0.995, 0.98, "仅限标量数值仿真", fontsize=6.3, color=COLORS["neutral"], ha="right", va="top")

    add_box(ax, (0.03, 0.57), 0.12, 0.18, "MNIST 振幅\n1 × 28 × 28", edge=COLORS["neutral"], face=COLORS["light"])
    add_arrow(ax, (0.15, 0.66), (0.205, 0.66))
    add_box(ax, (0.205, 0.57), 0.12, 0.18, "嵌入输入平面\n64 × 64", edge=COLORS["neutral"])
    add_arrow(ax, (0.325, 0.66), (0.38, 0.66))
    draw_phase_stack(ax, 0.38, 0.535)
    ax.text(0.455, 0.48, "FFT Rayleigh–Sommerfeld\n波长 0.75 mm；间距 30 mm", ha="center", va="top", fontsize=6.2, color=COLORS["neutral"])
    add_arrow(ax, (0.524, 0.66), (0.575, 0.66))
    add_box(ax, (0.575, 0.57), 0.12, 0.18, "输出光强\n64 × 64", edge=COLORS["neutral"])

    add_arrow(ax, (0.695, 0.68), (0.75, 0.80), color=COLORS["baseline"])
    add_box(ax, (0.75, 0.73), 0.12, 0.14, "10 个探测器\n能量", edge=COLORS["baseline"])
    add_arrow(ax, (0.87, 0.80), (0.94, 0.80), color=COLORS["baseline"])
    ax.text(0.995, 0.80, "类别\n分数", ha="right", va="center", fontsize=6.7, color=COLORS["baseline"], fontweight="bold")
    ax.text(0.75, 0.90, "基准 / 鲁棒 D²NN", fontsize=6.4, color=COLORS["baseline"], fontweight="bold")

    add_arrow(ax, (0.695, 0.64), (0.75, 0.55), color=COLORS["hybrid"])
    add_box(ax, (0.75, 0.47), 0.105, 0.14, "8 × 8 池化\n64 维特征", edge=COLORS["hybrid"])
    add_arrow(ax, (0.855, 0.54), (0.89, 0.54), color=COLORS["hybrid"])
    add_box(ax, (0.89, 0.47), 0.095, 0.14, "64→32→10 后端\n类别分数", edge=COLORS["hybrid"], fontsize=6.1)
    ax.text(0.75, 0.64, "混合模型", fontsize=6.4, color=COLORS["hybrid"], fontweight="bold")

    add_arrow(ax, (0.09, 0.56), (0.09, 0.27), color=COLORS["electronic"], style="--")
    add_arrow(ax, (0.09, 0.27), (0.75, 0.27), color=COLORS["electronic"], style="--")
    add_box(ax, (0.75, 0.20), 0.18, 0.14, "784→18→10 MLP\n类别分数", edge=COLORS["electronic"])
    ax.text(0.75, 0.37, "纯电子基线绕过光学传播", fontsize=6.4, color=COLORS["electronic"], fontweight="bold")
    add_arrow(ax, (0.93, 0.27), (0.975, 0.27), color=COLORS["electronic"])

    ax.text(0.03, 0.08, "光学兼容方法", fontsize=6.3, color=COLORS["neutral"], fontweight="bold")
    ax.plot([0.03, 0.695], [0.055, 0.055], color=COLORS["line"], lw=1.2)
    ax.text(0.995, 0.055, "所有维度均描述张量或模型接口", fontsize=5.9, color=COLORS["neutral"], ha="right", va="center")


def draw_training_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.97, "b", fontsize=9, fontweight="bold", va="top")
    ax.text(0.045, 0.97, "训练阶段扰动采样", fontsize=8.3, fontweight="bold", va="top")

    add_box(ax, (0.03, 0.40), 0.13, 0.22, "训练批次", edge=COLORS["neutral"], face=COLORS["light"])
    add_arrow(ax, (0.16, 0.51), (0.22, 0.51), color=COLORS["robust"])
    add_box(ax, (0.22, 0.34), 0.18, 0.34, "训练采样\n\n均匀：错位、间距\n高斯：相位、探测器\n不含相位量化", edge=COLORS["robust"], face="#F5F8FC", fontsize=6.2)
    ax.text(0.31, 0.73, "鲁棒 D²NN", ha="center", fontsize=6.4, color=COLORS["robust"], fontweight="bold")

    add_arrow(ax, (0.40, 0.51), (0.47, 0.51), color=COLORS["robust"])
    add_curve_arrow(ax, (0.16, 0.43), (0.47, 0.43), color=COLORS["neutral"], radius=0.42)
    ax.text(0.315, 0.13, "干净路径：基准与混合模型", ha="center", fontsize=6.1, color=COLORS["neutral"])
    add_box(ax, (0.47, 0.40), 0.14, 0.22, "光学前向", edge=COLORS["neutral"])
    add_arrow(ax, (0.61, 0.51), (0.68, 0.51))
    add_box(ax, (0.68, 0.34), 0.17, 0.34, "复合损失\n\nMSE 分数\n+ 0.1 CE logits\n+ 0.01 相位惩罚", edge=COLORS["hybrid"], face="#FCF6F6", fontsize=6.3)
    add_arrow(ax, (0.85, 0.51), (0.93, 0.51), color=COLORS["hybrid"])
    ax.text(0.995, 0.51, "更新\n参数", ha="right", va="center", fontsize=6.7, color=COLORS["hybrid"], fontweight="bold")

    add_curve_arrow(ax, (0.96, 0.42), (0.61, 0.40), color=COLORS["neutral"], radius=-0.58)
    ax.text(
        0.75,
        0.12,
        "通过数值传播反向传播",
        ha="center",
        fontsize=6.2,
        color=COLORS["neutral"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )


def main():
    fig = plt.figure(figsize=(7.2, 4.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.28, 1.0], hspace=0.04)
    draw_inference_panel(fig.add_subplot(grid[0]))
    draw_training_panel(fig.add_subplot(grid[1]))

    output_base = OUTPUT_DIR / "figure_1_method_overview"
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
