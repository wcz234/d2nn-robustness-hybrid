"""Render the protocol-bound robustness comparison from formal summary artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.5,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
    }
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = PROJECT_ROOT / "results" / "formal_mnist_v1" / "robustness_summary"
OUTPUT_DIR = Path(__file__).resolve().parent

CONDITIONS = [
    ("clean_continuous", "干净连续相位"),
    ("lateral_shift_0p25px", "横向错位 0.25 px"),
    ("lateral_shift_0p50px", "横向错位 0.50 px"),
    ("gap_error_0p1mm", "间距误差 0.1 mm"),
    ("gap_error_0p3mm", "间距误差 0.3 mm"),
    ("phase_noise_0p02rad", "相位噪声 0.02 rad"),
    ("phase_noise_0p05rad", "相位噪声 0.05 rad"),
    ("detector_noise_0p01", "探测器噪声 0.01"),
    ("detector_noise_0p03", "探测器噪声 0.03"),
    ("phase_quantization_16", "16 级相位量化"),
    ("phase_quantization_8", "8 级相位量化"),
    ("phase_quantization_4", "4 级相位量化"),
    ("mixed_nominal", "混合标称扰动"),
    ("mixed_stress", "混合强扰动"),
]
METHODS = {
    "baseline_d2nn": {"label": "基准 D²NN", "color": "#4D4D4D", "marker": "o"},
    "robust_d2nn": {"label": "鲁棒 D²NN", "color": "#3775BA", "marker": "s"},
    "hybrid": {"label": "混合模型", "color": "#B64342", "marker": "D"},
}
GROUP_BANDS = [
    (0, 0, "#F2F2F2"),
    (1, 4, "#EEF4FA"),
    (5, 8, "#F6F1F4"),
    (9, 11, "#F8F4E8"),
    (12, 13, "#EDF5F1"),
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def export_source_data(method_rows: list[dict], paired_rows: list[dict]) -> None:
    method_path = OUTPUT_DIR / "source_data_figure_2a.csv"
    with method_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["condition_id", "method_id", "n_training_seeds", "accuracy_mean", "ci95_low", "ci95_high"],
        )
        writer.writeheader()
        for row in method_rows:
            accuracy = row["metrics"]["accuracy"]
            writer.writerow(
                {
                    "condition_id": row["condition_id"],
                    "method_id": row["method_id"],
                    "n_training_seeds": accuracy["n_training_seeds"],
                    "accuracy_mean": accuracy["mean"],
                    "ci95_low": accuracy["ci95_t"][0],
                    "ci95_high": accuracy["ci95_t"][1],
                }
            )

    paired_path = OUTPUT_DIR / "source_data_figure_2b.csv"
    with paired_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["condition_id", "method_id", "reference_method_id", "n_training_seeds", "difference_pp", "ci95_low_pp", "ci95_high_pp"],
        )
        writer.writeheader()
        for row in paired_rows:
            accuracy = row["metrics"]["accuracy"]
            writer.writerow(
                {
                    "condition_id": row["condition_id"],
                    "method_id": row["right_method_id"],
                    "reference_method_id": row["left_method_id"],
                    "n_training_seeds": accuracy["n_training_seeds"],
                    "difference_pp": -100 * accuracy["mean"],
                    "ci95_low_pp": -100 * accuracy["ci95_t"][1],
                    "ci95_high_pp": -100 * accuracy["ci95_t"][0],
                }
            )


def add_group_bands(ax: plt.Axes) -> None:
    for start, end, color in GROUP_BANDS:
        ax.axhspan(start - 0.5, end + 0.5, color=color, zorder=0)


def plot_absolute_accuracy(ax: plt.Axes, indexed: dict[tuple[str, str], dict]) -> None:
    offsets = {"baseline_d2nn": -0.20, "robust_d2nn": 0.0, "hybrid": 0.20}
    add_group_bands(ax)
    for method_id, style in METHODS.items():
        means, lower, upper = [], [], []
        for condition_id, _ in CONDITIONS:
            estimate = indexed[(condition_id, method_id)]["metrics"]["accuracy"]
            means.append(100 * estimate["mean"])
            lower.append(100 * estimate["ci95_t"][0])
            upper.append(100 * estimate["ci95_t"][1])
        y = np.arange(len(CONDITIONS)) + offsets[method_id]
        means_array = np.asarray(means)
        ax.errorbar(
            means_array,
            y,
            xerr=np.vstack([means_array - lower, np.asarray(upper) - means_array]),
            fmt=style["marker"],
            ms=3.6,
            color=style["color"],
            ecolor=style["color"],
            elinewidth=0.9,
            capsize=1.8,
            capthick=0.8,
            label=style["label"],
            zorder=3,
        )
    ax.set_xlim(40, 106)
    ax.set_xlabel("准确率（%）")
    ax.set_yticks(np.arange(len(CONDITIONS)))
    ax.set_yticklabels([label for _, label in CONDITIONS])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.5, zorder=1)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, handletextpad=0.4, columnspacing=1.0)
    ax.set_title("预注册部署条件下的准确率", loc="left", pad=18, fontsize=7.2)


def plot_paired_effects(ax: plt.Axes, paired_index: dict[tuple[str, str], dict]) -> None:
    offsets = {"robust_d2nn": -0.11, "hybrid": 0.11}
    add_group_bands(ax)
    for method_id in ("robust_d2nn", "hybrid"):
        style = METHODS[method_id]
        means, lower, upper = [], [], []
        for condition_id, _ in CONDITIONS:
            estimate = paired_index[(condition_id, method_id)]["metrics"]["accuracy"]
            means.append(-100 * estimate["mean"])
            lower.append(-100 * estimate["ci95_t"][1])
            upper.append(-100 * estimate["ci95_t"][0])
        y = np.arange(len(CONDITIONS)) + offsets[method_id]
        means_array = np.asarray(means)
        ax.errorbar(
            means_array,
            y,
            xerr=np.vstack([means_array - lower, np.asarray(upper) - means_array]),
            fmt=style["marker"],
            ms=3.6,
            color=style["color"],
            ecolor=style["color"],
            elinewidth=0.9,
            capsize=1.8,
            capthick=0.8,
            label=style["label"],
            zorder=3,
        )
    ax.axvline(0, color="#767676", linestyle="--", linewidth=0.8, zorder=2)
    ax.set_xlim(-20, 47)
    ax.set_xlabel("相对基准的准确率差（百分点）")
    ax.set_yticks(np.arange(len(CONDITIONS)))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.5, zorder=1)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, handletextpad=0.4, columnspacing=1.0)
    ax.set_title("训练种子配对效应", loc="left", pad=18, fontsize=7.2)


def main() -> None:
    method_rows = read_jsonl(SUMMARY_DIR / "method_summary.jsonl")
    all_paired_rows = read_jsonl(SUMMARY_DIR / "paired_comparisons.jsonl")
    paired_rows = [
        row
        for row in all_paired_rows
        if row["left_method_id"] == "baseline_d2nn" and row["right_method_id"] in {"robust_d2nn", "hybrid"}
    ]
    if len(method_rows) != 42 or len(paired_rows) != 28:
        raise ValueError("formal Figure 1 requires 42 method-condition rows and 28 baseline-referenced paired rows")
    export_source_data(method_rows, paired_rows)
    method_index = {(row["condition_id"], row["method_id"]): row for row in method_rows}
    paired_index = {(row["condition_id"], row["right_method_id"]): row for row in paired_rows}

    width_mm = 183
    width_in = width_mm / 25.4
    height_in = 128 / 25.4
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(width_in, height_in),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.08},
        constrained_layout=True,
    )
    plot_absolute_accuracy(axes[0], method_index)
    plot_paired_effects(axes[1], paired_index)
    axes[0].text(-0.30, 1.08, "a", transform=axes[0].transAxes, fontsize=8, fontweight="bold", va="top")
    axes[1].text(-0.08, 1.08, "b", transform=axes[1].transAxes, fontsize=8, fontweight="bold", va="top")
    output_base = OUTPUT_DIR / "figure_2_robustness"
    fig.savefig(output_base.with_suffix(".svg"))
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600)
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
