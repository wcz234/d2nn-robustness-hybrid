"""Render model-size and CPU software-simulation boundaries from formal artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
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
    }
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "formal_mnist_v1"
OUTPUT_DIR = Path(__file__).resolve().parent
RUNS = {
    "baseline_d2nn": ["baseline_d2nn_seed42", "baseline_d2nn_seed43", "baseline_d2nn_seed44"],
    "robust_d2nn": ["robust_d2nn_seed42", "robust_d2nn_seed43", "robust_d2nn_seed44"],
    "hybrid": ["hybrid_seed42", "hybrid_seed43", "hybrid_seed44_retry1"],
    "electronic": ["electronic_seed42", "electronic_seed43", "electronic_seed44"],
}
METHODS = {
    "baseline_d2nn": {"label": "基准 D²NN", "color": "#4D4D4D", "marker": "o"},
    "robust_d2nn": {"label": "鲁棒 D²NN", "color": "#3775BA", "marker": "s"},
    "hybrid": {"label": "混合模型", "color": "#B64342", "marker": "D"},
    "electronic": {"label": "纯电子基线", "color": "#42949E", "marker": "^"},
}


def plain_log_tick(value: float, _position: int) -> str:
    labels = {1.0: "1", 10.0: "10", 100.0: "100", 1000.0: "1 000", 10000.0: "10 000"}
    return labels.get(float(value), "")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def representation_elements(schema_id: str) -> int:
    if ":" not in schema_id:
        raise ValueError(f"feature schema lacks dimensions: {schema_id}")
    dimensions = [int(value) for value in re.findall(r"\d+", schema_id.split(":", 1)[1])]
    if not dimensions:
        raise ValueError(f"feature schema lacks dimensions: {schema_id}")
    return int(np.prod(dimensions))


def load_complexity_rows() -> list[dict]:
    rows = []
    for method_id, runs in RUNS.items():
        manifests = [read_json(RESULTS_ROOT / "clean" / run / "evaluation_manifest.json") for run in runs]
        parameter_counts = {item["model"]["trainable_parameters"] for item in manifests}
        schema_ids = {item["model"]["feature_schema_id"] for item in manifests}
        if len(parameter_counts) != 1 or len(schema_ids) != 1:
            raise ValueError(f"model complexity is inconsistent across seeds for {method_id}")
        if {item["method_id"] for item in manifests} != {method_id}:
            raise ValueError(f"method identity mismatch for {method_id}")
        schema_id = next(iter(schema_ids))
        rows.append(
            {
                "method_id": method_id,
                "trainable_parameters": next(iter(parameter_counts)),
                "feature_schema_id": schema_id,
                "representation_elements": representation_elements(schema_id),
                "n_training_seeds": len(manifests),
            }
        )
    return rows


def export_source_data(complexity_rows: list[dict], seed_rows: list[dict]) -> None:
    with (OUTPUT_DIR / "source_data_figure_3a.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(complexity_rows[0]))
        writer.writeheader()
        writer.writerows(complexity_rows)
    fields = ["method_id", "training_seed", "accuracy", "cpu_median_seconds", "timing_repeat_count"]
    with (OUTPUT_DIR / "source_data_figure_3b.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in seed_rows:
            writer.writerow(
                {
                    "method_id": row["method_id"],
                    "training_seed": row["training_seed"],
                    "accuracy": row["metrics"]["accuracy"],
                    "cpu_median_seconds": row["metrics"]["cpu_median_seconds"],
                    "timing_repeat_count": row["timing_repeat_count"],
                }
            )


def plot_counts(ax: plt.Axes, complexity_rows: list[dict]) -> None:
    order = ["baseline_d2nn", "robust_d2nn", "hybrid", "electronic"]
    indexed = {row["method_id"]: row for row in complexity_rows}
    y = np.arange(len(order))
    parameters = np.asarray([indexed[method]["trainable_parameters"] for method in order])
    representations = np.asarray([indexed[method]["representation_elements"] for method in order])
    if np.any(parameters <= 0) or np.any(representations <= 0):
        raise ValueError("log-scale model counts must be strictly positive")
    for index, method_id in enumerate(order):
        color = METHODS[method_id]["color"]
        ax.plot([representations[index], parameters[index]], [index, index], color="#B8B8B8", lw=1.0, zorder=1)
        ax.scatter(parameters[index], index, s=30, color=color, marker="o", zorder=3)
        ax.scatter(representations[index], index, s=28, facecolor="white", edgecolor=color, marker="s", lw=1.0, zorder=3)
        ax.text(parameters[index] * 1.12, index, f"{parameters[index]:,}", va="center", color=color, fontsize=5.8)
        ax.text(representations[index] / 1.18, index, f"{representations[index]:,}", va="center", ha="right", color=color, fontsize=5.8)
    ax.set_xscale("log")
    ax.set_xlim(4, 35000)
    ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000, 10000]))
    ax.xaxis.set_major_formatter(FuncFormatter(plain_log_tick))
    ax.set_yticks(y)
    ax.set_yticklabels([METHODS[method]["label"] for method in order])
    ax.invert_yaxis()
    ax.set_xlabel("数量（对数尺度）")
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.5)
    ax.scatter([], [], color="#606060", marker="o", s=30, label="可训练参数")
    ax.scatter([], [], facecolor="white", edgecolor="#606060", marker="s", s=28, label="表示元素")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, handletextpad=0.4, columnspacing=1.0)
    ax.set_title("模型与表示规模", loc="left", pad=18, fontsize=7.2)


def plot_timing_accuracy(ax: plt.Axes, seed_rows: list[dict]) -> None:
    for method_id, style in METHODS.items():
        rows = sorted((row for row in seed_rows if row["method_id"] == method_id), key=lambda row: row["training_seed"])
        timing = np.asarray([row["metrics"]["cpu_median_seconds"] for row in rows])
        accuracy = 100 * np.asarray([row["metrics"]["accuracy"] for row in rows])
        if np.any(timing <= 0):
            raise ValueError("log-scale CPU timing must be strictly positive")
        ax.scatter(timing, accuracy, s=18, color=style["color"], marker=style["marker"], alpha=0.55, zorder=2)
        ax.scatter(
            np.mean(timing),
            np.mean(accuracy),
            s=46,
            color=style["color"],
            edgecolor="white",
            linewidth=0.7,
            marker=style["marker"],
            label=style["label"],
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.45, 120)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100]))
    ax.xaxis.set_major_formatter(FuncFormatter(plain_log_tick))
    ax.set_ylim(86.5, 98.0)
    ax.set_xlabel("10 000 样本 CPU 软件前向（s，对数尺度）")
    ax.set_ylabel("干净准确率（%）")
    ax.grid(color="#D8D8D8", linewidth=0.5)
    ax.legend(loc="upper left", ncol=2, handletextpad=0.4, columnspacing=0.9)
    ax.set_title("种子级模拟器开销与干净准确率", loc="left", pad=8, fontsize=7.2)


def main() -> None:
    complexity_rows = load_complexity_rows()
    seed_rows = read_jsonl(RESULTS_ROOT / "clean_summary_all_methods" / "seed_metrics.jsonl")
    if len(seed_rows) != 12 or {(row["method_id"], row["training_seed"]) for row in seed_rows} != {
        (method_id, seed) for method_id in RUNS for seed in (42, 43, 44)
    }:
        raise ValueError("formal Figure 2 requires all 12 clean seed-level records")
    export_source_data(complexity_rows, seed_rows)

    width_mm = 183
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(width_mm / 25.4, 91 / 25.4),
        gridspec_kw={"width_ratios": [0.95, 1.15], "wspace": 0.16},
        constrained_layout=True,
    )
    plot_counts(axes[0], complexity_rows)
    plot_timing_accuracy(axes[1], seed_rows)
    axes[0].text(-0.30, 1.10, "a", transform=axes[0].transAxes, fontsize=8, fontweight="bold", va="top")
    axes[1].text(-0.17, 1.10, "b", transform=axes[1].transAxes, fontsize=8, fontweight="bold", va="top")
    output_base = OUTPUT_DIR / "figure_3_complexity"
    fig.savefig(output_base.with_suffix(".svg"))
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600)
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
