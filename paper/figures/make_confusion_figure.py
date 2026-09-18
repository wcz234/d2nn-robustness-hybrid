"""Render clean-condition class confusion from formal per-seed metrics."""

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
        "font.size": 7,
        "axes.linewidth": 0.7,
    }
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "formal_mnist_v1" / "clean"
OUTPUT_DIR = Path(__file__).resolve().parent
RUNS = {
    "baseline_d2nn": ["baseline_d2nn_seed42", "baseline_d2nn_seed43", "baseline_d2nn_seed44"],
    "robust_d2nn": ["robust_d2nn_seed42", "robust_d2nn_seed43", "robust_d2nn_seed44"],
    "hybrid": ["hybrid_seed42", "hybrid_seed43", "hybrid_seed44_retry1"],
    "electronic": ["electronic_seed42", "electronic_seed43", "electronic_seed44"],
}
METHODS = {
    "baseline_d2nn": "基准 D²NN",
    "robust_d2nn": "鲁棒 D²NN",
    "hybrid": "混合模型",
    "electronic": "纯电子基线",
}


def load_metrics(method_id: str, run_name: str) -> dict:
    path = RESULTS_ROOT / run_name / "clean_metrics.json"
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    if matrix.shape != (10, 10):
        raise ValueError(f"expected 10x10 confusion matrix for {method_id}/{run_name}")
    if not np.all(matrix.sum(axis=1) > 0):
        raise ValueError(f"empty true-label row for {method_id}/{run_name}")
    return metrics


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / matrix.sum(axis=1, keepdims=True)


def collect_data() -> tuple[dict[str, list[tuple[int, np.ndarray, float]]], list[dict[str, object]]]:
    grouped: dict[str, list[tuple[int, np.ndarray, float]]] = {}
    source_rows: list[dict[str, object]] = []
    for method_id, run_names in RUNS.items():
        if len(run_names) != 3:
            raise ValueError(f"method {method_id} must contain three training seeds")
        grouped[method_id] = []
        for run_name in run_names:
            metrics = load_metrics(method_id, run_name)
            seed = int(metrics["training_seed"])
            matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
            normalized = row_normalize(matrix)
            grouped[method_id].append((seed, normalized, float(metrics["accuracy"])))
            for true_label in range(10):
                for predicted_label in range(10):
                    source_rows.append(
                        {
                            "method_id": method_id,
                            "method_label": METHODS[method_id],
                            "training_seed": seed,
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "count": int(matrix[true_label, predicted_label]),
                            "row_fraction": normalized[true_label, predicted_label],
                        }
                    )
    return grouped, source_rows


def export_source_data(rows: list[dict[str, object]]) -> None:
    path = OUTPUT_DIR / "source_data_figure_4.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_panel(ax: plt.Axes, method_id: str, records: list[tuple[int, np.ndarray, float]], panel_label: str) -> None:
    mean_matrix = np.mean([record[1] for record in records], axis=0)
    mean_accuracy = 100 * np.mean([record[2] for record in records])
    image = ax.imshow(mean_matrix, cmap="Blues", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    ax.set_title(f"{METHODS[method_id]}\n准确率 {mean_accuracy:.2f}%", fontsize=8, pad=7)
    ax.set_xticks(range(10), labels=range(10), fontsize=6)
    ax.set_yticks(range(10), labels=range(10), fontsize=6)
    ax.set_xlabel("预测数字", fontsize=7)
    ax.set_ylabel("真实数字", fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(-0.18, 1.12, panel_label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    return image


def main() -> None:
    grouped, source_rows = collect_data()
    export_source_data(source_rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.9), constrained_layout=True)
    images = []
    for ax, (method_id, records), panel_label in zip(axes.flat, grouped.items(), "abcd"):
        images.append(draw_panel(ax, method_id, records, panel_label))
    colorbar = fig.colorbar(images[-1], ax=axes.ravel().tolist(), shrink=0.84, pad=0.025)
    colorbar.set_label("逐行归一化频率", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)
    output_base = OUTPUT_DIR / "figure_4_confusion"
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
