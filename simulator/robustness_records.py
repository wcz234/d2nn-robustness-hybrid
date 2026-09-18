"""Machine-readable prediction and per-draw records for robustness evaluation."""

from __future__ import annotations

import hashlib
import json
from statistics import mean, stdev

import torch

from perturbations import PerturbationDraw


PREDICTION_SCHEMA_VERSION = "robustness-prediction/v1"
DRAW_METRICS_SCHEMA_VERSION = "robustness-draw-metrics/v2"
MACRO_F1_CONVENTION = "all_score_classes_zero_division_0"


def tensor_sha256(tensor: torch.Tensor | None) -> str | None:
    if tensor is None:
        return None
    values = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def macro_f1_from_confusion(confusion: torch.Tensor) -> float:
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1] or confusion.shape[0] < 1:
        raise ValueError("confusion matrix must be non-empty and square")
    if (confusion < 0).any():
        raise ValueError("confusion matrix counts must be non-negative")
    values = confusion.to(dtype=torch.float64)
    true_positive = values.diag()
    false_positive = values.sum(dim=0) - true_positive
    false_negative = values.sum(dim=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    class_f1 = torch.where(denominator > 0, 2 * true_positive / denominator, 0.0)
    return float(class_f1.mean().item())


def optical_draw_record(draw: PerturbationDraw) -> dict:
    layers = [
        {
            "lateral_shift_px": list(layer.lateral_shift_px),
            "gap_spacing_error_m": layer.axial_shift_m,
            "phase_noise_sha256": tensor_sha256(layer.phase_noise_rad),
        }
        for layer in draw.layers
    ]
    payload = {"phase_quantization_levels": draw.phase_quantization_levels, "layers": layers}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {**payload, "fingerprint_sha256": hashlib.sha256(canonical).hexdigest()}


def prediction_rows(
    *,
    context: dict,
    indices: torch.Tensor,
    targets: torch.Tensor,
    scores: torch.Tensor,
    contrasts: torch.Tensor,
    include_scores: bool,
    detector_noise_enabled: bool,
) -> list[dict]:
    predictions = scores.argmax(dim=1)
    rows = []
    for offset, dataset_index in enumerate(indices.tolist()):
        target = int(targets[offset].item())
        prediction = int(predictions[offset].item())
        row = {
            **context,
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "dataset_index": dataset_index,
            "sample_id": str(dataset_index),
            "target": target,
            "prediction": prediction,
            "correct": prediction == target,
            "target_score": float(scores[offset, target].item()),
            "max_score": float(scores[offset].max().item()),
            "contrast": float(contrasts[offset].item()),
            "detector_noise_id": (
                f"draw-{context['draw_id']}/sample-{dataset_index}" if detector_noise_enabled else None
            ),
        }
        if include_scores:
            row["scores"] = [float(value) for value in scores[offset].tolist()]
        rows.append(row)
    return rows


def draw_metric_record(*, context: dict, totals: dict, optical_record: dict | None, detector_table) -> dict:
    sample_count = totals["sample_count"]
    confusion = totals["confusion_matrix"]
    if confusion is None or int(confusion.sum().item()) != sample_count:
        raise ValueError("confusion matrix does not match the evaluated sample count")
    return {
        **context,
        "schema_version": DRAW_METRICS_SCHEMA_VERSION,
        "n_samples": sample_count,
        "n_correct": totals["correct"],
        "accuracy": totals["correct"] / sample_count,
        "macro_f1": macro_f1_from_confusion(confusion),
        "macro_f1_convention": MACRO_F1_CONVENTION,
        "confusion_matrix": confusion.tolist(),
        "mean_cross_entropy": totals["cross_entropy"] / sample_count,
        "mean_contrast": totals["contrast"] / sample_count,
        "optical_draw_fingerprint_sha256": None if optical_record is None else optical_record["fingerprint_sha256"],
        "optical_draw": optical_record,
        "detector_noise_scope": "disabled" if detector_table is None else "per_sample_feature_table",
        "detector_noise_shape": None if detector_table is None else list(detector_table.shape),
        "detector_noise_sha256": tensor_sha256(detector_table),
    }


def summarize_draw_metrics(draw_metrics: list[dict]) -> dict:
    if not draw_metrics:
        raise ValueError("at least one draw metric is required")
    accuracies = [item["accuracy"] for item in draw_metrics]
    contrasts = [item["mean_contrast"] for item in draw_metrics]
    return {
        "draw_count": len(draw_metrics),
        "accuracy_mean": mean(accuracies),
        "accuracy_sd_across_draws": stdev(accuracies) if len(accuracies) > 1 else None,
        "accuracy_min": min(accuracies),
        "accuracy_max": max(accuracies),
        "contrast_mean": mean(contrasts),
        "contrast_sd_across_draws": stdev(contrasts) if len(contrasts) > 1 else None,
        "statistical_unit_note": "draw variability for one trained model; not an independent training replicate",
    }
