"""Deterministic, audit-friendly robustness evaluation primitives."""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler

from perturbations import PerturbationConfig, PerturbationDraw, PerturbationSampler
from robustness_plan import plan_perturbation_config, validate_condition_plan
from robustness_records import (
    draw_metric_record as _draw_metric_record,
    optical_draw_record,
    prediction_rows as _prediction_rows,
    summarize_draw_metrics,
)


def _model_real_dtype(model) -> torch.dtype:
    parameter = next(model.parameters(), None)
    return torch.float32 if parameter is None else parameter.dtype


def materialize_optical_draw(model, config: PerturbationConfig, *, seed: int, device="cpu") -> PerturbationDraw:
    optical_config = PerturbationConfig(
        lateral_shift_max_px=config.lateral_shift_max_px,
        axial_shift_max_m=config.axial_shift_max_m,
        phase_noise_std_rad=config.phase_noise_std_rad,
        phase_quantization_levels=config.phase_quantization_levels,
    )
    return PerturbationSampler(optical_config, seed=seed).sample(
        num_layers=len(model.layers),
        size=model.size,
        dtype=_model_real_dtype(model),
        device=device,
    )


def build_detector_noise_table(
    *,
    sample_count: int,
    feature_count: int,
    std_relative: float,
    seed: int,
) -> torch.Tensor | None:
    if sample_count < 1 or feature_count < 1:
        raise ValueError("sample_count and feature_count must be positive")
    if not math.isfinite(std_relative) or std_relative < 0:
        raise ValueError("std_relative must be finite and non-negative")
    if std_relative == 0:
        return None
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((sample_count, feature_count), generator=generator, dtype=torch.float32) * std_relative


def _compose_batch_draw(optical_draw: PerturbationDraw, detector_noise: torch.Tensor | None) -> PerturbationDraw:
    return PerturbationDraw(
        layers=optical_draw.layers,
        phase_quantization_levels=optical_draw.phase_quantization_levels,
        detector_noise_relative=detector_noise,
    )


def validate_evaluation_loader(loader) -> int:
    sample_count = len(loader.dataset)
    if sample_count < 1:
        raise ValueError("robustness evaluation requires a non-empty dataset")
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("robustness evaluation requires a sequential, non-shuffled loader")
    if getattr(loader, "drop_last", False):
        raise ValueError("robustness evaluation requires drop_last=False")
    return sample_count


def _validate_batch_indices(indices, *, sample_count: int, seen_indices: set[int], draw_id: int) -> torch.Tensor:
    index_values = torch.as_tensor(indices, dtype=torch.long, device="cpu")
    if index_values.ndim != 1:
        raise ValueError("dataset indices must be a one-dimensional tensor")
    values = index_values.tolist()
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate dataset indices within draw {draw_id} and one batch")
    duplicates = seen_indices.intersection(values)
    if duplicates:
        raise ValueError(f"duplicate dataset indices within draw {draw_id}: {sorted(duplicates)}")
    if values and (min(values) < 0 or max(values) >= sample_count):
        raise ValueError("dataset indices must address the frozen selected test scope")
    seen_indices.update(values)
    return index_values


def _ensure_finite_outputs(scores: torch.Tensor, logits: torch.Tensor, contrasts: torch.Tensor) -> None:
    for name, values in (("scores", scores), ("logits", logits), ("contrast", contrasts)):
        if not torch.isfinite(values).all():
            raise ValueError(f"model produced non-finite {name} during robustness evaluation")


def _evaluate_batch(*, model, data, target, indices, batch_draw, context, include_scores) -> tuple[list[dict], dict]:
    target = target.to(context["device"])
    model_input = data.to(context["device"])
    if batch_draw is None:
        result = model.forward_with_metrics(model_input, target=target)
    else:
        result = model.forward_with_metrics(model_input, target=target, perturbation_draw=batch_draw)
    scores = result["scores"]
    logits = result["logits"]
    contrasts = result["contrast"]
    _ensure_finite_outputs(scores, logits, contrasts)
    rows = _prediction_rows(
        context={key: value for key, value in context.items() if key != "device"},
        indices=indices,
        targets=target.detach().cpu(),
        scores=scores.detach().cpu(),
        contrasts=contrasts.detach().cpu(),
        include_scores=include_scores,
        detector_noise_enabled=batch_draw is not None and batch_draw.detector_noise_relative is not None,
    )
    predictions = scores.argmax(dim=1)
    num_classes = int(scores.shape[1])
    encoded_pairs = target * num_classes + predictions
    totals = {
        "sample_count": len(rows),
        "correct": sum(int(row["correct"]) for row in rows),
        "cross_entropy": float(F.cross_entropy(logits, target, reduction="sum").item()),
        "contrast": float(contrasts.sum().item()),
        "confusion_matrix": torch.bincount(
            encoded_pairs,
            minlength=num_classes**2,
        ).reshape(num_classes, num_classes).cpu(),
    }
    return rows, totals


def _add_totals(destination: dict, source: dict) -> None:
    for key in ("sample_count", "correct", "cross_entropy", "contrast"):
        destination[key] += source[key]
    confusion = source["confusion_matrix"]
    if destination["confusion_matrix"] is None:
        destination["confusion_matrix"] = confusion.clone()
    else:
        destination["confusion_matrix"] += confusion


def _empty_draw_totals() -> dict:
    return {
        "sample_count": 0,
        "correct": 0,
        "cross_entropy": 0.0,
        "contrast": 0.0,
        "confusion_matrix": None,
    }


@torch.inference_mode()
def _evaluate_one_draw(
    *,
    model,
    loader,
    device,
    context: dict,
    optical_draw: PerturbationDraw,
    detector_table: torch.Tensor | None,
    include_scores: bool,
    prediction_sink: Callable[[dict], None],
) -> dict:
    sample_count = validate_evaluation_loader(loader)
    totals = _empty_draw_totals()
    seen_indices: set[int] = set()
    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            raise ValueError("robustness datasets must return data, target, and stable dataset_index")
        data, target, indices = batch
        index_values = _validate_batch_indices(
            indices,
            sample_count=sample_count,
            seen_indices=seen_indices,
            draw_id=context["draw_id"],
        )
        detector_noise = None if detector_table is None else detector_table.index_select(0, index_values).to(device)
        rows, batch_totals = _evaluate_batch(
            model=model,
            data=data,
            target=target,
            indices=index_values,
            batch_draw=_compose_batch_draw(optical_draw, detector_noise),
            context={**context, "device": device},
            include_scores=include_scores,
        )
        for row in rows:
            prediction_sink(row)
        _add_totals(totals, batch_totals)
    if totals["sample_count"] != sample_count or len(seen_indices) != sample_count:
        raise ValueError("prediction row count does not match the evaluated dataset size")
    return _draw_metric_record(
        context=context,
        totals=totals,
        optical_record=optical_draw_record(optical_draw),
        detector_table=detector_table,
    )


def evaluate_condition(
    *,
    model,
    loader,
    device,
    condition: dict,
    evaluation_id: str,
    include_scores: bool,
    prediction_sink: Callable[[dict], None],
    record_context: dict | None = None,
) -> tuple[list[dict], dict]:
    validate_condition_plan(condition)
    model.eval()
    config = plan_perturbation_config(condition)
    draw_metrics = []
    for draw_plan in condition["draws"]:
        context = {
            **(record_context or {}),
            "evaluation_id": evaluation_id,
            "condition_id": condition["condition_id"],
            **draw_plan,
        }
        optical_draw = materialize_optical_draw(model, config, seed=draw_plan["optical_seed"], device=device)
        detector_table = build_detector_noise_table(
            sample_count=len(loader.dataset),
            feature_count=model.detector_feature_count,
            std_relative=config.detector_noise_std_relative,
            seed=draw_plan["detector_seed"],
        )
        draw_metrics.append(
            _evaluate_one_draw(
                model=model,
                loader=loader,
                device=device,
                context=context,
                optical_draw=optical_draw,
                detector_table=detector_table,
                include_scores=include_scores,
                prediction_sink=prediction_sink,
            )
        )
    return draw_metrics, summarize_draw_metrics(draw_metrics)


def _validate_clean_only_condition(condition: dict) -> None:
    validate_condition_plan(condition)
    if plan_perturbation_config(condition) != PerturbationConfig():
        raise ValueError("clean-only evaluation rejects optical, quantization, and detector perturbations")


@torch.inference_mode()
def evaluate_clean_condition(
    *,
    model,
    loader,
    device,
    condition: dict,
    evaluation_id: str,
    include_scores: bool,
    prediction_sink: Callable[[dict], None],
    record_context: dict | None = None,
) -> tuple[list[dict], dict]:
    _validate_clean_only_condition(condition)
    model.eval()
    draw_plan = condition["draws"][0]
    context = {
        **(record_context or {}),
        "evaluation_id": evaluation_id,
        "condition_id": condition["condition_id"],
        **draw_plan,
    }
    sample_count = validate_evaluation_loader(loader)
    totals = _empty_draw_totals()
    seen_indices: set[int] = set()
    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            raise ValueError("clean evaluation datasets must return data, target, and stable dataset_index")
        data, target, indices = batch
        index_values = _validate_batch_indices(
            indices,
            sample_count=sample_count,
            seen_indices=seen_indices,
            draw_id=context["draw_id"],
        )
        rows, batch_totals = _evaluate_batch(
            model=model,
            data=data,
            target=target,
            indices=index_values,
            batch_draw=None,
            context={**context, "device": device},
            include_scores=include_scores,
        )
        for row in rows:
            prediction_sink(row)
        _add_totals(totals, batch_totals)
    if totals["sample_count"] != sample_count or len(seen_indices) != sample_count:
        raise ValueError("prediction row count does not match the evaluated dataset size")
    metric = _draw_metric_record(
        context=context,
        totals=totals,
        optical_record=None,
        detector_table=None,
    )
    return [metric], summarize_draw_metrics([metric])
