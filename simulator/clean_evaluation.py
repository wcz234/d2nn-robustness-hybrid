"""Clean-only classification contracts and CPU software timing."""

from __future__ import annotations

import math
from statistics import median
from time import perf_counter
from typing import Callable

import torch

from perturbations import PerturbationConfig
from robustness_evaluation import validate_evaluation_loader
from robustness_plan import build_draw_plans, serialized_perturbation_config


CLEAN_CONDITION_ID = "clean_continuous"
CLEAN_OPTICAL_SEED = 0
CLEAN_DETECTOR_SEED = 0
METHOD_VARIANTS = {
    "baseline_d2nn": "d2nn",
    "robust_d2nn": "d2nn",
    "hybrid": "hybrid",
    # Quantization-aware training changes the training procedure, not the architecture,
    # so a QAT hybrid checkpoint still reports model variant "hybrid" while being tracked
    # as its own method in the cohort.
    "hybrid_qat": "hybrid",
    # Phase filtering is likewise a training-only regularization on the same architecture.
    "phase_filtered_d2nn": "d2nn",
    # Ablation: hybrid optical front end with a linear electronic readout.
    "hybrid_linear_head": "hybrid",
    "electronic": "electronic",
    "lenet5": "lenet5",
}
METHOD_TRAINING_PERTURBATIONS_ENABLED = {
    "baseline_d2nn": False,
    "robust_d2nn": True,
    "hybrid": False,
    # Training-time phase quantization is recorded separately from the optical/detector
    # perturbation flag: `training_perturbations.enabled` stays false for these, while
    # `train_phase_quantization_levels` / `train_phase_filter_std_px` carry the identity.
    "hybrid_qat": False,
    "phase_filtered_d2nn": False,
    "hybrid_linear_head": False,
    "electronic": False,
    "lenet5": False,
}
TIMING_SCHEMA_VERSION = "cpu-software-simulation-timing/v1"


def validate_method_target(method_id: str, target) -> None:
    expected_variant = METHOD_VARIANTS.get(method_id)
    if expected_variant is None:
        raise ValueError(f"unsupported clean-evaluation method_id {method_id!r}")
    if target.model_variant != expected_variant:
        raise ValueError(
            f"method {method_id!r} requires model variant {expected_variant!r}, "
            f"not {target.model_variant!r}"
        )
    training = target.manifest.get("training_perturbations")
    if not isinstance(training, dict) or not isinstance(training.get("enabled"), bool):
        raise ValueError("checkpoint manifest requires boolean training_perturbations.enabled")
    expected_enabled = METHOD_TRAINING_PERTURBATIONS_ENABLED[method_id]
    if training["enabled"] is not expected_enabled:
        raise ValueError(
            f"method {method_id!r} requires training_perturbations.enabled={expected_enabled}"
        )


def build_clean_condition(
    *,
    condition_id: str = CLEAN_CONDITION_ID,
    optical_seed: int = CLEAN_OPTICAL_SEED,
    detector_seed: int = CLEAN_DETECTOR_SEED,
) -> dict:
    return {
        "condition_id": condition_id,
        "perturbation_config": serialized_perturbation_config(PerturbationConfig()),
        "draws": build_draw_plans(
            condition_id=condition_id,
            draw_count=1,
            optical_seed=optical_seed,
            detector_seed=detector_seed,
        ),
    }


def linear_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = probability * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    fraction = rank - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


@torch.inference_mode()
def _run_clean_forward_pass(*, model, loader, device: torch.device) -> int:
    expected_samples = validate_evaluation_loader(loader)
    observed_samples = 0
    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            raise ValueError("clean timing datasets must return data, target, and stable dataset_index")
        data, target, _ = batch
        result = model.forward_with_metrics(data.to(device), target=target.to(device))
        scores = result.get("scores")
        if not isinstance(scores, torch.Tensor) or scores.ndim != 2 or scores.shape[0] != data.shape[0]:
            raise ValueError("model timing pass requires a two-dimensional scores tensor")
        observed_samples += int(scores.shape[0])
    if observed_samples != expected_samples:
        raise ValueError("timing pass sample count does not match the selected test scope")
    return observed_samples


def measure_clean_inference(
    *,
    model,
    loader,
    device: torch.device,
    warmup_repeats: int,
    measurement_repeats: int,
    clock: Callable[[], float] | None = None,
) -> dict:
    if device.type != "cpu":
        raise ValueError("clean software timing is defined only for CPU simulation")
    if warmup_repeats < 0 or measurement_repeats < 1:
        raise ValueError("warmup_repeats must be non-negative and measurement_repeats positive")
    timer = perf_counter if clock is None else clock
    was_training = model.training
    model.eval()
    try:
        for _ in range(warmup_repeats):
            _run_clean_forward_pass(model=model, loader=loader, device=device)
        elapsed_seconds = []
        sample_count = len(loader.dataset)
        for _ in range(measurement_repeats):
            started = timer()
            observed_samples = _run_clean_forward_pass(model=model, loader=loader, device=device)
            elapsed = timer() - started
            if observed_samples != sample_count or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("invalid clean CPU software-timing observation")
            elapsed_seconds.append(elapsed)
    finally:
        model.train(was_training)
    q1_seconds = linear_quantile(elapsed_seconds, 0.25)
    q3_seconds = linear_quantile(elapsed_seconds, 0.75)
    median_seconds = median(elapsed_seconds)
    return {
        "schema_version": TIMING_SCHEMA_VERSION,
        "scope": "cpu_software_simulation_overhead",
        "claim_boundary": "not_physical_optical_hardware_latency_energy_or_efficiency",
        "measurement_scope": (
            "full_selected_test_loader_forward_with_metrics_including_data_iteration_"
            "and_shape_validation_excluding_metric_aggregation_and_artifact_io"
        ),
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "warmup_repeats": warmup_repeats,
        "measurement_repeats": measurement_repeats,
        "sample_count_per_repeat": sample_count,
        "elapsed_seconds": elapsed_seconds,
        "median_seconds": median_seconds,
        "q1_seconds": q1_seconds,
        "q3_seconds": q3_seconds,
        "iqr_seconds": q3_seconds - q1_seconds,
        "median_seconds_per_sample": median_seconds / sample_count,
    }
