"""Auditable manifest construction for one clean-only evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import torch

from robustness_io import attach_manifest_integrity
from robustness_plan import SIMULATION_SCOPE
from training_provenance import cli_configuration, environment_manifest, file_sha256


CLEAN_EVALUATION_SCHEMA_VERSION = "clean-evaluation/v1"
CLEAN_EVALUATION_SOURCE_FILES = (
    "artifacts.py",
    "clean_evaluation.py",
    "clean_evaluation_manifest.py",
    "d2nn.py",
    "evaluate_clean.py",
    "model_variants.py",
    "perturbations.py",
    "robustness_evaluation.py",
    "robustness_io.py",
    "robustness_plan.py",
    "robustness_records.py",
    "robustness_target.py",
    "tasks.py",
    "training_provenance.py",
)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: file_sha256(root / name) for name in CLEAN_EVALUATION_SOURCE_FILES}


def _model_record(target) -> dict:
    parameters = list(target.model.parameters())
    return {
        "variant": target.model_variant,
        "class_name": type(target.model).__name__,
        "training_seed": target.training_seed,
        "feature_schema_id": target.feature_schema_id,
        "detector_feature_count": getattr(target.model, "detector_feature_count", None),
        "score_semantics": target.score_semantics,
        "total_parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
    }


def _metric_summary(metric: dict) -> dict:
    fields = (
        "n_samples",
        "n_correct",
        "accuracy",
        "macro_f1",
        "macro_f1_convention",
        "confusion_matrix",
        "mean_cross_entropy",
        "mean_contrast",
    )
    return {field: metric[field] for field in fields}


def _evidence_status(*, selection: dict, protocol_binding: dict | None) -> str:
    if selection["selection"] != "full_test":
        return "subset_smoke_only_not_a_main_result"
    if protocol_binding is None:
        return "unregistered_full_test_requires_protocol_and_cross_training_seed_summary"
    return "protocol_bound_full_test_single_seed_requires_cross_training_seed_summary"


def build_clean_evaluation_manifest(
    *,
    args,
    target,
    selection: dict,
    artifacts: dict,
    metric: dict,
    timing: dict,
    evaluation_id: str,
    protocol_binding: dict | None,
    metrics_device: str = "cpu",
    timing_device: str = "cpu",
) -> dict:
    device = torch.device("cpu")
    manifest = {
        "schema_version": CLEAN_EVALUATION_SCHEMA_VERSION,
        "scope": SIMULATION_SCOPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_id": evaluation_id,
        "method_id": args.method_id,
        "evidence_status": _evidence_status(selection=selection, protocol_binding=protocol_binding),
        "formal_protocol": protocol_binding,
        "checkpoint": {
            "path": str(target.checkpoint_path.resolve()),
            "sha256": target.checkpoint_sha256,
        },
        "sample_scope": selection,
        "model": _model_record(target),
        "metrics": _metric_summary(metric),
        "timing": timing,
        "device_assignment": {
            "metrics_device": metrics_device,
            "timing_device": timing_device,
            "rationale": (
                "clean metrics and per-sample predictions may run on an accelerator; CPU "
                "software timing is always measured on CPU because that overhead is the "
                "measured quantity, not an implementation detail"
            ),
            "metrics_device_parity": (
                "cpu_and_cuda_clean_metrics_verified_bit_identical_on_this_cohort"
                if metrics_device == "cuda"
                else "not_applicable"
            ),
            "timing_claim_boundary": "cpu_software_simulation_overhead_only",
        },
        "statistics": {
            "independent_unit": "training_seed",
            "timing_repeats_role": "runtime_precision_only_not_independent_replicates",
            "reporting_requirement": "aggregate_clean_metrics_across_planned_training_seeds",
        },
        "runtime": {
            "device": metrics_device,
            "batch_size": args.batch_size,
            "num_workers": 0,
            "cpu_threads": args.cpu_threads,
            "deterministic_algorithms": metrics_device == "cpu",
            "include_scores": args.include_scores,
        },
        "artifacts": artifacts,
        "evaluation_source_files_sha256": _source_hashes(),
        "environment": environment_manifest(device, Path(__file__).parent),
        "cli_configuration": cli_configuration(args),
    }
    return attach_manifest_integrity(manifest)
