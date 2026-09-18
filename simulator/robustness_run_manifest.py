"""Provenance manifest construction for one robustness-evaluation run."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from robustness_io import attach_manifest_integrity
from robustness_plan import SIMULATION_SCOPE
from training_provenance import cli_configuration, environment_manifest, file_sha256


EVALUATION_SCHEMA_VERSION = "robustness-evaluation/v2"
EVALUATION_SOURCE_FILES = (
    "artifacts.py",
    "d2nn.py",
    "evaluate_robustness.py",
    "model_variants.py",
    "perturbations.py",
    "robustness_cohort.py",
    "robustness_evaluation.py",
    "robustness_io.py",
    "robustness_plan.py",
    "robustness_records.py",
    "robustness_run_manifest.py",
    "robustness_target.py",
    "tasks.py",
    "training_provenance.py",
)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: file_sha256(root / name) for name in EVALUATION_SOURCE_FILES}


def _model_record(target) -> dict:
    manifest = target.manifest
    return {
        "variant": target.model_variant,
        "class_name": type(target.model).__name__,
        "training_seed": target.training_seed,
        "feature_schema_id": target.feature_schema_id,
        "detector_feature_count": target.model.detector_feature_count,
        "score_semantics": target.score_semantics,
        "activation_type": manifest.get("activation_type"),
        "activation_positions": manifest.get("activation_positions"),
        "activation_hparams": manifest.get("activation_hparams"),
        "propagation_backend": manifest.get("propagation_backend"),
        "propagation_chunk_size": manifest.get("propagation_chunk_size"),
    }


def build_evaluation_manifest(
    *,
    args,
    plan_path,
    plan,
    target,
    device,
    selection,
    artifacts,
    summaries,
    evaluation_id,
):
    evidence_status = (
        "subset_smoke_only_not_a_main_result"
        if selection["selection"] == "test_prefix_subset"
        else "full_test_evaluation_requires_cross_training_seed_summary"
    )
    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "scope": SIMULATION_SCOPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_id": evaluation_id,
        "method_id": args.method_id,
        "evidence_status": evidence_status,
        "plan": {"path": str(plan_path.resolve()), "sha256": file_sha256(plan_path), "plan_id": plan["plan_id"]},
        "checkpoint": {"path": str(target.checkpoint_path.resolve()), "sha256": target.checkpoint_sha256},
        "cohort": plan["cohort"],
        "sample_scope": selection,
        "model": _model_record(target),
        "runtime": {
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "deterministic_algorithms": True,
            "include_scores": args.include_scores,
        },
        "sampling_protocol": plan["sampling_protocol"],
        "statistics": plan["statistics"],
        "condition_summaries": summaries,
        "artifacts": artifacts,
        "evaluation_source_files_sha256": _source_hashes(),
        "environment": environment_manifest(device, Path(__file__).parent),
        "cli_configuration": cli_configuration(args),
    }
    return attach_manifest_integrity(manifest)
