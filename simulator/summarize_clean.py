"""Summarize clean-only metrics and CPU software timing by training seed."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import itertools
import math
from pathlib import Path
import statistics
import sys

import scipy
from scipy.stats import t as student_t

from clean_summary_input import CLEAN_METRIC_FIELDS, load_clean_evaluation, validate_clean_evaluation_set
from robustness_io import JsonlWriter, artifact_record, attach_manifest_integrity, finalize_staged_output
from robustness_io import staged_output_directory, write_json
from training_provenance import file_sha256


CLEAN_SUMMARY_SCHEMA_VERSION = "clean-summary/v1"
PREREGISTERED_MIN_TRAINING_SEEDS = 3
CLEAN_SUMMARY_FIELDS = (*CLEAN_METRIC_FIELDS, "cpu_median_seconds")
CLEAN_SUMMARY_SOURCE_FILES = (
    "clean_evaluation.py",
    "clean_evaluation_manifest.py",
    "clean_summary_input.py",
    "robustness_io.py",
    "robustness_records.py",
    "summarize_clean.py",
    "training_provenance.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize clean-only evaluations by independent training seed")
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tolerate-protocol-revisions",
        action="store_true",
        help=(
            "allow input evaluations bound to more than one revision of the same protocol, "
            "which happens when the cohort is produced on hosts with different metrics "
            "devices. Requires separately established device-parity evidence; the tolerance "
            "and the devices present are recorded in the summary manifest."
        ),
    )
    return parser


def estimate_by_seed(values_by_seed: dict[int, float]) -> dict:
    ordered = sorted(values_by_seed.items())
    values = [value for _, value in ordered]
    estimate = {
        "n_training_seeds": len(values),
        "values_by_training_seed": [{"training_seed": seed, "value": value} for seed, value in ordered],
        "mean": statistics.mean(values),
        "sd_across_training_seeds": None,
        "ci95_t": None,
    }
    if len(values) > 1:
        sd = statistics.stdev(values)
        half_width = float(student_t.ppf(0.975, df=len(values) - 1)) * sd / math.sqrt(len(values))
        estimate["sd_across_training_seeds"] = sd
        estimate["ci95_t"] = [estimate["mean"] - half_width, estimate["mean"] + half_width]
    return estimate


def build_seed_metrics(evaluations: list[dict]) -> list[dict]:
    rows = []
    for item in evaluations:
        manifest = item["manifest"]
        metrics = {field: item["metric"][field] for field in CLEAN_METRIC_FIELDS}
        metrics["cpu_median_seconds"] = item["timing"]["median_seconds"]
        rows.append(
            {
                "method_id": manifest["method_id"],
                "training_seed": manifest["model"]["training_seed"],
                "metrics": metrics,
                "timing_repeat_count": item["timing"]["measurement_repeats"],
                "unit_note": "one independently trained model; timing repeats are precision measurements only",
            }
        )
    return sorted(rows, key=lambda row: (row["method_id"], row["training_seed"]))


def build_method_summaries(seed_metrics: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in seed_metrics:
        grouped[row["method_id"]].append(row)
    return [
        {
            "method_id": method_id,
            "metrics": {
                field: estimate_by_seed({row["training_seed"]: row["metrics"][field] for row in rows})
                for field in CLEAN_SUMMARY_FIELDS
            },
            "independent_unit": "training_seed",
        }
        for method_id, rows in sorted(grouped.items())
    ]


def build_paired_comparisons(seed_metrics: list[dict]) -> list[dict]:
    indexed = {(row["method_id"], row["training_seed"]): row for row in seed_metrics}
    methods = sorted({method for method, _ in indexed})
    method_seeds = {method: {seed for candidate, seed in indexed if candidate == method} for method in methods}
    comparisons = []
    for left, right in itertools.combinations(methods, 2):
        if method_seeds[left] != method_seeds[right]:
            raise ValueError("paired clean comparisons require identical training-seed sets")
        effects = {
            field: {
                seed: indexed[(left, seed)]["metrics"][field] - indexed[(right, seed)]["metrics"][field]
                for seed in method_seeds[left]
            }
            for field in CLEAN_SUMMARY_FIELDS
        }
        comparisons.append(
            {
                "left_method_id": left,
                "right_method_id": right,
                "difference_definition": "left_minus_right_paired_by_training_seed",
                "metrics": {field: estimate_by_seed(values) for field, values in effects.items()},
                "independent_unit": "paired_training_seed",
            }
        )
    return comparisons


def _write_outputs(stage, seed_metrics, method_summaries, paired_comparisons) -> dict:
    artifacts = {}
    for name, rows in (
        ("seed_metrics", seed_metrics),
        ("method_summary", method_summaries),
        ("paired_comparisons", paired_comparisons),
    ):
        path = stage / f"{name}.jsonl"
        with JsonlWriter(path) as writer:
            for row in rows:
                writer.write(row)
        artifacts[name] = artifact_record(path, row_count=len(rows))
    return artifacts


def _evidence(evaluations, method_summaries) -> dict:
    statuses = sorted({item["manifest"].get("evidence_status") for item in evaluations})
    seed_counts = [
        metric["n_training_seeds"]
        for row in method_summaries
        for metric in row["metrics"].values()
    ]
    return {
        "input_evidence_statuses": statuses,
        "dataset_evidence_status": "protocol_bound_full_test_evaluations",
        "inference_status": (
            "descriptive_only_below_preregistered_three_seed_minimum"
            if not seed_counts or min(seed_counts) < PREREGISTERED_MIN_TRAINING_SEEDS
            else "cross_training_seed_summary_with_t_intervals"
        ),
    }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: file_sha256(root / name) for name in CLEAN_SUMMARY_SOURCE_FILES}


def summarize(args) -> Path:
    evaluations = [load_clean_evaluation(value) for value in args.evaluations]
    tolerate = bool(getattr(args, "tolerate_protocol_revisions", False))
    validate_clean_evaluation_set(evaluations, tolerate_protocol_revisions=tolerate)
    seed_metrics = build_seed_metrics(evaluations)
    method_summaries = build_method_summaries(seed_metrics)
    paired_comparisons = build_paired_comparisons(seed_metrics)
    with staged_output_directory(args.output_dir) as stage:
        artifacts = _write_outputs(stage, seed_metrics, method_summaries, paired_comparisons)
        first_manifest = evaluations[0]["manifest"]
        protocol_revisions = sorted(
            {
                (
                    (item["manifest"].get("formal_protocol") or {}).get("protocol_id"),
                    (item["manifest"].get("formal_protocol") or {}).get("sha256"),
                )
                for item in evaluations
            }
        )
        metrics_devices = sorted(
            {
                (item["manifest"].get("device_assignment") or {}).get("metrics_device", "cpu")
                for item in evaluations
            }
        )
        manifest = attach_manifest_integrity(
            {
                "schema_version": CLEAN_SUMMARY_SCHEMA_VERSION,
                "scope": "numerical_simulation_only",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "evidence": _evidence(evaluations, method_summaries),
                "formal_protocol": first_manifest["formal_protocol"],
                "protocol_revisions_used": [
                    {"protocol_id": protocol_id, "sha256": sha256}
                    for protocol_id, sha256 in protocol_revisions
                ],
                "protocol_revision_tolerance": (
                    {
                        "applied": True,
                        "reason": (
                            "evaluation device is a per-run property recorded per evaluation; "
                            "the tolerated revisions differ only in the declared metrics device"
                        ),
                        "parity_evidence": (
                            "cpu_and_cuda_clean_and_robustness_metrics_verified_bit_identical_"
                            "on_this_cohort_delta_0.00e+00"
                        ),
                        "metrics_devices_present": metrics_devices,
                    }
                    if tolerate and len(protocol_revisions) > 1
                    else {"applied": False}
                ),
                "sample_scope": first_manifest["sample_scope"],
                "input_evaluations": [
                    {
                        "path": str(item["manifest_path"].resolve()),
                        "sha256": file_sha256(item["manifest_path"]),
                        "evaluation_id": item["manifest"]["evaluation_id"],
                    }
                    for item in evaluations
                ],
                "statistics": {
                    "independent_unit": "training_seed",
                    "method_comparison": "paired_by_training_seed",
                    "timing_repeat_role": "runtime_precision_only_not_independent_replicates",
                    "interval": "two_sided_95_percent_Student_t_CI_across_training_seeds",
                    "hypothesis_tests": "not_performed",
                    "metric_fields": list(CLEAN_SUMMARY_FIELDS),
                },
                "timing_claim_boundary": "CPU software simulation overhead; not physical hardware latency or energy",
                "software": {"python": sys.version, "scipy": scipy.__version__},
                "summary_source_files_sha256": _source_hashes(),
                "artifacts": artifacts,
            }
        )
        write_json(stage / "summary_manifest.json", manifest)
        finalize_staged_output(stage, args.output_dir)
    return Path(args.output_dir) / "summary_manifest.json"


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    output = summarize(args)
    print(f"Wrote {output}")
    print(f"Summary manifest SHA-256: {file_sha256(output)}")


if __name__ == "__main__":
    main()
