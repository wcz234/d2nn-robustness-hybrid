"""Aggregate robustness draws by training seed, then compare methods by paired seed."""

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

from robustness_io import (
    JsonlWriter,
    artifact_record,
    attach_manifest_integrity,
    finalize_staged_output,
    staged_output_directory,
    write_json,
)
from robustness_summary_input import load_evaluation, validate_evaluation_set
from training_provenance import file_sha256


SUMMARY_SCHEMA_VERSION = "robustness-summary/v2"
PREREGISTERED_MIN_TRAINING_SEEDS = 3
SUMMARY_SOURCE_FILES = (
    "robustness_io.py",
    "robustness_plan.py",
    "robustness_records.py",
    "robustness_summary_input.py",
    "summarize_robustness.py",
    "training_provenance.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize robustness without treating draws as training replicates")
    parser.add_argument("--evaluations", nargs="+", required=True, help="evaluation manifests or their directories")
    parser.add_argument("--output-dir", required=True)
    return parser


def build_seed_metrics(evaluations: list[dict], metric_fields: tuple[str, ...]) -> list[dict]:
    records = []
    for item in evaluations:
        manifest = item["manifest"]
        grouped = defaultdict(list)
        for row in item["draw_rows"]:
            grouped[row["condition_id"]].append(row)
        for condition_id, rows in sorted(grouped.items()):
            rows.sort(key=lambda row: row["draw_id"])
            records.append(
                {
                    "method_id": manifest["method_id"],
                    "training_seed": manifest["model"]["training_seed"],
                    "condition_id": condition_id,
                    "draw_ids": [row["draw_id"] for row in rows],
                    "draw_count": len(rows),
                    "metrics": {field: statistics.mean(row[field] for row in rows) for field in metric_fields},
                    "unit_note": "one independently trained model; draws were aggregated before cross-seed inference",
                }
            )
    return sorted(records, key=lambda row: (row["method_id"], row["training_seed"], row["condition_id"]))


def _estimate_by_seed(values_by_seed: dict[int, float]) -> dict:
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


def build_method_summaries(seed_metrics: list[dict], metric_fields: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in seed_metrics:
        grouped[(row["method_id"], row["condition_id"])].append(row)
    summaries = []
    for (method_id, condition_id), rows in sorted(grouped.items()):
        summaries.append(
            {
                "method_id": method_id,
                "condition_id": condition_id,
                "metrics": {
                    field: _estimate_by_seed({row["training_seed"]: row["metrics"][field] for row in rows})
                    for field in metric_fields
                },
                "independent_unit": "training_seed",
            }
        )
    return summaries


def _draw_rows_by_identity(evaluations: list[dict]) -> dict:
    indexed = {}
    for item in evaluations:
        manifest = item["manifest"]
        method = manifest["method_id"]
        seed = manifest["model"]["training_seed"]
        for row in item["draw_rows"]:
            indexed[(method, seed, row["condition_id"], row["draw_id"])] = row
    return indexed


def _paired_condition_record(*, left, right, condition_id, seeds, indexed, metric_fields) -> dict:
    effects = {field: {} for field in metric_fields}
    for seed in sorted(seeds):
        left_ids = {key[3] for key in indexed if key[:3] == (left, seed, condition_id)}
        right_ids = {key[3] for key in indexed if key[:3] == (right, seed, condition_id)}
        if not left_ids or left_ids != right_ids:
            raise ValueError("paired methods require identical draw IDs for every training seed")
        for field in metric_fields:
            differences = [
                indexed[(left, seed, condition_id, draw_id)][field]
                - indexed[(right, seed, condition_id, draw_id)][field]
                for draw_id in sorted(left_ids)
            ]
            effects[field][seed] = statistics.mean(differences)
    return {
        "left_method_id": left,
        "right_method_id": right,
        "difference_definition": "left_minus_right_after_within_seed_paired_draw_aggregation",
        "condition_id": condition_id,
        "metrics": {field: _estimate_by_seed(effects[field]) for field in metric_fields},
        "independent_unit": "paired_training_seed",
    }


def build_paired_comparisons(evaluations: list[dict], metric_fields: tuple[str, ...]) -> list[dict]:
    indexed = _draw_rows_by_identity(evaluations)
    methods = sorted({key[0] for key in indexed})
    conditions = sorted({key[2] for key in indexed})
    method_seeds = {method: {key[1] for key in indexed if key[0] == method} for method in methods}
    comparisons = []
    for left, right in itertools.combinations(methods, 2):
        if method_seeds[left] != method_seeds[right]:
            raise ValueError("paired method comparisons require identical training-seed sets")
        for condition_id in conditions:
            comparisons.append(
                _paired_condition_record(
                    left=left,
                    right=right,
                    condition_id=condition_id,
                    seeds=method_seeds[left],
                    indexed=indexed,
                    metric_fields=metric_fields,
                )
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


def _summary_evidence_status(evaluations, method_summaries) -> dict:
    input_statuses = sorted({item["manifest"].get("evidence_status") for item in evaluations})
    seed_counts = [
        metric["n_training_seeds"]
        for row in method_summaries
        for metric in row["metrics"].values()
    ]
    return {
        "input_evidence_statuses": input_statuses,
        "dataset_evidence_status": (
            "subset_smoke_only_not_a_main_result"
            if "subset_smoke_only_not_a_main_result" in input_statuses
            else "full_test_evaluations"
        ),
        "inference_status": (
            "descriptive_only_below_preregistered_three_seed_minimum"
            if not seed_counts or min(seed_counts) < PREREGISTERED_MIN_TRAINING_SEEDS
            else "cross_training_seed_summary_with_t_intervals"
        ),
    }


def _summary_source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: file_sha256(root / name) for name in SUMMARY_SOURCE_FILES}


def summarize(args) -> Path:
    evaluations = [load_evaluation(value) for value in args.evaluations]
    validate_evaluation_set(evaluations)
    metric_fields = tuple(evaluations[0]["metric_fields"])
    seed_metrics = build_seed_metrics(evaluations, metric_fields)
    method_summaries = build_method_summaries(seed_metrics, metric_fields)
    paired_comparisons = build_paired_comparisons(evaluations, metric_fields)
    with staged_output_directory(args.output_dir) as stage:
        artifacts = _write_outputs(stage, seed_metrics, method_summaries, paired_comparisons)
        first_manifest = evaluations[0]["manifest"]
        manifest = attach_manifest_integrity(
            {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "scope": "numerical_simulation_only",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "evidence": _summary_evidence_status(evaluations, method_summaries),
                "plan": first_manifest["plan"],
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
                    "draw_handling": "aggregate_within_training_seed_before_cross_seed_summary",
                    "method_comparison": "paired_by_training_seed_condition_and_draw_id",
                    "interval": "two_sided_95_percent_Student_t_CI_across_training_seeds",
                    "hypothesis_tests": "not_performed",
                    "metric_fields": list(metric_fields),
                },
                "software": {"python": sys.version, "scipy": scipy.__version__},
                "summary_source_files_sha256": _summary_source_hashes(),
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
