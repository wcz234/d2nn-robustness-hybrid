"""Summarize deployment distributions or fixed-front controls by training seed."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import numpy as np

from robustness_io import read_json, write_json_exclusive
from robustness_summary_input import load_evaluation, validate_evaluation_set
from summarize_robustness import _estimate_by_seed
from training_provenance import file_sha256


def deployment_summary(evaluations, *, threshold, quantile, references):
    if not 0 <= threshold <= 1 or not 0 < quantile < 1:
        raise ValueError("threshold must be in [0,1] and quantile in (0,1)")
    validate_evaluation_set(evaluations)
    seed_records, grouped = [], defaultdict(dict)
    for evaluation in evaluations:
        method = evaluation["manifest"]["method_id"]
        seed = evaluation["manifest"]["model"]["training_seed"]
        conditions = defaultdict(list)
        for row in evaluation["draw_rows"]:
            conditions[row["condition_id"]].append(row["accuracy"])
        reference = references.get(method, "clean_continuous")
        if reference not in conditions:
            raise ValueError(f"reference condition {reference} is missing for {method}")
        reference_accuracy = statistics.mean(conditions[reference])
        for condition, values in conditions.items():
            metric = {
                "accuracy_mean": statistics.mean(values),
                "accuracy_lower_quantile": float(np.quantile(values, quantile)),
                "failure_probability": statistics.mean(float(v < threshold) for v in values),
                "accuracy_drop_from_nominal_pp": 100 * (reference_accuracy - statistics.mean(values)),
            }
            seed_records.append({"method_id": method, "training_seed": seed, "condition_id": condition, "nominal_reference_condition": reference, "n_deployment_draws": len(values), "metrics": metric})
            grouped[(method, condition)][seed] = metric
    summaries = [{"method_id": method, "condition_id": condition, "metrics": {metric: _estimate_by_seed({seed: values[metric] for seed, values in seeds.items()}) for metric in next(iter(seeds.values()))}} for (method, condition), seeds in sorted(grouped.items())]
    return {
        "schema_version": "deployment-distribution-summary/v1", "scope": "numerical_simulation_only",
        "statistics": {"independent_unit": "training_seed", "draws_aggregated_before_cross_seed_intervals": True, "lower_quantile": quantile, "failure_accuracy_threshold": threshold, "failure_definition": "accuracy_strictly_below_threshold", "tail_estimator": "empirical linear quantile, descriptive; training-seed CI does not resolve finite deployment-draw tail uncertainty", "nominal_reference_map": references},
        "evidence_status": "descriptive_or_subset_only" if any(len(s) < 3 for s in grouped.values()) or any("subset" in e["manifest"].get("evidence_status", "") for e in evaluations) else "cross_seed_distribution_summary_with_marginal_intervals",
        "seed_records": seed_records, "summaries": summaries,
    }


def readout_summary(runs):
    if not runs or any(r.get("schema_version") != "fixed-front-readout/v1" for r in runs):
        raise ValueError("fixed-front results are required")
    if any(r["source_phase_sha256_before"] != r["source_phase_sha256_after"] for r in runs):
        raise ValueError("source phase weights changed")
    seeds = [r["source_training_seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("source training seeds must be unique; heads are not independent replicates")
    signatures = {json.dumps({"phase_path": r["phase_path"], "configuration": {key: r["configuration"][key] for key in ("fronts", "heads", "pools", "epochs", "lr", "hidden_dim")}, "counts": {key: r["sample_scope"][key] for key in ("train_samples", "validation_samples", "test_samples")}}, sort_keys=True) for r in runs}
    if len(signatures) != 1:
        raise ValueError("readout configurations and sample counts must match across source seeds")
    indexed = {r["source_training_seed"]: {row["method_id"]: row["test_metrics"]["accuracy"] for row in r["records"]} for r in runs}
    methods = set(next(iter(indexed.values())))
    if any(set(values) != methods for values in indexed.values()):
        raise ValueError("all seeds must contain identical readout methods")
    summaries = [{"method_id": method, "accuracy": _estimate_by_seed({seed: values[method] for seed, values in indexed.items()})} for method in sorted(methods)]
    pairs = set()
    for method in methods:
        if method.startswith("trained_pool"):
            for other_front in ("random", "free", "input"):
                other = method.replace("trained_", f"{other_front}_", 1)
                if other in methods:
                    pairs.add((method, other))
        if method.endswith("_mlp"):
            other = method[:-4] + "_linear"
            if other in methods:
                pairs.add((method, other))
    comparisons = [{"left_method": a, "right_method": b, "accuracy_difference_pp": _estimate_by_seed({seed: 100 * (values[a] - values[b]) for seed, values in indexed.items()})} for a, b in sorted(pairs)]
    if all(r.get("fixed_detector_test_metrics") is not None for r in runs):
        fixed = {r["source_training_seed"]: r["fixed_detector_test_metrics"]["accuracy"] for r in runs}
        for method in sorted(m for m in methods if m.startswith("trained_")):
            comparisons.append({"left_method": method, "right_method": "source_fixed_detector", "accuracy_difference_pp": _estimate_by_seed({seed: 100 * (indexed[seed][method] - fixed[seed]) for seed in seeds})})
    return {"schema_version": "fixed-front-summary/v1", "scope": "numerical_simulation_only", "independent_unit": "source_training_seed", "n_source_training_seeds": len(seeds), "evidence_status": "descriptive_or_subset_only" if len(seeds) < 3 or any("subset" in r["evidence_status"] for r in runs) else "cross_source_seed_summary_with_marginal_intervals", "summaries": summaries, "paired_comparisons": comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    deployment = commands.add_parser("deployment")
    deployment.add_argument("--evaluations", nargs="+", required=True)
    deployment.add_argument("--failure-threshold", type=float, required=True)
    deployment.add_argument("--lower-quantile", type=float, default=0.05)
    deployment.add_argument("--reference-map", help="JSON mapping exact method IDs to nominal reference condition IDs")
    deployment.add_argument("--output", required=True)
    readouts = commands.add_parser("readouts")
    readouts.add_argument("--results", nargs="+", required=True)
    readouts.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "deployment":
        references = read_json(args.reference_map) if args.reference_map else {}
        evaluations = [load_evaluation(path) for path in args.evaluations]
        known_methods = {e["manifest"]["method_id"] for e in evaluations}
        if set(references) - known_methods:
            raise ValueError("reference map contains unknown method IDs")
        payload = deployment_summary(evaluations, threshold=args.failure_threshold, quantile=args.lower_quantile, references=references)
        inputs = [e["manifest_path"] for e in evaluations]
    else:
        inputs = [Path(path) for path in args.results]
        payload = readout_summary([read_json(path) for path in inputs])
    payload["input_files"] = [{"path": str(path), "sha256": file_sha256(path)} for path in inputs]
    payload["source_sha256"] = file_sha256(Path(__file__))
    write_json_exclusive(args.output, payload)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
