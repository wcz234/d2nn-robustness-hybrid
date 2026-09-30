"""Retrospective declared-vs-observed seed audit; never rewrites frozen evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robustness_io import write_json_exclusive
from training_provenance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
COHORTS = [
    ("docs/FORMAL_EXPERIMENT_PROTOCOL.json", "results/formal_mnist_v1/robustness_summary/seed_metrics.jsonl"),
    ("docs/FORMAL_EXPERIMENT_PROTOCOL_V2.json", "results/formal_mnist_v2/robustness_summary/seed_metrics.jsonl"),
    ("docs/FORMAL_EXPERIMENT_PROTOCOL_V3.json", "results/formal_fashion_mnist_v3/robustness_summary/seed_metrics.jsonl"),
    ("docs/FORMAL_EXPERIMENT_PROTOCOL_V4.json", "results/formal_mnist_v4/robustness_summary/seed_metrics.jsonl"),
]


def audit(root=ROOT):
    records = []
    for protocol_name, metrics_name in COHORTS:
        protocol_path, metrics_path = Path(root) / protocol_name, Path(root) / metrics_name
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        observed = {}
        for method in sorted({row["method_id"] for row in rows}):
            observed[method] = sorted({row["training_seed"] for row in rows if row["method_id"] == method})
        declared = sorted(protocol["training"]["paired_training_seeds"])
        records.append({"protocol": protocol_name, "protocol_sha256": file_sha256(protocol_path), "seed_metrics": metrics_name, "seed_metrics_sha256": file_sha256(metrics_path), "declared_training_seeds": declared, "declared_statistical_n": protocol["statistics"].get("n_training_seeds"), "observed_seeds_by_method": observed, "seed_alignment": "MATCH" if all(values == declared for values in observed.values()) else "MISMATCH", "note": "retrospective check only; does not establish preregistration timing or correct historical records"})
    return {"schema_version": "protocol-alignment-audit/v1", "scope": "retrospective_repository_audit", "records": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = audit()
    write_json_exclusive(args.output, payload)
    for record in payload["records"]:
        print(f"{record['protocol']}: {record['seed_alignment']} declared={record['declared_training_seeds']} observed={record['observed_seeds_by_method']}")


if __name__ == "__main__":
    main()
