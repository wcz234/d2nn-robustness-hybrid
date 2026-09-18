"""Integrity and pairing validation for robustness-summary inputs."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import torch

from robustness_io import read_json, strict_json_loads, verify_manifest_integrity
from robustness_records import MACRO_F1_CONVENTION, macro_f1_from_confusion
from training_provenance import file_sha256


CORE_METRIC_FIELDS = ("accuracy", "mean_cross_entropy", "mean_contrast")
EVALUATION_SCHEMAS = {
    "robustness-evaluation/v1": {
        "draw_schema": "robustness-draw-metrics/v1",
        "metric_fields": CORE_METRIC_FIELDS,
    },
    "robustness-evaluation/v2": {
        "draw_schema": "robustness-draw-metrics/v2",
        "metric_fields": (*CORE_METRIC_FIELDS, "macro_f1"),
    },
}


def _manifest_path(value) -> Path:
    path = Path(value)
    return path / "evaluation_manifest.json" if path.is_dir() else path


def _artifact_path(manifest_path: Path, record: dict) -> Path:
    relative = record.get("path")
    if not isinstance(relative, str) or Path(relative).name != relative:
        raise ValueError("evaluation artifact paths must be direct relative filenames")
    return manifest_path.parent / relative


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = strict_json_loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _verify_artifact(manifest_path: Path, record: dict) -> list[dict]:
    path = _artifact_path(manifest_path, record)
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise ValueError(f"artifact SHA-256 mismatch: {path}")
    rows = _read_jsonl(path)
    if len(rows) != record.get("row_count"):
        raise ValueError(f"artifact row-count mismatch: {path}")
    return rows


def _validate_confusion_metrics(row: dict) -> None:
    matrix = row.get("confusion_matrix")
    if not isinstance(matrix, list) or not matrix or any(not isinstance(values, list) for values in matrix):
        raise ValueError("draw metric confusion_matrix must be a non-empty matrix")
    size = len(matrix)
    if any(len(values) != size for values in matrix):
        raise ValueError("draw metric confusion_matrix must be square")
    if any(isinstance(count, bool) or not isinstance(count, int) for values in matrix for count in values):
        raise ValueError("draw metric confusion_matrix counts must be integers")
    counts = torch.tensor(matrix, dtype=torch.int64)
    if (counts < 0).any() or int(counts.sum().item()) != row["n_samples"]:
        raise ValueError("draw metric confusion_matrix sample counts are invalid")
    if int(counts.diag().sum().item()) != row["n_correct"]:
        raise ValueError("draw metric confusion_matrix correct count is invalid")
    expected = macro_f1_from_confusion(counts)
    if row.get("macro_f1_convention") != MACRO_F1_CONVENTION:
        raise ValueError("draw metric macro-F1 convention is invalid")
    if not math.isclose(row["macro_f1"], expected, abs_tol=1e-12):
        raise ValueError("draw metric macro-F1 cannot be reproduced from its confusion matrix")


def _validate_draw_row(row: dict, *, manifest: dict, schema: dict) -> tuple:
    metric_fields = schema["metric_fields"]
    required = ("condition_id", "draw_id", "n_samples", "n_correct", *metric_fields)
    if any(key not in row for key in required):
        raise ValueError("draw metric row is missing required fields")
    if row.get("evaluation_id") != manifest.get("evaluation_id"):
        raise ValueError("draw metric evaluation_id does not match its manifest")
    if row.get("method_id") != manifest.get("method_id"):
        raise ValueError("draw metric method_id does not match its manifest")
    training_seed = manifest.get("model", {}).get("training_seed")
    if row.get("training_seed") != training_seed:
        raise ValueError("draw metric training_seed does not match its manifest")
    if row["n_samples"] < 1 or row["n_correct"] < 0 or row["n_correct"] > row["n_samples"]:
        raise ValueError("draw metric sample counts are invalid")
    if not math.isclose(row["accuracy"], row["n_correct"] / row["n_samples"], abs_tol=1e-12):
        raise ValueError("draw accuracy cannot be reproduced from integer counts")
    if row.get("schema_version") != schema["draw_schema"]:
        raise ValueError("draw metric schema does not match its evaluation manifest")
    if any(not math.isfinite(float(row[field])) for field in metric_fields):
        raise ValueError("draw metrics must be finite")
    if "macro_f1" in metric_fields:
        _validate_confusion_metrics(row)
    return row["condition_id"], row["draw_id"]


def load_evaluation(value) -> dict:
    manifest_path = _manifest_path(value)
    manifest = read_json(manifest_path)
    schema = EVALUATION_SCHEMAS.get(manifest.get("schema_version"))
    if schema is None:
        raise ValueError(f"unsupported evaluation schema in {manifest_path}")
    if manifest.get("scope") != "numerical_simulation_only":
        raise ValueError(f"evaluation is not marked as numerical simulation in {manifest_path}")
    verify_manifest_integrity(manifest)
    rows = _verify_artifact(manifest_path, manifest.get("artifacts", {}).get("draw_metrics", {}))
    keys = [_validate_draw_row(row, manifest=manifest, schema=schema) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate condition/draw rows in evaluation artifacts")
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "draw_rows": rows,
        "metric_fields": schema["metric_fields"],
    }


def validate_evaluation_set(evaluations: list[dict]) -> None:
    if not evaluations:
        raise ValueError("at least one evaluation is required")
    plan_identities = {
        (item["manifest"].get("plan", {}).get("plan_id"), item["manifest"].get("plan", {}).get("sha256"))
        for item in evaluations
    }
    if len(plan_identities) != 1:
        raise ValueError("all evaluations must use the same frozen robustness plan")
    if len({tuple(item["metric_fields"]) for item in evaluations}) != 1:
        raise ValueError("all evaluations must expose the same metric fields")
    identities = []
    optical_fingerprints = defaultdict(set)
    draw_key_sets = []
    sample_counts = defaultdict(set)
    for item in evaluations:
        manifest = item["manifest"]
        identities.append((manifest.get("method_id"), manifest.get("model", {}).get("training_seed")))
        draw_key_sets.append(frozenset((row["condition_id"], row["draw_id"]) for row in item["draw_rows"]))
        for row in item["draw_rows"]:
            draw_key = (row["condition_id"], row["draw_id"])
            optical_fingerprints[draw_key].add(row.get("optical_draw_fingerprint_sha256"))
            sample_counts[draw_key].add(row["n_samples"])
    if any(method is None or seed is None for method, seed in identities) or len(set(identities)) != len(identities):
        raise ValueError("each method/training-seed evaluation must be present exactly once")
    if len(set(draw_key_sets)) != 1:
        raise ValueError("all evaluations must contain identical condition/draw IDs")
    if any(len(values) != 1 or None in values for values in optical_fingerprints.values()):
        raise ValueError("paired evaluations do not share identical optical draws")
    if any(len(values) != 1 for values in sample_counts.values()):
        raise ValueError("paired evaluations do not use identical sample counts")
