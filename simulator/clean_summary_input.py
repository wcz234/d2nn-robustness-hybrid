"""Integrity and metric validation for clean-only summary inputs."""

from __future__ import annotations

import math
from pathlib import Path
import statistics

import torch

from clean_evaluation import TIMING_SCHEMA_VERSION, linear_quantile
from clean_evaluation_manifest import CLEAN_EVALUATION_SCHEMA_VERSION
from robustness_io import read_json, strict_json_loads, verify_manifest_integrity
from robustness_records import MACRO_F1_CONVENTION, PREDICTION_SCHEMA_VERSION, macro_f1_from_confusion
from training_provenance import file_sha256


CLEAN_METRIC_FIELDS = ("accuracy", "mean_cross_entropy", "mean_contrast", "macro_f1")


def _manifest_path(value) -> Path:
    path = Path(value)
    return path / "evaluation_manifest.json" if path.is_dir() else path


def _artifact_path(manifest_path: Path, record: dict) -> Path:
    relative = record.get("path")
    if not isinstance(relative, str) or Path(relative).name != relative:
        raise ValueError("clean-evaluation artifact paths must be direct relative filenames")
    path = manifest_path.parent / relative
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise ValueError(f"artifact SHA-256 mismatch: {path}")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = strict_json_loads(line)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _validate_metric(metric: dict, manifest: dict) -> torch.Tensor:
    required = ("n_samples", "n_correct", "confusion_matrix", *CLEAN_METRIC_FIELDS)
    if any(field not in metric for field in required):
        raise ValueError("clean metric artifact is missing required fields")
    if metric.get("evaluation_id") != manifest.get("evaluation_id"):
        raise ValueError("clean metric evaluation_id does not match its manifest")
    if metric.get("method_id") != manifest.get("method_id"):
        raise ValueError("clean metric method_id does not match its manifest")
    if metric.get("training_seed") != manifest.get("model", {}).get("training_seed"):
        raise ValueError("clean metric training_seed does not match its manifest")
    if metric.get("condition_id") != "clean_continuous" or metric.get("optical_draw") is not None:
        raise ValueError("clean metric must contain no optical perturbation draw")
    if metric.get("detector_noise_scope") != "disabled":
        raise ValueError("clean metric must disable detector noise")
    matrix = metric["confusion_matrix"]
    if not isinstance(matrix, list) or not matrix or any(not isinstance(row, list) for row in matrix):
        raise ValueError("clean confusion_matrix must be a non-empty matrix")
    counts = torch.tensor(matrix, dtype=torch.int64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or (counts < 0).any():
        raise ValueError("clean confusion_matrix must be square with non-negative counts")
    if int(counts.sum()) != metric["n_samples"] or int(counts.diag().sum()) != metric["n_correct"]:
        raise ValueError("clean confusion_matrix counts do not match the metric totals")
    if not math.isclose(metric["accuracy"], metric["n_correct"] / metric["n_samples"], abs_tol=1e-12):
        raise ValueError("clean accuracy cannot be reproduced from integer counts")
    if metric.get("macro_f1_convention") != MACRO_F1_CONVENTION:
        raise ValueError("clean macro-F1 convention is invalid")
    if not math.isclose(metric["macro_f1"], macro_f1_from_confusion(counts), abs_tol=1e-12):
        raise ValueError("clean macro-F1 cannot be reproduced from its confusion matrix")
    if any(not math.isfinite(float(metric[field])) for field in CLEAN_METRIC_FIELDS):
        raise ValueError("clean metrics must be finite")
    return counts


def _validate_predictions(rows: list[dict], *, manifest: dict, metric: dict, expected_confusion: torch.Tensor) -> None:
    expected_count = metric["n_samples"]
    if len(rows) != expected_count:
        raise ValueError("clean prediction row count does not match n_samples")
    confusion = torch.zeros_like(expected_confusion)
    seen_indices = set()
    for row in rows:
        identity = (row.get("evaluation_id"), row.get("method_id"), row.get("training_seed"))
        expected_identity = (
            manifest.get("evaluation_id"),
            manifest.get("method_id"),
            manifest.get("model", {}).get("training_seed"),
        )
        if identity != expected_identity or row.get("schema_version") != PREDICTION_SCHEMA_VERSION:
            raise ValueError("clean prediction identity or schema does not match its manifest")
        index = row.get("dataset_index")
        target = row.get("target")
        prediction = row.get("prediction")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (index, target, prediction)):
            raise ValueError("clean prediction indices and labels must be integers")
        if index in seen_indices or not 0 <= target < confusion.shape[0] or not 0 <= prediction < confusion.shape[1]:
            raise ValueError("clean prediction contains duplicate index or invalid class label")
        if row.get("correct") is not (target == prediction):
            raise ValueError("clean prediction correct flag is inconsistent")
        seen_indices.add(index)
        confusion[target, prediction] += 1
    if seen_indices != set(range(expected_count)) or not torch.equal(confusion, expected_confusion):
        raise ValueError("clean predictions cannot reproduce the metric confusion_matrix")


def _validate_timing(timing: dict) -> None:
    if timing.get("schema_version") != TIMING_SCHEMA_VERSION or timing.get("device") != "cpu":
        raise ValueError("unsupported clean CPU timing schema or device")
    if timing.get("scope") != "cpu_software_simulation_overhead":
        raise ValueError("clean timing scope must remain software simulation overhead")
    repeats = timing.get("elapsed_seconds")
    if not isinstance(repeats, list) or len(repeats) != timing.get("measurement_repeats") or not repeats:
        raise ValueError("clean timing repeat count is invalid")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in repeats):
        raise ValueError("clean timing observations must be finite and non-negative")
    expected = {
        "median_seconds": statistics.median(repeats),
        "q1_seconds": linear_quantile(repeats, 0.25),
        "q3_seconds": linear_quantile(repeats, 0.75),
    }
    expected["iqr_seconds"] = expected["q3_seconds"] - expected["q1_seconds"]
    for field, value in expected.items():
        if not math.isclose(timing.get(field), value, abs_tol=1e-12):
            raise ValueError(f"clean timing {field} cannot be reproduced from repeats")
    sample_count = timing.get("sample_count_per_repeat")
    if not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("clean timing sample_count_per_repeat must be positive")
    if not math.isclose(timing.get("median_seconds_per_sample"), expected["median_seconds"] / sample_count, abs_tol=1e-12):
        raise ValueError("clean timing per-sample median is inconsistent")


def load_clean_evaluation(value) -> dict:
    manifest_path = _manifest_path(value)
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != CLEAN_EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported clean-evaluation schema in {manifest_path}")
    if manifest.get("scope") != "numerical_simulation_only":
        raise ValueError(f"clean evaluation is not numerical-simulation-only: {manifest_path}")
    verify_manifest_integrity(manifest)
    artifacts = manifest.get("artifacts", {})
    metric_record = artifacts.get("clean_metrics", {})
    metric = read_json(_artifact_path(manifest_path, metric_record))
    if metric_record.get("row_count") != 1:
        raise ValueError("clean metric artifact must contain one aggregate record")
    confusion = _validate_metric(metric, manifest)
    predictions_record = artifacts.get("predictions", {})
    predictions = _read_jsonl(_artifact_path(manifest_path, predictions_record))
    if len(predictions) != predictions_record.get("row_count"):
        raise ValueError("clean prediction artifact row-count mismatch")
    _validate_predictions(predictions, manifest=manifest, metric=metric, expected_confusion=confusion)
    timing = read_json(_artifact_path(manifest_path, artifacts.get("cpu_software_timing", {})))
    if timing != manifest.get("timing"):
        raise ValueError("clean timing artifact does not match its manifest")
    _validate_timing(timing)
    manifest_metrics = manifest.get("metrics", {})
    if any(manifest_metrics.get(field) != metric.get(field) for field in manifest_metrics):
        raise ValueError("clean metric artifact does not match its manifest summary")
    return {"manifest_path": manifest_path, "manifest": manifest, "metric": metric, "timing": timing}


def validate_clean_evaluation_set(
    evaluations: list[dict],
    *,
    tolerate_protocol_revisions: bool = False,
) -> None:
    if not evaluations:
        raise ValueError("at least one clean evaluation is required")
    identities = [
        (item["manifest"].get("method_id"), item["manifest"].get("model", {}).get("training_seed"))
        for item in evaluations
    ]
    if any(method is None or seed is None for method, seed in identities) or len(set(identities)) != len(identities):
        raise ValueError("each clean method/training-seed evaluation must be present exactly once")
    protocols = {
        (
            (item["manifest"].get("formal_protocol") or {}).get("protocol_id"),
            (item["manifest"].get("formal_protocol") or {}).get("sha256"),
        )
        for item in evaluations
    }
    if None in next(iter(protocols), (None, None)):
        raise ValueError("formal clean summaries require a frozen protocol binding")
    if len(protocols) != 1:
        # Tolerance is explicit and must be justified by the caller. It exists because the
        # evaluation device is a per-run property while the protocol hash is shared, so a
        # cohort produced partly on CUDA and partly on CPU records two revisions whose only
        # difference is the declared metrics device. Numerical equivalence must be
        # established separately before enabling this.
        if not tolerate_protocol_revisions:
            raise ValueError(
                "formal clean summaries require one shared frozen protocol "
                f"(found {len(protocols)} protocol bindings); pass explicit tolerance only "
                "with documented device-parity evidence"
            )
        # Revisions of one protocol may carry a device-variant suffix; require the same base id.
        base_ids = {
            str(protocol_id).split("-localcpu-metrics")[0] for protocol_id, _ in protocols
        }
        if len(base_ids) != 1:
            raise ValueError(
                "tolerated protocol revisions must share one protocol_id; found "
                f"{sorted(str(value) for value in {protocol_id for protocol_id, _ in protocols})}"
            )
    sample_scopes = {str(item["manifest"].get("sample_scope")) for item in evaluations}
    if len(sample_scopes) != 1 or evaluations[0]["manifest"]["sample_scope"].get("selection") != "full_test":
        raise ValueError("formal clean summaries require one shared full-test sample scope")
