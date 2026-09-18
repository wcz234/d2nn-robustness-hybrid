from types import SimpleNamespace

import pytest
import torch

from clean_evaluation import TIMING_SCHEMA_VERSION, linear_quantile
from robustness_io import JsonlWriter, artifact_record, attach_manifest_integrity, read_json, write_json
from robustness_records import MACRO_F1_CONVENTION, macro_f1_from_confusion
from summarize_clean import load_clean_evaluation, summarize


def _predictions(correct_count):
    rows = []
    for index in range(10):
        target = index % 2
        prediction = target if index < correct_count else 1 - target
        rows.append((index, target, prediction))
    return rows


def write_clean_evaluation(root, *, method, seed, correct_count, timing_base=1.0, protocol_hash="a" * 64):
    root.mkdir()
    evaluation_id = f"clean-{method}-seed-{seed}"
    predictions = _predictions(correct_count)
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    predictions_path = root / "predictions.jsonl"
    with JsonlWriter(predictions_path) as writer:
        for index, target, prediction in predictions:
            confusion[target, prediction] += 1
            writer.write(
                {
                    "schema_version": "robustness-prediction/v1",
                    "evaluation_id": evaluation_id,
                    "method_id": method,
                    "training_seed": seed,
                    "condition_id": "clean_continuous",
                    "draw_id": 0,
                    "dataset_index": index,
                    "target": target,
                    "prediction": prediction,
                    "correct": target == prediction,
                }
            )
    accuracy = correct_count / 10
    metric = {
        "schema_version": "robustness-draw-metrics/v2",
        "evaluation_id": evaluation_id,
        "method_id": method,
        "training_seed": seed,
        "condition_id": "clean_continuous",
        "draw_id": 0,
        "n_samples": 10,
        "n_correct": correct_count,
        "accuracy": accuracy,
        "macro_f1": macro_f1_from_confusion(confusion),
        "macro_f1_convention": MACRO_F1_CONVENTION,
        "confusion_matrix": confusion.tolist(),
        "mean_cross_entropy": 1.0 - accuracy,
        "mean_contrast": accuracy - 0.5,
        "optical_draw_fingerprint_sha256": None,
        "optical_draw": None,
        "detector_noise_scope": "disabled",
        "detector_noise_shape": None,
        "detector_noise_sha256": None,
    }
    metric_path = root / "clean_metrics.json"
    write_json(metric_path, metric)
    repeats = [timing_base, timing_base + 0.2, timing_base + 0.1]
    median_seconds = timing_base + 0.1
    timing = {
        "schema_version": TIMING_SCHEMA_VERSION,
        "scope": "cpu_software_simulation_overhead",
        "claim_boundary": "not_physical_optical_hardware_latency_energy_or_efficiency",
        "measurement_scope": "test",
        "device": "cpu",
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "warmup_repeats": 1,
        "measurement_repeats": 3,
        "sample_count_per_repeat": 10,
        "elapsed_seconds": repeats,
        "median_seconds": median_seconds,
        "q1_seconds": linear_quantile(repeats, 0.25),
        "q3_seconds": linear_quantile(repeats, 0.75),
        "iqr_seconds": linear_quantile(repeats, 0.75) - linear_quantile(repeats, 0.25),
        "median_seconds_per_sample": median_seconds / 10,
    }
    timing_path = root / "cpu_software_timing.json"
    write_json(timing_path, timing)
    metric_summary_fields = (
        "n_samples",
        "n_correct",
        "accuracy",
        "macro_f1",
        "macro_f1_convention",
        "confusion_matrix",
        "mean_cross_entropy",
        "mean_contrast",
    )
    manifest = attach_manifest_integrity(
        {
            "schema_version": "clean-evaluation/v1",
            "scope": "numerical_simulation_only",
            "evaluation_id": evaluation_id,
            "method_id": method,
            "evidence_status": "protocol_bound_full_test_single_seed_requires_cross_training_seed_summary",
            "formal_protocol": {"protocol_id": "formal-test", "sha256": protocol_hash, "path": "protocol.json"},
            "sample_scope": {"selection": "full_test", "sample_count": 10, "full_test_samples": 10},
            "model": {"training_seed": seed},
            "metrics": {field: metric[field] for field in metric_summary_fields},
            "timing": timing,
            "artifacts": {
                "predictions": artifact_record(predictions_path, row_count=10),
                "clean_metrics": artifact_record(metric_path, row_count=1),
                "cpu_software_timing": artifact_record(timing_path),
            },
        }
    )
    manifest_path = root / "evaluation_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def read_jsonl(path):
    return [__import__("json").loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_clean_summary_uses_training_seeds_and_builds_paired_effects(tmp_path):
    manifests = [
        write_clean_evaluation(tmp_path / "a1", method="method-a", seed=1, correct_count=8, timing_base=1.0),
        write_clean_evaluation(tmp_path / "a2", method="method-a", seed=2, correct_count=9, timing_base=1.2),
        write_clean_evaluation(tmp_path / "b1", method="method-b", seed=1, correct_count=7, timing_base=2.0),
        write_clean_evaluation(tmp_path / "b2", method="method-b", seed=2, correct_count=8, timing_base=2.2),
    ]
    output_dir = tmp_path / "summary"
    manifest_path = summarize(SimpleNamespace(evaluations=manifests, output_dir=output_dir))

    method_rows = read_jsonl(output_dir / "method_summary.jsonl")
    paired_rows = read_jsonl(output_dir / "paired_comparisons.jsonl")
    method_a = next(row for row in method_rows if row["method_id"] == "method-a")
    assert method_a["metrics"]["accuracy"]["mean"] == pytest.approx(0.85)
    assert method_a["metrics"]["accuracy"]["n_training_seeds"] == 2
    assert method_a["metrics"]["cpu_median_seconds"]["n_training_seeds"] == 2
    assert paired_rows[0]["metrics"]["accuracy"]["mean"] == pytest.approx(0.1)
    assert paired_rows[0]["metrics"]["cpu_median_seconds"]["mean"] == pytest.approx(-1.0)
    summary_manifest = read_json(manifest_path)
    assert summary_manifest["statistics"]["independent_unit"] == "training_seed"
    assert summary_manifest["statistics"]["timing_repeat_role"].endswith("not_independent_replicates")


def test_clean_loader_rejects_tampered_predictions(tmp_path):
    manifest_path = write_clean_evaluation(tmp_path / "evaluation", method="method-a", seed=1, correct_count=8)
    with (manifest_path.parent / "predictions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_clean_evaluation(manifest_path)


def test_clean_summary_requires_identical_seed_sets_for_method_pairing(tmp_path):
    manifests = [
        write_clean_evaluation(tmp_path / "a1", method="method-a", seed=1, correct_count=8),
        write_clean_evaluation(tmp_path / "a2", method="method-a", seed=2, correct_count=9),
        write_clean_evaluation(tmp_path / "b1", method="method-b", seed=1, correct_count=7),
    ]
    with pytest.raises(ValueError, match="identical training-seed sets"):
        summarize(SimpleNamespace(evaluations=manifests, output_dir=tmp_path / "summary"))


def test_clean_summary_requires_one_frozen_protocol(tmp_path):
    manifests = [
        write_clean_evaluation(tmp_path / "a", method="method-a", seed=1, correct_count=8),
        write_clean_evaluation(
            tmp_path / "b",
            method="method-a",
            seed=2,
            correct_count=9,
            protocol_hash="b" * 64,
        ),
    ]
    with pytest.raises(ValueError, match="shared frozen protocol"):
        summarize(SimpleNamespace(evaluations=manifests, output_dir=tmp_path / "summary"))
