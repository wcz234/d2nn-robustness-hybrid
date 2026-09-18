from types import SimpleNamespace

import pytest

from robustness_io import (
    JsonlWriter,
    artifact_record,
    attach_manifest_integrity,
    read_json,
    strict_json_loads,
    verify_manifest_integrity,
    write_json,
)
from summarize_robustness import load_evaluation, summarize


def write_evaluation(root, *, method, seed, accuracies, fingerprints=None, legacy=False):
    root.mkdir()
    evaluation_id = f"{method}-seed-{seed}"
    metrics_path = root / "draw_metrics.jsonl"
    fingerprints = fingerprints or [f"optical-draw-{index}" for index in range(len(accuracies))]
    with JsonlWriter(metrics_path) as writer:
        for draw_id, accuracy in enumerate(accuracies):
            n_samples = 100
            row = {
                    "schema_version": (
                        "robustness-draw-metrics/v1" if legacy else "robustness-draw-metrics/v2"
                    ),
                    "evaluation_id": evaluation_id,
                    "method_id": method,
                    "training_seed": seed,
                    "condition_id": "mixed-noise",
                    "draw_id": draw_id,
                    "n_samples": n_samples,
                    "n_correct": round(accuracy * n_samples),
                    "accuracy": accuracy,
                    "mean_cross_entropy": 1.0 - accuracy,
                    "mean_contrast": accuracy - 0.5,
                    "optical_draw_fingerprint_sha256": fingerprints[draw_id],
                }
            if not legacy:
                correct = round(accuracy * n_samples)
                incorrect = n_samples - correct
                row.update(
                    macro_f1=accuracy,
                    macro_f1_convention="all_score_classes_zero_division_0",
                    confusion_matrix=[
                        [correct // 2, incorrect // 2],
                        [incorrect - incorrect // 2, correct - correct // 2],
                    ],
                )
            writer.write(row)
    manifest = attach_manifest_integrity(
        {
            "schema_version": "robustness-evaluation/v1" if legacy else "robustness-evaluation/v2",
            "scope": "numerical_simulation_only",
            "evaluation_id": evaluation_id,
            "method_id": method,
            "evidence_status": "full_test_evaluation_requires_cross_training_seed_summary",
            "plan": {"plan_id": "shared-plan", "sha256": "a" * 64, "path": "plan.json"},
            "model": {"training_seed": seed},
            "artifacts": {"draw_metrics": artifact_record(metrics_path, row_count=len(accuracies))},
        }
    )
    manifest_path = root / "evaluation_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def read_jsonl(path):
    return [strict_json_loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_summary_aggregates_draws_before_training_seeds_and_builds_paired_effects(tmp_path):
    manifests = [
        write_evaluation(tmp_path / "a1", method="method-a", seed=1, accuracies=[0.8, 0.6]),
        write_evaluation(tmp_path / "a2", method="method-a", seed=2, accuracies=[0.9, 0.7]),
        write_evaluation(tmp_path / "b1", method="method-b", seed=1, accuracies=[0.7, 0.5]),
        write_evaluation(tmp_path / "b2", method="method-b", seed=2, accuracies=[0.8, 0.6]),
    ]
    output_dir = tmp_path / "summary"

    manifest_path = summarize(SimpleNamespace(evaluations=[str(path) for path in manifests], output_dir=output_dir))

    seed_rows = read_jsonl(output_dir / "seed_metrics.jsonl")
    method_rows = read_jsonl(output_dir / "method_summary.jsonl")
    paired_rows = read_jsonl(output_dir / "paired_comparisons.jsonl")
    assert [row["metrics"]["accuracy"] for row in seed_rows] == pytest.approx([0.7, 0.8, 0.6, 0.7])
    assert [row["metrics"]["macro_f1"] for row in seed_rows] == pytest.approx([0.7, 0.8, 0.6, 0.7])
    method_a = next(row for row in method_rows if row["method_id"] == "method-a")
    assert method_a["metrics"]["accuracy"]["mean"] == pytest.approx(0.75)
    assert method_a["metrics"]["accuracy"]["n_training_seeds"] == 2
    assert method_a["metrics"]["accuracy"]["ci95_t"] is not None
    paired = paired_rows[0]["metrics"]["accuracy"]
    assert paired["mean"] == pytest.approx(0.1)
    assert paired["sd_across_training_seeds"] == pytest.approx(0.0)
    assert paired["ci95_t"] == pytest.approx([0.1, 0.1])
    manifest = read_json(manifest_path)
    verify_manifest_integrity(manifest)
    assert manifest["statistics"]["independent_unit"] == "training_seed"
    assert manifest["statistics"]["hypothesis_tests"] == "not_performed"
    assert manifest["evidence"]["dataset_evidence_status"] == "full_test_evaluations"
    assert manifest["evidence"]["inference_status"] == (
        "descriptive_only_below_preregistered_three_seed_minimum"
    )
    assert "training_provenance.py" in manifest["summary_source_files_sha256"]
    assert manifest["statistics"]["metric_fields"] == [
        "accuracy",
        "mean_cross_entropy",
        "mean_contrast",
        "macro_f1",
    ]


def test_summary_loader_accepts_legacy_v1_without_macro_f1(tmp_path):
    manifest_path = write_evaluation(
        tmp_path / "legacy",
        method="method-a",
        seed=1,
        accuracies=[0.8],
        legacy=True,
    )

    evaluation = load_evaluation(manifest_path)
    summary_path = summarize(
        SimpleNamespace(evaluations=[str(manifest_path)], output_dir=tmp_path / "legacy-summary")
    )
    summary_manifest = read_json(summary_path)

    assert evaluation["metric_fields"] == ("accuracy", "mean_cross_entropy", "mean_contrast")
    assert summary_manifest["statistics"]["metric_fields"] == [
        "accuracy",
        "mean_cross_entropy",
        "mean_contrast",
    ]


def test_summary_rejects_tampered_draw_metrics(tmp_path):
    manifest_path = write_evaluation(tmp_path / "evaluation", method="method-a", seed=1, accuracies=[0.8])
    with (manifest_path.parent / "draw_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="SHA-256"):
        load_evaluation(manifest_path)


def test_summary_requires_identical_training_seed_sets_for_method_pairing(tmp_path):
    manifests = [
        write_evaluation(tmp_path / "a1", method="method-a", seed=1, accuracies=[0.8]),
        write_evaluation(tmp_path / "a2", method="method-a", seed=2, accuracies=[0.9]),
        write_evaluation(tmp_path / "b1", method="method-b", seed=1, accuracies=[0.7]),
    ]

    with pytest.raises(ValueError, match="identical training-seed sets"):
        summarize(SimpleNamespace(evaluations=[str(path) for path in manifests], output_dir=tmp_path / "summary"))


def test_summary_requires_identical_optical_draws(tmp_path):
    manifests = [
        write_evaluation(
            tmp_path / "a1",
            method="method-a",
            seed=1,
            accuracies=[0.8],
            fingerprints=["draw-a"],
        ),
        write_evaluation(
            tmp_path / "b1",
            method="method-b",
            seed=1,
            accuracies=[0.7],
            fingerprints=["draw-b"],
        ),
    ]

    with pytest.raises(ValueError, match="identical optical draws"):
        summarize(SimpleNamespace(evaluations=[str(path) for path in manifests], output_dir=tmp_path / "summary"))


def test_summary_requires_identical_draw_id_sets_across_training_seeds(tmp_path):
    manifests = [
        write_evaluation(tmp_path / "a1", method="method-a", seed=1, accuracies=[0.8]),
        write_evaluation(tmp_path / "a2", method="method-a", seed=2, accuracies=[0.9, 0.7]),
    ]

    with pytest.raises(ValueError, match="identical condition/draw IDs"):
        summarize(SimpleNamespace(evaluations=[str(path) for path in manifests], output_dir=tmp_path / "summary"))
