import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import evaluate_clean
from clean_evaluation import measure_clean_inference, validate_method_target
from perturbations import PerturbationConfig
from robustness_io import read_json, verify_manifest_integrity, write_json
from robustness_plan import serialized_perturbation_config
from training_provenance import file_sha256


class CleanDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.tensor([[[float(index)]]]), index % 3, index


class CleanClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.forward_calls = 0

    def forward_with_metrics(self, x, target=None):
        self.forward_calls += 1
        logits = torch.full((x.shape[0], 3), -2.0, device=x.device) + self.anchor
        preferred = x.flatten(1)[:, 0].long().remainder(3)
        logits.scatter_(1, preferred.unsqueeze(1), 2.0)
        scores = torch.softmax(logits, dim=1)
        contrast = scores.max(dim=1).values - scores.min(dim=1).values
        return {"scores": scores, "logits": logits, "contrast": contrast}


def make_target(tmp_path, *, variant="electronic", enabled=False, config=None):
    checkpoint = tmp_path / f"{variant}.pth"
    checkpoint.write_bytes(f"clean-{variant}".encode())
    model = CleanClassifier()
    return SimpleNamespace(
        model=model,
        manifest={
            "training_perturbations": {
                "enabled": enabled,
                "config": serialized_perturbation_config(PerturbationConfig()) if config is None else config,
            }
        },
        checkpoint_path=checkpoint,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        dataset_key="mnist",
        input_shape=(1, 28, 28),
        test_samples=4,
        training_seed=42,
        model_variant=variant,
        feature_schema_id="flattened_input:1x28x28",
        score_semantics="softmax_probability",
    )


def timing_record():
    return {
        "schema_version": "cpu-software-simulation-timing/v1",
        "scope": "cpu_software_simulation_overhead",
        "claim_boundary": "not_physical_optical_hardware_latency_energy_or_efficiency",
        "measurement_scope": "test",
        "device": "cpu",
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "warmup_repeats": 1,
        "measurement_repeats": 2,
        "sample_count_per_repeat": 4,
        "elapsed_seconds": [1.0, 1.2],
        "median_seconds": 1.1,
        "q1_seconds": 1.05,
        "q3_seconds": 1.15,
        "iqr_seconds": 0.1,
        "median_seconds_per_sample": 0.275,
    }


@pytest.mark.parametrize(
    ("method_id", "variant", "enabled"),
    [
        ("baseline_d2nn", "d2nn", False),
        ("robust_d2nn", "d2nn", True),
        ("hybrid", "hybrid", False),
        ("electronic", "electronic", False),
    ],
)
def test_method_contract_accepts_only_matching_variant_and_training_mode(tmp_path, method_id, variant, enabled):
    target = make_target(tmp_path, variant=variant, enabled=enabled)
    validate_method_target(method_id, target)
    target.model_variant = "hybrid" if variant != "hybrid" else "d2nn"
    with pytest.raises(ValueError, match="requires model variant"):
        validate_method_target(method_id, target)


def test_cpu_timing_excludes_warmups_and_restores_model_mode():
    model = CleanClassifier().train()
    loader = DataLoader(CleanDataset(), batch_size=2, shuffle=False)
    clock = iter([10.0, 11.0, 20.0, 22.0, 30.0, 34.0]).__next__

    timing = measure_clean_inference(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        warmup_repeats=2,
        measurement_repeats=3,
        clock=clock,
    )

    assert model.training is True
    assert model.forward_calls == 10
    assert timing["elapsed_seconds"] == [1.0, 2.0, 4.0]
    assert timing["median_seconds"] == 2.0
    assert timing["q1_seconds"] == 1.5
    assert timing["q3_seconds"] == 3.0
    assert timing["iqr_seconds"] == 1.5


def test_run_writes_clean_predictions_metrics_timing_and_manifest(tmp_path):
    target = make_target(tmp_path)
    output_dir = tmp_path / "clean-evaluation"
    args = evaluate_clean.build_parser().parse_args(
        [
            "--checkpoint",
            str(target.checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--method-id",
            "electronic",
            "--batch-size",
            "2",
            "--include-scores",
        ]
    )
    loader = DataLoader(CleanDataset(), batch_size=2, shuffle=False)
    selection = {"selection": "full_test", "sample_count": 4, "full_test_samples": 4}
    with mock.patch("evaluate_clean.load_robustness_target", return_value=target), mock.patch(
        "evaluate_clean.build_indexed_test_loader", return_value=(loader, selection)
    ), mock.patch("evaluate_clean.measure_clean_inference", return_value=timing_record()), mock.patch(
        "clean_evaluation_manifest.environment_manifest", return_value={"test": True}
    ):
        manifest_path = evaluate_clean.run_clean_evaluation(args)

    manifest = read_json(manifest_path)
    verify_manifest_integrity(manifest)
    assert manifest["evidence_status"].startswith("unregistered_full_test")
    assert manifest["model"]["variant"] == "electronic"
    assert manifest["model"]["detector_feature_count"] is None
    assert manifest["metrics"]["accuracy"] == 1.0
    assert manifest["metrics"]["macro_f1"] == 1.0
    assert manifest["metrics"]["mean_cross_entropy"] > 0
    assert manifest["timing"]["median_seconds"] == 1.1
    for artifact in manifest["artifacts"].values():
        assert file_sha256(output_dir / artifact["path"]) == artifact["sha256"]
    predictions = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 4
    assert all(row["prediction"] == row["target"] and "scores" in row for row in predictions)


def _formal_protocol(*, condition_id="clean_continuous"):
    return {
        "protocol_schema_version": "formal-simulation-protocol/v1",
        "protocol_id": "clean-test-v1",
        "scope": "numerical_simulation_only",
        "dataset": {"key": "mnist", "test_samples": 4},
        "methods": [
            {"method_id": "electronic", "model_variant": "electronic", "training_perturbations": None}
        ],
        "evaluation": {
            "sample_scope": "full_test",
            "batch_size": 2,
            "num_workers": 0,
            "device": "cpu",
            "master_optical_seed": 62001,
            "master_detector_seed": 62002,
            "deterministic_draw_count": 1,
        },
        "conditions": [
            {
                "condition_id": condition_id,
                "draw_count": 1,
                "perturbation_config": serialized_perturbation_config(PerturbationConfig()),
            }
        ],
    }


def test_protocol_binding_rejects_renamed_clean_condition(tmp_path):
    target = make_target(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, _formal_protocol(condition_id="renamed-clean"))
    args = evaluate_clean.build_parser().parse_args(
        [
            "--checkpoint",
            str(target.checkpoint_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--method-id",
            "electronic",
            "--protocol",
            str(protocol_path),
            "--batch-size",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="clean_continuous"):
        evaluate_clean._load_protocol_binding(args, target)


def test_nonempty_output_is_rejected_before_dataset_loading(tmp_path):
    target = make_target(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("preserve", encoding="utf-8")
    args = evaluate_clean.build_parser().parse_args(
        [
            "--checkpoint",
            str(target.checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--method-id",
            "electronic",
        ]
    )
    with mock.patch("evaluate_clean.load_robustness_target", return_value=target), mock.patch(
        "evaluate_clean.build_indexed_test_loader"
    ) as dataset_loader:
        with pytest.raises(ValueError, match="empty or absent"):
            evaluate_clean.run_clean_evaluation(args)
    dataset_loader.assert_not_called()
