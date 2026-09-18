import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import evaluate_robustness
from d2nn import detector_contrast, normalized_detector_logits
from perturbations import PerturbationConfig
from robustness_io import read_json, verify_manifest_integrity, write_json
from robustness_plan import build_robustness_plan
from training_provenance import file_sha256


class CliDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.tensor([[[float(index)]]]), index % 3, index


class CliClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.size = 4
        self.detector_feature_count = 3
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward_with_metrics(self, x, target=None, perturbation_draw=None):
        del perturbation_draw
        scores = torch.full((x.shape[0], 3), 0.2, device=x.device)
        preferred = x.flatten(1)[:, 0].long().remainder(3)
        scores.scatter_(1, preferred.unsqueeze(1), 1.0)
        result = {"scores": scores, "logits": normalized_detector_logits(scores)}
        if target is not None:
            result["contrast"] = detector_contrast(scores, target)
        return result


def make_target(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"cli-test-checkpoint")
    optical_config = {
        "wavelength": 0.75e-3,
        "layer_distance": 30e-3,
        "pixel_size": 0.4e-3,
        "input_distance": 30e-3,
        "output_distance": 30e-3,
        "size": 4,
        "num_layers": 2,
    }
    manifest = {
        "model_version": "rs_v1",
        "optical_config": optical_config,
        "activation_type": "none",
        "activation_positions": [],
        "activation_hparams": {},
        "propagation_backend": "fft",
        "propagation_chunk_size": None,
    }
    return SimpleNamespace(
        model=CliClassifier(),
        manifest=manifest,
        checkpoint_path=checkpoint,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        dataset_key="mnist",
        input_shape=(1, 28, 28),
        test_samples=5,
        training_seed=42,
        model_variant="d2nn",
        feature_schema_id="class_detector_energy:3",
        score_semantics="raw_detector_energy",
    )


def make_plan(target):
    return build_robustness_plan(
        target=target,
        condition_id="clean-smoke",
        config=PerturbationConfig(),
        draw_count=1,
        optical_seed=101,
        detector_seed=202,
        sample_limit=4,
    )


def test_create_plan_writes_frozen_plan_and_refuses_overwrite(tmp_path):
    target = make_target(tmp_path)
    output = tmp_path / "plan.json"
    args = evaluate_robustness.build_parser().parse_args(
        [
            "create-plan",
            "--reference-checkpoint",
            str(target.checkpoint_path),
            "--output",
            str(output),
            "--condition-id",
            "clean-smoke",
            "--sample-limit",
            "4",
        ]
    )
    with mock.patch("evaluate_robustness.load_robustness_target", return_value=target):
        assert evaluate_robustness.create_plan(args) == output

    payload = read_json(output)
    assert payload["sample_scope"]["selection"] == "test_prefix_subset"
    assert payload["reference_checkpoint"]["sha256"] == target.checkpoint_sha256
    with mock.patch("evaluate_robustness.load_robustness_target") as target_loader:
        with pytest.raises(FileExistsError, match="overwrite"):
            evaluate_robustness.create_plan(args)
    target_loader.assert_not_called()


def test_create_protocol_plan_writes_one_multi_condition_plan(tmp_path):
    target = make_target(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol = {
        "protocol_schema_version": "formal-simulation-protocol/v1",
        "protocol_id": "formal-test-v1",
        "scope": "numerical_simulation_only",
        "dataset": {"key": "mnist", "test_samples": 5},
        "evaluation": {
            "sample_scope": "full_test",
            "master_optical_seed": 62001,
            "master_detector_seed": 62002,
        },
        "conditions": [
            {
                "condition_id": "clean",
                "draw_count": 1,
                "perturbation_config": {
                    "lateral_shift_max_px": 0.0,
                    "gap_spacing_error_max_m": 0.0,
                    "phase_noise_std_rad": 0.0,
                    "phase_quantization_levels": None,
                    "detector_noise_std_relative": 0.0,
                },
            }
        ],
    }
    write_json(protocol_path, protocol)
    output = tmp_path / "protocol-plan.json"
    args = evaluate_robustness.build_parser().parse_args(
        [
            "create-protocol-plan",
            "--reference-checkpoint",
            str(target.checkpoint_path),
            "--protocol",
            str(protocol_path),
            "--output",
            str(output),
        ]
    )

    with mock.patch("evaluate_robustness.load_robustness_target", return_value=target):
        assert evaluate_robustness.create_protocol_plan(args) == output

    payload = read_json(output)
    assert payload["sample_scope"]["selection"] == "full_test"
    assert payload["conditions"][0]["condition_id"] == "clean"
    assert payload["formal_protocol"]["sha256"] == file_sha256(protocol_path)


def test_run_writes_hashed_auditable_artifacts(tmp_path):
    target = make_target(tmp_path)
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, make_plan(target))
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    args = evaluate_robustness.build_parser().parse_args(
        [
            "run",
            "--plan",
            str(plan_path),
            "--checkpoint",
            str(target.checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--method-id",
            "baseline-d2nn",
            "--batch-size",
            "3",
            "--include-scores",
        ]
    )
    loader = DataLoader(CliDataset(), batch_size=3, shuffle=False)
    selection = {"selection": "test_prefix_subset", "sample_count": 4, "full_test_samples": 5}
    with mock.patch("evaluate_robustness.load_robustness_target", return_value=target), mock.patch(
        "evaluate_robustness.build_indexed_test_loader", return_value=(loader, selection)
    ), mock.patch("robustness_run_manifest.environment_manifest", return_value={"test": True}):
        manifest_path = evaluate_robustness.run_evaluation(args)

    manifest = read_json(manifest_path)
    verify_manifest_integrity(manifest)
    assert manifest["evidence_status"] == "subset_smoke_only_not_a_main_result"
    assert manifest["model"]["training_seed"] == 42
    assert manifest["statistics"]["independent_unit"] == "training_seed"
    assert "training_provenance.py" in manifest["evaluation_source_files_sha256"]
    assert manifest["artifacts"]["predictions"]["row_count"] == 4
    assert manifest["artifacts"]["draw_metrics"]["row_count"] == 1
    for record in manifest["artifacts"].values():
        assert file_sha256(output_dir / record["path"]) == record["sha256"]


def test_run_rejects_nonempty_output_before_dataset_loading(tmp_path):
    target = make_target(tmp_path)
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, make_plan(target))
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    args = evaluate_robustness.build_parser().parse_args(
        [
            "run",
            "--plan",
            str(plan_path),
            "--checkpoint",
            str(target.checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--method-id",
            "baseline-d2nn",
        ]
    )
    with mock.patch("evaluate_robustness.load_robustness_target", return_value=target), mock.patch(
        "evaluate_robustness.build_indexed_test_loader"
    ) as dataset_loader:
        with pytest.raises(ValueError, match="empty or absent"):
            evaluate_robustness.run_evaluation(args)
    dataset_loader.assert_not_called()


def test_checkpoint_preflight_failure_happens_before_dataset_loading(tmp_path):
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, make_plan(make_target(tmp_path)))
    args = evaluate_robustness.build_parser().parse_args(
        [
            "run",
            "--plan",
            str(plan_path),
            "--checkpoint",
            str(tmp_path / "invalid.pth"),
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--method-id",
            "baseline-d2nn",
        ]
    )
    with mock.patch(
        "evaluate_robustness.load_robustness_target", side_effect=ValueError("invalid checkpoint identity")
    ), mock.patch("evaluate_robustness.build_indexed_test_loader") as dataset_loader:
        with pytest.raises(ValueError, match="identity"):
            evaluate_robustness.run_evaluation(args)
    dataset_loader.assert_not_called()
