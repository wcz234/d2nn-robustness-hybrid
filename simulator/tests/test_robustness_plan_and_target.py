import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from perturbations import PerturbationConfig
from model_variants import ElectronicMLPClassifier
from robustness_plan import (
    build_robustness_plan,
    build_robustness_plan_from_protocol,
    plan_identity_sha256,
    validate_robustness_plan,
    validate_target_against_plan,
)
from robustness_target import load_robustness_target


def fake_target(**overrides):
    optical_config = {
        "wavelength": 0.75e-3,
        "layer_distance": 30e-3,
        "pixel_size": 0.4e-3,
        "input_distance": 30e-3,
        "output_distance": 30e-3,
        "size": 8,
        "num_layers": 2,
    }
    values = {
        "manifest": {"model_version": "rs_v1", "optical_config": optical_config},
        "checkpoint_sha256": "a" * 64,
        "model_variant": "d2nn",
        "training_seed": 42,
        "dataset_key": "mnist",
        "input_shape": (1, 28, 28),
        "test_samples": 10000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def build_plan(target=None, config=None, **overrides):
    options = {
        "target": target or fake_target(),
        "condition_id": "clean",
        "config": config or PerturbationConfig(),
        "draw_count": 1,
        "optical_seed": 101,
        "detector_seed": 202,
        "sample_limit": None,
    }
    options.update(overrides)
    return build_robustness_plan(**options)


def refresh_plan_id(plan):
    plan["plan_id"] = f"robustness-plan-{plan_identity_sha256(plan)[:20]}"


def test_plan_freezes_cohort_scope_seeds_and_statistical_boundary():
    plan = build_plan(sample_limit=32)

    validate_robustness_plan(plan)
    assert plan["scope"] == "numerical_simulation_only"
    assert plan["sample_scope"] == {
        "selection": "test_prefix_subset",
        "sample_count": 32,
        "full_test_samples": 10000,
    }
    assert plan["statistics"]["independent_unit"] == "training_seed"
    assert plan["sampling_protocol"]["detector_noise_scope"] == "per_sample_feature_table"
    assert plan["conditions"][0]["draws"][0]["optical_seed"] != plan["conditions"][0]["draws"][0][
        "detector_seed"
    ]


@pytest.mark.parametrize("field", ["sampling_protocol", "statistics"])
def test_plan_rejects_rewritten_protocol_semantics(field):
    plan = build_plan()
    plan[field] = {"rewritten": "claim"}
    refresh_plan_id(plan)

    with pytest.raises(ValueError, match=field):
        validate_robustness_plan(plan)


def test_plan_rejects_unknown_or_missing_perturbation_fields():
    plan = build_plan()
    plan["conditions"][0]["perturbation_config"]["gap_error_typo"] = 1e-4
    with pytest.raises(ValueError, match="unknown"):
        validate_robustness_plan(plan)

    plan = build_plan()
    del plan["conditions"][0]["perturbation_config"]["phase_noise_std_rad"]
    with pytest.raises(ValueError, match="missing"):
        validate_robustness_plan(plan)


def test_plan_rejects_meaningless_or_nonpositive_gap_ranges():
    single_layer = fake_target()
    single_layer.manifest = copy.deepcopy(single_layer.manifest)
    single_layer.manifest["optical_config"]["num_layers"] = 1
    with pytest.raises(ValueError, match="single-layer"):
        build_plan(target=single_layer, config=PerturbationConfig(axial_shift_max_m=1e-4), condition_id="gap")

    with pytest.raises(ValueError, match="smaller than"):
        build_plan(
            config=PerturbationConfig(axial_shift_max_m=30e-3),
            condition_id="gap",
            draw_count=2,
        )


def test_deterministic_conditions_require_one_draw():
    with pytest.raises(ValueError, match="exactly one draw"):
        build_plan(draw_count=2)


def test_formal_protocol_builds_one_shared_multi_condition_plan():
    protocol = {
        "protocol_schema_version": "formal-simulation-protocol/v1",
        "protocol_id": "formal-test-v1",
        "scope": "numerical_simulation_only",
        "dataset": {"key": "mnist", "test_samples": 10000},
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
            },
            {
                "condition_id": "phase-noise",
                "draw_count": 3,
                "perturbation_config": {
                    "lateral_shift_max_px": 0.0,
                    "gap_spacing_error_max_m": 0.0,
                    "phase_noise_std_rad": 0.05,
                    "phase_quantization_levels": None,
                    "detector_noise_std_relative": 0.0,
                },
            },
        ],
    }

    plan = build_robustness_plan_from_protocol(
        target=fake_target(),
        protocol=protocol,
        protocol_sha256="b" * 64,
    )

    assert plan["sample_scope"]["selection"] == "full_test"
    assert [condition["condition_id"] for condition in plan["conditions"]] == ["clean", "phase-noise"]
    assert len(plan["conditions"][1]["draws"]) == 3
    assert plan["formal_protocol"]["sha256"] == "b" * 64
    validate_robustness_plan(plan)


def test_target_must_match_the_frozen_optical_cohort():
    plan = build_plan()
    different = fake_target(dataset_key="fashion_mnist")
    with pytest.raises(ValueError, match="dataset_key"):
        validate_target_against_plan(different, plan)


def base_checkpoint_manifest(checkpoint_path: Path) -> dict:
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return {
        "task": "classification",
        "model_version": "rs_v1",
        "seed": 42,
        "model": {
            "variant": "d2nn",
            "class_name": "D2NN",
            "total_parameters": 128,
            "variant_config": {},
        },
        "data_split": {"dataset_key": "mnist", "input_shape": [1, 28, 28], "test_samples": 10000},
        "training_perturbations": {"config": {}},
        "artifacts": {"checkpoint": {"sha256": digest}},
        "optical_config": {
            "wavelength": 0.75e-3,
            "layer_distance": 30e-3,
            "pixel_size": 0.4e-3,
            "input_distance": 30e-3,
            "output_distance": 30e-3,
            "size": 8,
            "num_layers": 2,
        },
    }


def write_checkpoint_case(tmp_path, mutate):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not loaded for preflight failures")
    manifest = base_checkpoint_manifest(checkpoint)
    mutate(manifest)
    checkpoint.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest.update(task="imaging"), "classification checkpoints"),
        (lambda manifest: manifest["model"].update(variant="unsupported"), "unsupported"),
        (lambda manifest: manifest.pop("model_version"), "model_version"),
        (lambda manifest: manifest["artifacts"]["checkpoint"].update(sha256="0" * 64), "SHA-256"),
    ],
)
def test_ineligible_or_unidentified_checkpoints_fail_before_state_loading(tmp_path, mutate, message):
    checkpoint = write_checkpoint_case(tmp_path, mutate)
    with mock.patch("robustness_target.load_checkpoint_state_dict") as state_loader:
        with pytest.raises(ValueError, match=message):
            load_robustness_target(checkpoint, device=torch.device("cpu"))
    state_loader.assert_not_called()


@pytest.mark.parametrize("field", ["dataset_key", "test_samples", "input_shape"])
def test_missing_dataset_identity_is_explicit_before_state_loading(tmp_path, field):
    checkpoint = write_checkpoint_case(tmp_path, lambda manifest: manifest["data_split"].pop(field))
    with mock.patch("robustness_target.load_checkpoint_state_dict") as state_loader:
        with pytest.raises(ValueError, match=field):
            load_robustness_target(checkpoint, device=torch.device("cpu"))
    state_loader.assert_not_called()


def test_missing_training_seed_is_explicit_before_state_loading(tmp_path):
    checkpoint = write_checkpoint_case(tmp_path, lambda manifest: manifest.pop("seed"))
    with mock.patch("robustness_target.load_checkpoint_state_dict") as state_loader:
        with pytest.raises(ValueError, match="seed"):
            load_robustness_target(checkpoint, device=torch.device("cpu"))
    state_loader.assert_not_called()


def test_electronic_checkpoint_loads_without_optical_config(tmp_path):
    checkpoint = tmp_path / "electronic.pth"
    model = ElectronicMLPClassifier(input_shape=(1, 28, 28), hidden_dim=18, num_classes=10)
    torch.save(model.state_dict(), checkpoint)
    manifest = base_checkpoint_manifest(checkpoint)
    manifest["model"] = {
        "variant": "electronic",
        "class_name": "ElectronicMLPClassifier",
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "variant_config": {"input_shape": [1, 28, 28], "hidden_dim": 18, "num_classes": 10},
    }
    manifest.pop("optical_config")
    manifest["artifacts"]["checkpoint"]["sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest["training_perturbations"]["config"] = {
        "lateral_shift_max_px": 0.0,
        "gap_spacing_error_max_m": 0.0,
        "phase_noise_std_rad": 0.0,
        "phase_quantization_levels": None,
        "detector_noise_std_relative": 0.0,
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")

    target = load_robustness_target(checkpoint, device=torch.device("cpu"))

    assert target.model_variant == "electronic"
    assert target.optics is None
    assert target.feature_schema_id == "flattened_input:1x28x28"
