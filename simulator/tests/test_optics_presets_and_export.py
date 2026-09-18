import json
import math

import numpy as np
import pytest
import torch

from artifacts import (
    CLASSIFIER_D2NN2018_THZ_OPTICS,
    checkpoint_manifest_path,
    optical_config_dict,
    save_manifest,
)
from d2nn import D2NN
from export_phase_plate import build_parser as build_export_parser
from export_phase_plate import main as export_phase_plate_main
from train import build_parser as build_train_parser
from train import validate_training_args


def test_d2nn2018_thz_preset_uses_documented_pure_simulation_scale():
    optics = CLASSIFIER_D2NN2018_THZ_OPTICS

    assert optics.wavelength == pytest.approx(0.75e-3)
    assert optics.pixel_size == pytest.approx(0.4e-3)
    assert optics.layer_distance == pytest.approx(30e-3)
    assert optics.input_distance == pytest.approx(30e-3)
    assert optics.output_distance == pytest.approx(30e-3)


def test_d2nn2018_thz_accepts_multilayer_small_simulation_but_not_imaging():
    simulation_args = build_train_parser().parse_args(
        ["--optics-preset", "d2nn2018_thz", "--size", "64", "--layers", "3"]
    )
    validate_training_args(simulation_args)

    imaging_args = build_train_parser().parse_args(
        ["--task", "imaging", "--optics-preset", "d2nn2018_thz"]
    )
    with pytest.raises(ValueError, match="only supported for classification"):
        validate_training_args(imaging_args)


def _write_checkpoint(tmp_path, *, levels):
    optics = CLASSIFIER_D2NN2018_THZ_OPTICS.with_overrides(size=4, num_layers=1)
    model = D2NN(**optics.classifier_model_kwargs(), propagation_backend="fft")
    with torch.no_grad():
        model.layers[0].phase.copy_(torch.linspace(-0.4, 2 * math.pi + 0.4, 16).reshape(4, 4))

    checkpoint_path = tmp_path / "qat_demo.pth"
    torch.save(model.state_dict(), checkpoint_path)
    save_manifest(
        checkpoint_manifest_path(checkpoint_path),
        {
            "task": "classification",
            "optics_preset": "d2nn2018_thz",
            "optical_config": optical_config_dict(optics),
            "training_perturbations": {
                "config": {"phase_quantization_levels": levels},
            },
        },
    )
    return checkpoint_path


def test_qat_export_inherits_phase_levels_without_reusing_height_encoding(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path, levels=4)
    output_dir = tmp_path / "exports"

    export_phase_plate_main(
        [
            "--task",
            "classification",
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--height-quantization-levels",
            "16",
        ]
    )

    export_root = output_dir / checkpoint_path.stem
    phase_masks = np.load(export_root / "phase_masks.npy")
    metadata = json.loads((export_root / "metadata.json").read_text(encoding="utf-8"))
    step = 2 * math.pi / 4

    assert np.unique(np.round(phase_masks / step, decimals=5)).size <= 4
    assert np.allclose(np.remainder(phase_masks / step, 1), 0, atol=1e-5)
    assert metadata["phase_quantization"] == {
        "trained_levels": 4,
        "export_levels": 4,
        "source": "checkpoint_manifest",
    }
    assert metadata["height_quantization_levels"] == 16


def test_qat_export_rejects_phase_levels_that_disagree_with_manifest(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path, levels=4)
    parser_args = build_export_parser().parse_args(
        [
            "--task",
            "classification",
            "--checkpoint",
            str(checkpoint_path),
            "--phase-quantization-levels",
            "8",
        ]
    )
    assert parser_args.phase_quantization_levels == 8

    with pytest.raises(ValueError, match="do not match the QAT checkpoint manifest"):
        export_phase_plate_main(
            [
                "--task",
                "classification",
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                str(tmp_path / "exports"),
                "--phase-quantization-levels",
                "8",
            ]
        )


def test_non_qat_export_can_explicitly_quantize_phase_and_keep_legacy_height_alias(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path, levels=None)
    output_dir = tmp_path / "exports"

    export_phase_plate_main(
        [
            "--task",
            "classification",
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--phase-quantization-levels",
            "4",
            "--quantization-levels",
            "8",
        ]
    )

    export_root = output_dir / checkpoint_path.stem
    phase_masks = np.load(export_root / "phase_masks.npy")
    metadata = json.loads((export_root / "metadata.json").read_text(encoding="utf-8"))
    step = 2 * math.pi / 4

    assert np.allclose(np.remainder(phase_masks / step, 1), 0, atol=1e-5)
    assert metadata["phase_quantization"]["source"] == "explicit_non_qat_export"
    assert metadata["height_quantization_levels"] == 8


def test_invalid_height_levels_fail_before_creating_export_artifacts(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path, levels=4)
    output_dir = tmp_path / "exports"

    with pytest.raises(ValueError, match="height quantization levels"):
        export_phase_plate_main(
            [
                "--task",
                "classification",
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--height-quantization-levels",
                "1",
            ]
        )

    assert not (output_dir / checkpoint_path.stem).exists()
