import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import train


class TrainingManifestTests(unittest.TestCase):
    def test_training_writes_required_provenance_and_checkpoint_hash(self):
        class FakeDataset:
            def __init__(self, root, train, download, transform):
                self.train = train

            def __len__(self):
                return 12 if self.train else 4

            def __getitem__(self, index):
                return torch.zeros(1, 28, 28), index % 10

        class FakeClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.layers = torch.nn.ModuleList([torch.nn.Identity()])
                self.size = 8
                self.detector_feature_count = 10

        args = train.build_parser().parse_args(
            [
                "--epochs",
                "1",
                "--val-size",
                "2",
                "--layers",
                "1",
                "--size",
                "8",
                "--rs-backend",
                "fft",
                "--phase-quantization-levels",
                "4",
                "--deterministic",
            ]
        )
        dataset_config = {
            "dataset_cls": FakeDataset,
            "display_name": "Fake-MNIST",
            "checkpoint_name": "best_fake.pth",
            "paper_target": None,
            "grayscale": False,
        }
        fake_model = FakeClassifier()

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            train,
            "get_classification_dataset_config",
            return_value=dataset_config,
        ), mock.patch.object(
            train,
            "build_model_for_task",
            return_value=fake_model,
        ), mock.patch.object(
            train,
            "fit_classification_model",
            return_value=({"accuracy": 50.0, "contrast": 0.2, "epoch": 1}, None, {"train": {}, "val": {}}, {}),
        ) as fit_mock, mock.patch.object(
            train,
            "_run_classification_epoch",
            return_value={"accuracy": 51.0, "contrast": 0.21, "loss": 0.3},
        ) as test_epoch_mock:
            save_dir = Path(tmpdir)
            train.run_classification_training(args, torch.device("cpu"), Path("data"), save_dir)

            manifest_paths = list(save_dir.glob("*.json"))
            self.assertEqual(len(manifest_paths), 1)
            payload = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
            checkpoint_path = Path(payload["artifacts"]["checkpoint"]["path"])
            expected_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

        self.assertEqual(payload["checkpoint_sha256"], expected_hash)
        self.assertEqual(payload["data_split"]["train_samples"], 10)
        self.assertEqual(payload["data_split"]["validation_samples"], 2)
        self.assertEqual(payload["data_split"]["test_samples"], 4)
        self.assertEqual(payload["model"]["trainable_parameters"], 1)
        self.assertEqual(payload["test_condition"], "quantized_clean")
        perturbations = payload["training_perturbations"]
        self.assertEqual(perturbations["config"]["phase_quantization_levels"], 4)
        self.assertIn("gap_spacing_error_max_m", perturbations["config"])
        self.assertNotIn("axial_shift_max_m", perturbations["config"])
        self.assertFalse(perturbations["detector_noise_hardware_calibrated"])
        self.assertEqual(perturbations["rng_streams"], "separate_optical_and_detector")
        self.assertIn("source_files_sha256", payload["environment"])
        self.assertTrue(payload["cli_configuration"]["deterministic"])
        self.assertIsNotNone(fit_mock.call_args.kwargs["eval_perturbation_draw_factory"])
        self.assertIs(
            test_epoch_mock.call_args.kwargs["perturbation_draw_factory"],
            fit_mock.call_args.kwargs["eval_perturbation_draw_factory"],
        )


if __name__ == "__main__":
    unittest.main()
