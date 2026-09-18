import unittest
from unittest import mock

import torch

from artifacts import CLASSIFIER_PAPER_OPTICS
from model_variants import ElectronicMLPClassifier, HybridD2NNClassifier
from perturbations import PerturbationConfig, derive_detector_seed
import train
import train_core
from training_runtime import (
    build_classification_model,
    build_perturbation_factories,
    classification_run_suffix,
    resolve_perturbation_seeds,
    resolve_training_perturbation_config,
)


class TrainingCliIntegrationTests(unittest.TestCase):
    def test_default_cli_preserves_standard_d2nn_without_perturbations(self):
        args = train.build_parser().parse_args([])
        config = resolve_training_perturbation_config(args)

        self.assertEqual(args.model_variant, "d2nn")
        self.assertEqual(config, PerturbationConfig())
        self.assertEqual(
            resolve_perturbation_seeds(args),
            {
                "optical_perturbation_seed": args.seed,
                "detector_noise_seed": derive_detector_seed(args.seed),
            },
        )
        self.assertIsNone(classification_run_suffix(args, config, resolve_perturbation_seeds(args)))

    def test_cli_accepts_hybrid_and_reproducibility_options(self):
        args = train.build_parser().parse_args(
            [
                "--model-variant",
                "hybrid",
                "--hybrid-pool-size",
                "4",
                "--hybrid-hidden-dim",
                "12",
                "--optical-perturbation-seed",
                "91",
                "--detector-noise-seed",
                "92",
                "--train-lateral-shift-max-px",
                "0.5",
                "--train-gap-spacing-error-max-m",
                "0.0001",
                "--phase-quantization-levels",
                "8",
                "--deterministic",
            ]
        )

        self.assertEqual(args.model_variant, "hybrid")
        self.assertEqual(args.hybrid_pool_size, 4)
        self.assertEqual(args.hybrid_hidden_dim, 12)
        self.assertEqual(
            resolve_perturbation_seeds(args),
            {"optical_perturbation_seed": 91, "detector_noise_seed": 92},
        )
        self.assertTrue(args.deterministic)
        config = resolve_training_perturbation_config(args)
        self.assertEqual(config.phase_quantization_levels, 8)
        self.assertEqual(config.axial_shift_max_m, 0.0001)

    def test_validation_rejects_ignored_or_invalid_variant_configuration(self):
        cases = (
            (["--model-variant", "electronic", "--train-lateral-shift-max-px", "0.1"], "not applicable"),
            (["--model-variant", "electronic", "--rs-backend", "fft"], "not applicable"),
            (["--model-variant", "hybrid", "--size", "8", "--hybrid-pool-size", "9"], "cannot exceed"),
            (["--task", "imaging", "--model-variant", "hybrid"], "only supported for classification"),
            (["--phase-quantization-levels", "1"], "integer >= 2"),
        )
        for argv, message in cases:
            with self.subTest(argv=argv), self.assertRaisesRegex(ValueError, message):
                train.validate_training_args(train.build_parser().parse_args(argv))

    def test_model_builder_exposes_all_classification_variants(self):
        parser = train.build_parser()
        optics = CLASSIFIER_PAPER_OPTICS.with_overrides(size=8, num_layers=1)
        common = {
            "optics": optics,
            "input_shape": (1, 28, 28),
            "activation_type": "none",
            "activation_positions": (),
            "activation_hparams": {},
            "propagation_backend": "fft",
            "propagation_chunk_size": None,
        }
        sentinel = object()
        d2nn_builder = mock.Mock(return_value=sentinel)

        standard = build_classification_model(
            args=parser.parse_args(["--size", "8", "--layers", "1"]),
            d2nn_builder=d2nn_builder,
            **common,
        )
        hybrid = build_classification_model(
            args=parser.parse_args(
                ["--model-variant", "hybrid", "--size", "8", "--layers", "1", "--hybrid-pool-size", "2"]
            ),
            d2nn_builder=mock.Mock(),
            **common,
        )
        electronic = build_classification_model(
            args=parser.parse_args(["--model-variant", "electronic"]),
            d2nn_builder=mock.Mock(),
            **common,
        )

        self.assertIs(standard, sentinel)
        self.assertIsInstance(hybrid, HybridD2NNClassifier)
        self.assertEqual(hybrid.detector_feature_count, 4)
        self.assertIsInstance(electronic, ElectronicMLPClassifier)
        self.assertEqual(electronic.input_shape, (1, 28, 28))

    def test_seeded_train_factory_and_quantized_clean_eval_factory_are_distinct(self):
        optics = CLASSIFIER_PAPER_OPTICS.with_overrides(size=8, num_layers=1)
        model = HybridD2NNClassifier(
            **optics.classifier_model_kwargs(),
            propagation_backend="fft",
            pool_size=2,
            hidden_dim=4,
        )
        config = PerturbationConfig(
            lateral_shift_max_px=0.5,
            phase_quantization_levels=4,
            detector_noise_std_relative=0.1,
        )
        seeds = {"optical_perturbation_seed": 17, "detector_noise_seed": 18}
        first_train, first_eval = build_perturbation_factories(model, config, **seeds)
        repeated_train, _ = build_perturbation_factories(model, config, **seeds)

        first = first_train(batch_size=3, dtype=torch.float32, device="cpu")
        repeated = repeated_train(batch_size=3, dtype=torch.float32, device="cpu")
        quantized_clean = first_eval(batch_size=3, dtype=torch.float32, device="cpu")

        self.assertEqual(first.layers[0].lateral_shift_px, repeated.layers[0].lateral_shift_px)
        self.assertTrue(torch.equal(first.detector_noise_relative, repeated.detector_noise_relative))
        self.assertEqual(quantized_clean.phase_quantization_levels, 4)
        self.assertIsNone(quantized_clean.detector_noise_relative)
        self.assertEqual(quantized_clean.layers[0].lateral_shift_px, (0.0, 0.0))

    def test_detector_shape_does_not_advance_the_optical_rng_stream(self):
        class FakeModel:
            def __init__(self, detector_feature_count):
                self.layers = (object(), object())
                self.size = 6
                self.detector_feature_count = detector_feature_count

        config = PerturbationConfig(
            lateral_shift_max_px=0.5,
            phase_noise_std_rad=0.1,
            detector_noise_std_relative=0.2,
        )
        seeds = {"optical_perturbation_seed": 31, "detector_noise_seed": 32}
        d2nn_factory, _ = build_perturbation_factories(FakeModel(10), config, **seeds)
        hybrid_factory, _ = build_perturbation_factories(FakeModel(16), config, **seeds)

        d2nn_draws = [d2nn_factory(batch_size=2, dtype=torch.float32, device="cpu") for _ in range(2)]
        hybrid_draws = [hybrid_factory(batch_size=2, dtype=torch.float32, device="cpu") for _ in range(2)]

        for d2nn_draw, hybrid_draw in zip(d2nn_draws, hybrid_draws):
            for d2nn_layer, hybrid_layer in zip(d2nn_draw.layers, hybrid_draw.layers):
                self.assertEqual(d2nn_layer.lateral_shift_px, hybrid_layer.lateral_shift_px)
                self.assertTrue(torch.equal(d2nn_layer.phase_noise_rad, hybrid_layer.phase_noise_rad))
        self.assertEqual(d2nn_draws[0].detector_noise_relative.shape, (2, 10))
        self.assertEqual(hybrid_draws[0].detector_noise_relative.shape, (2, 16))

    def test_epoch_runner_requests_one_draw_per_batch(self):
        class TinyClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(1, 10)
                self.layers = ()
                self.seen_draws = []

            def forward_with_metrics(self, data, target=None, perturbation_draw=None):
                self.seen_draws.append(perturbation_draw)
                logits = self.classifier(data)
                scores = torch.softmax(logits, dim=1)
                return {"scores": scores, "logits": logits, "contrast": torch.zeros(data.size(0))}

        draws = []

        def draw_factory(*, batch_size, dtype, device):
            draw = {"batch_size": batch_size, "dtype": dtype, "device": str(device)}
            draws.append(draw)
            return draw

        model = TinyClassifier()
        loader = [
            (torch.ones(2, 1), torch.tensor([0, 1])),
            (torch.ones(1, 1), torch.tensor([2])),
        ]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        train_core.run_classification_epoch(
            model,
            loader,
            torch.device("cpu"),
            optimizer=optimizer,
            perturbation_draw_factory=draw_factory,
        )

        self.assertEqual([draw["batch_size"] for draw in draws], [2, 1])
        self.assertEqual(model.seen_draws, draws)

if __name__ == "__main__":
    unittest.main()
