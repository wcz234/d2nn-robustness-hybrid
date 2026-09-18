"""Classification runtime helpers for reproducible simulation experiments."""

from __future__ import annotations

from typing import Any, Callable

import torch

import robustness_plan
from model_variants import ElectronicMLPClassifier, HybridD2NNClassifier, LeNet5Classifier
from perturbations import (
    LayerPerturbation,
    PerturbationConfig,
    PerturbationDraw,
    PerturbationSampler,
    derive_detector_seed,
)


MODEL_VARIANTS = ("d2nn", "hybrid", "electronic", "lenet5")

PURE_ELECTRONIC_VARIANTS = ("electronic", "lenet5")
LENET5_CONV_CHANNELS = (6, 16)


def resolve_training_perturbation_config(args) -> PerturbationConfig:
    return PerturbationConfig(
        lateral_shift_max_px=args.train_lateral_shift_max_px,
        axial_shift_max_m=args.train_gap_spacing_error_max_m,
        phase_noise_std_rad=args.train_phase_noise_std_rad,
        phase_quantization_levels=args.phase_quantization_levels,
        detector_noise_std_relative=args.train_detector_noise_std_relative,
        train_phase_quantization_levels=getattr(args, "train_phase_quantization_levels", None),
        train_phase_filter_std_px=getattr(args, "train_phase_filter_std_px", 0.0),
    )


def perturbations_enabled(config: PerturbationConfig) -> bool:
    """True when any configured perturbation is actually active.

    This must stay an explicit field check rather than a whole-dataclass comparison:
    `phase_quantization_levels` is a *deployment* condition and
    `train_phase_quantization_levels` is a training-only knob, so neither may flip
    the `training_perturbations.enabled` flag that the clean-evaluation contract reads.
    """
    return any(
        value > 0
        for value in (
            config.lateral_shift_max_px,
            config.axial_shift_max_m,
            config.phase_noise_std_rad,
            config.detector_noise_std_relative,
        )
    )


def quantization_aware_training_enabled(config: PerturbationConfig) -> bool:
    """True when phase quantization is applied during training."""
    return config.train_phase_quantization_levels is not None


def phase_filter_training_enabled(config: PerturbationConfig) -> bool:
    """True when phase filtering is applied during training."""
    return config.train_phase_filter_std_px > 0


def training_only_regularization_enabled(config: PerturbationConfig) -> bool:
    """True when any training-only phase regularization knob is active."""
    return quantization_aware_training_enabled(config) or phase_filter_training_enabled(config)


def resolve_perturbation_seeds(args) -> dict[str, int]:
    optical_seed = args.seed if args.optical_perturbation_seed is None else int(args.optical_perturbation_seed)
    return {
        "optical_perturbation_seed": optical_seed,
        "detector_noise_seed": (
            derive_detector_seed(optical_seed)
            if args.detector_noise_seed is None
            else int(args.detector_noise_seed)
        ),
    }


def _electronic_optics_options_are_set(args) -> bool:
    return args.optics_preset != "paper" or any(
        value is not None
        for value in (
            args.wavelength,
            args.layer_distance,
            args.pixel_size,
            args.input_distance,
            args.output_distance,
            args.propagation_chunk_size,
        )
    ) or args.rs_backend != "direct"


def _electronic_activation_options_are_set(args) -> bool:
    return args.activation_type != "none" or any(
        value is not None
        for value in (
            args.activation_positions,
            args.activation_placement,
            args.activation_preset,
            args.activation_threshold,
            args.activation_temperature,
            args.activation_gain_min,
            args.activation_gain_max,
            args.activation_gamma,
            args.activation_responsivity,
            args.activation_emission_phase_mode,
        )
    )


def validate_variant_configuration(args, perturbation_config: PerturbationConfig) -> None:
    if args.model_variant != "d2nn" and args.task != "classification":
        raise ValueError("--model-variant hybrid/electronic/lenet5 is only supported for classification")
    if args.task != "classification" and perturbations_enabled(perturbation_config):
        raise ValueError("Training perturbations are only supported for classification")
    if args.model_variant == "hybrid" and args.hybrid_pool_size > args.size:
        raise ValueError("--hybrid-pool-size cannot exceed the optical --size")
    if args.hybrid_pool_size < 1 or args.hybrid_hidden_dim < 1 or args.electronic_hidden_dim < 1:
        raise ValueError("Model head dimensions must be positive integers")
    if args.lenet5_hidden_dim < 1:
        raise ValueError("--lenet5-hidden-dim must be a positive integer")
    if args.model_variant not in PURE_ELECTRONIC_VARIANTS:
        return
    if perturbations_enabled(perturbation_config):
        raise ValueError("Optical/detector perturbations are not applicable to pure-electronic variants")
    if _electronic_activation_options_are_set(args):
        raise ValueError("Optical activation options are not applicable to pure-electronic variants")
    if _electronic_optics_options_are_set(args):
        raise ValueError("Optics and propagation overrides are not applicable to pure-electronic variants")


def build_classification_model(
    *,
    args,
    optics,
    input_shape,
    activation_type,
    activation_positions,
    activation_hparams,
    propagation_backend,
    propagation_chunk_size,
    d2nn_builder: Callable[..., torch.nn.Module],
):
    if args.model_variant == "d2nn":
        return d2nn_builder(
            "classification",
            optics,
            activation_type=activation_type,
            activation_positions=activation_positions,
            activation_hparams=activation_hparams,
            propagation_chunk_size=propagation_chunk_size,
            propagation_backend=propagation_backend,
        )
    if args.model_variant == "hybrid":
        return HybridD2NNClassifier(
            **optics.classifier_model_kwargs(),
            activation_type=activation_type,
            activation_positions=activation_positions,
            activation_hparams=activation_hparams,
            propagation_chunk_size=propagation_chunk_size,
            propagation_backend=propagation_backend,
            pool_size=args.hybrid_pool_size,
            hidden_dim=args.hybrid_hidden_dim,
            linear_head=getattr(args, "hybrid_linear_head", False),
        )
    if args.model_variant == "lenet5":
        return LeNet5Classifier(
            input_shape=input_shape,
            hidden_dim=args.lenet5_hidden_dim,
            conv_channels=LENET5_CONV_CHANNELS,
        )
    return ElectronicMLPClassifier(input_shape=input_shape, hidden_dim=args.electronic_hidden_dim)


class PerBatchPerturbationFactory:
    """Generate one seeded device/noise draw for every training batch."""

    def __init__(
        self,
        model,
        config: PerturbationConfig,
        *,
        optical_seed: int,
        detector_seed: int,
        phase_quantization_levels: int | None = None,
    ):
        self.model = model
        self.config = config
        self.optical_seed = int(optical_seed)
        self.detector_seed = int(detector_seed)
        # Training-only quantization (QAT). `None` falls back to the deployment config.
        self.phase_quantization_levels = phase_quantization_levels
        self.draws_generated = 0
        self._sampler = PerturbationSampler(
            config,
            self.optical_seed,
            detector_seed=self.detector_seed,
        )

    def __call__(self, *, batch_size: int, dtype: torch.dtype, device) -> PerturbationDraw:
        draw = self._sampler.sample(
            num_layers=len(self.model.layers),
            size=self.model.size,
            batch_size=batch_size if self.config.detector_noise_std_relative > 0 else None,
            detector_features=(
                self.model.detector_feature_count if self.config.detector_noise_std_relative > 0 else None
            ),
            dtype=dtype,
            device=device,
            phase_quantization_levels=self.phase_quantization_levels,
        )
        self.draws_generated += 1
        return draw


class QuantizedCleanDrawFactory:
    """Apply deployment quantization during validation/test without random noise."""

    def __init__(self, model, levels: int):
        self._draw = PerturbationDraw(
            layers=tuple(LayerPerturbation() for _ in model.layers),
            phase_quantization_levels=levels,
        )

    def __call__(self, *, batch_size: int, dtype: torch.dtype, device) -> PerturbationDraw:
        del batch_size, dtype, device
        return self._draw


def build_perturbation_factories(
    model,
    config: PerturbationConfig,
    *,
    optical_perturbation_seed: int,
    detector_noise_seed: int,
):
    needs_train_factory = perturbations_enabled(config) or training_only_regularization_enabled(config)
    needs_eval_factory = config.phase_quantization_levels is not None
    if not needs_train_factory and not needs_eval_factory:
        return None, None
    train_factory = PerBatchPerturbationFactory(
        model,
        config,
        optical_seed=optical_perturbation_seed,
        detector_seed=detector_noise_seed,
        phase_quantization_levels=config.train_phase_quantization_levels,
    )
    eval_factory = None
    if config.phase_quantization_levels is not None:
        eval_factory = QuantizedCleanDrawFactory(model, config.phase_quantization_levels)
    return train_factory, eval_factory


def classification_run_suffix(args, config: PerturbationConfig, perturbation_seeds: dict[str, int]) -> str | None:
    if args.run_name:
        return None
    parts = []
    if args.model_variant != "d2nn":
        parts.append(f"model-{args.model_variant}")
        if args.model_variant == "hybrid":
            parts.extend((f"pool-{args.hybrid_pool_size}", f"hidden-{args.hybrid_hidden_dim}"))
        else:
            parts.append(f"hidden-{args.electronic_hidden_dim}")
    if perturbations_enabled(config):
        parts.append("train-perturbed")
        for key, value in serialized_perturbation_config(config).items():
            if value not in (None, 0, 0.0):
                parts.append(f"{key.replace('_', '-')}-{value}")
        parts.append(f"oseed-{perturbation_seeds['optical_perturbation_seed']}")
        parts.append(f"dseed-{perturbation_seeds['detector_noise_seed']}")
    return "__".join(parts) or None

def model_manifest(model, args, input_shape) -> dict[str, Any]:
    all_parameters = list(model.parameters())
    payload = {
        "variant": args.model_variant,
        "class_name": type(model).__name__,
        "total_parameters": sum(parameter.numel() for parameter in all_parameters),
        "trainable_parameters": sum(parameter.numel() for parameter in all_parameters if parameter.requires_grad),
        "phase_smoothness_regularizer_domain": (
            "not_applicable" if args.model_variant in PURE_ELECTRONIC_VARIANTS else "latent_continuous_phase"
        ),
    }
    if args.model_variant == "hybrid":
        payload["variant_config"] = {
            "pool_size": model.pool_size,
            "hidden_dim": model.hidden_dim,
            "linear_head": model.linear_head,
            "detector_feature_count": model.detector_feature_count,
        }
    elif args.model_variant == "lenet5":
        payload["variant_config"] = {
            "input_shape": list(input_shape),
            "conv_channels": list(model.conv_channels),
            "hidden_dim": model.hidden_dim,
            "num_classes": model.num_classes,
        }
    elif args.model_variant == "electronic":
        payload["variant_config"] = {
            "input_shape": list(input_shape),
            "hidden_dim": model.hidden_dim,
            "num_classes": model.num_classes,
        }
    else:
        payload["variant_config"] = {}
    return payload


def serialized_perturbation_config(config: PerturbationConfig) -> dict[str, Any]:
    """Serialize the *deployment* perturbation semantics of a config.

    Re-exports the canonical `robustness_plan` implementation so the checkpoint manifest
    payload and the protocol/plan payload cannot drift apart. It lists only the five
    deployment fields; `train_phase_quantization_levels` is a training-only
    quantization-aware knob and is recorded separately in the perturbation manifest.
    """
    return robustness_plan.serialized_perturbation_config(config)


def perturbation_manifest(config: PerturbationConfig, seeds: dict[str, int], train_factory) -> dict[str, Any]:
    metric_condition = "quantized_clean" if config.phase_quantization_levels is not None else "clean_continuous"
    stochastic_optical = any(
        value > 0
        for value in (
            config.lateral_shift_max_px,
            config.axial_shift_max_m,
            config.phase_noise_std_rad,
        )
    )
    detector_noise_enabled = config.detector_noise_std_relative > 0
    stochastic_training = stochastic_optical or detector_noise_enabled
    qat_levels = config.train_phase_quantization_levels
    return {
        "config": serialized_perturbation_config(config),
        "enabled": perturbations_enabled(config),
        "seeds": dict(seeds),
        "rng_streams": "separate_optical_and_detector",
        "training_sampling_scope": "per_batch" if stochastic_training else "fixed_quantization_or_disabled",
        "training_layer_perturbation_scope": (
            "shared_within_batch_resampled_per_batch" if stochastic_optical else "disabled"
        ),
        "training_detector_noise_scope": "per_sample_feature" if detector_noise_enabled else "disabled",
        "training_draws_generated": 0 if train_factory is None else train_factory.draws_generated,
        "training_phase_quantization_levels": qat_levels,
        "training_phase_filter_std_px": config.train_phase_filter_std_px,
        "phase_filter_training": (
            f"train_only_gaussian_std_{config.train_phase_filter_std_px}_px_validation_and_test_clean_continuous"
            if config.train_phase_filter_std_px > 0
            else "disabled"
        ),
        "quantization_aware_training": (
            f"train_only_{qat_levels}_levels_validation_and_test_clean_continuous"
            if qat_levels is not None
            else "disabled"
        ),
        "validation_test_condition": metric_condition,
        "validation_test_random_perturbations": False,
        "gap_spacing_error_semantics": "independent_inter_layer_gap_spacing_error",
        "detector_noise_semantics": "synthetic_relative_additive_simulation_noise",
        "detector_noise_hardware_calibrated": False,
    }
