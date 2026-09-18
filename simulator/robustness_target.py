"""Strict checkpoint reconstruction and indexed test data for robustness evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from artifacts import (
    CLASSIFIER_PAPER_OPTICS,
    build_model_for_task,
    ensure_checkpoint_version,
    infer_architecture,
    load_checkpoint_state_dict,
    read_checkpoint_manifest,
    resolve_optics,
)
from model_variants import ElectronicMLPClassifier, HybridD2NNClassifier, LeNet5Classifier
from tasks import MODEL_VERSION, build_classification_transform, get_classification_dataset_config
from tasks import resolve_activation_config, resolve_propagation_config
from training_provenance import file_sha256


OPTICAL_CONFIG_FIELDS = (
    "wavelength",
    "layer_distance",
    "pixel_size",
    "input_distance",
    "output_distance",
    "size",
    "num_layers",
)


@dataclass(frozen=True)
class LoadedRobustnessTarget:
    model: torch.nn.Module
    manifest: dict
    checkpoint_path: Path
    checkpoint_sha256: str
    dataset_key: str
    input_shape: tuple[int, ...]
    test_samples: int
    training_seed: int
    model_variant: str
    feature_schema_id: str
    score_semantics: str
    optics: object


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index


def _required_mapping(parent: dict, key: str) -> dict:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint manifest requires object field {key!r}")
    return value


def _required_string(parent: dict, key: str, *, context: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires non-empty string field {key!r}")
    return value


def _required_int(parent: dict, key: str, *, context: str, minimum: int = 0) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} requires integer field {key!r} >= {minimum}")
    return value


def _data_identity(manifest: dict, data_manifest: dict) -> tuple[str, tuple[int, ...], int, int]:
    dataset_key = _required_string(data_manifest, "dataset_key", context="checkpoint data_split")
    raw_shape = data_manifest.get("input_shape")
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 3:
        raise ValueError("checkpoint data_split.input_shape must contain three dimensions")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in raw_shape):
        raise ValueError("checkpoint data_split.input_shape dimensions must be positive integers")
    test_samples = _required_int(data_manifest, "test_samples", context="checkpoint data_split", minimum=1)
    training_seed = _required_int(manifest, "seed", context="checkpoint manifest")
    return dataset_key, tuple(raw_shape), test_samples, training_seed


def _validate_model_manifest(model_manifest: dict) -> None:
    _required_string(model_manifest, "class_name", context="checkpoint model")
    _required_int(model_manifest, "total_parameters", context="checkpoint model", minimum=1)
    variant = model_manifest.get("variant")
    if variant == "hybrid":
        config = _required_mapping(model_manifest, "variant_config")
        pool_size = _required_int(config, "pool_size", context="hybrid variant_config", minimum=1)
        _required_int(config, "hidden_dim", context="hybrid variant_config", minimum=1)
        feature_count = _required_int(config, "detector_feature_count", context="hybrid variant_config", minimum=1)
        if feature_count != pool_size**2:
            raise ValueError("hybrid detector_feature_count must equal pool_size squared")
    elif variant == "electronic":
        config = _required_mapping(model_manifest, "variant_config")
        input_shape = config.get("input_shape")
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError("electronic variant_config.input_shape must contain three dimensions")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in input_shape):
            raise ValueError("electronic variant_config.input_shape dimensions must be positive integers")
        _required_int(config, "hidden_dim", context="electronic variant_config", minimum=1)
        _required_int(config, "num_classes", context="electronic variant_config", minimum=1)
    elif variant == "lenet5":
        config = _required_mapping(model_manifest, "variant_config")
        input_shape = config.get("input_shape")
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError("lenet5 variant_config.input_shape must contain three dimensions")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in input_shape):
            raise ValueError("lenet5 variant_config.input_shape dimensions must be positive integers")
        conv_channels = config.get("conv_channels")
        if not isinstance(conv_channels, list) or len(conv_channels) != 2:
            raise ValueError("lenet5 variant_config.conv_channels must contain two stage widths")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in conv_channels):
            raise ValueError("lenet5 variant_config.conv_channels widths must be positive integers")
        _required_int(config, "hidden_dim", context="lenet5 variant_config", minimum=1)
        _required_int(config, "num_classes", context="lenet5 variant_config", minimum=1)


def _expected_checkpoint_sha256(manifest: dict) -> str:
    artifacts = _required_mapping(manifest, "artifacts")
    checkpoint = _required_mapping(artifacts, "checkpoint")
    expected = checkpoint.get("sha256") or manifest.get("checkpoint_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("checkpoint manifest requires a SHA-256 checkpoint identity")
    return expected.lower()


def _validate_manifest_identity(manifest: dict, checkpoint_path: Path) -> tuple[dict, dict, dict]:
    if manifest.get("task") != "classification":
        raise ValueError("robust optical evaluation only supports classification checkpoints")
    ensure_checkpoint_version(
        manifest,
        expected_version=MODEL_VERSION,
        checkpoint_path=checkpoint_path,
        allow_missing=False,
    )
    model_manifest = _required_mapping(manifest, "model")
    data_manifest = _required_mapping(manifest, "data_split")
    variant = model_manifest.get("variant")
    if variant not in ("d2nn", "hybrid", "electronic", "lenet5"):
        raise ValueError(f"classification evaluation does not support model variant {variant!r}")
    perturbation_manifest = _required_mapping(manifest, "training_perturbations")
    perturbation_config = _required_mapping(perturbation_manifest, "config")
    if variant in ("electronic", "lenet5") and any(
        value not in (None, 0, 0.0) for value in perturbation_config.values()
    ):
        raise ValueError("pure-electronic checkpoints cannot contain optical or detector training perturbations")
    _validate_model_manifest(model_manifest)
    _data_identity(manifest, data_manifest)
    return model_manifest, data_manifest, perturbation_manifest


def _validate_optical_config(manifest: dict, state_dict: dict) -> None:
    optical_config = _required_mapping(manifest, "optical_config")
    missing = [field for field in OPTICAL_CONFIG_FIELDS if optical_config.get(field) is None]
    if missing:
        raise ValueError(f"checkpoint optical_config is missing required fields: {', '.join(missing)}")
    for field in OPTICAL_CONFIG_FIELDS[:5]:
        value = optical_config[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"checkpoint optical_config.{field} must be finite and positive")
    for field in OPTICAL_CONFIG_FIELDS[5:]:
        _required_int(optical_config, field, context="checkpoint optical_config", minimum=1)
    architecture = infer_architecture(state_dict)
    if int(optical_config["size"]) != architecture["size"]:
        raise ValueError("checkpoint optical size does not match state_dict architecture")
    if int(optical_config["num_layers"]) != architecture["num_layers"]:
        raise ValueError("checkpoint layer count does not match state_dict architecture")


def _build_model(*, manifest, model_manifest, state_dict, checkpoint_path, device):
    if model_manifest["variant"] == "electronic":
        variant_config = _required_mapping(model_manifest, "variant_config")
        model = ElectronicMLPClassifier(
            input_shape=tuple(variant_config["input_shape"]),
            hidden_dim=variant_config["hidden_dim"],
            num_classes=variant_config["num_classes"],
        )
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval(), None
    if model_manifest["variant"] == "lenet5":
        variant_config = _required_mapping(model_manifest, "variant_config")
        model = LeNet5Classifier(
            input_shape=tuple(variant_config["input_shape"]),
            conv_channels=tuple(variant_config["conv_channels"]),
            hidden_dim=variant_config["hidden_dim"],
            num_classes=variant_config["num_classes"],
        )
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval(), None
    optics = resolve_optics(
        CLASSIFIER_PAPER_OPTICS,
        state_dict=state_dict,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
    )
    activation_type, activation_positions, activation_hparams = resolve_activation_config(manifest=manifest)
    propagation_backend, propagation_chunk_size = resolve_propagation_config(manifest=manifest)
    common = {
        "activation_type": activation_type,
        "activation_positions": activation_positions,
        "activation_hparams": activation_hparams,
        "propagation_backend": propagation_backend,
        "propagation_chunk_size": propagation_chunk_size,
    }
    if model_manifest["variant"] == "d2nn":
        model = build_model_for_task("classification", optics, **common)
    else:
        variant_config = _required_mapping(model_manifest, "variant_config")
        model = HybridD2NNClassifier(
            **optics.classifier_model_kwargs(),
            pool_size=variant_config["pool_size"],
            hidden_dim=variant_config["hidden_dim"],
            linear_head=bool(variant_config.get("linear_head", False)),
            **common,
        )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), optics


def _validate_reconstructed_model(model, model_manifest: dict) -> None:
    if type(model).__name__ != model_manifest.get("class_name"):
        raise ValueError("reconstructed model class does not match checkpoint manifest")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if total_parameters != model_manifest.get("total_parameters"):
        raise ValueError("reconstructed model parameter count does not match checkpoint manifest")
    variant_config = model_manifest.get("variant_config") or {}
    expected_features = variant_config.get("detector_feature_count")
    if expected_features is not None and getattr(model, "detector_feature_count", None) != expected_features:
        raise ValueError("reconstructed detector feature count does not match checkpoint manifest")


def load_robustness_target(checkpoint_path, *, device) -> LoadedRobustnessTarget:
    checkpoint_path = Path(checkpoint_path)
    manifest = read_checkpoint_manifest(checkpoint_path)
    if not isinstance(manifest, dict):
        raise ValueError("robustness evaluation requires an adjacent checkpoint manifest")
    model_manifest, data_manifest, _ = _validate_manifest_identity(manifest, checkpoint_path)
    dataset_key, input_shape, test_samples, training_seed = _data_identity(manifest, data_manifest)
    expected_hash = _expected_checkpoint_sha256(manifest)
    actual_hash = file_sha256(checkpoint_path)
    if actual_hash != expected_hash:
        raise ValueError("checkpoint SHA-256 does not match its manifest")
    state_dict = load_checkpoint_state_dict(checkpoint_path, map_location="cpu")
    if model_manifest["variant"] not in ("electronic", "lenet5"):
        _validate_optical_config(manifest, state_dict)
    model, optics = _build_model(
        manifest=manifest,
        model_manifest=model_manifest,
        state_dict=state_dict,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    _validate_reconstructed_model(model, model_manifest)
    variant = model_manifest["variant"]
    if variant == "d2nn":
        feature_schema = "class_detector_energy:10"
        score_semantics = "raw_detector_energy"
    elif variant == "hybrid":
        feature_schema = f"pooled_intensity:{model.pool_size}x{model.pool_size}"
        score_semantics = "softmax_probability"
    elif variant == "lenet5":
        variant_config = model_manifest.get("variant_config") or {}
        feature_schema = f"cnn_penultimate_features:{variant_config.get('hidden_dim')}"
        score_semantics = "softmax_probability"
    else:
        feature_schema = f"flattened_input:{'x'.join(str(value) for value in input_shape)}"
        score_semantics = "softmax_probability"
    return LoadedRobustnessTarget(
        model=model,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=actual_hash,
        dataset_key=dataset_key,
        input_shape=input_shape,
        test_samples=test_samples,
        training_seed=training_seed,
        model_variant=variant,
        feature_schema_id=feature_schema,
        score_semantics=score_semantics,
        optics=optics,
    )


def build_indexed_test_loader(
    target: LoadedRobustnessTarget,
    *,
    data_dir,
    batch_size: int,
    num_workers: int,
    sample_limit: int | None,
):
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    dataset_config = get_classification_dataset_config(target.dataset_key)
    transform = build_classification_transform(dataset_config)
    dataset = dataset_config["dataset_cls"](data_dir, train=False, download=True, transform=transform)
    if len(dataset) != target.test_samples:
        raise ValueError("test dataset size does not match checkpoint manifest")
    if tuple(int(value) for value in dataset[0][0].shape) != target.input_shape:
        raise ValueError("test input shape does not match checkpoint manifest")
    if sample_limit is not None:
        if sample_limit < 1 or sample_limit > len(dataset):
            raise ValueError("sample_limit must be between 1 and the full test-set size")
        dataset = Subset(dataset, range(sample_limit))
    indexed = IndexedDataset(dataset)
    loader = DataLoader(indexed, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    selection = "full_test" if sample_limit is None else "test_prefix_subset"
    return loader, {"selection": selection, "sample_count": len(indexed), "full_test_samples": target.test_samples}
