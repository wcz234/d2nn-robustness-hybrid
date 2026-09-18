"""
Unified D2NN training entrypoint and lifecycle orchestration.

This module owns the public training CLI, runtime setup, task dispatch,
and end-to-end training flow. Shared epoch math stays in ``train_core.py``,
and task-specific builders/config helpers stay in layered internal modules.
"""

import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from artifacts import (
    CLASSIFICATION_ONLY_OPTICS_PRESETS,
    CLASSIFICATION_OPTICS_PRESETS,
    LAB_CLASSIFICATION_OPTICS_PRESETS,
    build_model_for_task,
    checkpoint_manifest_path,
    checkpoint_variant_path,
    derive_experiment_run_name,
    experiment_manifest_fields,
    resolve_training_optics_preset,
    save_manifest,
)
from tasks import (
    MODEL_VERSION,
    build_classification_transform,
    build_imaging_loaders,
    build_imaging_training_model,
    classification_split_lengths,
    execute_experiment_grid,
    evaluate_imaging,
    fit_classification_model,
    fit_imaging_model,
    format_experiment_grid_commands,
    get_classification_dataset_config,
    print_model_summary,
    resolve_activation_config,
    resolve_propagation_config,
)
from train_core import run_classification_epoch
from training_provenance import (
    checkpoint_manifest as build_checkpoint_provenance,
    cli_configuration,
    data_split_manifest,
    environment_manifest,
)
from training_runtime import (
    MODEL_VARIANTS,
    build_classification_model,
    build_perturbation_factories,
    classification_run_suffix,
    model_manifest,
    perturbation_manifest,
    resolve_perturbation_seeds,
    resolve_training_perturbation_config,
    validate_variant_configuration,
)


EXPERIMENT_GRID_CHOICES = [
    "coherent_amplitude_positions",
    "coherent_amplitude_presets",
    "coherent_phase_presets",
    "coherent_activation_mechanisms",
    "incoherent_intensity_presets",
    "activation_mechanisms",
]

def validate_training_args(args):
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.val_size < 1:
        raise ValueError("--val-size must be at least 1")

    perturbation_config = resolve_training_perturbation_config(args)
    validate_variant_configuration(args, perturbation_config)

    if args.optics_preset in CLASSIFICATION_ONLY_OPTICS_PRESETS and args.task != "classification":
        raise ValueError(f"Optics preset {args.optics_preset!r} is only supported for classification")

    if args.optics_preset not in LAB_CLASSIFICATION_OPTICS_PRESETS:
        return

    if args.layers != 1:
        raise ValueError("The current lab optics presets are restricted to single-layer (--layers 1) runs.")

    if args.size != 200:
        raise ValueError("The current lab optics presets are calibrated for size 200 and cannot be combined with --size overrides.")

    if any(
        value is not None
        for value in (
            args.wavelength,
            args.layer_distance,
            args.pixel_size,
            args.input_distance,
            args.output_distance,
        )
    ):
        raise ValueError(
            "When using a lab optics preset, do not override --wavelength/--layer-distance/--pixel-size/--input-distance/--output-distance manually."
        )


def resolve_loader_runtime_config(args, device):
    num_workers = int(args.num_workers)
    return {
        "device": str(device),
        "allow_tf32": bool(args.allow_tf32 and device.type == "cuda"),
        "deterministic_requested": bool(args.deterministic),
        "num_workers": num_workers,
        "pin_memory": bool(args.pin_memory and device.type == "cuda"),
        "prefetch_factor": int(args.prefetch_factor) if num_workers > 0 else None,
    }


def resolve_run_identity(
    *,
    args,
    save_dir,
    checkpoint_name,
    activation_type,
    activation_positions,
    activation_hparams,
    propagation_backend,
    propagation_chunk_size,
    loss_config=None,
    run_name_suffix=None,
):
    resolved_run_name = derive_experiment_run_name(
        run_name=args.run_name,
        experiment_stage=args.experiment_stage,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        seed=args.seed,
        loss_config=loss_config,
        propagation_backend=propagation_backend,
        propagation_chunk_size=propagation_chunk_size,
        optics_preset=args.optics_preset,
        layer_count=args.layers,
    )
    if not args.run_name and run_name_suffix:
        resolved_run_name = "__".join(part for part in (resolved_run_name, run_name_suffix) if part)
    checkpoint_path = checkpoint_variant_path(Path(save_dir) / checkpoint_name, resolved_run_name)
    return resolved_run_name, checkpoint_path


def build_common_manifest_fields(
    *,
    args,
    checkpoint_path,
    run_name,
    optics,
    activation_type,
    activation_positions,
    activation_hparams,
    propagation_backend,
    propagation_chunk_size,
    runtime_config,
    loss_config=None,
):
    return experiment_manifest_fields(
        checkpoint_path=checkpoint_path,
        run_name=run_name,
        experiment_stage=args.experiment_stage,
        seed=args.seed,
        optics=optics,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        model_version=MODEL_VERSION,
        loss_config=loss_config,
        propagation_backend=propagation_backend,
        propagation_chunk_size=propagation_chunk_size,
        runtime_config=runtime_config,
        optics_preset=args.optics_preset,
    )

# Backward-compatible alias kept for existing tests and legacy introspection.
_run_classification_epoch = run_classification_epoch


def run_classification_training(args, device, data_dir, save_dir):
    dataset_cfg = get_classification_dataset_config(args.dataset)
    optics = resolve_training_optics_preset("classification", args.optics_preset).with_overrides(
        size=args.size,
        num_layers=args.layers,
        wavelength=args.wavelength,
        layer_distance=args.layer_distance,
        pixel_size=args.pixel_size,
        input_distance=args.input_distance,
        output_distance=args.output_distance,
    )
    activation_type, activation_positions, activation_hparams = resolve_activation_config(args)
    propagation_backend, propagation_chunk_size = resolve_propagation_config(args=args)
    loss_config = {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma}
    runtime_config = resolve_loader_runtime_config(args, device)
    perturbation_config = resolve_training_perturbation_config(args)
    perturbation_seeds = resolve_perturbation_seeds(args)

    print(f"Dataset: {dataset_cfg['display_name']}")

    transform = build_classification_transform(dataset_cfg)
    source_train_set = dataset_cfg["dataset_cls"](data_dir, train=True, download=True, transform=transform)
    test_set = dataset_cfg["dataset_cls"](data_dir, train=False, download=True, transform=transform)
    input_shape = tuple(int(value) for value in source_train_set[0][0].shape)
    train_len, val_len = classification_split_lengths(len(source_train_set), val_size=args.val_size)
    train_set, val_set = torch.utils.data.random_split(
        source_train_set,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader_common = {
        "batch_size": args.batch_size,
        "num_workers": runtime_config["num_workers"],
        "pin_memory": runtime_config["pin_memory"],
    }
    if runtime_config["num_workers"] > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = runtime_config["prefetch_factor"]
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_common,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_common)
    test_loader = DataLoader(test_set, shuffle=False, **loader_common)

    model = build_classification_model(
        args=args,
        optics=optics,
        input_shape=input_shape,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        propagation_chunk_size=propagation_chunk_size,
        propagation_backend=propagation_backend,
        d2nn_builder=build_model_for_task,
    ).to(device)
    train_draw_factory, eval_draw_factory = build_perturbation_factories(
        model,
        perturbation_config,
        **perturbation_seeds,
    )
    run_name_suffix = classification_run_suffix(args, perturbation_config, perturbation_seeds)
    resolved_run_name, checkpoint_path = resolve_run_identity(
        args=args,
        save_dir=save_dir,
        checkpoint_name=dataset_cfg["checkpoint_name"],
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        propagation_backend=propagation_backend,
        propagation_chunk_size=propagation_chunk_size,
        loss_config=loss_config,
        run_name_suffix=run_name_suffix,
    )
    print_model_summary(type(model).__name__, model, task="classification")

    if resolved_run_name:
        print(f"Run name: {resolved_run_name}")

    best_checkpoint_metrics, best_state_dict, history, last_activation_stats = fit_classification_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        loss_config=loss_config,
        checkpoint_path=checkpoint_path,
        train_perturbation_draw_factory=train_draw_factory,
        eval_perturbation_draw_factory=eval_draw_factory,
    )
    if best_checkpoint_metrics["epoch"] != args.epochs:
        model.load_state_dict(best_state_dict if best_state_dict is not None else torch.load(checkpoint_path, weights_only=True))
    torch.save(model.state_dict(), checkpoint_path)
    test_metrics = _run_classification_epoch(
        model,
        test_loader,
        device,
        optimizer=None,
        perturbation_draw_factory=eval_draw_factory,
        **loss_config,
    )
    manifest_path = checkpoint_manifest_path(checkpoint_path)
    checkpoint_provenance = build_checkpoint_provenance(checkpoint_path)
    data_provenance = data_split_manifest(
        dataset_key=args.dataset,
        dataset_name=dataset_cfg["display_name"],
        input_shape=input_shape,
        source_train_set=source_train_set,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        seed=args.seed,
    )
    model_provenance = model_manifest(model, args, input_shape)
    perturbation_provenance = perturbation_manifest(
        perturbation_config,
        perturbation_seeds,
        train_draw_factory,
    )
    manifest_data = {
        "task": "classification",
        "dataset": dataset_cfg["display_name"],
        "paper_target_accuracy": dataset_cfg["paper_target"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "best_val_accuracy": best_checkpoint_metrics["accuracy"],
        "best_val_contrast": best_checkpoint_metrics["contrast"],
        "best_epoch": best_checkpoint_metrics["epoch"],
        "test_accuracy": test_metrics["accuracy"],
        "test_contrast": test_metrics["contrast"],
        "test_loss": test_metrics["loss"],
        "test_condition": perturbation_provenance["validation_test_condition"],
        "history": history,
        **build_common_manifest_fields(
            args=args,
            checkpoint_path=checkpoint_path,
            run_name=resolved_run_name,
            optics=optics,
            activation_type=activation_type,
            activation_positions=activation_positions,
            activation_hparams=activation_hparams,
            propagation_backend=propagation_backend,
            propagation_chunk_size=propagation_chunk_size,
            runtime_config=runtime_config,
            loss_config=loss_config,
        ),
        "activation_diagnostics": last_activation_stats,
        "cli_configuration": cli_configuration(args),
        "data_split": data_provenance,
        "model": model_provenance,
        "training_perturbations": perturbation_provenance,
        "environment": environment_manifest(device, Path(__file__).parent),
        "artifacts": {
            "checkpoint": checkpoint_provenance,
            "manifest": {"path": str(manifest_path)},
        },
        "checkpoint_sha256": checkpoint_provenance["sha256"],
    }
    save_manifest(manifest_path, manifest_data)
    paper_target = dataset_cfg["paper_target"]
    print(
        f"\nTest accuracy: {test_metrics['accuracy']:.2f}% | Test contrast: {test_metrics['contrast']:.4f} "
        f"(paper target: {f'{paper_target:.2f}%' if paper_target is not None else 'n/a'}, saved to {checkpoint_path.name})"
    )


def run_imaging_training(args, device, data_dir, save_dir):
    runtime_config = resolve_loader_runtime_config(args, device)
    dataset_cfg, train_loader, val_loader, test_loader = build_imaging_loaders(args, data_dir, runtime_config)
    print(f"Dataset: {dataset_cfg['display_name']}")

    (
        model,
        optics,
        activation_type,
        activation_positions,
        activation_hparams,
        propagation_backend,
        propagation_chunk_size,
    ) = build_imaging_training_model(args, device)
    resolved_run_name, checkpoint_path = resolve_run_identity(
        args=args,
        save_dir=save_dir,
        checkpoint_name=dataset_cfg["checkpoint_name"],
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        propagation_backend=propagation_backend,
        propagation_chunk_size=propagation_chunk_size,
    )
    print_model_summary("D2NNImager", model, task="imaging")
    if resolved_run_name:
        print(f"Run name: {resolved_run_name}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_val_loss, best_epoch, best_state_dict, last_activation_stats = fit_imaging_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
    )
    if best_epoch != args.epochs:
        model.load_state_dict(best_state_dict if best_state_dict is not None else torch.load(checkpoint_path, weights_only=True))
    test_loss = evaluate_imaging(model, test_loader, criterion, device)
    manifest_data = {
        "task": "imaging",
        "dataset": dataset_cfg["display_name"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "image_size": args.image_size,
        "input_fraction": args.input_fraction,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "test_mse": test_loss,
        **build_common_manifest_fields(
            args=args,
            checkpoint_path=checkpoint_path,
            run_name=resolved_run_name,
            optics=optics,
            activation_type=activation_type,
            activation_positions=activation_positions,
            activation_hparams=activation_hparams,
            propagation_backend=propagation_backend,
            propagation_chunk_size=propagation_chunk_size,
            runtime_config=runtime_config,
        ),
        "activation_diagnostics": last_activation_stats,
    }
    save_manifest(checkpoint_manifest_path(checkpoint_path), manifest_data)
    print(f"\nTest MSE: {test_loss:.4f} (saved to {checkpoint_path.name})")


def run_training_task(args, device, data_dir, save_dir):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(bool(args.deterministic))
    if device.type == "cuda":
        allow_tf32 = bool(args.allow_tf32)
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        torch.backends.cudnn.deterministic = bool(args.deterministic)
        torch.backends.cudnn.benchmark = not args.deterministic
    print(f"Seed: {args.seed}")
    task_runners = {
        "classification": run_classification_training,
        "imaging": run_imaging_training,
    }
    task_runners[args.task](args, device, data_dir, save_dir)


def build_parser():
    parser = argparse.ArgumentParser(description="D2NN training")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "imaging"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="classification: mnist/fashion-mnist/cifar10-gray/cifar10-rgb; imaging: stl10/imagefolder",
    )
    parser.add_argument("--image-root", type=str, default=None, help="root for imagefolder mode")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val-size", type=int, default=5000, help="classification validation split size")
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--size", type=int, default=200, help="Network pixel resolution (NxN)")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--input-fraction", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0, help="classification MSE loss weight")
    parser.add_argument("--beta", type=float, default=0.1, help="classification cross-entropy loss weight")
    parser.add_argument("--gamma", type=float, default=0.01, help="classification regularization loss weight")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--print-experiment-grid",
        type=str,
        default=None,
        choices=EXPERIMENT_GRID_CHOICES,
        help="print a predefined experiment command grid and exit",
    )
    parser.add_argument(
        "--run-experiment-grid",
        type=str,
        default=None,
        choices=EXPERIMENT_GRID_CHOICES,
        help="run a predefined experiment grid sequentially",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="optional experiment suffix used to keep checkpoints/manifests separate",
    )
    parser.add_argument(
        "--experiment-stage",
        type=str,
        default="baseline",
        help="high-level experiment stage label recorded in manifests",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for splits, loaders, and training")
    parser.add_argument(
        "--model-variant",
        type=str,
        default="d2nn",
        choices=MODEL_VARIANTS,
        help=(
            "classification model: optical D2NN, hybrid optical/electronic, "
            "pure electronic MLP, or pure electronic LeNet-5-style CNN"
        ),
    )
    parser.add_argument("--hybrid-pool-size", type=int, default=8)
    parser.add_argument("--hybrid-hidden-dim", type=int, default=32)
    parser.add_argument("--electronic-hidden-dim", type=int, default=18)
    parser.add_argument("--lenet5-hidden-dim", type=int, default=120)
    parser.add_argument(
        "--optical-perturbation-seed",
        type=int,
        default=None,
        help="optical lateral/gap/phase perturbation RNG seed; defaults to --seed",
    )
    parser.add_argument(
        "--detector-noise-seed",
        type=int,
        default=None,
        help="detector-noise RNG seed; defaults to a stable seed derived from the resolved optical seed",
    )
    parser.add_argument("--train-lateral-shift-max-px", type=float, default=0.0)
    parser.add_argument(
        "--train-gap-spacing-error-max-m",
        type=float,
        default=0.0,
        help="independent inter-layer gap-spacing error bound; not absolute layer-position displacement",
    )
    parser.add_argument("--train-phase-noise-std-rad", type=float, default=0.0)
    parser.add_argument(
        "--phase-quantization-levels",
        type=int,
        default=None,
        help="deployment phase levels used during training, validation, and test",
    )
    parser.add_argument(
        "--train-phase-quantization-levels",
        type=int,
        default=None,
        help=(
            "quantization-aware training: quantize phase during training only, while "
            "validation and test stay clean continuous. Models trained this way are "
            "evaluated on the frozen phase-quantization conditions."
        ),
    )
    parser.add_argument("--train-detector-noise-std-relative", type=float, default=0.0)
    parser.add_argument(
        "--train-phase-filter-std-px",
        type=float,
        default=0.0,
        help=(
            "phase-filtering regularization: low-pass filter the phase during training only, "
            "while validation and test stay clean continuous. Models trained this way are " 
            "evaluated on the frozen optical conditions."
        ),
    )
    parser.add_argument(
        "--hybrid-linear-head",
        action="store_true",
        help=(
            "hybrid ablation: replace the 64-32-10 electronic MLP with a single linear map " 
            "from pooled features to classes."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="request deterministic PyTorch algorithms; unsupported operations fail explicitly",
    )
    parser.add_argument(
        "--activation-type",
        type=str,
        default="none",
        choices=["none", "identity", "coherent_amplitude", "coherent_phase", "incoherent_intensity"],
        help="optional field activation inserted after selected diffractive layers",
    )
    parser.add_argument(
        "--activation-positions",
        type=str,
        default=None,
        help="comma-separated 1-based layer indices after which activations are inserted",
    )
    parser.add_argument(
        "--activation-placement",
        type=str,
        default=None,
        choices=["front", "mid", "back", "all"],
        help="named placement alias resolved from the current layer count",
    )
    parser.add_argument(
        "--activation-preset",
        type=str,
        default=None,
        choices=["conservative", "balanced", "aggressive"],
        help="optional preset for activation hyperparameters (applies to all activation types)",
    )
    parser.add_argument("--activation-threshold", type=float, default=None)
    parser.add_argument("--activation-temperature", type=float, default=None)
    parser.add_argument("--activation-gain-min", type=float, default=None)
    parser.add_argument("--activation-gain-max", type=float, default=None)
    parser.add_argument("--activation-gamma", type=float, default=None)
    parser.add_argument("--activation-responsivity", type=float, default=None)
    parser.add_argument("--activation-emission-phase-mode", type=str, default=None)
    parser.add_argument(
        "--rs-backend",
        type=str,
        default="direct",
        choices=["direct", "fft"],
        help="Rayleigh-Sommerfeld propagation backend used during forward propagation",
    )
    parser.add_argument(
        "--propagation-chunk-size",
        type=int,
        default=None,
        help="direct-backend target chunk size; ignored by the FFT backend",
    )
    parser.add_argument("--allow-tf32", action="store_true", help="enable TF32 matmul/cuDNN acceleration on CUDA")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--pin-memory", action="store_true", help="pin host memory for CUDA transfers")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when workers > 0")
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--layer-distance", type=float, default=None)
    parser.add_argument("--pixel-size", type=float, default=None)
    parser.add_argument("--input-distance", type=float, default=None)
    parser.add_argument("--output-distance", type=float, default=None)
    parser.add_argument(
        "--optics-preset",
        type=str,
        default="paper",
        choices=sorted(CLASSIFICATION_OPTICS_PRESETS),
        help=(
            "named classification optics preset; d2nn2018_thz is a pure-simulation scale, while lab presets "
            "remain fixed to classification, --layers 1, --size 200, and no manual optics overrides"
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.task == "imaging" and args.dataset == "mnist":
        args.dataset = "stl10"
    validate_training_args(args)
    if args.print_experiment_grid:
        for command in format_experiment_grid_commands(args.print_experiment_grid, args):
            print(command)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    repo_root = Path(__file__).parent
    data_dir = repo_root / "data"
    save_dir = repo_root / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.run_experiment_grid:
        execute_experiment_grid(
            args.run_experiment_grid,
            args,
            lambda run_args: run_training_task(run_args, device, data_dir, save_dir),
        )
        return

    run_training_task(args, device, data_dir, save_dir)


if __name__ == "__main__":
    main()
