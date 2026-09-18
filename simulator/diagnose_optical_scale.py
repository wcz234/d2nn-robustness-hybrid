"""Generate a machine-readable numerical diagnostic for D2NN optical scales."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from artifacts import CLASSIFICATION_OPTICS_PRESETS, optical_config_dict
from d2nn import D2NN
from train_core import classification_composite_loss


DEFAULT_PRESETS = ("paper", "d2nn2018_thz")


def build_parser():
    parser = argparse.ArgumentParser(description="Numerical D2NN optical-scale gradient diagnostic")
    parser.add_argument("--presets", nargs="+", choices=sorted(CLASSIFICATION_OPTICS_PRESETS), default=DEFAULT_PRESETS)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rs-backend", choices=["direct", "fft"], default="fft")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _validate_args(args) -> None:
    for name in ("size", "layers", "batch_size", "num_threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def _synthetic_batch(*, batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = torch.rand((batch_size, 1, 28, 28), generator=generator)
    targets = torch.arange(batch_size, dtype=torch.long).remainder(10)
    return inputs, targets


def _fresnel_number(*, size: int, pixel_size: float, wavelength: float, distance: float) -> float:
    half_width = size * pixel_size / 2
    return half_width**2 / (wavelength * distance)


def _phase_gradient_metrics(model, loss, *, retain_graph: bool) -> dict[str, float | bool]:
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=retain_graph)
    gradients = torch.cat([layer.phase.grad.detach().abs().reshape(-1) for layer in model.layers])
    return {
        "abs_mean": float(gradients.mean()),
        "abs_max": float(gradients.max()),
        "finite": bool(torch.isfinite(gradients).all()),
    }


def _dimensionless_metrics(optics, *, size: int) -> dict[str, float]:
    common = {"size": size, "pixel_size": optics.pixel_size, "wavelength": optics.wavelength}
    return {
        "input_plane_fresnel_number": _fresnel_number(**common, distance=optics.input_distance),
        "inter_layer_fresnel_number": _fresnel_number(**common, distance=optics.layer_distance),
        "output_plane_fresnel_number": _fresnel_number(**common, distance=optics.output_distance),
    }


def _loss_and_gradient_metrics(model, result, targets, args) -> dict[str, float | bool]:
    loss_terms = classification_composite_loss(
        result,
        targets,
        model,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )
    data_loss = args.alpha * loss_terms["mse"] + args.beta * loss_terms["ce"]
    data_gradients = _phase_gradient_metrics(model, data_loss, retain_graph=True)
    composite_gradients = _phase_gradient_metrics(model, loss_terms["total"], retain_graph=False)
    if not all(bool(torch.isfinite(value)) for value in loss_terms.values()):
        raise ValueError("non-finite diagnostic loss")
    if not data_gradients["finite"] or not composite_gradients["finite"]:
        raise ValueError("non-finite phase gradients")
    return {
        "composite_loss": float(loss_terms["total"].detach()),
        "classification_mse_loss": float(loss_terms["mse"].detach()),
        "cross_entropy_loss": float(loss_terms["ce"].detach()),
        "phase_regularization_loss": float(loss_terms["reg"].detach()),
        "data_loss_phase_gradient_abs_mean": data_gradients["abs_mean"],
        "data_loss_phase_gradient_abs_max": data_gradients["abs_max"],
        "composite_phase_gradient_abs_mean": composite_gradients["abs_mean"],
        "composite_phase_gradient_abs_max": composite_gradients["abs_max"],
        "finite_phase_gradients": True,
    }


def diagnose_preset(*, preset_name: str, args, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
    optics = CLASSIFICATION_OPTICS_PRESETS[preset_name].with_overrides(size=args.size, num_layers=args.layers)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        model = D2NN(
            **optics.classifier_model_kwargs(),
            propagation_backend=args.rs_backend,
        ).train()
        result = model.forward_with_metrics(inputs, target=targets)
        loss_metrics = _loss_and_gradient_metrics(model, result, targets, args)

    intensity = result["intensity"].detach()
    detector_scores = result["scores"].detach()
    if not bool(torch.isfinite(intensity).all()):
        raise ValueError(f"non-finite diagnostic values for preset {preset_name!r}")

    return {
        "optical_config": optical_config_dict(optics),
        "dimensionless": _dimensionless_metrics(optics, size=args.size),
        "metrics": {
            **loss_metrics,
            "mean_output_intensity": float(intensity.mean()),
            "mean_detector_score": float(detector_scores.mean()),
            "finite_output": bool(torch.isfinite(intensity).all()),
        },
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    ratio = numerator / denominator
    return ratio if math.isfinite(ratio) else None


def _diagnose_presets(args, inputs, targets) -> dict[str, dict]:
    previous_num_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(args.num_threads)
        torch.use_deterministic_algorithms(True)
        return {
            preset_name: diagnose_preset(
                preset_name=preset_name,
                args=args,
                inputs=inputs,
                targets=targets,
            )
            for preset_name in args.presets
        }
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_num_threads)


def _comparison_metrics(presets) -> dict[str, float | None]:
    if "paper" not in presets or "d2nn2018_thz" not in presets:
        return {}
    paper_metrics = presets["paper"]["metrics"]
    thz_metrics = presets["d2nn2018_thz"]["metrics"]
    return {
        "d2nn2018_thz_to_paper_mean_output_intensity_ratio": _safe_ratio(
            thz_metrics["mean_output_intensity"], paper_metrics["mean_output_intensity"]
        ),
        "d2nn2018_thz_to_paper_data_loss_phase_gradient_abs_mean_ratio": _safe_ratio(
            thz_metrics["data_loss_phase_gradient_abs_mean"],
            paper_metrics["data_loss_phase_gradient_abs_mean"],
        ),
    }


def _diagnostic_configuration(args, targets) -> dict:
    return {
        "presets": list(args.presets),
        "size": args.size,
        "layers": args.layers,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "synthetic_input_seed": args.seed + 1,
        "synthetic_input": {
            "shape": [args.batch_size, 1, 28, 28],
            "distribution": "torch.rand uniform on [0, 1)",
            "targets": "cyclic class indices arange(batch_size) mod 10",
            "class_counts": torch.bincount(targets, minlength=10).tolist(),
        },
        "rs_backend": args.rs_backend,
        "num_threads": args.num_threads,
        "loss": {
            "name": "classification_composite_loss",
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "data_gradient_excludes_phase_regularization": True,
        },
    }


def build_diagnostic(args) -> dict:
    _validate_args(args)
    inputs, targets = _synthetic_batch(batch_size=args.batch_size, seed=args.seed + 1)
    presets = _diagnose_presets(args, inputs, targets)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "numerical_simulation_only",
        "claim_boundary": "diagnostic signal scale, not physical-device performance",
        "configuration": _diagnostic_configuration(args, targets),
        "interpretation_notes": [
            "Data-loss gradients use the current normalized-detector implementation, including its 1e-8 numerical floor.",
            "Mean output intensity is comparable only within runs sharing batch size and grid resolution.",
        ],
        "presets": presets,
        "comparisons": _comparison_metrics(presets),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = build_diagnostic(args)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Saved optical-scale diagnostic to: {args.output}")
    else:
        print(serialized)
    return payload


if __name__ == "__main__":
    main()
