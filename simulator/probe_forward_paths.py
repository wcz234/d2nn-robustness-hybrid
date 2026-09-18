"""Disambiguate the forward paths of QAT and phase-filtered checkpoints.

Motivation (external assessment item #4): a QAT checkpoint records
`training_phase_quantization_levels = 4` while its validation/test condition is
`clean_continuous`. The headline "QAT clean accuracy = 61.74%" therefore measures the
*unquantized latent parameters* of a model that was optimized under a four-level
constraint -- a configuration that is never deployed. Three distinct paths must be
reported separately:

  P1  latent-continuous test   : forward with the learned continuous phase, no
                                 quantization, no perturbation. This is what the
                                 current protocol measures as "clean".
  P2  four-level nominal test  : forward with the reconstructed four-level quantized
                                 mask, no misalignment/noise. This is the deployed
                                 device with perfect calibration -- the fair figure of
                                 merit for a QAT model.
  P3  four-level disturbed test: the frozen four-level robustness condition, which adds
                                 the deployment draws. Taken from the existing
                                 robustness evaluation.

The script also reports learned-phase statistics, which bear on the "training smoothed
the phase" hypothesis.

Read-only with respect to every frozen artifact; it writes only its own JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

import torch

from perturbations import LayerPerturbation, PerturbationDraw, quantize_phase_uniform
from robustness_target import build_indexed_test_loader, load_robustness_target

T_975_DF2 = 4.302652729749454
SEEDS = (42, 43, 44, 45, 46)


def classifier_scores(model, loader, device):
    """Return per-sample argmax predictions for one full pass."""
    model.eval()
    predictions = []
    with torch.inference_mode():
        for data, _target, _index in loader:
            out = model.forward_with_metrics(data.to(device))
            predictions.append(out["scores"].argmax(dim=1).cpu())
    return torch.cat(predictions)


def accuracy(predictions, targets):
    return float((predictions == targets).float().mean())


def quantized_draw(model, levels, device):
    """A draw that applies uniform phase quantization to every layer, nothing else."""
    return PerturbationDraw(
        layers=tuple(LayerPerturbation() for _ in model.layers),
        phase_quantization_levels=levels,
    )


def phase_statistics(model, levels):
    """Learned-phase statistics relevant to the 'smoothing' hypothesis."""
    stats = {}
    for index, layer in enumerate(model.layers, start=1):
        phase = layer.phase.detach().cpu()
        quantized = quantize_phase_uniform(phase, levels, straight_through=False)
        residual = (phase - quantized).abs()
        # Total variation is the discrete proxy for spatial smoothness.
        tv = (phase.diff(dim=0).abs().mean() + phase.diff(dim=1).abs().mean()) / 2
        stats[f"layer_{index}"] = {
            "phase_std_rad": float(phase.std()),
            "quantization_residual_mean_rad": float(residual.mean()),
            "quantization_residual_max_rad": float(residual.max()),
            "total_variation_rad_per_pixel": float(tv),
        }
    return stats


def evaluate_checkpoint(checkpoint, device, levels, method_id, data_dir):
    target = load_robustness_target(checkpoint, device=torch.device("cpu"))
    model = target.model.to(device)
    loader, selection = build_indexed_test_loader(
        target, data_dir=data_dir, batch_size=128, num_workers=0, sample_limit=None
    )
    targets = torch.tensor([target_value for _, target_value, _ in loader.dataset])

    # P1: latent continuous, no perturbation.
    p1 = accuracy(classifier_scores(model, loader, device), targets)

    # P2: quantized mask, nominal (no misalignment, no noise).
    draw = quantized_draw(model, levels, device)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for data, _t, _i in loader:
            out = model.forward_with_metrics(data.to(device), perturbation_draw=draw)
            predictions.append(out["scores"].argmax(dim=1).cpu())
    p2 = accuracy(torch.cat(predictions), targets)

    return {
        "method_id": method_id,
        "checkpoint": str(checkpoint),
        "training_seed": target.training_seed,
        "sample_scope": selection,
        "latent_continuous_test": p1,
        "quantized_nominal_test": p2,
        "gap_nominal_minus_latent": p2 - p1,
        "phase_statistics": phase_statistics(model, levels),
    }


def summarize(rows, key):
    values = [row[key] * 100 for row in rows]
    mean = statistics.mean(values)
    if len(values) < 2:
        return {"mean": mean, "values": values, "ci95_t": None}
    sd = statistics.stdev(values)
    half = T_975_DF2 * sd / math.sqrt(len(values))
    return {"mean": mean, "sd": sd, "values": values, "ci95_t": [mean - half, mean + half]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True, help="directory holding the checkpoints")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    device = torch.device(args.device)
    data_dir = args.data_dir or str(pathlib.Path(__file__).resolve().parent / "data")
    artifacts = pathlib.Path(args.artifacts)

    # Ordered so the decisive comparison (QAT vs non-QAT hybrid) lands first; the plain
    # D2NN checkpoints are the most expensive per pass because they run the direct-space
    # backend over the full test set.
    groups = {
        "hybrid_qat": "hybrid_qat",
        "hybrid": "hybrid",
        "baseline_d2nn": "baseline_d2nn",
    }

    report = {"levels": args.levels, "device": str(device), "groups": {}}
    for label, tag in groups.items():
        rows = []
        for seed in SEEDS:
            checkpoint = artifacts / f"best_mnist.formal_mnist_v2_{tag}_seed{seed}.pth"
            if not checkpoint.is_file():
                print(f"  missing {checkpoint.name}", file=sys.stderr)
                continue
            row = evaluate_checkpoint(checkpoint, device, args.levels, label, data_dir)
            rows.append(row)
            print(
                f"  {label:14s} seed{row['training_seed']}  "
                f"P1 latent={row['latent_continuous_test'] * 100:6.2f}  "
                f"P2 {args.levels}-level nominal={row['quantized_nominal_test'] * 100:6.2f}"
            )
        report["groups"][label] = {
            "per_checkpoint": rows,
            "latent_continuous_test": summarize(rows, "latent_continuous_test"),
            "quantized_nominal_test": summarize(rows, "quantized_nominal_test"),
        }
        print()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
