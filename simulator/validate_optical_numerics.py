"""Independent propagation and fixed-geometry subpixel-shift diagnostics.

Refinement changes numerical samples, never physical aperture or phase-cell size.
The reference plate is piecewise constant and unit modulus at each fine sample;
it is not claimed to describe manufacturing edge profiles or vector diffraction.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from artifacts import CLASSIFIER_D2NN2018_THZ_OPTICS
from d2nn import D2NN, RayleighSommerfeldPropagation
from perturbations import LayerPerturbation, PerturbationDraw, shift_complex_modulation
from robustness_io import write_json_exclusive
from robustness_target import build_indexed_test_loader, load_robustness_target
from training_provenance import environment_manifest, file_sha256


def angular_spectrum(field, *, wavelength, distance, pixel_size, padding=3):
    """Band-limited propagating-wave ASM, independently of the RS kernel.

    Rectangular sampling cutoffs supplement the propagating-frequency disk.
    See Matsushima & Shimobaba, Optics Express 17, 19662 (2009).
    """
    if field.ndim != 3 or padding < 1 or distance <= 0 or wavelength <= 0 or pixel_size <= 0:
        raise ValueError("ASM requires a batched field and positive physical parameters")
    height, width = field.shape[-2:]
    padded_h, padded_w = height * padding, width * padding
    top, left = (padded_h - height) // 2, (padded_w - width) // 2
    padded = F.pad(field, (left, padded_w - width - left, top, padded_h - height - top))
    fy = torch.fft.fftfreq(padded_h, d=pixel_size, device=field.device, dtype=field.real.dtype)
    fx = torch.fft.fftfreq(padded_w, d=pixel_size, device=field.device, dtype=field.real.dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radial = 1 - wavelength**2 * (xx.square() + yy.square())
    cutoff_x = 1 / (wavelength * math.sqrt(1 + (2 * distance / (padded_w * pixel_size))**2))
    cutoff_y = 1 / (wavelength * math.sqrt(1 + (2 * distance / (padded_h * pixel_size))**2))
    keep = (radial >= 0) & (xx.abs() <= cutoff_x) & (yy.abs() <= cutoff_y)
    transfer = torch.exp(1j * (2 * math.pi * distance / wavelength) * radial.clamp_min(0).sqrt()) * keep
    result = torch.fft.ifft2(torch.fft.fft2(padded) * transfer)
    return result[:, top:top + height, left:left + width]


def translated_piecewise_plate(phase, *, factor, shift_px):
    """Evaluate a translated coarse phase cell at each fine-grid sample center."""
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("refinement factor must be a positive integer")
    if phase.ndim != 2 or not all(math.isfinite(v) for v in shift_px):
        raise ValueError("plate requires a 2D phase and finite shift")
    height, width = phase.shape
    source_x = torch.floor((torch.arange(width * factor, device=phase.device) + 0.5) / factor - shift_px[0]).long()
    source_y = torch.floor((torch.arange(height * factor, device=phase.device) + 0.5) / factor - shift_px[1]).long()
    valid = (source_y[:, None] >= 0) & (source_y[:, None] < height) & (source_x[None, :] >= 0) & (source_x[None, :] < width)
    values = phase[source_y.clamp(0, height - 1)[:, None], source_x.clamp(0, width - 1)[None, :]]
    return torch.where(valid, torch.exp(1j * values), torch.ones_like(values, dtype=torch.complex64))


@torch.no_grad()
def refined_intensity(model, inputs, *, factor, shifts):
    """Propagate the same physical input/plates on a finer numerical grid."""
    if model.activation_type != "none":
        raise ValueError("refinement diagnostic currently supports phase-only fronts")
    if len(shifts) != len(model.layers):
        raise ValueError("one shift per phase plate is required")
    coarse = model._embed_input(inputs)
    field = coarse.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
    size = model.size * factor
    pixel_size = model.input_propagation.pixel_size / factor

    def propagate(u, module):
        return RayleighSommerfeldPropagation(
            size, module.wavelength, module.distance, pixel_size, backend="fft"
        ).to(u.device)(u)

    field = propagate(field, model.input_propagation)
    for layer, shift in zip(model.layers, shifts):
        field = field * translated_piecewise_plate(layer.phase, factor=factor, shift_px=shift)
        if layer.propagation is not None:
            field = propagate(field, layer.propagation)
    field = propagate(field, model.output_propagation)
    fine_intensity = field.abs().square()
    # Integrate over the original detector cells; original pixel area is common.
    return F.avg_pool2d(fine_intensity.unsqueeze(1), factor, factor).squeeze(1)


def gaussian_reference():
    """Compare ASM against analytic paraxial Gaussian intensity on a resolved grid."""
    size, pixel_size, wavelength, distance, waist = 128, 5e-6, 532e-9, 0.01, 60e-6
    axis = (torch.arange(size, dtype=torch.float64) - (size - 1) / 2) * pixel_size
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    radius2 = xx.square() + yy.square()
    field = torch.exp(-radius2 / waist**2).to(torch.complex128).unsqueeze(0)
    output = angular_spectrum(field, wavelength=wavelength, distance=distance, pixel_size=pixel_size)
    rayleigh_range = math.pi * waist**2 / wavelength
    waist_z = waist * math.sqrt(1 + (distance / rayleigh_range)**2)
    expected = (waist / waist_z)**2 * torch.exp(-2 * radius2 / waist_z**2)
    actual = output[0].abs().square()
    rs = RayleighSommerfeldPropagation(size, wavelength, distance, pixel_size, backend="fft")(field)[0].abs().square()
    return {
        "relative_intensity_l2": float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)),
        "relative_collected_power_error": float((actual.sum() / expected.sum() - 1).abs()),
        "rs_relative_intensity_l2": float(torch.linalg.vector_norm(rs - expected) / torch.linalg.vector_norm(expected)),
        "reference_scope": "paraxial Gaussian intensity, not arbitrary-plate validation",
        "parameters": {"size": size, "pixel_size_m": pixel_size, "wavelength_m": wavelength, "distance_m": distance, "waist_m": waist},
    }


@torch.no_grad()
def run(args):
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    target = None
    if args.checkpoint:
        target = load_robustness_target(args.checkpoint, device=torch.device("cpu"))
        if target.model_variant not in ("d2nn", "hybrid"):
            raise ValueError("a phase-only optical checkpoint is required")
        model = target.model
        loader, selection = build_indexed_test_loader(target, data_dir=args.data_dir, batch_size=args.sample_limit, num_workers=0, sample_limit=args.sample_limit)
        inputs, labels, _ = next(iter(loader))
    else:
        optics = CLASSIFIER_D2NN2018_THZ_OPTICS.with_overrides(size=args.size, num_layers=args.layers)
        model = D2NN(**optics.classifier_model_kwargs(), propagation_backend="fft").eval()
        # Deliberately exercise abrupt phase boundaries; this is a numerical probe.
        for layer in model.layers:
            layer.phase.uniform_(0, 2 * math.pi)
        inputs, labels = torch.rand(args.sample_limit, 1, 28, 28), None
        selection = {"selection": "synthetic_uniform_inputs", "sample_count": len(inputs)}
    if sorted(set(args.factors)) != args.factors or min(args.factors) < 1 or len(args.factors) < 2:
        raise ValueError("at least two strictly increasing positive refinement factors are required")
    records = []
    for shift in args.shifts:
        shifts = [(shift, 0.0)] * len(model.layers)
        draw = PerturbationDraw(tuple(LayerPerturbation(lateral_shift_px=s) for s in shifts))
        legacy = model.output_intensity(inputs, draw)
        reference = refined_intensity(model, inputs, factor=args.factors[-1], shifts=shifts)
        for implementation, factor in [("legacy_bilinear_complex", 1)] + [("piecewise_unit_modulus", f) for f in args.factors]:
            intensity = legacy if implementation.startswith("legacy") else refined_intensity(model, inputs, factor=factor, shifts=shifts)
            record = {
                "shift_px": shift, "shift_m": shift * model.input_propagation.pixel_size,
                "implementation": implementation, "refinement_factor": factor,
                "relative_intensity_l2_to_finest": float(torch.linalg.vector_norm(intensity - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-30)),
                "mean_collected_power": float(intensity.sum(dim=(-2, -1)).mean() * model.input_propagation.pixel_size**2),
                "reference_is_converged": "not_assumed; inspect successive refinements",
            }
            if labels is not None:
                if target.model_variant == "hybrid":
                    features = model.detector_features_from_intensity(intensity)
                    logits = model.electronic_head(features / features.mean(1, keepdim=True).clamp_min(1e-8))
                else:
                    logits = model.detect_from_intensity(intensity)
                record["accuracy"] = float((logits.argmax(1) == labels).float().mean())
            if implementation.startswith("legacy"):
                transmission = shift_complex_modulation(torch.exp(1j * model.layers[0].phase), shifts[0])
                record["first_plate_mean_modulus"] = float(transmission.abs().mean())
                record["first_plate_min_modulus"] = float(transmission.abs().min())
            records.append(record)
    # This comparison diagnoses discretization; agreement alone is not physical proof.
    field = model._embed_input(inputs)
    prop = model.input_propagation
    rs = prop(field)
    asm = angular_spectrum(field, wavelength=prop.wavelength, distance=prop.distance, pixel_size=prop.pixel_size, padding=args.asm_padding)
    propagation_refinements = []
    for factor in args.factors:
        fine_field = field.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
        fine_rs = RayleighSommerfeldPropagation(model.size * factor, prop.wavelength, prop.distance, prop.pixel_size / factor, backend="fft")(fine_field)
        fine_asm = angular_spectrum(fine_field, wavelength=prop.wavelength, distance=prop.distance, pixel_size=prop.pixel_size / factor, padding=args.asm_padding)
        rs_integrated = F.avg_pool2d(fine_rs.abs().square().unsqueeze(1), factor).squeeze(1)
        asm_integrated = F.avg_pool2d(fine_asm.abs().square().unsqueeze(1), factor).squeeze(1)
        propagation_refinements.append({"refinement_factor": factor, "relative_intensity_l2_rs_vs_asm": float(torch.linalg.vector_norm(rs_integrated - asm_integrated) / torch.linalg.vector_norm(asm_integrated).clamp_min(1e-30))})
    payload = {
        "schema_version": "optical-numerics/v1", "scope": "numerical_simulation_only",
        "evidence_status": "numerical_diagnostic_not_a_training_cohort",
        "checkpoint_sha256": None if target is None else target.checkpoint_sha256,
        "phase_path": "latent_continuous_only",
        "sample_scope": selection, "gaussian_reference": gaussian_reference(),
        "rs_vs_asm_relative_intensity_l2": float(torch.linalg.vector_norm(rs.abs().square() - asm.abs().square()) / torch.linalg.vector_norm(asm.abs().square()).clamp_min(1e-30)),
        "shift_records": records,
        "propagation_refinement_records": propagation_refinements,
        "claim_boundary": "fixed geometry; piecewise scalar phase plates; no hardware or convergence claim",
        "source_sha256": file_sha256(Path(__file__)),
        "environment": environment_manifest(torch.device("cpu"), Path(__file__).parent),
        "configuration": vars(args),
    }
    write_json_exclusive(args.output, payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--asm-padding", type=int, default=3,
                        help="ASM computational-window padding; sweep separately from sampling refinement")
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--shifts", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args(argv)
    if args.sample_limit < 1 or args.num_threads < 1 or args.asm_padding < 1:
        parser.error("sample-limit, num-threads and asm-padding must be positive")
    payload = run(args)
    print(f"Wrote {args.output}; Gaussian relative L2={payload['gaussian_reference']['relative_intensity_l2']:.6g}")


if __name__ == "__main__":
    main()
