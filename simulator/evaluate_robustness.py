"""Create frozen plans and run classification robustness simulations."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from perturbations import PerturbationConfig, derive_detector_seed
from robustness_evaluation import evaluate_condition
from robustness_io import (
    JsonlWriter,
    artifact_record,
    ensure_output_available,
    finalize_staged_output,
    read_json,
    staged_output_directory,
    write_json,
    write_json_exclusive,
)
from robustness_plan import (
    build_robustness_plan,
    build_robustness_plan_from_protocol,
    canonical_json_sha256,
    validate_robustness_plan,
    validate_target_against_plan,
)
from robustness_run_manifest import build_evaluation_manifest
from robustness_target import build_indexed_test_loader, load_robustness_target
from training_provenance import file_sha256


def _non_empty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("value must be non-empty")
    return value.strip()


def _add_perturbation_arguments(parser) -> None:
    parser.add_argument("--lateral-shift-max-px", type=float, default=0.0)
    parser.add_argument("--gap-spacing-error-max-m", type=float, default=0.0)
    parser.add_argument("--phase-noise-std-rad", type=float, default=0.0)
    parser.add_argument("--phase-quantization-levels", type=int, default=None)
    parser.add_argument("--detector-noise-std-relative", type=float, default=0.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen, numerical-only D2NN robustness evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-plan", help="freeze one condition before evaluation")
    create.add_argument("--reference-checkpoint", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--condition-id", required=True, type=_non_empty)
    create.add_argument("--draw-count", type=int, default=1)
    create.add_argument("--optical-seed", type=int, default=42)
    create.add_argument("--detector-seed", type=int, default=None)
    create.add_argument("--sample-limit", type=int, default=None)
    _add_perturbation_arguments(create)

    create_protocol = commands.add_parser(
        "create-protocol-plan",
        help="freeze every optical condition in a formal simulation protocol",
    )
    create_protocol.add_argument("--reference-checkpoint", required=True)
    create_protocol.add_argument("--protocol", required=True)
    create_protocol.add_argument("--output", required=True)

    run = commands.add_parser("run", help="evaluate a checkpoint against an existing frozen plan")
    run.add_argument("--plan", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--method-id", required=True, type=_non_empty)
    run.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--num-workers", type=int, default=0)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--include-scores", action="store_true")
    return parser


def _config_from_args(args) -> PerturbationConfig:
    return PerturbationConfig(
        lateral_shift_max_px=args.lateral_shift_max_px,
        axial_shift_max_m=args.gap_spacing_error_max_m,
        phase_noise_std_rad=args.phase_noise_std_rad,
        phase_quantization_levels=args.phase_quantization_levels,
        detector_noise_std_relative=args.detector_noise_std_relative,
    )


def create_plan(args) -> Path:
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing plan: {output_path}")
    target = load_robustness_target(args.reference_checkpoint, device=torch.device("cpu"))
    detector_seed = derive_detector_seed(args.optical_seed) if args.detector_seed is None else args.detector_seed
    plan = build_robustness_plan(
        target=target,
        condition_id=args.condition_id,
        config=_config_from_args(args),
        draw_count=args.draw_count,
        optical_seed=args.optical_seed,
        detector_seed=detector_seed,
        sample_limit=args.sample_limit,
    )
    write_json_exclusive(output_path, plan)
    return output_path


def create_protocol_plan(args) -> Path:
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing plan: {output_path}")
    protocol_path = Path(args.protocol)
    target = load_robustness_target(args.reference_checkpoint, device=torch.device("cpu"))
    plan = build_robustness_plan_from_protocol(
        target=target,
        protocol=read_json(protocol_path),
        protocol_sha256=file_sha256(protocol_path),
    )
    write_json_exclusive(output_path, plan)
    return output_path


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device(name)


def _evaluation_id(*, method_id: str, training_seed: int, plan_id: str, checkpoint_sha256: str) -> str:
    identity = {
        "method_id": method_id,
        "training_seed": training_seed,
        "plan_id": plan_id,
        "checkpoint_sha256": checkpoint_sha256,
    }
    return f"robustness-evaluation-{canonical_json_sha256(identity)[:20]}"


def _load_frozen_test_data(args, target, plan):
    scope = plan["sample_scope"]
    sample_limit = scope["sample_count"] if scope["selection"] != "full_test" else None
    loader, selection = build_indexed_test_loader(
        target,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_limit=sample_limit,
    )
    if selection != scope:
        raise ValueError("loaded dataset selection does not match the frozen robustness plan")
    return loader, selection


def _record_context(args, target) -> dict:
    return {
        "method_id": args.method_id,
        "model_id": target.checkpoint_path.stem,
        "model_variant": target.model_variant,
        "training_seed": target.training_seed,
        "checkpoint_sha256": target.checkpoint_sha256,
        "feature_schema_id": target.feature_schema_id,
        "score_semantics": target.score_semantics,
    }


def _run_conditions(*, stage, plan, target, loader, device, args, evaluation_id):
    prediction_path = stage / "predictions.jsonl"
    metrics_path = stage / "draw_metrics.jsonl"
    condition_summaries = []
    with JsonlWriter(prediction_path) as predictions, JsonlWriter(metrics_path) as metrics:
        for condition in plan["conditions"]:
            draw_metrics, summary = evaluate_condition(
                model=target.model,
                loader=loader,
                device=device,
                condition=condition,
                evaluation_id=evaluation_id,
                include_scores=args.include_scores,
                prediction_sink=predictions.write,
                record_context=_record_context(args, target),
            )
            for row in draw_metrics:
                metrics.write(row)
            condition_summaries.append({"condition_id": condition["condition_id"], **summary})
        prediction_rows = predictions.row_count
        metric_rows = metrics.row_count
    expected_draws = sum(len(condition["draws"]) for condition in plan["conditions"])
    expected_predictions = expected_draws * len(loader.dataset)
    if prediction_rows != expected_predictions or metric_rows != expected_draws:
        raise ValueError("robustness artifact row counts do not match the frozen plan")
    artifacts = {
        "predictions": artifact_record(prediction_path, row_count=prediction_rows),
        "draw_metrics": artifact_record(metrics_path, row_count=metric_rows),
    }
    return artifacts, condition_summaries


def run_evaluation(args) -> Path:
    plan_path = Path(args.plan)
    plan = read_json(plan_path)
    validate_robustness_plan(plan)
    device = _resolve_device(args.device)
    target = load_robustness_target(args.checkpoint, device=device)
    validate_target_against_plan(target, plan)
    ensure_output_available(args.output_dir)
    loader, selection = _load_frozen_test_data(args, target, plan)
    evaluation_id = _evaluation_id(
        method_id=args.method_id,
        training_seed=target.training_seed,
        plan_id=plan["plan_id"],
        checkpoint_sha256=target.checkpoint_sha256,
    )
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with staged_output_directory(args.output_dir) as stage:
            artifacts, summaries = _run_conditions(
                stage=stage,
                plan=plan,
                target=target,
                loader=loader,
                device=device,
                args=args,
                evaluation_id=evaluation_id,
            )
            manifest = build_evaluation_manifest(
                args=args,
                plan_path=plan_path,
                plan=plan,
                target=target,
                device=device,
                selection=selection,
                artifacts=artifacts,
                summaries=summaries,
                evaluation_id=evaluation_id,
            )
            write_json(stage / "evaluation_manifest.json", manifest)
            finalize_staged_output(stage, args.output_dir)
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
    return Path(args.output_dir) / "evaluation_manifest.json"


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "create-plan":
        output = create_plan(args)
    elif args.command == "create-protocol-plan":
        output = create_protocol_plan(args)
    else:
        output = run_evaluation(args)
    print(f"Wrote {output}")
    if output.name == "evaluation_manifest.json":
        print(f"Evaluation manifest SHA-256: {file_sha256(output)}")


if __name__ == "__main__":
    main()
