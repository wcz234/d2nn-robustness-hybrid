"""Run clean-only numerical classification evaluation for all formal methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from clean_evaluation import (
    CLEAN_CONDITION_ID,
    METHOD_VARIANTS,
    build_clean_condition,
    measure_clean_inference,
    validate_method_target,
)
from clean_evaluation_manifest import build_clean_evaluation_manifest
from perturbations import PerturbationConfig
from robustness_evaluation import evaluate_clean_condition
from robustness_io import (
    JsonlWriter,
    artifact_record,
    ensure_output_available,
    finalize_staged_output,
    read_json,
    staged_output_directory,
    write_json,
)
from robustness_plan import SIMULATION_SCOPE, canonical_json_sha256, serialized_perturbation_config
from robustness_target import build_indexed_test_loader, load_robustness_target
from training_provenance import file_sha256


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean-only numerical classification evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-id", required=True, choices=tuple(METHOD_VARIANTS))
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--sample-limit", type=_positive_int, default=None)
    parser.add_argument("--cpu-threads", type=_positive_int, default=1)
    parser.add_argument(
        "--metrics-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help=(
            "device used to compute clean metrics and per-sample predictions. CPU timing "
            "is always measured on CPU because that quantity is the measurement itself."
        ),
    )
    parser.add_argument("--warmup-repeats", type=_non_negative_int, default=1)
    parser.add_argument("--measurement-repeats", type=_positive_int, default=5)
    parser.add_argument("--include-scores", action="store_true")
    return parser


def _validate_protocol_method(protocol: dict, *, args, target) -> None:
    methods = protocol.get("methods")
    if not isinstance(methods, list):
        raise ValueError("formal protocol requires a methods list")
    matches = [method for method in methods if method.get("method_id") == args.method_id]
    if len(matches) != 1 or matches[0].get("model_variant") != target.model_variant:
        raise ValueError("formal protocol method identity does not match the checkpoint")
    expected_config = matches[0].get("training_perturbations")
    actual_config = target.manifest["training_perturbations"]["config"]
    clean_config = serialized_perturbation_config(PerturbationConfig())
    required_config = clean_config if expected_config is None else expected_config
    if actual_config != required_config:
        raise ValueError("checkpoint training perturbations do not match the formal protocol")


def _validate_protocol_runtime(protocol: dict, *, args, target) -> dict:
    dataset = protocol.get("dataset")
    evaluation = protocol.get("evaluation")
    if not isinstance(dataset, dict) or not isinstance(evaluation, dict):
        raise ValueError("formal protocol requires dataset and evaluation objects")
    if dataset.get("key") != target.dataset_key or dataset.get("test_samples") != target.test_samples:
        raise ValueError("formal protocol dataset identity does not match the checkpoint")
    if evaluation.get("sample_scope") != "full_test" or args.sample_limit is not None:
        raise ValueError("protocol-bound clean evaluation requires the full test set")
    if evaluation.get("num_workers") != 0:
        raise ValueError("formal protocol clean evaluation requires num_workers=0")
    # `device` governs the metrics pass; CPU software timing is always CPU.
    protocol_device = evaluation.get("device")
    if protocol_device not in ("cpu", "cuda"):
        raise ValueError("formal protocol clean evaluation device must be 'cpu' or 'cuda'")
    if protocol_device != args.metrics_device:
        raise ValueError(
            f"protocol requires metrics device {protocol_device!r} but --metrics-device is "
            f"{args.metrics_device!r}"
        )
    if evaluation.get("batch_size") != args.batch_size:
        raise ValueError("--batch-size does not match the formal protocol")
    if evaluation.get("deterministic_draw_count") != 1:
        raise ValueError("formal protocol clean evaluation requires deterministic_draw_count=1")
    for field in ("master_optical_seed", "master_detector_seed"):
        value = evaluation.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"formal protocol evaluation.{field} must be a non-negative integer")
    return evaluation


def _clean_protocol_condition(protocol: dict) -> dict:
    clean_config = serialized_perturbation_config(PerturbationConfig())
    conditions = protocol.get("conditions")
    if not isinstance(conditions, list):
        raise ValueError("formal protocol requires a conditions list")
    matches = [condition for condition in conditions if condition.get("condition_id") == CLEAN_CONDITION_ID]
    if len(matches) != 1:
        raise ValueError(f"formal protocol requires exactly one {CLEAN_CONDITION_ID!r} condition")
    if matches[0].get("perturbation_config") != clean_config or matches[0].get("draw_count") != 1:
        raise ValueError("formal protocol clean_continuous must be zero-perturbation with one draw")
    return matches[0]


def _load_protocol_binding(args, target) -> tuple[dict | None, dict]:
    if args.protocol is None:
        return None, build_clean_condition()
    protocol_path = Path(args.protocol)
    protocol = read_json(protocol_path)
    if protocol.get("protocol_schema_version") != "formal-simulation-protocol/v1":
        raise ValueError("unsupported formal protocol schema version")
    if protocol.get("scope") != SIMULATION_SCOPE:
        raise ValueError("formal protocol must be numerical-simulation-only")
    _validate_protocol_method(protocol, args=args, target=target)
    evaluation = _validate_protocol_runtime(protocol, args=args, target=target)
    condition = _clean_protocol_condition(protocol)
    binding = {
        "path": str(protocol_path.resolve()),
        "sha256": file_sha256(protocol_path),
        "protocol_id": protocol.get("protocol_id"),
    }
    return binding, build_clean_condition(
        condition_id=condition["condition_id"],
        optical_seed=evaluation["master_optical_seed"],
        detector_seed=evaluation["master_detector_seed"],
    )


def _evaluation_id(*, args, target, selection: dict, protocol_binding: dict | None) -> str:
    identity = {
        "schema_version": "clean-evaluation/v1",
        "method_id": args.method_id,
        "training_seed": target.training_seed,
        "checkpoint_sha256": target.checkpoint_sha256,
        "sample_scope": selection,
        "formal_protocol_sha256": None if protocol_binding is None else protocol_binding["sha256"],
    }
    return f"clean-evaluation-{canonical_json_sha256(identity)[:20]}"


def _record_context(*, args, target) -> dict:
    return {
        "method_id": args.method_id,
        "model_id": target.checkpoint_path.stem,
        "model_variant": target.model_variant,
        "training_seed": target.training_seed,
        "checkpoint_sha256": target.checkpoint_sha256,
        "feature_schema_id": target.feature_schema_id,
        "score_semantics": target.score_semantics,
    }


def _write_clean_artifacts(
    *,
    stage,
    args,
    target,
    loader,
    condition,
    evaluation_id,
    metrics_device,
    timing_device,
):
    # Timing is deliberately measured on CPU regardless of the metrics device: the
    # recorded quantity is CPU software-simulation overhead, which is the measurement
    # itself and cannot be moved to an accelerator without changing what is measured.
    # The model is still on CPU here and is only moved afterwards.
    timing = measure_clean_inference(
        model=target.model,
        loader=loader,
        device=timing_device,
        warmup_repeats=args.warmup_repeats,
        measurement_repeats=args.measurement_repeats,
    )
    if metrics_device != timing_device:
        target.model.to(metrics_device)
    prediction_path = stage / "predictions.jsonl"
    with JsonlWriter(prediction_path) as predictions:
        metrics, _ = evaluate_clean_condition(
            model=target.model,
            loader=loader,
            device=metrics_device,
            condition=condition,
            evaluation_id=evaluation_id,
            include_scores=args.include_scores,
            prediction_sink=predictions.write,
            record_context=_record_context(args=args, target=target),
        )
        prediction_rows = predictions.row_count
    if prediction_rows != len(loader.dataset) or len(metrics) != 1:
        raise ValueError("clean-evaluation artifact row counts do not match the selected test scope")
    metric_path = stage / "clean_metrics.json"
    timing_path = stage / "cpu_software_timing.json"
    write_json(metric_path, metrics[0])
    write_json(timing_path, timing)
    artifacts = {
        "predictions": artifact_record(prediction_path, row_count=prediction_rows),
        "clean_metrics": artifact_record(metric_path, row_count=1),
        "cpu_software_timing": artifact_record(timing_path),
    }
    return artifacts, metrics[0], timing


def _resolve_metrics_device(args) -> torch.device:
    device = torch.device(args.metrics_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--metrics-device cuda was requested but CUDA is unavailable")
    return device


def run_clean_evaluation(args) -> Path:
    metrics_device = _resolve_metrics_device(args)
    timing_device = torch.device("cpu")
    # Load on the CPU so the checkpoint identity checks and the timing path stay
    # device-independent, then move the model for the metrics pass.
    target = load_robustness_target(args.checkpoint, device=torch.device("cpu"))
    validate_method_target(args.method_id, target)
    protocol_binding, condition = _load_protocol_binding(args, target)
    ensure_output_available(args.output_dir)
    loader, selection = build_indexed_test_loader(
        target,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        sample_limit=args.sample_limit,
    )
    evaluation_id = _evaluation_id(
        args=args,
        target=target,
        selection=selection,
        protocol_binding=protocol_binding,
    )
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(args.cpu_threads)
        if metrics_device.type == "cpu":
            torch.use_deterministic_algorithms(True)
        with staged_output_directory(args.output_dir) as stage:
            artifacts, metric, timing = _write_clean_artifacts(
                stage=stage,
                args=args,
                target=target,
                loader=loader,
                condition=condition,
                evaluation_id=evaluation_id,
                metrics_device=metrics_device,
                timing_device=timing_device,
            )
            manifest = build_clean_evaluation_manifest(
                args=args,
                target=target,
                selection=selection,
                artifacts=artifacts,
                metric=metric,
                timing=timing,
                evaluation_id=evaluation_id,
                protocol_binding=protocol_binding,
                metrics_device=str(metrics_device),
                timing_device=str(timing_device),
            )
            write_json(stage / "evaluation_manifest.json", manifest)
            finalize_staged_output(stage, args.output_dir)
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
        torch.set_num_threads(previous_threads)
    return Path(args.output_dir) / "evaluation_manifest.json"


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    output = run_clean_evaluation(args)
    print(f"Wrote {output}")
    print(f"Evaluation manifest SHA-256: {file_sha256(output)}")


if __name__ == "__main__":
    main()
