"""Freeze a prospective v5 design and generate dependency-ordered command jobs.

No historical protocol is rewritten. Generating jobs never starts training.
The design uses fixed learning rate/budgets, not test-selected hyperparameters.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import os
import subprocess

from robustness_io import read_json, write_json_exclusive
from robustness_plan import canonical_json_sha256
from training_provenance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "baseline_d2nn": [],
    "robust_shift": ["--train-lateral-shift-max-px", "0.5"],
    "robust_gap": ["--train-gap-spacing-error-max-m", "0.0001"],
    "robust_phase": ["--train-phase-noise-std-rad", "0.05"],
    "robust_detector": ["--train-detector-noise-std-relative", "0.01"],
    "robust_joint": ["--train-lateral-shift-max-px", "0.5", "--train-gap-spacing-error-max-m", "0.0001", "--train-phase-noise-std-rad", "0.05", "--train-detector-noise-std-relative", "0.01"],
    "hybrid": ["--model-variant", "hybrid"],
    "hybrid_robust": ["--model-variant", "hybrid", "--train-lateral-shift-max-px", "0.5", "--train-gap-spacing-error-max-m", "0.0001", "--train-phase-noise-std-rad", "0.05", "--train-detector-noise-std-relative", "0.01"],
    "hybrid_qat_legacy": ["--model-variant", "hybrid", "--train-phase-quantization-levels", "4"],
    "hybrid_qat_deployed": ["--model-variant", "hybrid", "--train-phase-quantization-levels", "4", "--validate-training-phase"],
    "phase_filtered_legacy": ["--train-phase-filter-std-px", "1.0"],
    "phase_filtered_deployed": ["--train-phase-filter-std-px", "1.0", "--validate-training-phase"],
}


def condition(name, draws=20, **changes):
    config = {"lateral_shift_max_px": 0.0, "gap_spacing_error_max_m": 0.0, "phase_noise_std_rad": 0.0, "phase_quantization_levels": None, "detector_noise_std_relative": 0.0}
    filter_std = changes.pop("phase_filter_std_px", None)
    config.update(changes)
    stochastic = any(config[key] > 0 for key in ("lateral_shift_max_px", "gap_spacing_error_max_m", "phase_noise_std_rad", "detector_noise_std_relative"))
    result = {"condition_id": name, "draw_count": draws if stochastic else 1, "perturbation_config": config}
    if filter_std is not None:
        result["phase_filter_std_px"] = filter_std
    return result


def build_protocol(*, dataset, seeds, budgets, stochastic_draws=20):
    if dataset not in ("mnist", "fashion-mnist") or not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("dataset and distinct non-negative seeds are required")
    if not budgets or len(set(budgets)) != len(budgets) or min(budgets) < 1 or stochastic_draws < 2:
        raise ValueError("distinct positive budgets and at least two stochastic draws are required")
    conditions = [condition("clean_continuous")]
    for value in (0.125, 0.25, 0.5, 0.75, 1.0, 2.0):
        conditions.append(condition(f"shift_{value:g}px", stochastic_draws, lateral_shift_max_px=value))
    for value in (0.0001, 0.0003, 0.001):
        conditions.append(condition(f"gap_{value:g}m", stochastic_draws, gap_spacing_error_max_m=value))
    for value in (0.02, 0.05, 0.1, 0.2):
        conditions.append(condition(f"phase_noise_{value:g}rad", stochastic_draws, phase_noise_std_rad=value))
    for value in (0.01, 0.03, 0.1):
        conditions.append(condition(f"detector_noise_{value:g}", stochastic_draws, detector_noise_std_relative=value))
    for value in (32, 16, 8, 4, 2):
        conditions.append(condition(f"phase_quantization_{value}", phase_quantization_levels=value))
    mixed = {"lateral_shift_max_px": 0.5, "gap_spacing_error_max_m": 0.0001, "phase_noise_std_rad": 0.05, "detector_noise_std_relative": 0.01}
    conditions += [condition("mixed", stochastic_draws, **mixed), condition("qat4_mixed", stochastic_draws, phase_quantization_levels=4, **mixed), condition("filtered1_nominal", phase_filter_std_px=1.0), condition("filtered1_mixed", stochastic_draws, phase_filter_std_px=1.0, **mixed)]
    return {
        "protocol_schema_version": "formal-simulation-protocol/v1", "protocol_id": f"supplementary-{dataset}-v5",
        "scope": "numerical_simulation_only", "status": "prospective_frozen_design_not_completed_results",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"key": dataset, "test_samples": 10000, "validation_samples": 5000},
        "training": {"paired_training_seeds": seeds, "epoch_budgets": budgets, "batch_size": 128, "learning_rate": 0.01, "size": 64, "layers": 3, "optics_preset": "d2nn2018_thz", "propagation_backend": "fft", "validation_selection": "accuracy_then_contrast_then_later_epoch_on_explicit_phase_path", "scheduler": "StepLR(step_size=5,gamma=0.5)", "budget_interpretation": "equal epoch count, not equal wall-clock compute; convergence must be inspected, not assumed", "hyperparameter_selection": "none: fixed learning rate; any later validation-only tuning needs a new protocol"},
        "methods": [{"method_id": name, "training_cli": values} for name, values in METHODS.items()],
        "evaluation": {"sample_scope": "full_test", "master_optical_seed": 72001, "master_detector_seed": 72002, "failure_accuracy_threshold": 0.9 if dataset == "mnist" else 0.75, "lower_quantile": 0.05, "deployment_tail_interpretation": "20 draws give descriptive tails; use a separately frozen 100+ draw confirmatory plan for precision"},
        "conditions": conditions,
        "statistics": {"independent_unit": "training_seed", "n_training_seeds": len(seeds), "draw_handling": "aggregate distributions within seed, then summarize across seeds", "intervals": "marginal Student-t; no equivalence or simultaneous significance claims", "nominal_reference_by_method": {"hybrid_qat_deployed": "phase_quantization_4", "phase_filtered_deployed": "filtered1_nominal"}},
        "fixed_front_controls": {"source_method": "baseline_d2nn", "source_budget": max(budgets), "pools": [4, 8], "heads": ["linear", "mlp"], "fronts": ["trained", "random", "free", "input"], "readout_epochs": 30, "readout_learning_rate": 0.003, "independent_unit": "source_training_seed"},
        "numerical_gate": {"refinement_factors": [1, 2, 4, 8], "keep_physical_geometry_fixed": True, "required_interpretation": "independent propagation and successive refinements must be assessed before physical robustness claims"},
        "excluded_claims": ["hardware latency", "power/energy advantage", "calibrated detector noise", "general validity outside tested datasets and geometry"],
    }


def validate_protocol(protocol):
    if protocol.get("status") != "prospective_frozen_design_not_completed_results":
        raise ValueError("a prospective supplementary protocol is required")
    training = protocol["training"]
    if protocol["statistics"]["n_training_seeds"] != len(training["paired_training_seeds"]):
        raise ValueError("protocol seed count mismatch")
    if len({m["method_id"] for m in protocol["methods"]}) != len(protocol["methods"]):
        raise ValueError("method IDs must be unique")
    if "baseline_d2nn" not in {m["method_id"] for m in protocol["methods"]}:
        raise ValueError("baseline checkpoint is needed for the shared evaluation plan")
    for method in protocol["methods"]:
        if method["method_id"] not in METHODS or method["training_cli"] != METHODS[method["method_id"]]:
            raise ValueError("unsupported or rewritten method training arguments")
    from train import build_parser, validate_training_args
    for method in protocol["methods"]:
        args = build_parser().parse_args(["--task", "classification", "--dataset", protocol["dataset"]["key"], "--size", str(training["size"]), "--layers", str(training["layers"]), "--rs-backend", training["propagation_backend"], "--optics-preset", training["optics_preset"], *method["training_cli"]])
        validate_training_args(args)


def generate_jobs(protocol, protocol_path, python):
    validate_protocol(protocol)
    dataset = protocol["dataset"]["key"]
    dataset_id = dataset.replace("-", "_")
    training = protocol["training"]
    root = ROOT
    artifacts = root / "artifacts" / f"supplementary_{dataset_id}_v5"
    results = root / "results" / f"supplementary_{dataset_id}_v5"
    protocol_path = str(Path(protocol_path).resolve())
    jobs, checkpoints, evaluations = [], {}, []

    def job(name, stage, argv, requires=(), outputs=()):
        jobs.append({"job_id": name, "stage": stage, "cwd": str(root), "argv": [str(x) for x in argv], "requires": list(requires), "expected_outputs": [str(x) for x in outputs]})

    for budget in training["epoch_budgets"]:
        for seed in training["paired_training_seeds"]:
            for method in protocol["methods"]:
                name = f"v5_{dataset_id}_{method['method_id']}_e{budget}_seed{seed}"
                prefix = "best_mnist" if dataset == "mnist" else "best_fashion_mnist"
                checkpoint = artifacts / f"{prefix}.{name}.pth"
                checkpoints[(method["method_id"], budget, seed)] = (name, checkpoint)
                argv = [python, "simulator/train.py", "--task", "classification", "--dataset", dataset, "--epochs", budget, "--seed", seed, "--batch-size", training["batch_size"], "--lr", training["learning_rate"], "--size", training["size"], "--layers", training["layers"], "--optics-preset", training["optics_preset"], "--rs-backend", training["propagation_backend"], "--run-name", name, "--experiment-stage", f"supplementary_{dataset_id}_v5", "--save-dir", artifacts, *method["training_cli"]]
                job(name, "training", argv, outputs=[checkpoint, checkpoint.with_suffix(".json")])
    reference_job, reference_checkpoint = checkpoints[("baseline_d2nn", training["epoch_budgets"][0], training["paired_training_seeds"][0])]
    plan = results / "robustness_plan.json"
    job("create_shared_plan", "plan", [python, "simulator/evaluate_robustness.py", "create-protocol-plan", "--reference-checkpoint", reference_checkpoint, "--protocol", protocol_path, "--output", plan], [reference_job], [plan])
    for (method, budget, seed), (name, checkpoint) in checkpoints.items():
        output = results / "evaluations" / name
        job(f"eval_{name}", "evaluation", [python, "simulator/evaluate_robustness.py", "run", "--checkpoint", checkpoint, "--plan", plan, "--method-id", f"{method}_e{budget}", "--output-dir", output, "--device", "cpu"], [name, "create_shared_plan"], [output / "evaluation_manifest.json"])
        evaluations.append(output)
    eval_dependencies = [j["job_id"] for j in jobs if j["stage"] == "evaluation"]
    job("summarize_robustness", "summary", [python, "simulator/summarize_robustness.py", "--evaluations", *evaluations, "--output-dir", results / "robustness_summary"], eval_dependencies, [results / "robustness_summary" / "summary_manifest.json"])
    for seed in training["paired_training_seeds"]:
        source_job, checkpoint = checkpoints[("baseline_d2nn", max(training["epoch_budgets"]), seed)]
        job(f"readouts_seed{seed}", "readouts", [python, "simulator/fixed_front_readout.py", "--checkpoint", checkpoint, "--output-dir", artifacts / f"readouts_seed{seed}", "--epochs", 30, "--lr", 0.003], [source_job], [artifacts / f"readouts_seed{seed}" / "readout_results.json"])
        job(f"numerics_seed{seed}", "numerics", [python, "simulator/validate_optical_numerics.py", "--checkpoint", checkpoint, "--sample-limit", 4, "--output", results / f"numerics_seed{seed}.json"], [source_job], [results / f"numerics_seed{seed}.json"])
    job("summarize_readouts", "summary", [python, "simulator/summarize_supplementary.py", "readouts", "--results", *[artifacts / f"readouts_seed{seed}" / "readout_results.json" for seed in training["paired_training_seeds"]], "--output", results / "readout_summary.json"], [f"readouts_seed{seed}" for seed in training["paired_training_seeds"]], [results / "readout_summary.json"])
    return jobs


def run_job_file(args):
    protocol, bundle = read_json(args.protocol), read_json(args.jobs)
    if bundle.get("protocol_sha256") != canonical_json_sha256(protocol):
        raise ValueError("job bundle does not match the frozen protocol")
    if bundle.get("schema_version") != "supplementary-command-jobs/v1":
        raise ValueError("unsupported job bundle")
    # Reject rewritten commands; only the interpreter path may differ between environments.
    expected = generate_jobs(protocol, args.protocol, bundle["jobs"][0]["argv"][0])
    if bundle["jobs"] != expected:
        raise ValueError("job commands differ from the frozen design; regenerate in this checkout")
    indexed = {j["job_id"]: j for j in expected}
    if args.job_id not in indexed:
        raise ValueError(f"unknown job ID {args.job_id}")
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    sources = {p.name: file_sha256(p) for p in sorted((ROOT / "simulator").glob("*.py"))}
    code_hash = canonical_json_sha256(sources)
    visited = set()

    def completed(job):
        path = state_dir / f"{job['job_id']}.json"
        if not path.exists():
            return False
        state = read_json(path)
        if state["job_sha256"] != canonical_json_sha256(job) or state["code_sha256"] != code_hash:
            raise ValueError("saved job state belongs to different code/configuration; use a new experiment output tree")
        if not all(Path(item["path"]).is_file() and file_sha256(Path(item["path"])) == item["sha256"] for item in state["outputs"]):
            raise ValueError("saved job outputs are missing or changed; inspect the experiment before resuming")
        return True

    def execute(name):
        if name in visited:
            return
        job = indexed[name]
        if completed(job):
            visited.add(name)
            return
        for dependency in job["requires"]:
            if args.with_dependencies:
                execute(dependency)
            elif not completed(indexed[dependency]):
                raise ValueError(f"dependency {dependency} is incomplete; run it first or use --with-dependencies")
        if any(Path(path).exists() for path in job["expected_outputs"]):
            raise FileExistsError(f"untracked output already exists for {name}; refusing to overwrite")
        argv = [sys.executable, *job["argv"][1:]]
        environment = dict(os.environ, OMP_NUM_THREADS=str(args.num_threads), MPLCONFIGDIR=str(state_dir.resolve() / "matplotlib"))
        print(f"Running {name}: {shlex.join(argv)}", flush=True)
        subprocess.run(argv, cwd=job["cwd"], env=environment, check=True)
        if not all(Path(path).is_file() for path in job["expected_outputs"]):
            raise RuntimeError(f"job {name} did not create its expected outputs")
        state = {"job_sha256": canonical_json_sha256(job), "code_sha256": code_hash, "executed_argv": argv, "outputs": [{"path": path, "sha256": file_sha256(Path(path))} for path in job["expected_outputs"]]}
        write_json_exclusive(state_dir / f"{name}.json", state)
        visited.add(name)
    execute(args.job_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--dataset", choices=["mnist", "fashion-mnist"], default="mnist")
    freeze.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    freeze.add_argument("--budgets", type=int, nargs="+", default=[3, 30])
    freeze.add_argument("--stochastic-draws", type=int, default=20)
    freeze.add_argument("--output", required=True)
    jobs_parser = commands.add_parser("jobs")
    jobs_parser.add_argument("--protocol", required=True)
    jobs_parser.add_argument("--python", default=sys.executable)
    jobs_parser.add_argument("--output", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--protocol", required=True)
    runner.add_argument("--jobs", required=True)
    runner.add_argument("--job-id", required=True)
    runner.add_argument("--state-dir", required=True)
    runner.add_argument("--with-dependencies", action="store_true")
    runner.add_argument("--num-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        protocol = build_protocol(dataset=args.dataset, seeds=args.seeds, budgets=args.budgets, stochastic_draws=args.stochastic_draws)
        validate_protocol(protocol)
        write_json_exclusive(args.output, protocol)
        print(f"Frozen design {args.output}; {len(protocol['methods']) * len(args.seeds) * len(args.budgets)} training runs; {sum(c['draw_count'] for c in protocol['conditions'])} deployment draws/model")
    elif args.command == "jobs":
        protocol = read_json(args.protocol)
        jobs = generate_jobs(protocol, args.protocol, args.python)
        write_json_exclusive(args.output, {"schema_version": "supplementary-command-jobs/v1", "protocol_sha256": canonical_json_sha256(protocol), "execution": "not_started; commands must be run in dependency order", "jobs": jobs})
        print(f"Wrote {len(jobs)} jobs to {args.output}. No training was started.")
        for job in jobs[:3]:
            print(shlex.join(job["argv"]))
    else:
        if args.num_threads < 1:
            parser.error("num-threads must be positive")
        run_job_file(args)


if __name__ == "__main__":
    main()
