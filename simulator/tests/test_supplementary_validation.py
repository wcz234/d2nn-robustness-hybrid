import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from artifacts import CLASSIFIER_D2NN2018_THZ_OPTICS
from d2nn import D2NN
from fixed_front_readout import extract_features, fit_readout, frozen_front, phase_identity
from perturbations import PerturbationConfig, shift_complex_modulation
from robustness_evaluation import _compose_batch_draw, _validate_clean_only_condition
from robustness_plan import build_draw_plans, validate_condition_plan
from robustness_records import optical_draw_record
from robustness_summary_input import load_evaluation
from supplementary_experiments import build_protocol, generate_jobs, run_job_file
from summarize_supplementary import deployment_summary, readout_summary
import train
from training_runtime import build_perturbation_factories, classification_run_suffix, perturbation_manifest
from validate_optical_numerics import gaussian_reference, refined_intensity, translated_piecewise_plate


def model(size=8):
    return D2NN(**CLASSIFIER_D2NN2018_THZ_OPTICS.with_overrides(size=size, num_layers=1).classifier_model_kwargs(), propagation_backend="fft").eval()


def test_deployment_validation_matches_qat_and_filter_without_stochastic_draws():
    optical = model()
    config = PerturbationConfig(train_phase_quantization_levels=4, train_phase_filter_std_px=1.0, lateral_shift_max_px=0.5, detector_noise_std_relative=0.1)
    seeds = {"optical_perturbation_seed": 42, "detector_noise_seed": 43}
    train_factory, eval_factory = build_perturbation_factories(optical, config, **seeds, validate_training_phase=True)
    draw = eval_factory(batch_size=2, dtype=torch.float32, device="cpu")
    assert draw.phase_quantization_levels == 4
    assert draw.phase_filter_std_px == 1.0
    assert draw.layers[0].lateral_shift_px == (0, 0)
    assert draw.detector_noise_relative is None
    manifest = perturbation_manifest(config, seeds, train_factory, validate_training_phase=True)
    assert manifest["nominal_phase_config"] == {"phase_quantization_levels": 4, "phase_filter_std_px": 1.0}
    assert manifest["validation_test_condition"] == "nominal_deployment_phase"
    _, legacy_eval = build_perturbation_factories(optical, config, **seeds)
    assert legacy_eval is None


def test_unnamed_qat_filter_and_linear_head_have_distinct_checkpoint_suffixes():
    parser = train.build_parser()
    suffixes = []
    for argv in ([], ["--train-phase-quantization-levels", "4"], ["--train-phase-filter-std-px", "1"], ["--model-variant", "hybrid"], ["--model-variant", "hybrid", "--hybrid-linear-head"], ["--train-phase-quantization-levels", "4", "--validate-training-phase"]):
        args = parser.parse_args(argv)
        suffixes.append(classification_run_suffix(args, PerturbationConfig(train_phase_quantization_levels=args.train_phase_quantization_levels, train_phase_filter_std_px=args.train_phase_filter_std_px), {"optical_perturbation_seed": 42, "detector_noise_seed": 43}))
    assert len(set(suffixes)) == len(suffixes)
    with pytest.raises(ValueError, match="requires"):
        train.validate_training_args(parser.parse_args(["--validate-training-phase"]))


def test_subpixel_interpolation_at_phase_boundary_is_not_unit_modulus():
    phase = torch.tensor([[0.0, torch.pi], [0.0, torch.pi]])
    legacy = shift_complex_modulation(torch.exp(1j * phase), (0.5, 0.0))
    fine = translated_piecewise_plate(phase, factor=4, shift_px=(0.5, 0.0))
    assert legacy.abs().min() < 1e-5
    assert torch.allclose(fine.abs(), torch.ones_like(fine.real), atol=1e-6)
    # Fixed aperture: refining doubles samples, not the number/size of phase cells.
    unshifted = translated_piecewise_plate(phase, factor=2, shift_px=(0.0, 0.0))
    assert torch.allclose(unshifted, torch.exp(1j * phase).repeat_interleave(2, 0).repeat_interleave(2, 1))


def test_refinement_factor_one_clean_matches_existing_optical_path():
    optical = model()
    inputs = torch.rand(2, 1, 28, 28)
    expected = optical.output_intensity(inputs)
    refined = refined_intensity(optical, inputs, factor=1, shifts=[(0.0, 0.0)])
    assert torch.allclose(refined, expected, atol=1e-7, rtol=1e-5)


def test_independent_asm_matches_resolved_analytic_gaussian_intensity():
    reference = gaussian_reference()
    assert reference["relative_intensity_l2"] < 2e-5
    assert reference["relative_collected_power_error"] < 2e-5


def test_readout_fit_does_not_update_any_frozen_optical_weight():
    source = model()
    front = frozen_front(source, "trained", seed=42)
    before = phase_identity(front)
    loader = DataLoader(TensorDataset(torch.rand(12, 1, 28, 28), torch.arange(12) % 10), batch_size=4)
    features, labels = extract_features(front, loader, kind="trained", pools=[2])
    head, history = fit_readout(features["pool2"], labels, features["pool2"], labels, kind="linear", seed=42, epochs=2, learning_rate=0.01, batch_size=4, hidden_dim=32, device="cpu")
    assert phase_identity(front) == before == phase_identity(source)
    assert all(not parameter.requires_grad for parameter in front.parameters())
    assert history["best_epoch"] in (1, 2)
    assert all(torch.isfinite(p).all() for p in head.parameters())


def test_filter_path_is_preserved_in_batch_draw_fingerprint_and_clean_guard():
    optical = model()
    _, factory = build_perturbation_factories(optical, PerturbationConfig(train_phase_filter_std_px=1.0), optical_perturbation_seed=42, detector_noise_seed=43, validate_training_phase=True)
    draw = factory(batch_size=2, dtype=torch.float32, device="cpu")
    assert _compose_batch_draw(draw, None).phase_filter_std_px == 1.0
    assert optical_draw_record(draw)["phase_filter_std_px"] == 1.0
    condition = {"condition_id": "filtered", "perturbation_config": {"lateral_shift_max_px": 0.0, "gap_spacing_error_max_m": 0.0, "phase_noise_std_rad": 0.0, "phase_quantization_levels": None, "detector_noise_std_relative": 0.0}, "phase_filter_std_px": 1.0, "draws": build_draw_plans(condition_id="filtered", draw_count=1, optical_seed=42, detector_seed=43)}
    validate_condition_plan(condition)
    with pytest.raises(ValueError, match="rejects"):
        _validate_clean_only_condition(condition)
    condition["phase_filter_std_px"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_condition_plan(condition)


def test_prospective_protocol_jobs_resolve_flags_and_dependencies(tmp_path):
    protocol = build_protocol(dataset="mnist", seeds=[42, 43], budgets=[3, 30], stochastic_draws=20)
    jobs = generate_jobs(protocol, tmp_path / "protocol.json", "python")
    known = set()
    for job in jobs:
        assert set(job["requires"]) <= known
        known.add(job["job_id"])
        if job["stage"] == "training":
            args = train.build_parser().parse_args(job["argv"][2:])
            train.validate_training_args(args)
            if "qat_deployed" in job["job_id"]:
                assert args.validate_training_phase
    conditions = {c["condition_id"]: c for c in protocol["conditions"]}
    assert conditions["phase_quantization_4"]["draw_count"] == 1
    assert conditions["qat4_mixed"]["draw_count"] == 20
    assert protocol["statistics"]["n_training_seeds"] == 2


def test_readout_summary_rejects_treating_heads_as_training_replicates():
    run = {"schema_version": "fixed-front-readout/v1", "source_training_seed": 42, "source_phase_sha256_before": "a", "source_phase_sha256_after": "a"}
    with pytest.raises(ValueError, match="unique"):
        readout_summary([run, copy.deepcopy(run)])


def test_job_runner_verifies_completion_and_rejects_changed_outputs(tmp_path, monkeypatch):
    import json
    import supplementary_experiments as module
    from robustness_plan import canonical_json_sha256
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "simulator").mkdir()
    (tmp_path / "simulator" / "source.py").write_text("# stable code\n")
    protocol = build_protocol(dataset="mnist", seeds=[42], budgets=[1], stochastic_draws=2)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    jobs = generate_jobs(protocol, protocol_path, "python")
    bundle = {"schema_version": "supplementary-command-jobs/v1", "protocol_sha256": canonical_json_sha256(protocol), "jobs": jobs}
    bundle_path = tmp_path / "jobs.json"
    bundle_path.write_text(json.dumps(bundle))
    calls = []
    def fake_run(argv, **kwargs):
        calls.append(argv)
        for path in jobs[0]["expected_outputs"]:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("test output")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = SimpleNamespace(protocol=protocol_path, jobs=bundle_path, job_id=jobs[0]["job_id"], state_dir=tmp_path / "state", with_dependencies=False, num_threads=2)
    run_job_file(args)
    run_job_file(args)
    assert len(calls) == 1
    args.job_id = "create_shared_plan"
    Path(jobs[0]["expected_outputs"][0]).write_text("changed")
    with pytest.raises(ValueError, match="missing or changed"):
        run_job_file(args)
    bundle["jobs"][0]["argv"].append("--injected")
    bundle_path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="commands differ"):
        run_job_file(args)


def test_deployment_tails_aggregate_within_seed_before_confidence_intervals(tmp_path):
    from test_robustness_summary import write_evaluation
    inputs = [load_evaluation(write_evaluation(tmp_path / f"seed{seed}", method="a", seed=seed, accuracies=values)) for seed, values in [(42, [0.8, 0.6]), (43, [0.9, 0.7])]]
    result = deployment_summary(inputs, threshold=0.75, quantile=0.05, references={"a": "mixed-noise"})
    summary = result["summaries"][0]["metrics"]
    assert summary["accuracy_mean"]["mean"] == pytest.approx(0.75)
    assert summary["accuracy_mean"]["n_training_seeds"] == 2
    assert summary["failure_probability"]["mean"] == pytest.approx(0.5)
    assert summary["accuracy_lower_quantile"]["mean"] == pytest.approx(0.66)
    assert summary["accuracy_drop_from_nominal_pp"]["mean"] == 0
    assert result["evidence_status"] == "descriptive_or_subset_only"


@pytest.mark.parametrize("version", [2, 3, 4])
def test_legacy_builders_use_docs_inputs_and_do_not_modify_frozen_protocols(tmp_path, version):
    import importlib
    builder = importlib.import_module(f"build_protocol_v{version}")
    root = Path(builder.ROOT)
    snapshots = {p: p.read_bytes() for p in (root / "docs").glob("FORMAL_EXPERIMENT_PROTOCOL*.json")}
    output = tmp_path / f"derived{version}.json"
    builder.main(["--output", str(output)])
    import json
    derived = json.loads(output.read_text())
    assert derived["status"] == "regenerated_design_not_original_preregistration"
    assert len(derived["training"]["paired_training_seeds"]) == derived["statistics"]["n_training_seeds"]
    assert all(p.read_bytes() == content for p, content in snapshots.items())
    with pytest.raises(FileExistsError):
        builder.main(["--output", str(output)])
