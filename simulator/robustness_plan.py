"""Frozen robustness-evaluation plan construction and validation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from perturbations import PerturbationConfig
from robustness_cohort import build_sample_scope, cohort_from_target, validate_cohort, validate_sample_scope


PLAN_SCHEMA_VERSION = "robustness-plan/v1"
SIMULATION_SCOPE = "numerical_simulation_only"
SEED_MODULUS = 2**63
PERTURBATION_FIELDS = frozenset(
    {
        "lateral_shift_max_px",
        "gap_spacing_error_max_m",
        "phase_noise_std_rad",
        "phase_quantization_levels",
        "detector_noise_std_relative",
    }
)
FROZEN_SAMPLING_PROTOCOL = {
    "optical_draw_scope": "one_fixed_draw_per_draw_id_across_selected_test_scope",
    "detector_noise_scope": "per_sample_feature_table",
    "detector_noise_semantics": "synthetic_relative_additive_simulation_noise",
    "feature_schema_pairing": "optical_draws_are_paired; detector_tables_are_feature_schema_specific",
}
FROZEN_STATISTICS_PROTOCOL = {
    "independent_unit": "training_seed",
    "draw_role": "deployment_perturbation_variability_not_independent_training_replicates",
}


def canonical_json_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_seed(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < SEED_MODULUS:
        raise ValueError(f"{name} must be an integer in [0, {SEED_MODULUS})")
    return value


def stable_protocol_seed(master_seed: int, *, condition_id: str, draw_id: int, stream: str) -> int:
    _validate_seed("master_seed", master_seed)
    payload = f"{master_seed}|{condition_id}|{int(draw_id)}|{stream}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % SEED_MODULUS


def build_draw_plans(*, condition_id: str, draw_count: int, optical_seed: int, detector_seed: int) -> list[dict]:
    if isinstance(draw_count, bool) or not isinstance(draw_count, int) or draw_count < 1:
        raise ValueError("draw_count must be a positive integer")
    _validate_seed("optical_seed", optical_seed)
    _validate_seed("detector_seed", detector_seed)
    return [
        {
            "draw_id": draw_id,
            "optical_seed": stable_protocol_seed(
                optical_seed,
                condition_id=condition_id,
                draw_id=draw_id,
                stream="optical",
            ),
            "detector_seed": stable_protocol_seed(
                detector_seed,
                condition_id=condition_id,
                draw_id=draw_id,
                stream="detector",
            ),
        }
        for draw_id in range(draw_count)
    ]


def serialized_perturbation_config(config: PerturbationConfig) -> dict[str, Any]:
    return {
        "lateral_shift_max_px": config.lateral_shift_max_px,
        "gap_spacing_error_max_m": config.axial_shift_max_m,
        "phase_noise_std_rad": config.phase_noise_std_rad,
        "phase_quantization_levels": config.phase_quantization_levels,
        "detector_noise_std_relative": config.detector_noise_std_relative,
    }


def plan_perturbation_config(condition: dict) -> PerturbationConfig:
    config = condition.get("perturbation_config")
    if not isinstance(config, dict):
        raise ValueError("condition perturbation_config must be an object")
    missing = sorted(PERTURBATION_FIELDS - config.keys())
    unknown = sorted(config.keys() - PERTURBATION_FIELDS)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        raise ValueError(f"condition perturbation_config fields are invalid ({'; '.join(details)})")
    return PerturbationConfig(
        lateral_shift_max_px=config["lateral_shift_max_px"],
        axial_shift_max_m=config["gap_spacing_error_max_m"],
        phase_noise_std_rad=config["phase_noise_std_rad"],
        phase_quantization_levels=config["phase_quantization_levels"],
        detector_noise_std_relative=config["detector_noise_std_relative"],
    )


def validate_condition_plan(condition: dict) -> None:
    if not isinstance(condition, dict) or not isinstance(condition.get("condition_id"), str):
        raise ValueError("condition_id is required")
    if not condition["condition_id"].strip():
        raise ValueError("condition_id must be non-empty")
    config = plan_perturbation_config(condition)
    draws = condition.get("draws")
    if not isinstance(draws, list) or not draws:
        raise ValueError("condition draws must be a non-empty list")
    if any(not isinstance(draw, dict) for draw in draws):
        raise ValueError("every condition draw must be an object")
    draw_ids = [draw.get("draw_id") for draw in draws]
    if draw_ids != list(range(len(draws))):
        raise ValueError("draw IDs must be contiguous integers starting at zero")
    for draw in draws:
        _validate_seed("optical_seed", draw.get("optical_seed"))
        _validate_seed("detector_seed", draw.get("detector_seed"))
    stochastic = any(
        value > 0
        for value in (
            config.lateral_shift_max_px,
            config.axial_shift_max_m,
            config.phase_noise_std_rad,
            config.detector_noise_std_relative,
        )
    )
    if not stochastic and len(draws) != 1:
        raise ValueError("deterministic clean/quantization-only conditions require exactly one draw")


def _plan_identity_payload(plan: dict) -> dict[str, Any]:
    payload = {
        key: plan[key]
        for key in (
            "schema_version",
            "scope",
            "reference_checkpoint",
            "cohort",
            "sample_scope",
            "sampling_protocol",
            "statistics",
            "conditions",
        )
    }
    if "formal_protocol" in plan:
        payload["formal_protocol"] = plan["formal_protocol"]
    return payload


def plan_identity_sha256(plan: dict) -> str:
    return canonical_json_sha256(_plan_identity_payload(plan))


def _build_condition(*, condition_id, config, draw_count, optical_seed, detector_seed) -> dict:
    return {
        "condition_id": condition_id,
        "perturbation_config": serialized_perturbation_config(config),
        "seed_protocol": {
            "name": "sha256_derived_independent_streams/v1",
            "master_optical_seed": optical_seed,
            "master_detector_seed": detector_seed,
        },
        "draws": build_draw_plans(
            condition_id=condition_id,
            draw_count=draw_count,
            optical_seed=optical_seed,
            detector_seed=detector_seed,
        ),
    }


def _assemble_plan(*, target, conditions, sample_limit, formal_protocol=None) -> dict[str, Any]:
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "scope": SIMULATION_SCOPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_checkpoint": {
            "sha256": target.checkpoint_sha256,
            "model_variant": target.model_variant,
            "training_seed": target.training_seed,
        },
        "cohort": cohort_from_target(target),
        "sample_scope": build_sample_scope(full_test_samples=target.test_samples, sample_limit=sample_limit),
        "sampling_protocol": dict(FROZEN_SAMPLING_PROTOCOL),
        "statistics": dict(FROZEN_STATISTICS_PROTOCOL),
        "conditions": conditions,
    }
    if formal_protocol is not None:
        plan["formal_protocol"] = formal_protocol
    plan["plan_id"] = f"robustness-plan-{plan_identity_sha256(plan)[:20]}"
    validate_robustness_plan(plan)
    return plan


def build_robustness_plan(
    *,
    target,
    condition_id: str,
    config: PerturbationConfig,
    draw_count: int,
    optical_seed: int,
    detector_seed: int,
    sample_limit: int | None,
) -> dict[str, Any]:
    condition = _build_condition(
        condition_id=condition_id,
        config=config,
        draw_count=draw_count,
        optical_seed=optical_seed,
        detector_seed=detector_seed,
    )
    return _assemble_plan(target=target, conditions=[condition], sample_limit=sample_limit)


def build_robustness_plan_from_protocol(*, target, protocol: dict, protocol_sha256: str) -> dict[str, Any]:
    if protocol.get("protocol_schema_version") != "formal-simulation-protocol/v1":
        raise ValueError("formal protocol schema_version is unsupported")
    if protocol.get("scope") != SIMULATION_SCOPE:
        raise ValueError("formal protocol must be numerical-simulation-only")
    if not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64:
        raise ValueError("formal protocol requires a SHA-256 identity")
    dataset = protocol.get("dataset")
    evaluation = protocol.get("evaluation")
    if not isinstance(dataset, dict) or not isinstance(evaluation, dict):
        raise ValueError("formal protocol requires dataset and evaluation objects")
    if dataset.get("key") != target.dataset_key or dataset.get("test_samples") != target.test_samples:
        raise ValueError("reference checkpoint does not match the formal protocol dataset")
    if evaluation.get("sample_scope") != "full_test":
        raise ValueError("formal robustness protocol must evaluate the full test set")
    optical_seed = _validate_seed("master_optical_seed", evaluation.get("master_optical_seed"))
    detector_seed = _validate_seed("master_detector_seed", evaluation.get("master_detector_seed"))
    specifications = protocol.get("conditions")
    if not isinstance(specifications, list) or not specifications:
        raise ValueError("formal protocol requires a non-empty conditions list")
    conditions = []
    for specification in specifications:
        if not isinstance(specification, dict):
            raise ValueError("every formal protocol condition must be an object")
        raw_config = specification.get("perturbation_config")
        if not isinstance(raw_config, dict) or set(raw_config) != PERTURBATION_FIELDS:
            raise ValueError("formal protocol condition perturbation fields are invalid")
        config = PerturbationConfig(
            lateral_shift_max_px=raw_config["lateral_shift_max_px"],
            axial_shift_max_m=raw_config["gap_spacing_error_max_m"],
            phase_noise_std_rad=raw_config["phase_noise_std_rad"],
            phase_quantization_levels=raw_config["phase_quantization_levels"],
            detector_noise_std_relative=raw_config["detector_noise_std_relative"],
        )
        conditions.append(
            _build_condition(
                condition_id=specification.get("condition_id"),
                config=config,
                draw_count=specification.get("draw_count"),
                optical_seed=optical_seed,
                detector_seed=detector_seed,
            )
        )
    formal_protocol = {
        "protocol_schema_version": protocol["protocol_schema_version"],
        "protocol_id": protocol.get("protocol_id"),
        "sha256": protocol_sha256.lower(),
    }
    return _assemble_plan(
        target=target,
        conditions=conditions,
        sample_limit=None,
        formal_protocol=formal_protocol,
    )


def validate_robustness_plan(plan: dict) -> None:
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(f"robustness plan requires schema_version {PLAN_SCHEMA_VERSION!r}")
    if plan.get("scope") != SIMULATION_SCOPE:
        raise ValueError("robustness plan must be numerical-simulation-only")
    if not isinstance(plan.get("created_at_utc"), str) or not plan["created_at_utc"]:
        raise ValueError("robustness plan requires created_at_utc")
    if plan.get("sampling_protocol") != FROZEN_SAMPLING_PROTOCOL:
        raise ValueError("robustness plan sampling_protocol does not match the frozen protocol")
    if plan.get("statistics") != FROZEN_STATISTICS_PROTOCOL:
        raise ValueError("robustness plan statistics does not match the frozen protocol")
    validate_cohort(plan.get("cohort"))
    validate_sample_scope(plan)
    conditions = plan.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("robustness plan requires at least one condition")
    for condition in conditions:
        validate_condition_plan(condition)
    condition_ids = [condition["condition_id"] for condition in conditions]
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("condition IDs must be unique within a robustness plan")
    expected_id = f"robustness-plan-{plan_identity_sha256(plan)[:20]}"
    if plan.get("plan_id") != expected_id:
        raise ValueError("robustness plan_id does not match the frozen plan contents")
    _validate_gap_bounds(plan)


def _validate_gap_bounds(plan: dict) -> None:
    optical_config = plan["cohort"]["optical_config"]
    layer_distance = float(optical_config["layer_distance"])
    num_layers = int(optical_config["num_layers"])
    for condition in plan["conditions"]:
        gap_bound = plan_perturbation_config(condition).axial_shift_max_m
        if num_layers < 2 and gap_bound > 0:
            raise ValueError("gap-spacing error is not applicable to a single-layer optical model")
        if gap_bound >= layer_distance:
            raise ValueError("gap-spacing error bound must be smaller than the nominal layer distance")


def validate_target_against_plan(target, plan: dict) -> None:
    validate_robustness_plan(plan)
    expected = plan["cohort"]
    actual = cohort_from_target(target)
    for key in ("task", "model_version", "dataset_key", "input_shape", "full_test_samples", "optical_config"):
        if actual[key] != expected[key]:
            raise ValueError(f"checkpoint does not match robustness-plan cohort field {key!r}")
    if target.model_variant not in expected.get("compatible_model_variants", []):
        raise ValueError("checkpoint model variant is not compatible with this robustness plan")
