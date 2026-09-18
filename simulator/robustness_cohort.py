"""Optical cohort and selected-test-scope contracts for robustness plans."""

from __future__ import annotations

from typing import Any


OPTICAL_CONFIG_FIELDS = (
    "wavelength",
    "layer_distance",
    "pixel_size",
    "input_distance",
    "output_distance",
    "size",
    "num_layers",
)


def cohort_from_target(target) -> dict[str, Any]:
    optical_config = target.manifest.get("optical_config") or {}
    return {
        "task": "classification",
        "model_version": target.manifest.get("model_version"),
        "dataset_key": target.dataset_key,
        "input_shape": list(target.input_shape),
        "full_test_samples": target.test_samples,
        "optical_config": {field: optical_config.get(field) for field in OPTICAL_CONFIG_FIELDS},
        "optical_draw_dtype": "float32",
        # Training-only regularizations (QAT, phase filtering) and the linear-head ablation
        # keep the base architecture, so the optical cohort stays d2nn + hybrid here.
        "compatible_model_variants": ["d2nn", "hybrid"],
    }


def build_sample_scope(*, full_test_samples: int, sample_limit: int | None) -> dict[str, Any]:
    if sample_limit is None:
        return {
            "selection": "full_test",
            "sample_count": full_test_samples,
            "full_test_samples": full_test_samples,
        }
    if isinstance(sample_limit, bool) or not isinstance(sample_limit, int) or not 1 <= sample_limit <= full_test_samples:
        raise ValueError("sample_limit must be between 1 and the full test-set size")
    return {
        "selection": "test_prefix_subset",
        "sample_count": sample_limit,
        "full_test_samples": full_test_samples,
    }


def validate_cohort(cohort: dict) -> None:
    if not isinstance(cohort, dict) or cohort.get("task") != "classification":
        raise ValueError("plan cohort must describe a classification task")
    if not isinstance(cohort.get("dataset_key"), str) or not cohort["dataset_key"]:
        raise ValueError("plan cohort requires dataset_key")
    input_shape = cohort.get("input_shape")
    if not isinstance(input_shape, list) or len(input_shape) != 3 or any(int(value) < 1 for value in input_shape):
        raise ValueError("plan cohort input_shape must contain three positive dimensions")
    if not isinstance(cohort.get("full_test_samples"), int) or cohort["full_test_samples"] < 1:
        raise ValueError("plan cohort requires a positive full_test_samples value")
    optical_config = cohort.get("optical_config")
    if not isinstance(optical_config, dict) or any(optical_config.get(field) is None for field in OPTICAL_CONFIG_FIELDS):
        raise ValueError("plan cohort optical_config is incomplete")


def validate_sample_scope(plan: dict) -> None:
    scope = plan.get("sample_scope")
    if not isinstance(scope, dict):
        raise ValueError("plan sample_scope must be an object")
    selection = scope.get("selection")
    sample_count = scope.get("sample_count")
    full_count = plan["cohort"]["full_test_samples"]
    if selection not in ("full_test", "test_prefix_subset"):
        raise ValueError("plan sample_scope selection is invalid")
    if not isinstance(sample_count, int) or not 1 <= sample_count <= full_count:
        raise ValueError("plan sample_scope sample_count is invalid")
    if scope.get("full_test_samples") != full_count:
        raise ValueError("plan sample_scope full_test_samples does not match the cohort")
    if selection == "full_test" and sample_count != full_count:
        raise ValueError("full_test plans must include the complete test set")
    if selection == "test_prefix_subset" and sample_count >= full_count:
        raise ValueError("test_prefix_subset plans must be smaller than the full test set")
