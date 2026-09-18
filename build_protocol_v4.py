"""Create FORMAL_EXPERIMENT_PROTOCOL_V4.json: MNIST robustness controls and ablations.

v1, v2 and v3 stay frozen and untouched. v4 adds three methods to the MNIST design:

  phase_filtered_d2nn  - phase-filtering regularization (external review item #5), the
                         training-only low-pass filter used by misalignment-resilient
                         diffractive networks.
  hybrid_linear_head   - ablation A (item #6): identical optical front end and identical
                         64-dimensional pooled features, but a single linear map instead of
                         the 64-32-10 MLP, which isolates nonlinear electronic processing.
  hybrid_pool4         - ablation C (item #6): 4x4 pooled features (16 dimensions) with the
                         MLP head, which isolates representation width.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V2_PATH = ROOT / "FORMAL_EXPERIMENT_PROTOCOL_V2.json"
V4_PATH = ROOT / "FORMAL_EXPERIMENT_PROTOCOL_V4.json"

V4_SEEDS = [42, 43, 44]

PHASE_FILTERED_METHOD = {
    "method_id": "phase_filtered_d2nn",
    "model_variant": "d2nn",
    "training_perturbations": None,
    "phase_filter_training": {
        "train_phase_filter_std_px": 1.0,
        "kernel": "separable_gaussian_replicate_padding",
        "applies_to": "training_only",
        "validation_and_test_condition": "clean_continuous",
    },
    "robustness_plan_applicability": "all_frozen_optical_conditions",
    "rationale": (
        "External review item #5: the related-work section discusses phase filtering but the "
        "experiment never runs it. This adds the cheapest same-protocol control. Phase "
        "filtering is a training-only regularization here, matching how QAT is handled: the "
        "deployed mask stays continuous, so clean metrics remain comparable to the other "
        "methods and the effect appears only through the learned phase structure."
    ),
}

LINEAR_HEAD_METHOD = {
    "method_id": "hybrid_linear_head",
    "model_variant": "hybrid",
    "hybrid_pool_size": 8,
    "hybrid_linear_head": True,
    "training_perturbations": None,
    "robustness_plan_applicability": "all_frozen_optical_conditions",
    "rationale": (
        "External review item #6 variant A: the hybrid model changes both the pooled "
        "representation width and the electronic head. Holding the 8x8 pooled representation "
        "fixed while replacing the 64-32-10 MLP with a single linear map isolates the "
        "contribution of nonlinear electronic processing."
    ),
}

POOL4_METHOD = {
    "method_id": "hybrid_pool4",
    "model_variant": "hybrid",
    "hybrid_pool_size": 4,
    "hybrid_hidden_dim": 32,
    "training_perturbations": None,
    "robustness_plan_applicability": "all_frozen_optical_conditions",
    "rationale": (
        "External review item #6 variant C: 4x4 pooled features (16 dimensions) with the same "
        "MLP head width, which isolates representation width at fixed head architecture."
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    v2 = json.loads(V2_PATH.read_text(encoding="utf-8"))

    v4 = deepcopy(v2)
    v4["protocol_id"] = "formal-mnist-v4-20260917"
    v4["supersedes"] = {
        "protocol_id": v2["protocol_id"],
        "path": V2_PATH.name,
        "sha256": sha256(V2_PATH),
        "relationship": (
            "v4 reuses the v2 dataset, training budget, evaluation settings, 14 conditions "
            "and statistics, and appends three review-response methods."
        ),
    }
    v4["status"] = "frozen_before_v4_training"
    v4["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    v4["training"] = deepcopy(v2["training"])
    v4["training"]["paired_training_seeds"] = list(V4_SEEDS)
    v4["training"]["experiment_stage"] = "formal_mnist_v4"
    v4["training"]["output_dir"] = "artifacts/formal_mnist_v4"
    v4["methods"] = list(v2["methods"]) + [
        deepcopy(PHASE_FILTERED_METHOD),
        deepcopy(LINEAR_HEAD_METHOD),
        deepcopy(POOL4_METHOD),
    ]
    v4["statistics"] = deepcopy(v2["statistics"])
    v4["statistics"]["n_training_seeds"] = len(V4_SEEDS)
    v4["v4_change_log"] = [
        "Add 'phase_filtered_d2nn': training-only Gaussian phase filtering (item #5).",
        "Add 'hybrid_linear_head': pooled-feature ablation with a linear readout (item #6 A).",
        "Add 'hybrid_pool4': representation-width ablation at 4x4 pooling (item #6 C).",
        "MNIST v1/v2 and Fashion-MNIST v3 results are unaffected and stay frozen.",
    ]

    V4_PATH.write_text(json.dumps(v4, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {V4_PATH.name}")
    print(f"  protocol_id : {v4['protocol_id']}")
    print(f"  methods     : {[m['method_id'] for m in v4['methods']]}")
    print(f"  seeds       : {v4['training']['paired_training_seeds']}")
    print(f"  conditions  : {len(v4['conditions'])}  draws: {sum(c['draw_count'] for c in v4['conditions'])}")
    print(f"  sha256      : {sha256(V4_PATH)}")


if __name__ == "__main__":
    main()
