"""Create FORMAL_EXPERIMENT_PROTOCOL_V2.json by extending the frozen v1 protocol.

v1 is frozen evidence and must not be edited. v2 adds the LeNet-5 pure-electronic
reference baseline requested by external review. Everything else (dataset, training
budget, evaluation, conditions, statistics) is copied unchanged so that v1-cohort
results remain directly comparable.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V1_PATH = ROOT / "FORMAL_EXPERIMENT_PROTOCOL.json"
V2_PATH = ROOT / "FORMAL_EXPERIMENT_PROTOCOL_V2.json"

LENET5_METHOD = {
    "method_id": "lenet5",
    "model_variant": "lenet5",
    "lenet5_hidden_dim": 120,
    "training_perturbations": None,
    "robustness_plan_applicability": "clean_classification_only; optical perturbations are not applicable",
    "rationale": (
        "External review objection: the 784-18-10 MLP is too weak an electronic reference "
        "on MNIST, so 'hybrid is comparable to the electronic baseline' would be a baseline-"
        "selection artifact. LeNet-5 adds a conventional CNN reference at a comparable "
        "parameter scale to the hybrid model."
    ),
}

HYBRID_QAT_METHOD = {
    "method_id": "hybrid_qat",
    "model_variant": "hybrid",
    "hybrid_pool_size": 8,
    "hybrid_hidden_dim": 32,
    "training_perturbations": None,
    "quantization_aware_training": {
        "train_phase_quantization_levels": 4,
        "applies_to": "training_only",
        "validation_and_test_condition": "clean_continuous",
    },
    "robustness_plan_applicability": "all_frozen_optical_conditions",
    "rationale": (
        "External review question: is the four-level phase-quantization failure a property of "
        "the hybrid architecture, or of training without quantization? This method trains the "
        "same hybrid architecture with four-level phase quantization applied during training "
        "while validation and test stay clean, then faces the frozen quantization conditions. "
        "Deployment perturbations are disabled, which is what the clean-evaluation contract "
        "checks; the QAT identity is recorded in training_perturbations."
        "quantization_aware_training and training_phase_quantization_levels."
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    if any(method.get("method_id") == "lenet5" for method in v1["methods"]):
        raise ValueError("v1 already contains a lenet5 method; refusing to fork")

    v2 = deepcopy(v1)
    v2["protocol_id"] = "formal-mnist-v2-20260917"
    v2["supersedes"] = {
        "protocol_id": v1["protocol_id"],
        "path": V1_PATH.name,
        "sha256": sha256(V1_PATH),
        "relationship": (
            "v2 extends v1 by appending one method. Dataset, training budget, evaluation "
            "settings, conditions, and statistics are byte-identical to v1."
        ),
    }
    v2["status"] = "frozen_before_v2_training"
    v2["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()

    # Device assignment for the v2 cohort. Measured before adoption (device_parity_probe.py):
    # CPU and CUDA agree bit-for-bit on clean/lateral/quantization/mixed conditions while
    # CUDA is ~4.1x faster. CPU software-simulation timing stays on CPU because that
    # overhead is the measured quantity itself.
    v2["evaluation"]["device"] = "cuda"
    v2["evaluation"]["timing_device"] = "cpu"
    v2["evaluation"]["device_parity_evidence"] = {
        "probe": "device_parity_probe.py",
        "conditions_probed": [
            "clean_continuous",
            "lateral_shift_0p50px",
            "phase_quantization_4",
            "mixed_stress",
        ],
        "checkpoints_probed": ["baseline_d2nn_seed42", "hybrid_seed42", "hybrid_qat_seed42"],
        "worst_accuracy_delta": 0.0,
        "speedup": {"baseline_d2nn": 3.22, "hybrid": 4.45, "hybrid_qat": 5.28, "overall": 4.11},
        "conclusion": "cuda_metrics_bit_identical_to_cpu_on_this_cohort",
    }
    v2["evaluation"]["timing_device_rationale"] = (
        "CPU software-simulation overhead is the measured quantity, not an implementation detail"
    )
    v2["methods"] = list(v1["methods"]) + [deepcopy(HYBRID_QAT_METHOD), deepcopy(LENET5_METHOD)]
    v2["v2_change_log"] = [
        "Add method_id 'hybrid_qat': quantization-aware hybrid (four-level phase quantization "
        "during training, clean validation/test).",
        "Add method_id 'lenet5': LeNet-5-style pure-electronic CNN reference baseline.",
        "No change to dataset, training budget, conditions, or statistics.",
        "v1 results stay valid for the v1 cohort; the new methods are evaluated under v2.",
    ]

    V2_PATH.write_text(json.dumps(v2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {V2_PATH.name}")
    print(f"  protocol_id      : {v2['protocol_id']}")
    print(f"  supersedes       : {v2['supersedes']['protocol_id']} (sha256 {v2['supersedes']['sha256'][:12]}...)")
    print(f"  methods          : {[m['method_id'] for m in v2['methods']]}")
    print(f"  conditions       : {len(v2['conditions'])}")
    print(f"  sha256           : {sha256(V2_PATH)}")


if __name__ == "__main__":
    main()
