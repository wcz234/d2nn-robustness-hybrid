"""Create FORMAL_EXPERIMENT_PROTOCOL_V3.json: the Fashion-MNIST frozen comparison.

Protocol v1 (MNIST, n=3) and v2 (MNIST, n=5 + QAT + LeNet-5) stay frozen and untouched.
v3 keeps the same methods, training budget, evaluation runtime setting, 14 conditions and
statistics, and changes only the dataset identity plus the robustness-plan cohort binding.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V2_PATH = ROOT / "FORMAL_EXPERIMENT_PROTOCOL_V2.json"
V3_PATH = ROOT / "FORMAL_EXPERIMENT_PROTOCOL_V3.json"

# Optical-compatible methods only: the Fashion-MNIST comparison targets the claim about
# condition-selective robustness and hybrid readout, which are optical claims. Pure
# electronic references on MNIST are already reported under v2.
V3_METHODS = ("baseline_d2nn", "robust_d2nn", "hybrid")
V3_SEEDS = [42, 43, 44]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    v2 = json.loads(V2_PATH.read_text(encoding="utf-8"))

    v3 = deepcopy(v2)
    v3["protocol_id"] = "formal-fashion-mnist-v3-20260917"
    v3["supersedes"] = {
        "protocol_id": v2["protocol_id"],
        "path": V2_PATH.name,
        "sha256": sha256(V2_PATH),
        "relationship": (
            "v3 reuses the v2 training budget, evaluation runtime settings, 14 conditions "
            "and statistics, and changes the dataset identity to Fashion-MNIST."
        ),
    }
    v3["status"] = "frozen_before_v3_training"
    v3["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()

    v3["dataset"] = deepcopy(v2["dataset"])
    # The robustness-plan cohort validator compares this against the dataset_key stored in
    # the training manifest, which is the dataset CLI token verbatim ("fashion-mnist"),
    # not the normalized identifier. MNIST hides this because both forms coincide.
    v3["dataset"]["key"] = "fashion-mnist"
    v3["dataset"]["normalized_key"] = "fashion_mnist"
    v3["dataset"]["display_name"] = "Fashion-MNIST"
    v3["dataset"]["note"] = (
        "Same provider, split policy and sample counts as the MNIST cohorts; only the "
        "image content differs, which makes this a second-dataset replication of the "
        "frozen protocol rather than a new experimental design."
    )

    v3["training"] = deepcopy(v2["training"])
    v3["training"]["paired_training_seeds"] = list(V3_SEEDS)
    v3["training"]["experiment_stage"] = "formal_fashion_mnist_v3"
    v3["training"]["output_dir"] = "artifacts/formal_fashion_mnist_v3"

    keep = set(V3_METHODS)
    v3["methods"] = [deepcopy(method) for method in v2["methods"] if method["method_id"] in keep]
    found = {method["method_id"] for method in v3["methods"]}
    if found != keep:
        raise ValueError(f"v3 method selection mismatch: {sorted(found)} vs {sorted(keep)}")

    v3["statistics"] = deepcopy(v2["statistics"])
    v3["statistics"]["n_training_seeds"] = len(V3_SEEDS)

    v3["v3_change_log"] = [
        "Dataset identity changed to Fashion-MNIST; split policy and sample counts unchanged.",
        "Methods limited to the three optical-compatible methods used by the robustness plan.",
        "Training seeds 42/43/44, matching the v1 cohort size for a like-for-like replication.",
        "MNIST v1 and v2 results are unaffected and remain the primary cohort.",
    ]

    V3_PATH.write_text(json.dumps(v3, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {V3_PATH.name}")
    print(f"  protocol_id : {v3['protocol_id']}")
    print(f"  dataset     : {v3['dataset']['key']} ({v3['dataset']['test_samples']} test samples)")
    print(f"  methods     : {[m['method_id'] for m in v3['methods']]}")
    print(f"  seeds       : {v3['training']['paired_training_seeds']}")
    print(f"  conditions  : {len(v3['conditions'])}  draws: {sum(c['draw_count'] for c in v3['conditions'])}")
    print(f"  evaluation  : device={v3['evaluation']['device']} timing={v3['evaluation']['timing_device']}")
    print(f"  sha256      : {sha256(V3_PATH)}")


if __name__ == "__main__":
    main()
