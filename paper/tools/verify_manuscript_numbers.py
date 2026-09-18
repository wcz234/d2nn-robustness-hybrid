"""Cross-check every number the manuscript states against the frozen per-seed records.

For each (A, B, condition) comparison reported in the text, the identity
    mean(A) - mean(B) == mean(per-seed A - per-seed B)
must hold whenever both sides come from the same cohort, and the reported interval must be
centred on the reported mean. This catches exactly the two defects the external assessment
identified, plus any others of the same class.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics

# Resolve relative to the repository root so the tools work from any checkout.
ROOT = pathlib.Path(__file__).resolve().parents[2] / "results"
T_975_DF2 = 4.302652729749454
SEEDS = (42, 43, 44)


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def seed_acc(path):
    return {
        (r["method_id"], r["condition_id"], r["training_seed"]): r["metrics"]["accuracy"]
        for r in load(path)
    }


SOURCES = {
    "v4_ablation": seed_acc(ROOT / "formal_mnist_v4/robustness_summary/seed_metrics.jsonl"),
    "v4_reference": seed_acc(ROOT / "formal_mnist_v4/reference_robustness_summary/seed_metrics.jsonl"),
    "v3": seed_acc(ROOT / "formal_fashion_mnist_v3/robustness_summary/seed_metrics.jsonl"),
}

# Every comparison the manuscript reports, with the source cohort for each side.
CLAIMS = [
    # section 3.7 / Table 9
    ("v4_ablation", "hybrid_linear_head", "v4_reference", "hybrid", "clean_continuous", 0.91, -0.21, 2.02),
    ("v4_ablation", "hybrid_linear_head", "v4_reference", "hybrid", "lateral_shift_0p50px", 1.14, 0.43, 1.85),
    ("v4_ablation", "hybrid_linear_head", "v4_reference", "hybrid", "mixed_stress", 1.79, 0.61, 2.97),
    ("v4_ablation", "hybrid_linear_head", "v4_reference", "baseline_d2nn", "clean_continuous", 6.96, 5.79, 8.13),
    ("v4_ablation", "hybrid_pool4", "v4_reference", "hybrid", "clean_continuous", 0.35, -1.56, 2.25),
    ("v4_ablation", "hybrid_pool4", "v4_reference", "hybrid", "lateral_shift_0p50px", -0.27, -2.06, 1.51),
    ("v4_ablation", "hybrid_pool4", "v4_reference", "hybrid", "mixed_stress", 0.18, -1.99, 2.35),
    ("v4_ablation", "phase_filtered_d2nn", "v4_reference", "baseline_d2nn", "clean_continuous", -11.63, -13.20, -10.06),
    ("v4_ablation", "phase_filtered_d2nn", "v4_reference", "baseline_d2nn", "lateral_shift_0p25px", -8.99, -10.07, -7.90),
    ("v4_ablation", "phase_filtered_d2nn", "v4_reference", "baseline_d2nn", "lateral_shift_0p50px", -9.18, -9.57, -8.79),
    ("v4_ablation", "phase_filtered_d2nn", "v4_reference", "baseline_d2nn", "phase_quantization_4", -1.21, -9.22, 6.79),
    # section 3.6 / Fashion-MNIST
    ("v3", "hybrid", "v3", "baseline_d2nn", "clean_continuous", 3.80, 0.58, 7.01),
    ("v3", "hybrid", "v3", "baseline_d2nn", "lateral_shift_0p50px", 4.27, 0.75, 7.80),
    ("v3", "robust_d2nn", "v3", "baseline_d2nn", "clean_continuous", -1.72, -3.33, -0.10),
    ("v3", "robust_d2nn", "v3", "baseline_d2nn", "lateral_shift_0p25px", -1.07, -3.19, 1.05),
    ("v3", "robust_d2nn", "v3", "baseline_d2nn", "lateral_shift_0p50px", -0.38, -2.54, 1.79),
    ("v3", "robust_d2nn", "v3", "baseline_d2nn", "mixed_stress", 1.39, -1.70, 4.48),
    ("v3", "robust_d2nn", "v3", "baseline_d2nn", "phase_quantization_4", 21.67, -5.76, 49.10),
    ("v3", "robust_d2nn", "v3", "hybrid", "phase_quantization_4", 16.49, 9.37, 23.61),
]

TOL = 0.006  # rounding tolerance for two-decimal reporting
failures = 0

print(f"{'comparison':52s} {'reported':>9s} {'computed':>9s} {'CI centre':>10s} {'status':>10s}")
for sa, ma, sb, mb, cond, rep_mean, rep_lo, rep_hi in CLAIMS:
    a, b = SOURCES[sa], SOURCES[sb]
    diffs = [(a[(ma, cond, s)] - b[(mb, cond, s)]) * 100 for s in SEEDS]
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    half = T_975_DF2 * sd / len(diffs) ** 0.5
    lo, hi = round(mean - half, 2), round(mean + half, 2)
    centre_ok = abs((rep_lo + rep_hi) / 2 - rep_mean) < 0.011
    value_ok = abs(mean - rep_mean) < TOL and abs(lo - rep_lo) < 0.011 and abs(hi - rep_hi) < 0.011
    status = "OK" if (value_ok and centre_ok) else "FAIL"
    if status == "FAIL":
        failures += 1
    label = f"{ma} - {mb} @ {cond}"
    print(f"{label:52s} {rep_mean:9.2f} {mean:9.2f} {(rep_lo+rep_hi)/2:10.2f} {status:>10s}")
    if status == "FAIL":
        print(f"{'':52s} reported CI [{rep_lo}, {rep_hi}]  computed CI [{lo}, {hi}]  per-seed={[round(d,2) for d in diffs]}")

print()
print(f"claims checked: {len(CLAIMS)}   failures: {failures}")
raise SystemExit(1 if failures else 0)
