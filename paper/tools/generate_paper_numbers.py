"""Generate every number Table 9 and the Fashion-MNIST paragraph need, from frozen records.

Purpose: eliminate hand-transcription, which produced a cross-cohort reference column and a
sign-dropped confidence-interval bound in the previous revision.

Paired differences are always defined as (method A - the stated reference) over the SAME
three training seeds, and every reported interval is a two-sided 95% Student-t interval with
df = 2 computed from the per-seed differences.
"""

from __future__ import annotations

import json
import pathlib
import statistics

# Resolve relative to the repository root so the tools work from any checkout.
ROOT = pathlib.Path(__file__).resolve().parents[2] / "results"
V4 = ROOT / "formal_mnist_v4"
V3 = ROOT / "formal_fashion_mnist_v3"
T_975_DF2 = 4.302652729749454
SEEDS = (42, 43, 44)

COND_LABEL = {
    "clean_continuous": "干净连续相位",
    "lateral_shift_0p25px": "0.25 像素横向错位",
    "lateral_shift_0p50px": "0.50 像素横向错位",
    "mixed_stress": "混合强扰动",
    "phase_quantization_4": "四级相位量化",
}


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def seed_acc(path):
    return {
        (row["method_id"], row["condition_id"], row["training_seed"]): row["metrics"]["accuracy"]
        for row in load(path)
    }


def t_interval(diffs):
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    half = T_975_DF2 * sd / len(diffs) ** 0.5
    return mean, mean - half, mean + half


def main() -> None:
    abl = seed_acc(V4 / "robustness_summary" / "seed_metrics.jsonl")
    ref = seed_acc(V4 / "reference_robustness_summary" / "seed_metrics.jsonl")

    print("### TABLE 9 — values to place in the table (mean over seeds 42-44, %)")
    cols = [
        ("hybrid_linear_head", "hybrid_linear_head"),
        ("hybrid_pool4", "hybrid_pool4"),
        ("phase_filtered_d2nn", "phase_filtered_d2nn"),
        ("hybrid", "hybrid"),
        ("baseline_d2nn", "baseline_d2nn"),
    ]
    header = f"{'condition':16s}" + "".join(f"{name:>14s}" for _, name in cols)
    print(header)
    for cond in COND_LABEL:
        cells = ""
        for method, _ in cols:
            src = ref if method in ("hybrid", "baseline_d2nn") else abl
            cells += f"{statistics.mean(src[(method, cond, s)] * 100 for s in SEEDS):14.2f}"
        print(f"{COND_LABEL[cond]:16s}{cells}")

    print()
    print("### TABLE 9 PROSE — same-seed paired differences (A - reference), 95% t CI")
    prose_pairs = [
        ("hybrid_linear_head", "hybrid", ("clean_continuous", "lateral_shift_0p25px",
                                          "lateral_shift_0p50px", "mixed_stress")),
        ("hybrid_linear_head", "baseline_d2nn", ("clean_continuous",)),
        ("hybrid_pool4", "hybrid", ("clean_continuous", "lateral_shift_0p50px", "mixed_stress")),
        ("phase_filtered_d2nn", "baseline_d2nn", ("clean_continuous", "lateral_shift_0p25px",
                                                  "lateral_shift_0p50px", "phase_quantization_4")),
    ]
    for a, b, conds in prose_pairs:
        for cond in conds:
            diffs = [(abl[(a, cond, s)] - ref[(b, cond, s)]) * 100 for s in SEEDS]
            mean, lo, hi = t_interval(diffs)
            crosses = "crosses 0" if lo <= 0 <= hi else "excludes 0"
            print(f"  {a} - {b} @ {COND_LABEL[cond]}")
            print(f"    per-seed = {[round(d, 2) for d in diffs]}  mean = {mean:+.2f}  CI = [{lo:+.2f}, {hi:+.2f}]  ({crosses})")

    print()
    print("### SECTION 3.6 (Fashion-MNIST) — same-seed paired differences, 95% t CI")
    v3 = seed_acc(V3 / "robustness_summary" / "seed_metrics.jsonl")
    for a, b in (("hybrid", "baseline_d2nn"), ("robust_d2nn", "baseline_d2nn"), ("robust_d2nn", "hybrid")):
        for cond in COND_LABEL:
            diffs = [(v3[(a, cond, s)] - v3[(b, cond, s)]) * 100 for s in SEEDS]
            mean, lo, hi = t_interval(diffs)
            crosses = "crosses 0" if lo <= 0 <= hi else "excludes 0"
            print(f"  {a} - {b} @ {COND_LABEL[cond]:16s} mean={mean:+7.2f}  CI=[{lo:+7.2f}, {hi:+7.2f}]  ({crosses})")
        print()

    print("### Fashion-MNIST per-method means and seed values")
    for method in ("baseline_d2nn", "robust_d2nn", "hybrid"):
        for cond in ("clean_continuous", "phase_quantization_4"):
            vals = [v3[(method, cond, s)] * 100 for s in SEEDS]
            print(f"  {method:14s} {COND_LABEL[cond]:16s} mean={statistics.mean(vals):6.2f}  seeds={[round(v, 2) for v in vals]}")


if __name__ == "__main__":
    main()
