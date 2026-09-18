# Results draft audit

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Hybrid improves clean accuracy over baseline D2NN by 6.06 percentage points | `FORMAL-CLEAN-PAIRED-001` | Supported |
| Hybrid clean accuracy exceeds electronic by 1.03 percentage points, but the paired interval crosses zero | `FORMAL-CLEAN-PAIRED-003` | Supported; no resolved ordering |
| Robust training trades 1.33 clean percentage points for selected perturbation gains | `FORMAL-CLEAN-PAIRED-002`, `FORMAL-ROBUST-PAIRED-001`, `FORMAL-ROBUST-PAIRED-003`, `FORMAL-ROBUST-PAIRED-004` | Supported |
| Hybrid has the highest mean accuracy in 13 of 14 optical-compatible conditions | `FORMAL-ROBUST-RANK-001` | Supported as a mean ranking, not a significance claim |
| Hybrid exceeds baseline under mixed stress by 9.72 percentage points | `FORMAL-ROBUST-PAIRED-002` | Supported |
| Extreme four-level phase quantization leaves hybrid-versus-robust ordering unresolved | Formal paired comparison; hybrid-minus-robust 95% CI -41.38 to 32.06 percentage points | Supported |
| CPU timing is software simulation overhead rather than hardware latency | `FORMAL-CLEAN-BASELINE-004`, `FORMAL-CLEAN-ROBUST-004`, `FORMAL-CLEAN-HYBRID-004`, `FORMAL-CLEAN-ELECTRONIC-004` | Supported within the stated boundary |

## Structural audit

- Setup anchoring: pass. Dataset, methods, condition plan, independent unit, interval, and test policy are stated.
- Head-to-head comparison: pass. All four methods are named, and electronic is restricted to clean evaluation.
- Deep-dive disaggregation: pass. The text separates displacement, stochastic perturbations, phase quantization, and mixed stress.
- Takeaway synthesis: pass. Every result cluster ends with a bounded implication.
- Robustness boundary: pass. Results describe numerical simulation only.
- Ablation: partial. The four model variants expose architecture/training contributions, but no component-removal ablation isolates the hybrid pooling size or electronic-head depth.
- Generalization: limited to MNIST and the frozen perturbation plan.

## Sentence-level style audit

| Category | Violations found | Fixed | Remaining |
|---|---:|---:|---:|
| Negation-first constructions | 2 | 2 | 0 |
| Throat-clearing | 0 | 0 | 0 |
| Unsupported hedging | 0 | 0 | 0 |
| Generic adjectives | 3 | 3 | 0 |
| Sentences over 40 words | 4 | 4 | 0 |
| Passive voice that obscures agency | 2 | 2 | 0 |
| Missing external citations | 0 | 0 | 0; Results contains only internally measured claims |

## Reviewer risks

1. Three training seeds yield wide t intervals, especially under four-level phase quantization.
2. The study includes no physical optical platform, real edge device, power measurement, or energy measurement.
3. The electronic baseline does not enter optical perturbation evaluation because those perturbations are undefined for its architecture.
4. CPU timing compares software execution paths and cannot establish deployment latency.
5. MNIST alone does not support claims about cross-dataset generalization.

