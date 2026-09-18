# Related Work and Discussion audit

## Related Work structure

- Placement: standalone post-evaluation section; final venue may fold it into the Introduction.
- Categories: optical readout; hybrid/edge inference; robustness; quantization/coherence/simulation.
- Strongest neighboring methods included: misalignment vaccination, phase filtering, parallel subnetworks, structural robustness, sharpness-aware training, normalized cutoff frequency, and quantization-aware training.
- Positioning sentence: protocol-bound numerical comparison, not new hardware or a new perturbation primitive.
- Length: below the 10,000-character ceiling.

## Claim-evidence boundaries

- Prior physical measurements appear only as background and are never transferred to this study.
- The Discussion labels the hybrid representation explanation as an interpretation that requires ablation.
- The four-level quantization mechanism remains unresolved.
- The study does not claim reproduction of external algorithms or hardware.
- Open-source projects guide implementation and reproducibility context; they do not serve as experimental evidence for this paper's results.

## Sentence-level style audit

| Category | Violations found | Fixed | Remaining |
|---|---:|---:|---:|
| Negation-first constructions | 3 | 3 | 0 |
| Throat-clearing | 0 | 0 | 0 |
| Unsupported hedging | 2 | 2 | 0 |
| Generic adjectives | 4 | 4 | 0 |
| Sentences over 40 words | 5 | 5 | 0 |
| Passive voice that obscures agency | 2 | 2 | 0 |
| Technical claims without citations | 7 | 7 | 0 |

## Remaining reviewer risks

1. The manuscript lacks direct reimplementations of the strongest recent robustness methods.
2. The mechanism behind quantization tolerance remains untested.
3. A standalone Related Work section may not match a final Nature-family target and may need integration into the Introduction.

