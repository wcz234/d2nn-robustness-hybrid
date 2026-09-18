# Figure 2 robustness contract

> The filename preserves the original artifact prefix; the integrated manuscript assigns this result to Figure 2.

- Core conclusion: Perturbation-aware training improves selected deployment conditions, while the hybrid model retains the highest mean accuracy in most conditions; extreme four-level phase quantization remains seed-sensitive.
- Archetype: quantitative grid.
- Target/output: generic Nature-leaning double-column figure; 183 mm wide; editable SVG/PDF plus 600 dpi TIFF and PNG preview.
- Backend: Python/matplotlib only.
- Panel a: absolute accuracy across 14 preregistered conditions for baseline D2NN, robust D2NN, and hybrid models.
- Panel b: paired accuracy difference relative to baseline D2NN for robust D2NN and hybrid models.
- Hero evidence: full-condition accuracy profile in panel a.
- Validation evidence: paired-by-training-seed effect intervals in panel b.
- Independent unit: training seed (`n = 3` per method).
- Interval: two-sided 95% Student-t confidence interval across training seeds.
- Hypothesis tests: none.
- Source data: formal robustness summary JSONL files; all 42 method-condition rows are used, and all 28 baseline-referenced paired rows are used.
- Reviewer risk: three seeds produce wide intervals for unstable conditions; intervals are not clipped to the probability range and must not be interpreted as physical bounds.
