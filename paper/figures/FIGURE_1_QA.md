# Figure 2 robustness QA

> The filename preserves the original artifact prefix; the integrated manuscript assigns this result to Figure 2.

- Static source preflight: 14 pass, 0 warn, 0 fail.
- Backend: Python/matplotlib only.
- Final width: 183 mm.
- Source-data coverage: 42/42 method-condition rows and 28/28 baseline-referenced paired rows.
- Independent unit: training seed (`n = 3`).
- Intervals: two-sided 95% Student-t confidence intervals across training seeds.
- Hypothesis tests: none.
- SVG editable text: 40 `<text>` nodes detected.
- PDF export: generated with TrueType font embedding requested (`pdf.fonttype = 42`).
- TIFF export: 4322 x 3023 px at 600 dpi.
- PNG preview: 2161 x 1511 px.
- Visual inspection: condition labels, legends, panel labels, intervals, axes, and footnote do not overlap at the exported size.
- Integrity note: four-level phase-quantization intervals extend beyond the probability range because the unbounded small-sample t interval is shown without clipping. The manuscript must describe this as seed instability rather than a physical probability bound.
