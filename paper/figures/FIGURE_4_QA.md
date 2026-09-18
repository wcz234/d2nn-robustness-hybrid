# Figure 4 confusion QA

- Static source preflight: 14 pass, 0 warn, 0 fail under strict mode.
- Backend: Python/matplotlib only.
- Final width: 182.9 mm.
- Source-data coverage: 4 methods x 3 training seeds x 10 x 10 cells; no rows excluded.
- Independent unit: training seed (`n = 3` per method).
- Normalization: each seed-level confusion matrix is normalized by true-label row, then averaged across seeds.
- SVG editable text: enabled with `svg.fonttype = none`.
- PDF export: generated with TrueType font embedding requested (`pdf.fonttype = 42`).
- TIFF export: generated at 600 dpi; PNG is a 300 dpi preview.
- Visual inspection: four heatmaps, shared colorbar, panel labels, and Chinese axis labels do not overlap at the exported size.
- Interpretation boundary: matrices describe the fixed MNIST numerical simulation cohort and do not represent a physical optical or edge-device experiment.
