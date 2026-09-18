# Figure 3 complexity contract

> The filename preserves the original artifact prefix; the integrated manuscript assigns this result to Figure 3.

- Core conclusion: The hybrid model improves clean optical classification with a small parameter increase and a 64-element optical-to-electronic representation, while CPU wall-clock values characterize the simulator rather than deployment hardware.
- Archetype: quantitative grid.
- Target/output: generic Nature-leaning double-column figure; 183 mm wide; editable SVG/PDF plus 600 dpi TIFF and PNG preview.
- Backend: Python/matplotlib only.
- Panel a: trainable parameter count and representation dimensionality for all four methods.
- Panel b: per-training-seed clean accuracy against the median duration of a complete 10,000-sample CPU software forward.
- Independent unit: training seed (`n = 3` per method).
- Timing repeats: precision measurements only; not independent replicates.
- Source data: 12 protocol-bound clean evaluation manifests and the formal clean seed summary.
- Reviewer risk: operations differ between electronic and optical-compatible models; wall-clock positions must not be interpreted as physical optical latency, edge-device latency, power, or energy efficiency.
