# Figure 4 confusion contract

## Core conclusion

The hybrid readout concentrates clean-condition predictions closer to the diagonal across MNIST classes, while the robust optical model retains class-specific confusion patterns that differ from the clean baseline.

- Archetype: quantitative grid.
- Backend: Python/matplotlib only.
- Target/output: generic SCI figure; 183 mm wide; editable SVG/PDF plus 600 dpi TIFF and PNG preview.
- Panels a-d: mean row-normalized 10 x 10 confusion matrices for baseline D²NN, robust D²NN, hybrid, and electronic models.
- Source data: the 12 formal clean `clean_metrics.json` files, exported as `source_data_figure_4.csv`.
- Statistics: each panel averages three independent training seeds; test samples are not treated as independent replicates.
- Integrity boundary: this figure reports clean MNIST numerical predictions only and makes no physical-device claim.
