# Figure 1 method-overview QA

- Static source preflight: 14 pass, 0 warn, 0 fail under strict mode.
- Backend: Python/matplotlib only.
- Final width: 182.9 mm.
- Source-data coverage: not applicable; the figure is a code-defined schematic.
- Protocol cross-check: 64x64 planes, three phase layers, 0.75 mm wavelength, 30 mm distances, 8x8 pooling, 64-to-32-to-10 hybrid head, and 784-to-18-to-10 electronic baseline match the frozen implementation.
- Training cross-check: robust training samples displacement, gap, phase, and detector perturbations; phase quantization is explicitly excluded from training.
- SVG editable text: 52 `<text>` nodes detected.
- PDF export: generated with TrueType font embedding requested (`pdf.fonttype = 42`).
- TIFF export: 4389 x 2889 px at 600 dpi.
- Visual inspection: labels, arrows, branches, component boxes, and panel boundaries do not overlap at the exported size.
- Interpretation boundary: panel a states “scalar simulation only”; the schematic does not depict a manufactured optical device.
