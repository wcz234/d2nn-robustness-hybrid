# Figure 1 method-overview contract

> The filename records the figure's original planning slot. The manuscript assigns this first-appearance schematic to Figure 1.

## Core conclusion

The protocol separates optical propagation, readout capacity, and training-time perturbation sampling, which allows the evaluation to attribute gains without implying physical-device performance.

## Figure architecture

- Archetype: schematic-led composite
- Backend: Python with matplotlib only
- Target: generic journal, double-column width
- Final size: 183 mm wide, approximately 119 mm high
- Panel a: numerical inference paths for baseline/robust D2NN, hybrid, and electronic models
- Panel b: clean and perturbation-aware training paths through the shared composite objective

## Evidence hierarchy

- Hero evidence: the shared optical front end branches into detector-energy and pooled-intensity readouts.
- Supporting evidence: the electronic baseline bypasses optical propagation.
- Mechanism evidence: robust training samples deployment perturbations before the optical forward pass.

## Integrity and reviewer risks

- This is a code-defined schematic and contains no experimental observations.
- Dimensions and hyperparameters must match the frozen protocol and model implementations.
- The figure must say “numerical simulation” and must not depict a physical optical prototype.
- Method colors must match Figures 1 and 2: baseline grey, robust blue, hybrid red, electronic teal.
- Source data: not applicable; the script and frozen protocol define all labels.
