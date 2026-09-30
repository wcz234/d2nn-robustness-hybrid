# Supplementary workflow validation

These are diagnostics and single-seed pipeline pilots, **not** a completed v5 cohort and
not replacements for the manuscript tables. No optical hardware was used.

## Software checks

- Python 3.12, PyTorch 2.10.0+cpu and torchvision 0.25.0+cpu installed from the new CPU
  requirements file: 283 tests and 10 subtests passed.
- Manuscript number verifier: 19 registered claims checked, zero failures. The Chinese
  Markdown changes are editorial; the shipped English/PDF/DOCX snapshots were not rebuilt.
- Default MNIST design generated 253 dependency-ordered jobs: 120 training runs and
  387 deployment draws per evaluated model. Generation did not execute those jobs.
- Historical protocol audit: v2 declares seeds 42–44 while stored metrics contain 42–46.
  The checked v1/v3/v4 seed lists match. Original frozen protocols/results are untouched.

## Synthetic numerical diagnostics

The two `optical_numerics_padding*.json` files use the same one synthetic input, seeded
random single phase plate, 16×16 grid, fixed THz geometry, and refinement factors 1/2/4/8.

| Quantity | Observed value | Interpretation |
|---|---:|---|
| ASM vs analytic paraxial Gaussian intensity, relative L2 | 1.53e-6 | Checks one resolved regime |
| RS vs the same analytic Gaussian, relative L2 | 2.69e-3 | Small but nonzero numerical discrepancy |
| RS vs ASM, original grid, padding 3 | 0.1381 | Sensitive to computational window |
| RS vs ASM, original grid, padding 7 | 0.0374 | Padding materially reduces discrepancy |
| RS vs ASM, refinement 8, padding 3 / 7 | 0.1274 / 0.0268 | Refinement alone does not settle window sensitivity |
| Piecewise-plate refinement 4 vs 8, across tested shifts | approximately 0.0023–0.0024 | The finest grid is still not certified converged |

At a half-pixel shift, legacy complex interpolation gives first-plate mean modulus about
0.653; the sampled piecewise phase-cell construction has unit modulus. Collected power
and intensity differences must be inspected alongside classifier changes. This single
synthetic probe establishes neither the sign nor the size of the effect on trained-model
robustness. Larger padding, additional samples/seeds and geometry checks remain necessary.

## Actual single-seed pipeline pilots

Both optical training pilots use the full MNIST 55,000/5,000/10,000 split, seed 42,
64×64/three phase plates, the THz preset, FFT backend, batch 128 and learning rate 0.01.
All CLI arguments, split identities, selected epoch, source/checkpoint hashes and package
versions are retained in the adjacent training manifests.

| Pilot | Budget | Selection/test path | Test accuracy |
|---|---:|---|---:|
| Baseline D2NN | 3 epochs, selected epoch 2 | Continuous clean | 89.26% |
| Hybrid QAT | 1 epoch | Nominal four-level phase path | 90.78% |

The baseline was trained with PyTorch 2.14.0+cpu, QAT with the pinned 2.10.0+cpu environment.
These budgets/environments differ, so their accuracies are **not a method comparison**.
The independent four-level evaluation is recorded in `qat4_evaluation/` and matches the
training manifest exactly: accuracy 0.9078 and contrast 0.743164668560028.
`qat4_distribution_pilot.json` exercises nominal-referenced aggregation and labels its one
training seed as descriptive evidence. Pure quantization
has one deterministic draw; it must not be represented as a stochastic deployment study.

The readout pilot uses the baseline checkpoint, 2,048 training / 512 validation / 512 test
samples, five readout epochs and both linear/MLP heads. All 18 readouts completed; source
phase hashes match before/after. `readout_pilot_summary.json` correctly records one source
seed and descriptive/subset evidence. The short head fits are not converged performance
comparisons. Head weight files and source optical checkpoints are local artifacts; they
are not distributed here. Regenerate them with the manifest CLI and the documented workflows.

`optical_numerics_baseline_seed42.json` refines that trained front on the first test sample.
`filter_evaluation/` checks explicit filtered deployment on 128 test samples of an otherwise
unfiltered baseline; it does not test the efficacy of training with filtering. The complete
prospective v5 plan loaded successfully with 26 conditions and 387 draws; its generation
does not mean those evaluations ran.

## Evidence still required

The 5-seed, 3/30-epoch MNIST/Fashion-MNIST cohorts, full-data frozen-front comparisons,
convergence studies and deployment-tail confirmation remain unexecuted. Twenty deployment
draws give descriptive tails; a separately frozen 100+ draw study is needed for precision.
Shot/read noise, photon budgets, measured manufacturing errors and a physical bench are
future research work. Follow `docs/SUPPLEMENTARY_EXPERIMENTS.md` before making new claims.
