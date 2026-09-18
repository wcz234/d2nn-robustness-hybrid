# Upstream attribution

`simulator/` in this repository is a fork of:

- **Repository:** https://github.com/yeungmkw/nonlinear-d2nn
- **License:** MIT (see `LICENSE` in this directory)
- **Pinned upstream commit:** `f1ae6e3e5076fcd2f60111ef4d42a2798d30ac8f`

## Contribution boundary

The upstream project provides the D²NN numerical propagation, classification training, detector-region
readout and experiment-artifact conventions. This study adds the protocol-bound robustness and
evaluation pipeline on top of it.

- `local_delta.patch` is `git diff f1ae6e3 <local HEAD>` and contains **41 changed files,
  7425 insertions, 106 deletions** relative to the pinned upstream commit. Applied to a clean
  checkout of `f1ae6e3`, it reproduces the `simulator/` tree shipped here.
- Upstream commit history is not vendored into this repository; the patch plus the pinned commit
  hash are the reproducible record of the contribution boundary.

## Notes on scope

The upstream README describes nonlinear-activation experiments, fabrication exports and a
single-layer lab-validation path. Those belong to the upstream project and are **not** part of the
study reported in `paper/`. The reported study uses the classification path with the
`d2nn2018_thz` numerical optics preset; no physical optical bench was used.

## Citation boundary

Upstream numbers (including any accuracy, latency or fabrication figures in the upstream README or
its historical artifacts) must not be cited as results of this study. All quantitative results here
come from the frozen protocols in `docs/` and the outputs in `results/`.
