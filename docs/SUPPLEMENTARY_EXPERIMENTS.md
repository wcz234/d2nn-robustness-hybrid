# Supplementary experiments

These workflows implement the next tests; implementation is not evidence that the full
study has run. Historical results and checkpoint identities are never replaced. Run
commands from the repository root and use an unused output path for every new experiment.

## 1. Numerical validation before interpreting displacement robustness

```bash
python simulator/validate_optical_numerics.py --output artifacts/numerics_synthetic.json
python simulator/validate_optical_numerics.py \
  --checkpoint artifacts/MY_BASELINE.pth --sample-limit 4 \
  --factors 1 2 4 8 --output artifacts/numerics_checkpoint.json
```

The script independently implements band-limited angular-spectrum propagation and checks
it against analytic paraxial Gaussian intensity. It compares RS and ASM across numerical
refinements and compares legacy bilinear complex-transmission shifts with translated,
piecewise-constant, unit-modulus phase cells. Aperture, original phase-cell pitch, distances,
input image, and detector area are held fixed; only numerical sampling is refined. Fine
intensity is integrated back over the original detector cells. Increasing the model's
`--size` without reducing numerical pitch would change geometry and is not this test.

The finest tested grid is a reference, not automatically a converged solution. Inspect
successive refinements, collected power, and classification changes; extend refinement if
differences remain material. Gaussian agreement validates one resolved regime, not every
phase plate. The probe excludes manufacturing edge profiles, partial coherence and vector
effects. It also does not model extra field outside the existing propagation window.
Repeat the independent calculation with `--asm-padding 3` and `--asm-padding 7` (and larger
values when necessary). The angular-spectrum sampling cutoff depends on padded width;
padding sensitivity must be resolved before interpreting RS/ASM discrepancies. The probe
uses rectangular frequency cutoffs, not an optimized two-dimensional support or a certified
alias-free solver. Background: [Matsushima and Shimobaba (2009)](https://doi.org/10.1364/OE.17.019662).

## 2. Deployment-consistent checkpoint selection

```bash
python simulator/train.py --task classification --dataset mnist \
  --size 64 --layers 3 --rs-backend fft --optics-preset d2nn2018_thz \
  --model-variant hybrid --train-phase-quantization-levels 4 \
  --validate-training-phase --epochs 30 --batch-size 128 --lr 0.01 \
  --seed 42 --run-name v5_qat_deployed_seed42 --save-dir ../artifacts/v5
```

`--validate-training-phase` applies the QAT quantizer and/or Gaussian phase filter during
validation/test, with training-time random errors disabled. The manifest records the nominal
phase configuration. Omitting the flag preserves legacy continuous-phase model selection.
For filtering, compare two independently trained recipes with identical initialization and
data order: training-only smoothing/continuous validation and smoothing with matching
nominal validation. Evaluate both latent and explicitly filtered deployment conditions.
If quantization and smoothing are combined, the operator order is quantize-then-smooth;
the smoothed effective phase is not necessarily restricted to the original discrete levels.

Nominal QAT accuracy, latent continuous accuracy and QAT-plus-mixed-error accuracy answer
different questions. Pure four-level quantization is deterministic and is evaluated once.
Use the `qat4_mixed` condition to test additional alignment/noise errors.

## 3. Frozen-front causal controls

```bash
python simulator/fixed_front_readout.py --checkpoint artifacts/MY_BASELINE.pth \
  --output-dir artifacts/readouts_seed42 --epochs 30 --lr 0.003
```

The source train/validation split is reconstructed and checked against its index hashes.
Every optical weight is frozen; feature extraction uses evaluation mode/no gradients and
checks phase hashes. The same source phases feed ten-region detector features and 4x4/8x8
intensity pools. Linear and MLP heads use the same initialization and shuffled order for
compatible dimensions. Each head is selected using validation accuracy then cross entropy;
the test dataset is constructed only after all head selection is complete.

Controls use seeded fixed random phases, zero-phase free propagation with the same geometry,
and direct pooling of the **same amplitude-encoded input squared**. This last control matches
the optical intensity convention and input embedding/resolution; it is not a native 784-pixel
image classifier. All heads receive mean-normalized intensity and optimize only readout
cross entropy. Comparing this frozen-source task with a jointly trained optical model does
not equalize the source front's original training objective.

Run this for each independently trained source checkpoint. Different heads or random fronts
derived from one checkpoint do not increase the independent sample size.

```bash
python simulator/summarize_supplementary.py readouts \
  --results artifacts/readouts_seed42/readout_results.json \
            artifacts/readouts_seed43/readout_results.json \
            artifacts/readouts_seed44/readout_results.json \
  --output results/v5_readout_summary.json
```

`--train-limit`, `--validation-limit`, and `--test-limit` allow a cheap pipeline pilot. Any
limit marks its evidence as a subset pilot. Such results must not populate manuscript tables.

## 4. Prospective factorial training and longer budgets

```bash
python simulator/supplementary_experiments.py freeze --dataset mnist \
  --seeds 42 43 44 45 46 --budgets 3 30 --stochastic-draws 20 \
  --output artifacts/v5_mnist_protocol.json
python simulator/supplementary_experiments.py jobs \
  --protocol artifacts/v5_mnist_protocol.json --output artifacts/v5_mnist_jobs.json
```

The default design has **120 optical training runs per dataset**, not a smoke test. It adds
single-shift, single-gap, single-phase-noise, single-detector-noise and joint training,
hybrid-plus-joint training, paired legacy/deployed QAT, and paired legacy/deployed phase
filtering. Repeat for `--dataset fashion-mnist` after the numerical/selection gates.

The generated JSON contains executable argv arrays, working directories and dependency
IDs. Job generation does not run commands. Training jobs precede the shared frozen plan;
evaluation jobs share that plan; readout/numerical jobs use the longer-budget baseline
source. GPU training follows the existing CLI's automatic device selection. Evaluation jobs
default to CPU; if adopting CUDA, record that choice and evaluate parity before drawing
cross-device conclusions. Commands and environments must be retained with actual results.

The two checked-in `docs/SUPPLEMENTARY_PROTOCOL_V5_*.json` files are the prospective designs.
Generate the job bundle in the checkout where it will run, then execute an individual job:

```bash
python simulator/supplementary_experiments.py jobs \
  --protocol docs/SUPPLEMENTARY_PROTOCOL_V5_MNIST.json --output artifacts/v5_jobs.json
python simulator/supplementary_experiments.py run \
  --protocol docs/SUPPLEMENTARY_PROTOCOL_V5_MNIST.json --jobs artifacts/v5_jobs.json \
  --job-id v5_mnist_baseline_d2nn_e3_seed42 --state-dir artifacts/v5_job_state
```

`--with-dependencies` executes prerequisites recursively and may start many training jobs.
The runner rejects edited commands, existing untracked outputs, changed code, or changed
completed outputs. Completion records contain command, code and output hashes. Re-running
a completed job skips it only after verification. After changing the experiment code, use
a separate checkout/output tree; do not mix old and new jobs in the same results directory.

Learning rate is fixed at 0.01 in both budgets, using the existing StepLR schedule. This
isolates training duration under one optimizer recipe, **not equal wall-clock compute** or
best-achievable architecture performance. Thirty epochs are not automatically convergence.
Inspect each history; any additional validation-only tuning/early stopping or extension
requires an explicit new design before looking at its test results.

The conditions include dense shifts, gap/phase/detector noise, 32/16/8/4/2-level quantization,
mixed error, QAT-plus-mixed error, and explicit filtered paths. Applying filtering to a
condition affects every method under that condition; it is not a hidden per-method change.
Nominal comparisons must name the relevant condition for each deployment assumption.

## 5. Deployment tails and performance loss relative to nominal

```bash
python simulator/summarize_supplementary.py deployment \
  --evaluations results/MY_EVALUATION_A results/MY_EVALUATION_B \
  --failure-threshold 0.90 --lower-quantile 0.05 \
  --reference-map artifacts/v5_reference_map.json \
  --output results/v5_deployment_distributions.json
```

The reference map uses exact method IDs, for example:

```json
{"hybrid_qat_deployed_e30":"phase_quantization_4",
 "phase_filtered_deployed_e30":"filtered1_nominal"}
```

Other methods default to `clean_continuous`. The threshold is predeclared in the design
(0.90 for MNIST and 0.75 for Fashion-MNIST), not selected after inspecting failures. Summaries
report mean accuracy, empirical lower quantile, failure probability and loss from the
method's named nominal path. Draws are summarized within each training seed before
cross-seed Student-t intervals. Twenty draws offer descriptive tails; use a separately
frozen 100+ draw follow-up for a more precise deployment distribution. A cross-training-seed
interval does not remove finite-draw uncertainty in a tail estimate. Marginal intervals
and a lack of detectable difference do not establish equivalence or simultaneous rankings.

## Remaining research work

Physical shot/read noise, fixed photon-budget comparisons, detector/ADC costs, broader
datasets, measured phase/position errors, and a real optical bench remain unimplemented
research extensions. Relative additive detector noise and CPU simulation time retain their
existing numerical-only meanings. No latency, power or energy advantage is claimed.
