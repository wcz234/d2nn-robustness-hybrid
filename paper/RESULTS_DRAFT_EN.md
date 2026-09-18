# Results

## Protocol-bound evaluation separates clean performance from deployment robustness

All results derive from numerical simulations on the complete 10,000-image MNIST test set. We evaluated three independently trained models per method using seeds 42, 43, and 44. The baseline D2NN, robust D2NN, and hybrid model shared one frozen 14-condition plan containing 34 deployment draws. The electronic model entered only the clean evaluation because optical perturbations do not apply to its architecture. We first aggregated deployment draws within each training seed and then summarized the three seed-level estimates. Error bars and intervals report two-sided 95% Student-t confidence intervals across training seeds; we performed no hypothesis tests. These marginal intervals do not provide simultaneous coverage across the 14 conditions.

## The hybrid model improves clean optical inference without establishing an advantage over the electronic baseline

The hybrid model achieved the highest clean accuracy among the three optical-compatible methods. Its mean accuracy reached 95.70% (95% CI, 94.72-96.67%), compared with 89.64% (88.81-90.46%) for the baseline D2NN and 88.31% (87.90-88.72%) for the robust D2NN. Pairing models by training seed yielded a 6.06-percentage-point hybrid gain over the baseline D2NN (5.19-6.93 percentage points). Macro-F1 followed the same ordering: 0.9565 for the hybrid model, 0.8947 for the baseline D2NN, and 0.8813 for the robust D2NN.

The electronic model achieved 94.67% clean accuracy (94.04-95.30%). The hybrid-electronic difference was 1.03 percentage points, but its paired interval crossed zero (-0.56 to 2.61 percentage points). These three training seeds therefore do not establish a consistent clean-accuracy advantage of the hybrid model over the electronic baseline.

**Takeaway.** The electronic head recovers 6.06 percentage points over detector-region classification under clean optical simulation. The current seed cohort supports parity, rather than a resolved advantage, between hybrid and electronic clean classification.

## Perturbation-aware training trades clean accuracy for condition-specific resilience

Perturbation-aware training reduced clean accuracy by 1.33 percentage points relative to the baseline D2NN (-1.75 to -0.90 percentage points). It recovered this trade-off under selected perturbations. At a 0.50-pixel lateral shift, the robust D2NN improved accuracy by 2.20 percentage points over the baseline (1.82-2.58 percentage points). Under mixed stress, the corresponding gain was 2.76 percentage points (0.96-4.56 percentage points), raising mean accuracy from 83.36% to 86.12%.

Phase quantization produced the largest but least stable robust-training gain. Four phase levels reduced baseline accuracy to 61.41%, whereas the robust D2NN achieved 79.32%. The paired gain was 17.91 percentage points, with a wide interval of 3.01-32.81 percentage points. At eight phase levels, the mean gain was 3.29 percentage points, but the paired interval crossed zero. Robust training did not improve every perturbation family: its mean accuracy remained 1.09-1.33 percentage points below the baseline under gap error, phase noise, and detector noise.

**Takeaway.** Perturbation-aware training redistributes accuracy toward lateral displacement, combined stress, and coarse phase quantization. It does not provide a uniform robustness gain, and its clean-accuracy cost remains visible outside those conditions.

## The hybrid architecture leads in most conditions but exposes an extreme-quantization boundary

The hybrid model produced the highest mean accuracy in 13 of the 14 preregistered conditions (Fig. 2a). At a 0.50-pixel lateral shift, it exceeded the baseline D2NN by 9.48 percentage points (8.30-10.65 percentage points). Under mixed stress, it reached 93.08% accuracy and exceeded the baseline by 9.72 percentage points (7.62-11.82 percentage points). The hybrid model also gained 9.76 percentage points at eight phase levels (5.85-13.66 percentage points).

Four-level phase quantization was the only condition in which the hybrid mean did not lead. The robust D2NN achieved 79.32% mean accuracy, while the hybrid model achieved 74.66%. The paired hybrid-minus-robust interval spanned -41.38 to 32.06 percentage points, so three seeds do not resolve their ordering. The unbounded t interval for hybrid accuracy also extended beyond the probability range. We retained the raw interval to expose seed instability rather than clipping it to an apparently physical bound.

![Protocol-bound robustness comparison.](figures/figure_1_robustness.png)

**Figure 2 | Accuracy across preregistered deployment conditions.** **a,** Mean accuracy for baseline D2NN, robust D2NN, and hybrid models. **b,** Paired accuracy differences for robust D2NN and hybrid models relative to baseline D2NN. Points denote means across three independently trained seeds; error bars show marginal two-sided 95% Student-t confidence intervals. Deployment draws were aggregated within each seed. No hypothesis tests were performed, and intervals were not adjusted for simultaneous coverage.

**Takeaway.** The hybrid model provides the broadest mean-accuracy advantage across the frozen plan. Extreme phase quantization remains a failure boundary because seed-level uncertainty prevents a stable ranking against the robust D2NN.

## Model size and CPU simulation overhead bound the edge-inference interpretation

The baseline and robust D2NNs each contain 12,288 trainable parameters and expose ten detector-region energies. The hybrid model contains 14,698 trainable parameters and exposes a 64-element pooled-intensity representation to its electronic head. The electronic baseline contains 14,320 trainable parameters and processes the 784-element input image directly. These counts characterize model and representation size; they do not measure communication energy or device memory traffic.

The electronic model required a mean of 0.696 s for a complete 10,000-sample CPU software forward, based on the median timing within each seed. The corresponding means were 60.41 s for the baseline D2NN, 63.24 s for the hybrid model, and 70.30 s for the robust D2NN. Baseline timing varied strongly across seeds, and the paired hybrid-baseline interval ranged from -56.39 to 62.05 s. The measurements therefore describe fixed-environment software simulation overhead, not single-sample latency, optical propagation time, edge-device latency, power, or energy efficiency.

![Model size and CPU software simulation boundary.](figures/figure_2_complexity.png)

**Figure 3 | Model size and CPU software simulation boundary.** **a,** Trainable parameter counts and representation elements for the four methods. **b,** Clean accuracy against the median duration of a complete 10,000-sample CPU software forward for each training seed. Small points denote individual training seeds, and large points denote method means. Five timing repeats within each seed measured runtime precision and were not treated as independent replicates. CPU wall-clock values do not represent physical optical latency, edge-device latency, power, or energy efficiency.

**Takeaway.** The hybrid model adds 2,410 trainable parameters and expands the optical-to-electronic representation from 10 to 64 values. CPU wall-clock measurements characterize the cost of the numerical simulator and cannot support a hardware-speed or energy-efficiency claim.
