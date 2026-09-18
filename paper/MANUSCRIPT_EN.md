# Protocol-bound robustness evaluation and hybrid optoelectronic inference in diffractive neural networks

`AUTHOR_INPUT_NEEDED: authors, affiliations, and corresponding-author details.`

# Abstract

Diffractive neural networks can partition inference between optical propagation and electronic processing, but their deployment claims depend on separating perturbation tolerance from readout capacity and hardware performance. We present a protocol-bound numerical comparison of baseline diffractive, perturbation-aware diffractive, hybrid optical-electronic, and electronic models on MNIST. Three independent training seeds support clean evaluation for all methods, while the optical-compatible methods share a frozen 14-condition plan containing 34 deployment draws over the complete 10,000-image test set. Perturbation-aware training reduced clean accuracy by 1.33 percentage points relative to the baseline, but improved accuracy by 2.20 points at a 0.50-pixel shift, 2.76 points under mixed stress, and 17.91 points under four-level phase quantization. The hybrid model improved clean accuracy by 6.06 points over the baseline and achieved the highest mean accuracy in 13 of 14 optical-compatible conditions. Under mixed stress, it reached 93.08% accuracy and exceeded the baseline by 9.72 points. Extreme phase quantization remained seed-sensitive and did not resolve the hybrid-versus-robust ordering. Parameter counts, representation dimensions, and CPU wall-clock measurements describe model structure and software simulation overhead only. The study provides a reproducible simulation benchmark for condition-specific robustness and optical-electronic partitioning without making physical latency, power, or energy claims.

# Introduction

Edge inference systems must preserve classification accuracy under restricted local computation and variable deployment conditions. Photonic architectures address this pressure by moving linear operations into optical propagation, while system designs divide storage and computation between remote, optical, and electronic components \cite{Sludds2022NetcastEdge,Zhou2021ReconfigurableDPU}. Their value depends on more than nominal accuracy: the inference path must expose which perturbations it tolerates, what representation crosses each computational boundary, and which performance claims come from hardware rather than simulation.

Diffractive deep neural networks implement learned transformations with passive phase layers and classify optical intensity at an output plane \cite{Lin2018AllOpticalD2NN}. Their physical structure introduces alignment, spacing, phase, detector, and quantization errors. Training-time misalignment sampling, phase filtering, parallel subnetworks, heterogeneous diffractive neurons, and sharpness-aware objectives provide established routes to robustness \cite{Mengu2020MisalignmentResilient,Wang2025PhaseFilteredD2NN,Wang2026ParallelSubnetworkD2NN,Zhu2026HybridRobustODNN,Xu2026SharpnessAwarePNN}. Hybrid optical-electronic classifiers provide another route by replacing fixed detector decisions with a learned electronic head \cite{Mengu2020DiffractiveIntegration,Chang2018HybridOpticalElectronicCNN}.

These routes create three coupled questions. First, perturbation-aware training can exchange clean accuracy for tolerance to selected errors, so a single robustness score hides where the trade occurs. Second, an electronic head changes both decision capacity and the size of the optical-electronic representation. Third, numerical propagation time cannot establish physical latency or energy efficiency. A defensible comparison must therefore keep data, training seeds, deployment draws, metrics, and claim boundaries fixed while separating optical-compatible robustness from clean-only electronic inference.

We call this structure a **protocol-bound optical-electronic comparison**. We train baseline D2NN, perturbation-aware D2NN, hybrid, and electronic models on the same MNIST split using three independent training seeds. The three optical-compatible methods share one frozen plan with 14 conditions and 34 deployment draws over the complete 10,000-image test set. We aggregate draws within each seed, pair cross-method differences by training seed, preserve sample-level predictions, and bind every summary to checkpoint, protocol, source, and artifact hashes. The electronic model enters only clean evaluation because optical perturbations do not apply to its architecture.

The comparison reveals a bounded robustness trade-off. Perturbation-aware training loses 1.33 clean percentage points relative to the baseline D2NN, but gains 2.20 points at a 0.50-pixel shift, 2.76 points under mixed stress, and 17.91 points under four-level phase quantization. The hybrid model gains 6.06 clean points over the baseline and has the highest mean accuracy in 13 of 14 optical-compatible conditions (Fig. 2). Under mixed stress, it reaches 93.08% and exceeds the baseline by 9.72 points. Four-level quantization remains the exception: three seeds do not resolve the hybrid-versus-robust ordering.

This paper makes three evidence-backed contributions:

1. **Auditable evaluation.** We provide a seed-paired numerical protocol that connects full-test predictions, 14 deployment conditions, and integrity-checked summaries without treating samples or draws as independent replicates.
2. **Condition-specific robustness.** We show that perturbation-aware training redistributes accuracy toward displacement, mixed stress, and coarse quantization rather than uniformly improving every perturbation family.
3. **Bounded hybrid inference.** We show that a 64-element pooled optical representation and electronic head improve mean accuracy across most frozen conditions, while parameter counts and CPU timing remain model and simulator descriptors rather than hardware-efficiency evidence (Fig. 3).

# Related Work

## Diffractive classification and optical readout

Diffractive deep neural networks established that trained passive phase layers can implement classification and imaging through optical propagation \cite{Lin2018AllOpticalD2NN}. Subsequent work improved the readout by assigning positive and negative detector pairs to each class and by distributing classes across parallel diffractive networks \cite{Li2019DifferentialDetection}. These studies define the optical classifier and detector-readout families represented by our baseline D2NN. They do not compare a common seed-level cohort across clean inference, multiple deployment perturbations, and a compact electronic head.

## Hybrid optical-electronic and edge-oriented inference

Hybrid systems divide inference between an optical front end and an electronic back end. Jointly optimized diffractive layers and electronic classifiers showed that compressed optical outputs can feed lightweight electronic networks \cite{Mengu2020DiffractiveIntegration}. Optimized diffractive optics have also implemented the first layer of an optical-electronic convolutional classifier \cite{Chang2018HybridOpticalElectronicCNN}. Reconfigurable diffractive processors and hardware-software co-designed multi-task networks extend this idea to programmable and shared optical front ends \cite{Zhou2021ReconfigurableDPU,Li2021RealtimeMultitaskD2NN}. Netcast addresses a different edge architecture by streaming weights over deployed fiber to a low-storage photonic receiver \cite{Sludds2022NetcastEdge}. These systems motivate optical-electronic partitioning, but their physical latency and energy measurements do not transfer to our passive free-space simulator.

## Robustness of diffractive and physical neural networks

Robust diffractive training has several established forms. Misalignment-resilient networks sample lateral and axial offsets during training \cite{Mengu2020MisalignmentResilient}. Random phase dropout regularizes active diffractive models through sampled phase masks \cite{Xiao2021RandomPhaseDropout}. Phase filtering, parallel subnetworks, and random misalignment training improve tolerance to multi-plane alignment errors in imaging systems \cite{Wang2025PhaseFilteredD2NN,Wang2026ParallelSubnetworkD2NN}. Heterogeneous diffractive-neuron sizes provide a structural route that avoids conventional perturbation vaccination \cite{Zhu2026HybridRobustODNN}. At a broader physical-network level, sharpness-aware training optimizes deployment robustness without fixing one error distribution in advance \cite{Xu2026SharpnessAwarePNN}. Normalized cutoff frequency offers a complementary criterion for predicting robustness from propagated spatial-frequency content \cite{Wang2026NormalizedCutoffRobustness}. Our perturbation-aware D2NN uses the established vaccination strategy; the contribution lies in its bounded comparison against clean and hybrid alternatives, not in inventing perturbation injection.

## Quantization, coherence, and reproducible simulation

Phase-limited quantization-aware training directly optimizes D2NNs for discrete phase levels \cite{Wang2025PLQAT}, while two-level optical encoding has supported minimalist diffractive architectures \cite{Liu2025MinimalistDONN}. Spatial-coherence simulations show that fully coherent illumination forms a specific modeling boundary rather than a universal optical condition \cite{Filipovich2024SpatialCoherence}. LightRidge provides an open end-to-end framework for differentiable optical modeling and architecture exploration \cite{Li2023LightRidge}. Our study evaluates post-training phase quantization under a fully coherent scalar propagation model; it does not implement quantization-aware training or partial-coherence propagation.

## Positioning

This work contributes a protocol-bound numerical comparison rather than a new optical component or physical prototype. The same MNIST split, training seeds, frozen deployment draws, artifact hashes, and seed-level statistics connect baseline D2NN, perturbation-aware D2NN, hybrid optical-electronic, and electronic inference. This structure exposes where robustness gains hold, where they reverse, and which claims remain outside simulation.

# Methods and Experimental Protocol

## Protocol-bound task and model cohort

We formulated MNIST classification as a ten-class mapping from a grayscale image $x \in [0,1]^{1\times28\times28}$ to class scores. The study compared four models under matched data splits, training budgets, and seeds. The baseline D2NN used clean training and detector-region readout. The robust D2NN sampled optical perturbations during training but retained the same readout. The hybrid model coupled a clean-trained D2NN front end to an electronic classifier. The electronic baseline used an MLP without optical propagation. Optical perturbations do not apply to the electronic architecture, so it entered only the clean evaluation.

## Scalar diffraction defines the optical front end

The optical models embedded each input as an amplitude field on a $64\times64$ plane. Three trainable phase layers then transformed the complex field, following the D2NN formulation \cite{Lin2018AllOpticalD2NN}. Layer $l$ stored a phase profile $\phi_l$. The simulator first applied optional uniform phase quantization $Q$ and then added phase noise $\epsilon_l$:

$$
\widetilde{\phi}_l = Q(\phi_l) + \epsilon_l, \qquad
T_l = \exp(i\widetilde{\phi}_l).
$$

This order matches the frozen evaluation code. A sampled lateral displacement translated $T_l$ without wraparound. Locations exposed outside the shifted phase plate received unit complex transmission.

We propagated the field with a scalar Rayleigh-Sommerfeld kernel. For propagation distance $d$, transverse offsets $(\Delta x,\Delta y)$, sampling pitch $\Delta$, and wavelength $\lambda$, we used

$$
h_d(\Delta x,\Delta y) =
-i\frac{\Delta^2}{\lambda}\frac{d}{r^2}\exp(ikr),
\qquad
r=\sqrt{\Delta x^2+\Delta y^2+d^2},
\qquad
k=\frac{2\pi}{\lambda}.
$$

The implementation evaluated the corresponding linear convolution with FFTs. We set $\lambda=0.75$ mm and $\Delta=0.4$ mm. We set the nominal input, inter-layer, and output propagation distances to 30 mm. These values define a numerical simulation configuration rather than a calibrated manufactured system.

We computed the output intensity as $I=|u|^2$. Ten fixed, ring-arranged detector masks $M_c$ produced class energies

$$
s_c=\sum M_c I.
$$

The baseline and robust D2NNs predicted the class with the largest detector energy. We defined their cross-entropy logits as

$$
z_c=\log\left(\frac{s_c}{\sum_j s_j+\varepsilon}+\varepsilon\right).
$$

## The electronic head preserves a wider optical representation

The hybrid model replaced the ten detector energies with an $8\times8$ adaptive average-pooled intensity map. It flattened the resulting 64 features and divided them by their per-sample mean. A $64\rightarrow32\rightarrow10$ MLP with a GELU hidden activation generated logits, and a softmax generated class scores. This interface separated the simulated optical front end from the electronic decision stage and exposed the representation dimension for comparison.

The electronic baseline processed the original image through a $784\rightarrow18\rightarrow10$ MLP with a GELU hidden activation. It provided a parameter-transparent clean-classification reference without an optical front end. We did not apply optical or detector perturbations to this model.

![Numerical inference and training paths.](figures/figure_1_method_overview.png)

**Figure 1 | Numerical inference paths and perturbation-aware training.** **a,** The baseline and robust D2NNs use detector-region energies, while the hybrid model feeds pooled intensity features to an electronic head. The electronic baseline bypasses optical propagation. **b,** The robust D2NN samples displacement, gap, phase, and detector perturbations during training. Baseline and hybrid training use the clean path, and phase quantization enters only the frozen evaluation. All optical operations represent scalar numerical simulation.

## Perturbation-aware training samples deployment variation

Each deployment draw sampled independent layer perturbations. Horizontal and vertical displacements followed symmetric uniform distributions over their configured pixel ranges. Inter-layer distance errors followed symmetric uniform distributions over their configured metric ranges. The final phase layer had no following inter-layer distance error. Phase noise and relative detector noise followed zero-mean Gaussian distributions.

Uniform phase quantization used $L\geq2$ levels over the wrapped $2\pi$ interval. The simulator added detector noise after optical measurement. It scaled the noise by each sample's mean detector feature and clipped the resulting measurements at zero. The same detector-noise rule applied to detector energies and hybrid pooled features.

We optimized the composite classification objective

$$
\mathcal{L}=\alpha\mathcal{L}_{\mathrm{MSE}}
+\beta\mathcal{L}_{\mathrm{CE}}
+\gamma R_{\mathrm{phase}}.
$$

$\mathcal{L}_{\mathrm{MSE}}$ compared normalized class scores with one-hot labels. $\mathcal{L}_{\mathrm{CE}}$ acted on model logits. $R_{\mathrm{phase}}$ averaged squared circular phase differences along both spatial axes. We fixed $\alpha=1$, $\beta=0.1$, and $\gamma=0.01$.

The baseline D2NN and hybrid model used clean training. The robust D2NN jointly sampled a maximum lateral shift of 0.5 pixels, a maximum inter-layer distance error of $10^{-4}$ m, phase noise with a 0.05-rad standard deviation, and relative detector noise with a 0.01 standard deviation. The robust-training distribution did not include phase quantization.

## Frozen data and training protocol

We used the 60,000 training images and 10,000 test images distributed by torchvision for MNIST. For each training seed, a deterministic `random_split` divided the training side into 55,000 training and 5,000 validation images. The test set entered only the frozen post-selection evaluations.

Every formal run used three epochs, batch size 128, learning rate 0.01, `num_workers=0`, and deterministic PyTorch algorithms. Seeds 42, 43, and 44 defined three independently trained models per method. We selected checkpoints lexicographically by clean validation accuracy, detector contrast, and later epoch. We fixed this rule before formal training.

## One plan controls all optical robustness evaluations

The three optical-compatible methods shared one frozen evaluation plan. It contained 14 conditions and 34 deployment draws covering clean inference, lateral displacement, inter-layer distance error, phase noise, relative detector noise, phase quantization, and two mixed-stress conditions. Every draw processed the complete 10,000-image test set. We identified the plan as `robustness-plan-a8120eb01ea7093419fe`; its file had SHA-256 `de04e6f7e86b216cc406228c3f2180e03b8b47d17ba3c0b3ab63d1764b03fc71`.

Each draw retained sample-level predictions, an integer confusion matrix, accuracy, macro-F1, mean cross-entropy, and mean detector contrast. We computed macro-F1 over all ten score classes and assigned zero to a class with a zero F1 denominator. A separate clean-only evaluator processed the electronic baseline under the same complete test set.

## Training seed is the independent statistical unit

We treated each independently trained model, indexed by training seed, as one independent unit. We first averaged deployment draws within a seed and then summarized seeds 42, 43, and 44. Cross-method contrasts paired models with the same seed. We reported seed-level values, means, standard deviations, and two-sided 95% Student-t confidence intervals.

The intervals describe marginal uncertainty across three training seeds. They do not provide simultaneous coverage across the 14 conditions, and three observations cannot support a reliable normality assessment. We performed no hypothesis tests and reported no p values or multiple-comparison correction. Test images, deployment draws, and timing repeats did not count as independent replicates.

## CPU measurements quantify software simulation overhead

The formal clean evaluator fixed PyTorch intra-operation threads to one. It performed one complete-test-set warmup followed by five complete-test-set `forward_with_metrics` measurements. Each measurement included DataLoader iteration and forward-output shape validation. It excluded metric aggregation and artifact writes. This scope differs from the earlier end-to-end pilot used only to budget deployment draws.

We retained all repeat durations and summarized their median and interquartile range within each seed. Cross-seed summaries used the per-seed medians. These measurements quantify fixed-environment CPU software simulation overhead. They do not measure single-image latency, physical optical propagation, edge-device latency, power, throughput, or energy efficiency.

## Artifact identities preserve reproducibility

Every formal training run retained its checkpoint, adjacent training manifest, standard-output and standard-error logs, and SHA-256 identities. Every clean or robustness evaluation retained its manifest, sample-level JSONL predictions, summary metrics, applicable timing records, and source hashes. Summary generation verified artifact sizes, hashes, protocol identity, and source identity before aggregation. We registered every reported numerical result in `EVIDENCE_LEDGER.md`; interrupted runs and non-comparable environments remained visible as excluded artifacts.

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

# Discussion

## Robustness gains follow the perturbation family rather than the method label

The frozen evaluation shows that perturbation-aware training redistributes performance instead of uniformly improving it. The robust D2NN sacrificed 1.33 clean percentage points and remained below the baseline under gap, phase, and detector noise. It recovered accuracy under the stronger lateral shift, mixed stress, and coarse phase quantization. This pattern agrees with the established role of training-time perturbation sampling in alignment resilience \cite{Mengu2020MisalignmentResilient}, while narrowing the claim to the distributions and optical model evaluated here.

The four-level quantization result requires the most caution. The robust D2NN improved its mean by 17.91 percentage points over the baseline, although the interval remained wide across three seeds. Quantization was not included in robust training, so distribution matching alone cannot explain this result. Stochastic optical perturbations could favor phase configurations that tolerate coarse discretization, but the present experiments did not measure phase-spectrum smoothness, normalized cutoff frequency, or loss-landscape sharpness. Phase-filtered networks, normalized-cutoff-frequency analysis, and sharpness-aware training provide concrete tests for this candidate explanation \cite{Wang2025PhaseFilteredD2NN,Wang2026NormalizedCutoffRobustness,Xu2026SharpnessAwarePNN}.

## The electronic head broadens the useful optical representation

The hybrid model led the mean accuracy in 13 of 14 optical-compatible conditions. Its 64 pooled-intensity values preserve more spatial information than the ten detector-region energies used by the pure D2NN. An electronic head can therefore learn class boundaries that the fixed detector partition cannot express. This interpretation is consistent with prior jointly optimized optical-electronic classifiers \cite{Mengu2020DiffractiveIntegration,Chang2018HybridOpticalElectronicCNN}, but our experiment does not isolate pooling resolution, head depth, or feature normalization. A component ablation must test whether the gain comes from representation width, nonlinear electronic processing, or their interaction.

Extreme phase quantization also defines the hybrid boundary. At four phase levels, the robust D2NN had a higher mean than the hybrid model, but the paired interval could not resolve their order. The hybrid seed estimates varied enough to produce an unbounded t interval beyond 100% accuracy. The instability indicates sensitivity to the learned optical features and prevents a general claim that an electronic head compensates every optical distortion.

## Model counts do not establish edge-device efficiency

The hybrid model adds 2,410 trainable parameters to the D2NN and passes 64 values across the optical-electronic boundary. These counts make the computational partition explicit, but they do not determine bandwidth, memory traffic, conversion cost, or energy. Photonic edge systems can achieve hardware benefits through architectures such as remote weight streaming or reconfigurable optical processing \cite{Sludds2022NetcastEdge,Zhou2021ReconfigurableDPU}. Our simulator does not implement those devices, so their measured performance cannot validate our timing results.

The CPU measurements reinforce this boundary. Optical-compatible models required about one minute for a complete software-simulated test-set forward, while the electronic model required less than one second. This difference measures numerical propagation cost in one software environment. It cannot estimate physical propagation time, sensor conversion, electronic-head latency, device throughput, power, or energy efficiency.

## Evidence boundaries and next experiments

Five constraints limit the conclusions. First, MNIST does not represent natural images or task diversity. Second, three training seeds provide limited precision, especially for extreme quantization. Third, the fully coherent scalar propagation model omits partial coherence, material dispersion, fabrication calibration, and detector-system details; coherence alone can change diffractive-network behavior \cite{Filipovich2024SpatialCoherence}. Fourth, the study compares matched in-repository variants rather than reimplementing phase-filtered, structural-hybrid, sharpness-aware, or quantization-aware methods \cite{Wang2025PLQAT,Wang2025PhaseFilteredD2NN,Zhu2026HybridRobustODNN,Xu2026SharpnessAwarePNN}. Fifth, no physical optical platform or edge device validates latency, power, or energy claims.

The next evidence milestone should expand the seed cohort and repeat the frozen comparison on Fashion-MNIST and a grayscale natural-image benchmark. A second milestone should add quantization-aware training, normalized-cutoff-frequency analysis, and pooling/head ablations without changing the current result set. Physical validation would then require measured alignment, detector noise, conversion latency, and energy under a documented optical platform. Until those experiments exist, the paper's contribution remains a reproducible numerical benchmark of robustness and optical-electronic partitioning.

# Conclusion

This study establishes a protocol-bound numerical comparison of baseline D2NN, perturbation-aware D2NN, hybrid optical-electronic, and electronic inference. The central result is a separation between condition-specific robustness and readout capacity. Perturbation-aware training trades 1.33 clean percentage points for gains under stronger displacement, mixed stress, and coarse phase quantization. The hybrid model gains 6.06 clean points over detector-region classification and leads the mean accuracy in 13 of 14 optical-compatible conditions.

The evidence also defines where these findings stop. Four-level phase quantization produces wide seed-level uncertainty and does not resolve the hybrid-versus-robust ranking. Three seeds and MNIST cannot support broad generalization. Fully coherent scalar propagation omits physical fabrication, calibration, partial coherence, device conversion, and measured detector behavior. CPU wall-clock values describe one software simulator and do not measure optical propagation time, edge latency, power, or energy efficiency.

The next decisive tests are larger seed cohorts, additional datasets, component ablations for the hybrid head, and direct comparisons with quantization-aware, phase-filtered, normalized-cutoff-frequency, and sharpness-aware methods. A later physical study must measure alignment, detector noise, conversion latency, and energy on a documented optical platform. Until then, the present contribution is a reproducible simulation benchmark that exposes robustness gains, reversals, and optical-electronic computation boundaries without converting numerical evidence into hardware claims.

# Data Availability

This study reused the publicly available MNIST dataset distributed through torchvision. The project records SHA-256 values for the eight downloaded MNIST files and for the deterministic training and validation index sets. Formal simulations generated sample-level predictions, confusion matrices, seed-level summaries, figure source tables, evaluation manifests, and integrity hashes.

The generated simulation outputs and figure source data currently remain in the project workspace. They have not yet received a persistent public repository identifier. Before submission, the authors must deposit the supporting outputs, source tables, metadata, and file manifest in a durable repository and replace `[DATA_REPOSITORY_DOI]` with the assigned identifier.

`AUTHOR_INPUT_NEEDED: DATA_REPOSITORY_DOI, repository name, dataset licence, and release version.`

# Code Availability

The analysis code records the frozen protocol, model definitions, training entry points, perturbation sampling, clean and robustness evaluation, seed-level aggregation, and figure generation. The current repository is local and has no verified public release or archival DOI. Before submission, the authors must publish a versioned release, archive it in a persistent repository, and replace `[CODE_REPOSITORY_URL]` and `[CODE_ARCHIVE_DOI]` with verified identifiers.

`AUTHOR_INPUT_NEEDED: CODE_REPOSITORY_URL, CODE_ARCHIVE_DOI, software licence, and release tag.`

# Ethics Statement

The study used a public benchmark and numerical simulation. It collected no new human-participant data, identifiable personal information, biological specimens, or physical-device measurements.

# Author Contributions

`AUTHOR_INPUT_NEEDED: author names and CRediT roles.`

# Funding

`AUTHOR_INPUT_NEEDED: funder names, grant numbers, or an explicit statement that the work received no specific funding.`

# Competing Interests

`AUTHOR_INPUT_NEEDED: competing-interest declaration from all authors.`

# Acknowledgements

`AUTHOR_INPUT_NEEDED: acknowledgements, if applicable.`

# References

The machine-readable bibliography is maintained in `../references.bib`.
