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
