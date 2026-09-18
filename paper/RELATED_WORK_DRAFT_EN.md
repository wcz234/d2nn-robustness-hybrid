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

