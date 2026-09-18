# Terminology Ledger

| Canonical term | Definition / boundary | Avoid |
|---|---|---|
| diffractive deep neural network (D2NN) | Optical model with trainable phase layers and scalar Rayleigh-Sommerfeld propagation | optical neural network as a blanket term |
| baseline D2NN | `method_id=baseline_d2nn`; clean training, optical detector readout | standard hardware D2NN |
| robust D2NN | `method_id=robust_d2nn`; training-time sampled deployment perturbations | universally robust D2NN |
| hybrid model | `method_id=hybrid`; D2NN optical front end plus 8x8 pooled features and 32-unit electronic head | hybrid optical-electronic hardware system |
| electronic baseline | `method_id=electronic`; 784-18-10 MLP with no optical front end | edge-device measurement |
| clean-only evaluation | Zero optical, quantization and detector perturbations; electronic models are evaluated only here | clean hardware experiment |
| deployment draw | One deterministic realization of a frozen numerical perturbation condition | independent replicate |
| training seed | Independent trained model and the formal statistical unit | test-sample replicate, timing repeat |
| macro-F1 | Class-average F1 over all 10 score classes; zero denominator maps to zero | weighted F1 unless explicitly requested |
| CPU software simulation overhead | Fixed-environment wall-clock timing of the numerical forward loop | physical optical latency, edge latency, power, energy efficiency |
| optical-compatible methods | baseline D2NN, robust D2NN and hybrid; the electronic baseline is excluded | four methods in the optical plan |
| numerical simulation only | Study boundary: no physical optical platform and no real edge-device measurement | fabricated, measured, deployed |
