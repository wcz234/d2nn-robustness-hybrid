# 论文架构与主张-证据映射（正式结果已冻结）

## One-sentence argument

在纯数值 MNIST 仿真中，我们用统一的扰动采样、光学/电子计算边界和 seed-level 统计协议比较 baseline D2NN、robust D2NN、hybrid 与 electronic；证据边界止于固定 CPU 软件仿真，不延伸到实体光学平台、真实边缘设备、功耗或能效。

## Planned sections

| Section | Claim job | Required evidence | Current status |
|---|---|---|---|
| Introduction | 边缘约束与 D2NN 部署扰动/计算分割的结构性问题 | 文献矩阵与最终结果预览 | final English draft complete |
| Related work | 区分 D2NN、错位感知训练、量化、混合光电和 NCF 判据 | `references.bib`, `LITERATURE_MATRIX.md`, DOI/URL ledger | English draft complete; 18 citation keys verified |
| Methods | 定义传播模型、扰动、hybrid/electronic 结构和 loss | `METHODS_AND_EXPERIMENTS_DRAFT.md`, source hashes | Chinese and English drafts complete; implementation cross-check complete |
| Experimental design | 固定 MNIST split、4 methods、3 seeds、14 conditions/34 draws | `FORMAL_EXPERIMENT_PROTOCOL.json`, frozen plan | verified |
| Results: clean baseline | 比较 clean accuracy/F1/CE/CPU overhead | 12 clean evaluation manifests and `summarize_clean.py` | complete; full summary and paired comparisons frozen |
| Results: robustness | 分解扰动退化并比较 robust training | 9 optical/hybrid robustness manifests + summary | complete; 14 conditions/34 draws frozen |
| Results: representation/complexity | 比较 trainable parameters, pooled feature size, head parameters | checkpoint manifests + deterministic complexity source data | complete; manuscript Figure 3 source data and exports generated |
| Results: class-level errors | 展示四类方法在干净 MNIST 条件下的类别混淆结构 | 12 clean evaluation manifests and confusion source data | complete; manuscript Figure 4 source data and exports generated |
| Discussion | 解释有限 seed 仿真证据与硬件边界 | frozen summary, failure logs, limitations | English draft complete |
| Conclusion | 只保留 summary 支持的 bounded claims | all metrics in `EVIDENCE_LEDGER.md` | English draft complete |

## Evidence gates

1. Do not report a method-level result until its evaluation manifest, artifact hashes and source hashes pass verification.
2. Treat training seed as `n`; do not count test samples, perturbation draws or timing repeats as independent replicates.
3. Keep electronic clean-only results outside the optical plan and optical summary.
4. Describe all timing values as CPU software simulation overhead; never as physical optical latency, edge-device latency, power or energy efficiency.
5. Preserve the interrupted hybrid seed44 checkpoint and logs as a failure artifact; use only the retry1 checkpoint with its adjacent manifest after strict loading succeeds.

## Figure/table plan

| Artifact | Figure/table role | Source |
|---|---|---|
| Architecture schematic | Optical front end, detector features, electronic head and electronic baseline | code-defined model variants; non-data diagram |
| Protocol matrix | Methods, seeds, conditions, draw counts and metric boundaries | `FORMAL_EXPERIMENT_PROTOCOL.json` |
| Clean comparison | Accuracy/F1/CE and CPU overhead by training seed | clean summary JSONL after all 12 evaluations |
| Robustness grid | Accuracy/F1 across 14 conditions with within-seed draw aggregation | robustness summary JSONL |
| Complexity trade-off | Parameters, representation dimensionality and timing boundary | manifests and verified timing artifacts |
| Class-confusion structure | Clean-condition per-class error patterns across four methods | formal clean confusion matrices and `make_confusion_figure.py` |
| Failure/limitation panel | Missing/interrupted artifacts and simulation-only boundary | logs, ledger and discussion notes |
