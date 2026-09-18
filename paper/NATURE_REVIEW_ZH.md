# 投稿前模拟审稿报告

## Review setup

- **Input scope：** 完整中文正文、三张正式图、冻结实验协议、正式结果汇总、证据台账和统计审计。
- **Assessment boundary：** 研究只包含公开 MNIST 上的完全相干标量数值仿真，没有实体光学平台、真实边缘设备、功耗或能效测量。
- **Shared manuscript claim summary：** 统一协议下，扰动感知训练产生条件选择性的鲁棒性收益；混合电子后端扩展光学表示的读出能力；参数量和 CPU 仿真开销不能替代硬件证据。
- **Visible evidence base：** 四方法 × 三训练种子的干净评估，三类光学兼容方法 × 三训练种子的 14 条件鲁棒性评估，完整 10 000 样本测试集，逐样本预测、manifest、哈希和图源 CSV。
- **Missing materials affecting confidence：** 第二公开数据集、更多训练种子、组件消融、其他鲁棒训练方法的同协议实现，以及物理平台验证。

## Reviewer 1

### Overall assessment

论文的最大优点是证据边界清楚。作者没有把 deployment draw、测试图像或计时重复当作独立样本，也没有把 CPU 软件计时包装成光学硬件性能。统一协议揭示了鲁棒训练收益随扰动族变化，这一结果具有方法评估价值。然而，三个训练种子和单一 MNIST 任务不足以稳定支撑极端量化与广泛鲁棒性结论。

### Who would be interested in the results, and why

衍射神经网络、光子计算、物理神经网络鲁棒训练和光电协同推理研究者会关注该结果，因为它展示了常用扰动注入方法在哪些条件下获益、在哪些条件下反转，并提供了可复算的冻结评估结构。

### Major strengths

- 冻结 14 条件、34 deployment draw 的统一方案降低了事后选择空间。
- 训练种子作为独立单位，部署采样先在种子内聚合。
- 四级相位量化的宽区间被保留，没有裁剪为表面合理范围。
- 正式结果保留逐样本预测、混淆矩阵、代码与产物哈希。

### Major concerns

**R1-M1 — [statistical-rigor]**

- **Claim pointer：** 扰动感知训练在横向错位、混合扰动和四级相位量化下获得条件选择性收益。
- **Evidence pointer：** 第 4.3 节、图 2、统计审计。
- **Concern：** 每种方法只有三个训练种子。四级相位量化虽有 17.91 个百分点的均值差，但区间为 3.01 至 32.81 个百分点，说明效应量对种子高度敏感。
- **Resolution test：** 扩大训练种子队列，并在不覆盖当前冻结结果的前提下复现主要条件；若无法扩展，应继续把结论限定为当前种子队列中的描述性结果。

**R1-M2 — [experimental-design]**

- **Claim pointer：** 混合模型的优势来自更丰富的光学表示和电子分类头。
- **Evidence pointer：** 第 5.2 节。
- **Concern：** 当前设计同时改变池化表示宽度、电子非线性和判决方式，不能区分三者的独立贡献。
- **Resolution test：** 增加池化分辨率、线性与非线性电子头、参数量匹配读出的组件消融，或者把机制表述进一步降为与现有结果相容的候选解释。

### Technical failings that need to be addressed before the case is established

核心比较本身成立，但若论文要主张通用鲁棒优化机制，必须增加种子数量和组件消融。当前证据只建立了协议内的经验模式。

### Assessment against Nature-style criteria

- **Originality：** 协议约束比较和证据边界具有原创组织价值；扰动注入与混合后端本身不是新方法。
- **Scientific importance：** 对本领域有实用价值，但尚未证明具有广泛科学影响。
- **Interdisciplinary readership：** 主要面向光子计算和物理神经网络社区。
- **Technical soundness：** 主要结果可追溯且统计单位正确；外部有效性和机制识别有限。
- **Readability：** 结构清楚，仿真与硬件边界表达充分。

### Recommendation posture

适合作为有边界的数值方法比较稿进一步修改；若定位为广泛影响力期刊，当前技术证据仍不足。

## Reviewer 2

### Overall assessment

论文诚实地区分了已有技术和本文贡献。真正的贡献不是提出新型 D2NN，而是将基准、扰动感知训练和混合读出置于同一可审计协议中。该定位可信，但创新性容易被审稿人判断为基准化工作，尤其是在只使用 MNIST 且没有复现更近期对照方法的情况下。

### Who would be interested in the results, and why

开发 D2NN 模拟器、鲁棒训练方法和混合光电分类器的研究者会将该工作作为实验设计与结果解释的参考。研究也可帮助工程团队避免把软件仿真时间误写成光学设备时延。

### Major strengths

- 相关工作覆盖扰动注入、相位滤波、量化感知、sharpness-aware 训练和混合光电网络。
- 创新表述没有宣称扰动注入或混合结构为本文首创。
- 电子基线只进入适用的干净条件比较，避免不恰当的光学扰动对照。

### Major concerns

**R2-M1 — [novelty-significance]**

- **Claim pointer：** 本文贡献是协议约束的鲁棒性与光电计算分割基准。
- **Evidence pointer：** 引言贡献列表、相关工作第 2.3 至 2.4 节。
- **Concern：** 统一协议具有价值，但当前只比较仓库内三类光学兼容方法，没有在同一代码和预算下实现相位滤波、量化感知训练或 sharpness-aware 等近期强对照。
- **Resolution test：** 至少增加一个近期强对照的同协议复现，或将论文定位明确收窄为基准、扰动注入与紧凑电子读出之间的受控案例研究。

**R2-M2 — [data-resource-quality]**

- **Claim pointer：** 该数值基准可用于分析边缘智能中的鲁棒性与光电分割。
- **Evidence pointer：** 摘要、结论、第 5.4 节。
- **Concern：** MNIST 的任务复杂度和输入统计与自然图像、传感器图像及实际边缘工作负载差异较大，限制了“边缘智能”定位的说服力。
- **Resolution test：** 在 Fashion-MNIST 和至少一个灰度自然图像或传感任务上重复冻结比较，或在标题和摘要中进一步突出“MNIST 数值基准”的范围。

### Technical failings that need to be addressed before the case is established

当前结果足以支持一个可复现 MNIST 数值基准，但不足以支持广泛的边缘智能方法结论。增加数据集和近期对照将显著提高论文竞争力。

### Assessment against Nature-style criteria

- **Originality：** 基准协议和证据审计有区别度，算法创新有限。
- **Scientific importance：** 目前属于领域内部的方法学价值。
- **Interdisciplinary readership：** 对光学计算以外读者的直接意义尚弱。
- **Technical soundness：** 结果链完整，但比较集合和数据范围偏窄。
- **Readability：** 论文能够让非专业读者理解主要权衡，但边缘应用动机强于实际任务证据。

### Recommendation posture

作为专业领域中文期刊稿具有基础；面向高影响力综合期刊时，创新性和外部验证尚未建立。

## Reviewer 3

### Overall assessment

稿件的叙事重点是“哪些结论可以由仿真支持，哪些不能”。这一立场清楚且负责任。图 1 使模型边界易于理解，图 2 展示条件选择性，图 3 阻止读者误解软件计时。不过，标题中的“面向边缘智能”容易让非专业读者期待真实设备或至少更接近边缘场景的数据，而正文实际是 MNIST 数值研究。

### Who would be interested in the results, and why

跨学科读者可能对“光学前端与电子后端如何分工”这一问题感兴趣。可复算的证据链也对研究可重复性和科学报告规范具有示范意义。

### Major strengths

- 摘要直接披露 MNIST、三个训练种子和数值仿真边界。
- 图 1 至图 3 均使用中文标签，并在图注中定义证据范围。
- 讨论没有把相容性解释写成已验证机制。

### Major concerns

**R3-M1 — [claim-moderation]**

- **Claim pointer：** 研究面向边缘智能，并讨论混合光电推理边界。
- **Evidence pointer：** 标题、摘要、第 5.3 至 5.4 节。
- **Concern：** 没有真实边缘设备、通信链路、传感器转换或自然图像任务时，“面向边缘智能”主要是应用动机而非已验证场景。
- **Resolution test：** 目标期刊确定后，在标题或副标题中加入“MNIST 数值仿真基准”，或者增加接近边缘任务的数据和系统级指标。

**R3-M2 — [writing-clarity]**

- **Claim pointer：** 论文旨在让非专业读者区分鲁棒性、读出能力和硬件性能。
- **Evidence pointer：** 摘要、引言前三段、图 1。
- **Concern：** `checkpoint`、`manifest`、`logits` 和 deployment draw 等术语对中文综合读者仍偏技术化，首次出现时缺少统一中文释义。
- **Resolution test：** 目标期刊排版时增加术语表或在首次出现处给出中文释义，同时保留必要英文标识以便复现。

### Technical failings that need to be addressed before the case is established

稿件的可读性足以支持专业期刊审阅，但综合读者所期待的应用证据与当前研究范围不匹配。标题定位和任务范围需要进一步对齐。

### Assessment against Nature-style criteria

- **Originality：** 证据边界和协议组织有新意，核心模型组件沿用已有思路。
- **Scientific importance：** 可重复性示范明确，但影响范围尚属专业领域。
- **Interdisciplinary readership：** 光电计算分割具有跨领域潜力，当前 MNIST 证据不足以激活广泛兴趣。
- **Technical soundness：** 报告严谨，未观察到硬件外推或数据夸大。
- **Readability：** 主线清楚；少量英文工程术语增加阅读门槛。

### Recommendation posture

支持在专业领域期刊中经过定位与术语修订后评审；广泛影响力主张目前尚未建立。

## Cross-review synthesis

### Consensus strengths

- 三位审稿人均认可冻结协议、训练种子级统计和完整证据链。
- 三位审稿人均认可论文主动限制硬件、功耗、时延和泛化主张。
- 混合模型与鲁棒 D2NN 的结果没有被写成无条件优势。

### Consensus technical risks

- **small-seed-generalization：** Reviewer 1 和 Reviewer 2 均认为三个训练种子限制结论稳定性与外部有效性。
- **scope-vs-edge-positioning：** Reviewer 2 和 Reviewer 3 均认为 MNIST 数值仿真不足以支撑宽泛的边缘智能定位。
- **novelty-vs-baseline-scope：** Reviewer 1 和 Reviewer 2 均指出当前贡献更接近受控比较与基准，而非新型鲁棒算法。

### Where emphasis differs across reviewers

- Reviewer 1 强调统计精度、机制消融和实验设计。
- Reviewer 2 强调近期强对照、数据集范围和创新性定位。
- Reviewer 3 强调标题承诺、跨学科可读性和中文术语。

### Broad-interest / significance readout

当前稿件对衍射神经网络和光子计算社区有清晰价值，但尚未显示立即且广泛的跨学科影响。其最有说服力的贡献是可审计的数值比较和结论边界，而不是新器件、新训练原理或真实边缘系统性能。

### Most important issues to resolve before a strong Nature-style case is established

1. 增加训练种子和至少一个额外公开数据集。
2. 增加混合后端组件消融及一个近期强鲁棒方法的同协议对照。
3. 对齐“边缘智能”标题承诺与实际证据范围。
4. 如主张真实部署价值，补充物理平台或真实设备测量；在此之前保持纯数值基准定位。

## Risk / unsupported claims

- 不支持将 CPU 软件计时解释为物理传播时延、设备延迟、吞吐、功耗或能效。
- 不支持将 13/14 条件的均值领先解释为经过多重比较校正的逐条件优势。
- 不支持将四级相位量化结果解释为稳定硬件容差。
- 不支持声称混合电子后端可补偿所有光学畸变。
- 不支持声称当前结果能够泛化到自然图像、真实边缘任务或实体光学平台。
- 当前材料足以评估数值研究本身，但目标期刊适配、作者声明和公开仓库仍未完成。
