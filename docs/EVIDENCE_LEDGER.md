# 证据台账

本文件是论文事实与数字的唯一登记入口。状态只允许使用：`待获取`、`已获取未核验`、`已核验`、`不适用`。正文引文按“BibTeX key → `references.bib` → `LITERATURE_MATRIX.md` → 本文件 B 表 DOI/URL”追溯。

## A. 本地证据现状

| 证据项 | 状态 | 权威来源 | 可支持的论文内容 | 备注 |
|---|---|---|---|---|
| 原始数据 | 已核验 | torchvision MNIST，文件位于第三方基线数据目录 | 数据集与冒烟实验设置 | 完整 60,000/10,000 MNIST；正式划分仍需独立固化 |
| 数据校验和 | 已核验 | `data/MNIST_SHA256SUMS.txt` 与 `simulator/data/MNIST/raw/` | 数据完整性 | 2026-08-01 独立重算 8 个原始/压缩文件 SHA-256 后全部对账；校验清单 SHA-256 `5fca1cd7423a425bbf2762aee4b1ce5599f3e3c8620e658d7dc47c9551f1be5c`；修正了登记文件中一个 65 字符的转录错误 |
| 数据划分清单 | 已核验 | `artifacts/baseline_pilot_thz/best_mnist.cpu_pilot_thz_64x3_seed42.json` | 固定训练/验证划分 | manifest 已记录 55,000/5,000 划分及训练、验证索引 SHA-256 |
| 机器可读书目 | 已核验 | `references.bib` | 正文 BibTeX key、书目信息与 DOI/URL | 19 个条目已与文献矩阵和本表 DOI 对账 |
| 文献证据矩阵 | 已核验 | `LITERATURE_MATRIX.md` | 外部方法、证据属性与论点边界 | 19 个引用条目的 key/DOI 已与 `references.bib` 一致 |
| 引用支撑等级审计 | 已核验 | `results/citation_audit/nature_citation_zh_v2/DIRECT_CITATIONS.md` | Nature 系列来源的支撑等级与误命中处置 | `nature-citation` 2.0.0 检索 19 条主张、24 条候选；19 条判定为题名相关但主题无关，1 条经摘要核验后采用（EXT-032），3 条确认范围内无候选 |
| D2NN 基线代码 | 已核验 | `third_party/nonlinear-d2nn` @ `f1ae6e3`；`simulator/tests/test_d2nn_core.py` | 方法与基线 | MIT；180 项核心测试按每组 20 项通过；整文件两次触发 60 秒硬超时，未观察到断言失败 |
| 鲁棒训练代码 | 已核验 | `simulator/perturbations.py`、`simulator/training_runtime.py` 及对应测试 | 鲁棒性优化方法 | 支持横向错位、层间距误差、相位噪声、相位量化和合成探测器噪声；正式 3-seed 鲁棒性结果已冻结 |
| 混合光电推理代码 | 已核验 | `simulator/model_variants.py` 及对应测试 | 系统设计 | 光学传播前端 + 池化探测特征 + 小型电子 MLP；正式 clean 与 robustness 模型比较已冻结 |
| 鲁棒性评估与统计管线 | 已核验 | `simulator/evaluate_robustness.py`、`simulator/summarize_robustness.py` 及对应测试 | 冻结评估计划、逐样本预测、哈希审计、按训练 seed 汇总和方法配对 | 计划强制固定采样/统计语义；draw 先在 seed 内聚合；预注册正式比较最低 3 个配对训练 seed；全套验证当前 `269 passed, 10 subtests passed` |
| Clean-only 评估管线 | 已核验 | `simulator/evaluate_clean.py`、`simulator/clean_evaluation.py`、`simulator/clean_evaluation_manifest.py` 及 `tests/test_clean_evaluation_cli.py` | 电子模型与 optical/hybrid 方法的 clean 指标、逐样本预测、固定 CPU 软件仿真计时 | 电子模型不进入 optical robustness plan；计时重复不是独立实验重复；四方法 12 个正式 clean 结果已生成并核验 |
| 正式 MNIST v1 实验协议 | 已核验 | `FORMAL_EXPERIMENT_PROTOCOL.json` | 正式方法、seed、训练预算、扰动条件、draw 数与统计边界 | SHA-256 `0c1d48d864fc02bbea64ae53c59d92f921cadf2edaee2cb6b73462a0bf386fe1`；2026-07-29 正式训练前冻结，后续不得静默覆盖 |
| 正式共享 robustness plan | 已核验 | `results/formal_mnist_v1/robustness_plan.json` | baseline D2NN、robust D2NN 与 hybrid 三类 optical-compatible 方法的统一 14 条件、34 draws、完整 10,000 测试集和跨方法配对身份 | `plan_id=robustness-plan-a8120eb01ea7093419fe`；文件 SHA-256 `DE04E6F7E86B216CC406228C3F2180E03B8B47D17BA3C0B3AB63D1764B03FC71`；仅为冻结计划，不是实验结果；electronic 只进入独立 clean-only 评估 |
| 正式 checkpoint 集合 | 已核验 | `artifacts/formal_mnist_v1/` 与训练日志 | 四方法 × 三训练 seed 的模型与相邻 manifest | baseline D2NN 3/3、robust D2NN 3/3、hybrid retry1 3/3、electronic 3/3；原 `hybrid_seed44.pth` 的 OOM 中断 artifact 和日志保留但不纳入正式 cohort；有效套件 12/12 |
| Checkpoint | 已核验 | `artifacts/baseline_pilot_thz/best_mnist.cpu_pilot_thz_64x3_seed42.pth` | 复现实验 | SHA-256 `571e87ca47319575e7457a0671c6e7ca9597540d465689991d64d00a861b683d`；单种子 pilot |
| 光学尺度诊断 | 已核验 | `results/diagnostics/optical_scale_64x3_seed42.json` | 数值尺度选择与负面诊断 | 仅支持固定合成输入下的仿真信号尺度比较，不支持硬件性能结论 |
| 工程 smoke 逐样本预测 | 已核验 | `results/robustness_smoke/pilot_mnist_003_subset32_mixed_schema_v2/predictions.jsonl` | 评估链路、混淆矩阵与批划分无关性核验 | 32 个测试前缀样本 × 2 个 draw，共 64 行；仅 smoke，禁止作为论文主结果 |
| 正式逐样本预测 | 已核验 | `results/formal_mnist_v1/clean/` 与 `results/formal_mnist_v1/robustness/` | 准确率、混淆矩阵和指标复算 | 四方法 12 个 clean 目录及三类光学兼容方法 9 个 robustness 目录均保存正式逐样本预测；非可比诊断目录不进入正式 cohort |
| 重复运行日志 | 已核验 | 正式训练、clean evaluation 与 robustness evaluation manifest | 均值、标准差、置信区间 | 每种方法三个独立训练 seed；部署 draw 与计时重复不作为独立重复 |
| 鲁棒性扫描结果 | 已核验 | `results/formal_mnist_v1/robustness_summary/` | 噪声、错位、量化敏感性 | 14 条件、34 deployment draws、完整 10,000 测试集；正式 summary hash 已登记 |
| CPU 延迟日志 | 已核验 | `results/formal_mnist_v1/clean/*/cpu_software_timing.json` | 固定环境下的软件仿真开销 | 四方法 12 个正式 clean 目录均包含固定 1 线程、1 次 warmup、5 次完整测试集 forward；禁止写成真实边缘设备或光学硬件性能 |
| 功耗或能效数据 | 不适用 | — | 不进入本论文实验结论 | 无真实设备测量；禁止估算或虚构为本文结果 |
| 边缘设备测量 | 不适用 | — | 不进入本论文实验结论 | 研究范围锁定为纯仿真 |
| 光学原型原始数据 | 不适用 | — | 不进入本论文实验结论 | 研究范围锁定为纯仿真 |

## B. 外部来源登记要求

每条论文或开源项目在采用前必须补齐以下字段。正文实际引用的论文还必须同时存在于 `references.bib` 和 `LITERATURE_MATRIX.md`，并以相同 DOI/URL 对齐：

| ID | 类型 | 题名/仓库 | DOI/URL | 作者/维护者 | 年份/版本 | 许可证 | 核验状态 | 用途 |
|---|---|---|---|---|---|---|---|---|
| EXT-001 | 论文 | All-optical machine learning using diffractive deep neural networks | https://doi.org/10.1126/science.aat8084 | Lin et al. | 2018 | 不适用 | 已核验 | D2NN 奠基工作 |
| EXT-002 | 论文 | Analysis of Diffractive Optical Neural Networks and Their Integration With Electronic Neural Networks | https://doi.org/10.1109/JSTQE.2019.2921376 | Mengu et al. | 2020 | 不适用 | 已核验 | 混合光电推理基线 |
| EXT-003 | 论文 | Misalignment resilient diffractive optical networks | https://doi.org/10.1515/nanoph-2020-0291 | Mengu et al. | 2020 | 不适用 | 已核验 | 随机错位感知训练 |
| EXT-004 | 论文 | Class-specific differential detection in diffractive optical neural networks improves inference accuracy | https://doi.org/10.1117/1.AP.1.4.046001 | Li et al. | 2019 | 不适用 | 已核验 | 差分探测读出 |
| EXT-005 | 论文 | Hybrid optical-electronic convolutional neural networks with optimized diffractive optics for image classification | https://doi.org/10.1038/s41598-018-30619-y | Chang et al. | 2018 | 不适用 | 已核验 | 光学前端与电子后端联合优化 |
| EXT-006 | 论文 | Large-scale neuromorphic optoelectronic computing with a reconfigurable diffractive processing unit | https://doi.org/10.1038/s41566-021-00796-w | Zhou et al. | 2021 | 不适用 | 已核验 | 可重构混合光电计算背景 |
| EXT-007 | 论文 | Real-time multi-task diffractive deep neural networks via hardware-software co-design | https://doi.org/10.1038/s41598-021-90221-7 | Li et al. | 2021 | 不适用 | 已核验 | 硬软协同与器件噪声背景 |
| EXT-008 | 论文 | Delocalized photonic deep learning on the internet's edge | https://doi.org/10.1126/science.abq8271 | Sludds et al. | 2022 | 不适用 | 已核验 | 边缘光子推理动机；仅引用文献报告值 |
| EXT-009 | 开源项目 | nonlinear-d2nn | https://github.com/yeungmkw/nonlinear-d2nn | yeungmkw | commit f1ae6e3 | MIT | 已核验 | 当前可运行 D2NN 训练骨架；所有指标独立重跑 |
| EXT-010 | 开源项目 | torchoptics | https://github.com/MatthewFilipovich/torchoptics | Matthew Filipovich | commit 34fe9c4 | MIT | 已核验 | 可微光传播交叉验证 |
| EXT-011 | 论文 | Optical random phase dropout in a diffractive deep neural network | https://doi.org/10.1364/OL.428761 | Xiao et al. | 2021 | 不适用 | 已核验 | 随机相位正则化参考 |
| EXT-012 | 论文 | Phase-limited quantization-aware training for diffractive deep neural networks | https://doi.org/10.1364/AO.548035 | Wang et al. | 2025 | 不适用 | 已核验 | 相位受限量化感知训练参考 |
| EXT-013 | 论文 | Role of spatial coherence in diffractive optical neural networks | https://doi.org/10.1364/OE.523619 | Filipovich et al. | 2024 | 不适用 | 已核验 | 空间相干性与传播假设边界 |
| EXT-014 | 论文 | LightRidge: An End-to-end Agile Design Framework for Diffractive Optical Neural Networks | https://doi.org/10.1145/3623278.3624757 | Li et al. | 2023 | 不适用 | 已核验 | D2NN 设计框架与器件量化参考 |
| EXT-015 | 论文 | Minimalist Optical Neural Computing: Optical Diffractive Neural Network by 2-level Quantized Pixel-Wise Optical Encoding | https://doi.org/10.1002/lpor.202402303 | Liu et al. | 2025 | 不适用 | 已核验 | 二级量化光学编码；限定量化创新表述 |
| EXT-016 | 论文 | Misalignment resilient phase-filtered diffractive deep neural networks | https://doi.org/10.1364/OE.558279 | Wang and Zhou | 2025 | 不适用 | 已核验 | 相位滤波错位鲁棒方法；限定鲁棒性创新表述 |
| EXT-017 | 论文 | Robustness of parallel subnetwork-filtered diffractive deep neural networks | https://doi.org/10.1364/OE.598903 | Wang and Zhou | 2026 | 不适用 | 已核验 | 并行子网错位鲁棒方法；限定鲁棒性创新表述 |
| EXT-018 | 论文 | Design of task-specific optical systems using broadband diffractive neural networks | https://doi.org/10.1038/s41377-019-0223-1 | Luo et al. | 2019 | 不适用 | 已核验 | 波长扰动与宽带传播背景 |
| EXT-019 | 论文 | Noise Aware Design Enables Robust Diffractive Deep Neural Network Designs in Visible Wavelengths | https://doi.org/10.1109/IPC57732.2023.10360509 | Hettiarachchi et al. | 2023 | 不适用 | 已核验 | 制造噪声感知训练背景 |
| EXT-020 | 论文 | Compact high-robustness diffractive neural network chip for water-immersed optical inference | https://doi.org/10.3788/COL202422.120002 | Luan et al. | 2024 | 不适用 | 已核验 | 环境域变化鲁棒性背景 |
| EXT-021 | 论文 | Compact eternal diffractive neural network chip for extreme environments | https://doi.org/10.1038/s44172-024-00211-6 | Dong et al. | 2024 | 不适用 | 已核验 | 极端环境与二值相位背景 |
| EXT-022 | 论文 | Space-efficient optical computing with an integrated chip diffractive neural network | https://doi.org/10.1038/s41467-022-28702-0 | Zhu et al. | 2022 | 不适用 | 已核验 | 集成光学计算的空间/能效动机；仅引用文献值 |
| EXT-023 | 论文 | LOEN: Lensless opto-electronic neural network empowered machine vision | https://doi.org/10.1038/s41377-022-00809-5 | Shi et al. | 2022 | 不适用 | 已核验 | 无透镜混合光电机器视觉背景 |
| EXT-024 | 论文 | 2bit Nonlinear Diffractive Deep Neural Network (2bit ND2NN) | https://doi.org/10.1016/j.optlastec.2024.111120 | Sun et al. | 2024 | 不适用 | 已核验 | 2-bit 非线性 D2NN；限定量化创新表述 |
| EXT-025 | 论文 | An optical neural network using less than 1 photon per multiplication | https://doi.org/10.1038/s41467-021-27774-8 | Wang et al. | 2022 | 不适用 | 已核验 | 低光子光计算动机；不得外推为本文能耗 |
| EXT-026 | 论文 | Edge Intelligence: Paving the Last Mile of Artificial Intelligence With Edge Computing | https://doi.org/10.1109/JPROC.2019.2918951 | Zhou et al. | 2019 | 不适用 | 已核验 | 边缘智能定义、约束与云边协同背景 |
| EXT-027 | 论文 | Physical neural networks using sharpness-aware training | https://doi.org/10.1038/s41467-026-68470-9 | Xu et al. | 2026 | CC BY 4.0 | 已核验 | 通用物理神经网络鲁棒训练近邻；限制无需误差先验与通用鲁棒优化的创新性表述 |
| EXT-028 | 开源项目 | Sharpness-Aware-Training | https://github.com/cuhkhuangslab/Sharpness-Aware-Training | CUHK Huang Lab | commit `9e52734f8e5829dad14fe0c1f943eb03ad00536b` | MIT | 已核验 | EXT-027 作者代码；仅完成仓库、提交和许可元数据核验，尚未本地复现 |
| EXT-029 | 论文 | Hybrid structure for enhancing robustness in optical diffractive neural networks without vaccination training | https://doi.org/10.2738/foe.2026.0012 | Zhu et al. | 2026 | 不适用 | 已核验 | 异质衍射神经元结构鲁棒性；其 hybrid 概念不等同于本文的混合光电后端 |
| EXT-030 | 论文 | Robustness criterion for diffractive deep neural networks using normalized cutoff frequency | https://doi.org/10.1364/OE.601000 | Wang et al. | 2026 | 不适用 | 已核验 | 归一化截止频率鲁棒性判据；限制通用鲁棒性评估创新表述 |
| EXT-031 | 开源项目 | Robustness-Criterion-for-Optical-Diffractive-Neural-Networks-Using-Normalized-Cutoff-Frequency | https://github.com/Rainbowseaaa/Robustness-Criterion-for-Optical-Diffractive-Neural-Networks-Using-Normalized-Cutoff-Frequency | Rainbowseaaa | commit `ee984268af4e8b78022198037797cf8c71a44ee4` | 未检测到许可证 | 已核验 | EXT-030 关联代码与数据；只作方法参考，不复用代码 |
| EXT-032 | 论文 | Wavefront-aberration-tolerant diffractive deep neural networks using volume holographic optical elements | https://doi.org/10.1038/s41598-024-82791-z | Hoshi et al. | 2025 | 不适用 | 已核验 | 制造/波前像差类扰动的训练期自适应；补足扰动感知训练谱系（`nature-citation` 2.0.0 检索并经摘要核验，等级=强支撑） |

包含实体实验的外部来源仅可用于相关工作、研究动机或建模依据；其硬件延迟、吞吐、功耗和能效不得登记为本文 Metric ID，也不得与本地仿真结果混合比较。

## C. 论文数字登记模板

任何进入摘要、结论或图表的数字都必须先登记：

| Metric ID | 数值 | 单位 | 数据文件 | 生成命令 | 配置 | 随机种子 | 环境 | 核验人/时间 |
|---|---:|---|---|---|---|---|---|---|
| SMOKE-MNIST-001 | 10.17 | test accuracy (%) | `results/manifests/baseline_smoke_mnist_seed42.json` | `python third_party/nonlinear-d2nn/train.py --task classification --dataset mnist --epochs 1 --size 32 --layers 1 --batch-size 256 --seed 42 --rs-backend fft --num-workers 0 ...` | manifest 内嵌 | 42 | Python 3.11.7 / PyTorch 2.10.0+cpu / CPU | 2026-07-25；仅流水线冒烟，禁止作为主结果 |
| PILOT-MNIST-002 | 10.06 | test accuracy (%) | `results/manifests/baseline_pilot_mnist_seed42.json` | `python simulator/train.py --task classification --dataset mnist --epochs 1 --size 64 --layers 3 --batch-size 128 --seed 42 --rs-backend fft --num-workers 0 ...` | manifest 内嵌；启动时源码哈希见 `results/README.md` | 42 | Python 3.11.7 / PyTorch 2.10.0+cpu / CPU | 2026-07-25；负面尺度诊断，禁止作为主结果 |
| OPTICAL-SCALE-DIAG-001 | 1.814153e12 | THz/paper mean-output-intensity ratio | `results/diagnostics/optical_scale_64x3_seed42.json` | `python simulator/diagnose_optical_scale.py --size 64 --layers 3 --batch-size 10 --seed 42 --rs-backend fft --num-threads 1 ...` | 固定合成输入，每类 1 个样本；同初始化；完整配置见 JSON | 42/输入 43 | Python 3.11.7 / PyTorch 2.10.0+cpu / CPU | 2026-07-25；数值尺度诊断，不得外推为物理效率 |
| OPTICAL-SCALE-DIAG-002 | 7.849283e4 | THz/paper data-loss phase-gradient mean ratio | `results/diagnostics/optical_scale_64x3_seed42.json` | 同上 | 数据损失为 `alpha*MSE + beta*CE`，排除相位正则；保留 logits `1e-8` 下限 | 42/输入 43 | Python 3.11.7 / PyTorch 2.10.0+cpu / CPU | 2026-07-25；仅支持当前数值实现的训练信号诊断 |
| PILOT-MNIST-003 | 87.66 | test accuracy (%) | `artifacts/baseline_pilot_thz/best_mnist.cpu_pilot_thz_64x3_seed42.json` | `python simulator/train.py --task classification --dataset mnist --epochs 1 --size 64 --layers 3 --batch-size 128 --lr 0.01 --seed 42 --rs-backend fft --num-workers 0 --deterministic --optics-preset d2nn2018_thz --experiment-stage baseline_pilot --run-name cpu_pilot_thz_64x3_seed42 --save-dir ../artifacts/baseline_pilot_thz` | manifest 内嵌；0.75 mm / 0.4 mm / 30 mm 纯仿真尺度 | 42 | Python 3.11.7 / PyTorch 2.10.0+cpu / CPU | 2026-07-25；单种子 1 epoch pilot，禁止作为正式主结果 |
| ROBUST-SMOKE-MNIST-004 | 96.875 | mean accuracy across draws (%) | `results/robustness_smoke/pilot_mnist_003_subset32_mixed_schema_v2_summary/method_summary.jsonl` | `python simulator/summarize_robustness.py --evaluations results/robustness_smoke/pilot_mnist_003_subset32_mixed_schema_v2 --output-dir results/robustness_smoke/pilot_mnist_003_subset32_mixed_schema_v2_summary` | 测试集前缀 32 个样本；2 个固定 mixed-perturbation draw；完整配置见冻结 plan | 训练 seed 42；draw seed 见 plan | Python 3.11.7 / PyTorch 2.10.0+cpu / SciPy 1.17.1 / CPU | 2026-07-28；单训练 seed、测试前缀工程 smoke，禁止作为正式结果或参数选择依据 |
| ROBUST-SMOKE-MNIST-005 | 0.512141577899456 | mean detector contrast across draws | `results/robustness_smoke/pilot_mnist_003_subset32_mixed_schema_v2_summary/method_summary.jsonl` | 同 ROBUST-SMOKE-MNIST-004 | 同 ROBUST-SMOKE-MNIST-004 | 同 ROBUST-SMOKE-MNIST-004 | 同 ROBUST-SMOKE-MNIST-004 | 2026-07-28；仅证明汇总链路可运行，禁止作为论文主结果 |
| ROBUST-SMOKE-MNIST-006 | 0.8654545454545455 | macro-F1 across draws | `results/robustness_smoke/pilot_mnist_003_subset32_mixed_schema_v2_summary/method_summary.jsonl` | 同 ROBUST-SMOKE-MNIST-004 | 全部 10 个输出类别；零分母类别 F1 记 0；混淆矩阵可重算 | 同 ROBUST-SMOKE-MNIST-004 | 同 ROBUST-SMOKE-MNIST-004 | 2026-07-28；测试前缀类别覆盖不足使其仅具管线诊断意义，禁止作为论文主结果 |
| FORMAL-CLEAN-ELECTRONIC-001 | 94.67 | mean clean test accuracy across training seeds (%) | `results/formal_mnist_v1/clean_summary_electronic_only/method_summary.jsonl` | `python simulator/summarize_clean.py --evaluations results/formal_mnist_v1/clean/electronic_seed42 results/formal_mnist_v1/clean/electronic_seed43 results/formal_mnist_v1/clean/electronic_seed44 --output-dir results/formal_mnist_v1/clean_summary_electronic_only` | 协议绑定完整 10,000 测试集；每 seed 独立训练；95% Student-t CI 94.0401%–95.2999% | 42/43/44 | 各 evaluation manifest 内嵌环境；CPU | 2026-08-01；正式电子基线局部结果；summary manifest SHA-256 `c1741b9474e224383c7f07214b85e37f0917fddef8e0f64daf6f7c5a725e58a8` |
| FORMAL-CLEAN-ELECTRONIC-002 | 0.946085058604949 | mean clean macro-F1 across training seeds | 同 FORMAL-CLEAN-ELECTRONIC-001 | 同 FORMAL-CLEAN-ELECTRONIC-001 | 全部 10 类、零分母类别 F1 记 0；逐样本预测重建混淆矩阵后复算；95% Student-t CI 0.939620–0.952550 | 42/43/44 | 同 FORMAL-CLEAN-ELECTRONIC-001 | 2026-08-01；正式电子基线局部结果 |
| FORMAL-CLEAN-ELECTRONIC-003 | 0.187209483961761 | mean cross-entropy across training seeds | 同 FORMAL-CLEAN-ELECTRONIC-001 | 同 FORMAL-CLEAN-ELECTRONIC-001 | 每 seed 完整测试集 mean CE 后跨训练 seed 求均值；95% Student-t CI 0.169961–0.204458 | 42/43/44 | 同 FORMAL-CLEAN-ELECTRONIC-001 | 2026-08-01；正式电子基线局部结果 |
| FORMAL-CLEAN-ELECTRONIC-004 | 0.6957211666352426 | mean of per-seed median seconds per complete 10,000-sample CPU software forward | 同 FORMAL-CLEAN-ELECTRONIC-001 | 同 FORMAL-CLEAN-ELECTRONIC-001 | PyTorch CPU intra-op threads=1；batch 128；每 seed 1 warmup + 5 measurement repeats；不含指标聚合和 artifact I/O；95% Student-t CI 0.566263–0.825180 s | 42/43/44 | 各 evaluation manifest 内嵌环境；固定环境 CPU 软件仿真 | 2026-08-01；不是单样本延迟、真实边缘设备延迟、光学硬件延迟、功耗或能效 |
| FORMAL-CLEAN-BASELINE-001 | 89.63666666666666 | mean clean test accuracy across training seeds (%) | `results/formal_mnist_v1/clean_summary_all_methods/method_summary.jsonl` | `python simulator/summarize_clean.py --evaluations <12 protocol-bound clean evaluation directories> --output-dir results/formal_mnist_v1/clean_summary_all_methods` | 完整 10,000 样本测试集；95% Student-t CI 88.8149%–90.4584%；summary manifest SHA-256 `315d549388eab0afbe7e86629576273d1b1433575ef5dac0fb6a3d2d84cc4fe8` | 42/43/44 | Python 3.11.7 / PyTorch 2.10.0 / CPU；12 个输入 manifest 的协议、源码哈希和运行环境一致 | 2026-08-11；正式 baseline D2NN clean 结果 |
| FORMAL-CLEAN-BASELINE-002 | 0.8947126287569379 | mean clean macro-F1 across training seeds | 同 FORMAL-CLEAN-BASELINE-001 | 同 FORMAL-CLEAN-BASELINE-001 | 全部 10 类；逐样本预测重建混淆矩阵后复算；95% Student-t CI 0.886533–0.902892 | 42/43/44 | 同 FORMAL-CLEAN-BASELINE-001 | 2026-08-11；正式 baseline D2NN clean 结果 |
| FORMAL-CLEAN-BASELINE-003 | 0.6461869439442952 | mean cross-entropy across training seeds | 同 FORMAL-CLEAN-BASELINE-001 | 同 FORMAL-CLEAN-BASELINE-001 | 每 seed 完整测试集 mean CE 后跨训练 seed 求均值；95% Student-t CI 0.608814–0.683560 | 42/43/44 | 同 FORMAL-CLEAN-BASELINE-001 | 2026-08-11；正式 baseline D2NN clean 结果 |
| FORMAL-CLEAN-BASELINE-004 | 60.40665703332828 | mean of per-seed median seconds per complete 10,000-sample CPU software forward | 同 FORMAL-CLEAN-BASELINE-001 | 同 FORMAL-CLEAN-BASELINE-001 | 每 seed 1 warmup + 5 measurement repeats；跨 seed SD 24.5462 s；墙钟波动大，不把 t 区间解释为物理延迟区间 | 42/43/44 | 同 FORMAL-CLEAN-BASELINE-001 | 2026-08-11；固定环境 CPU 软件仿真开销，不是硬件延迟、功耗或能效 |
| FORMAL-CLEAN-ROBUST-001 | 88.31 | mean clean test accuracy across training seeds (%) | 同 FORMAL-CLEAN-BASELINE-001 | 同 FORMAL-CLEAN-BASELINE-001 | 完整 10,000 样本测试集；95% Student-t CI 87.9033%–88.7167% | 42/43/44 | 同 FORMAL-CLEAN-BASELINE-001 | 2026-08-11；正式 robust D2NN clean 结果 |
| FORMAL-CLEAN-ROBUST-002 | 0.8813239600260546 | mean clean macro-F1 across training seeds | 同 FORMAL-CLEAN-BASELINE-001 | 同 FORMAL-CLEAN-BASELINE-001 | 全部 10 类；逐样本预测重建混淆矩阵后复算；95% Student-t CI 0.877717–0.884931 | 42/43/44 | 同 FORMAL-CLEAN-BASELINE-001 | 2026-08-11；正式 robust D2NN clean 结果 |
| FORMAL-CLEAN-ROBUST-003 | 0.7110996447881063 | mean cross-entropy across training seeds | 同 FORMAL-CLEAN-BASELINE-001 | 同 FORMAL-CLEAN-BASELINE-001 | 每 seed 完整测试集 mean CE 后跨训练 seed 求均值；95% Student-t CI 0.701985–0.720215 | 42/43/44 | 同 FORMAL-CLEAN-BASELINE-001 | 2026-08-11；正式 robust D2NN clean 结果 |
| FORMAL-CLEAN-ROBUST-004 | 70.29650076672745 | mean of per-seed median seconds per complete 10,000-sample CPU software forward | 同 FORMAL-CLEAN-BASELINE-001 | 同 FORMAL-CLEAN-BASELINE-001 | 每 seed 1 warmup + 5 measurement repeats；95% Student-t CI 53.1724–87.4206 s；仅作软件仿真开销 | 42/43/44 | 同 FORMAL-CLEAN-BASELINE-001 | 2026-08-11；不是硬件延迟、功耗或能效 |
| FORMAL-CLEAN-HYBRID-001 | 95.69666666666666 | mean clean test accuracy across training seeds (%) | 同 FORMAL-CLEAN-BASELINE-001 | 同 FORMAL-CLEAN-BASELINE-001 | 完整 10,000 样本测试集；95% Student-t CI 94.7198%–96.6735%；seed44 使用正式 retry1 checkpoint | 42/43/44 | 同 FORMAL-CLEAN-BASELINE-001 | 2026-08-11；正式 hybrid clean 结果 |
| FORMAL-CLEAN-HYBRID-002 | 0.9565316732481971 | mean clean macro-F1 across training seeds | 同 FORMAL-CLEAN-BASELINE-001 | 同 FORMAL-CLEAN-BASELINE-001 | 全部 10 类；逐样本预测重建混淆矩阵后复算；95% Student-t CI 0.946583–0.966481 | 42/43/44 | 同 FORMAL-CLEAN-BASELINE-001 | 2026-08-11；正式 hybrid clean 结果 |
| FORMAL-CLEAN-HYBRID-003 | 0.1371374305764834 | mean cross-entropy across training seeds | 同 FORMAL-CLEAN-BASELINE-001 | 同 FORMAL-CLEAN-BASELINE-001 | 每 seed 完整测试集 mean CE 后跨训练 seed 求均值；95% Student-t CI 0.124593–0.149682 | 42/43/44 | 同 FORMAL-CLEAN-BASELINE-001 | 2026-08-11；正式 hybrid clean 结果 |
| FORMAL-CLEAN-HYBRID-004 | 63.2405386333509 | mean of per-seed median seconds per complete 10,000-sample CPU software forward | 同 FORMAL-CLEAN-BASELINE-001 | 同 FORMAL-CLEAN-BASELINE-001 | 每 seed 1 warmup + 5 measurement repeats；95% Student-t CI 60.3063–66.1747 s；仅作软件仿真开销 | 42/43/44 | 同 FORMAL-CLEAN-BASELINE-001 | 2026-08-11；不是硬件延迟、功耗或能效 |
| FORMAL-CLEAN-PAIRED-001 | 6.06 | hybrid minus baseline D2NN clean accuracy (percentage points) | `results/formal_mnist_v1/clean_summary_all_methods/paired_comparisons.jsonl` | 同 FORMAL-CLEAN-BASELINE-001 | 按 training seed 配对；95% Student-t CI 5.1898–6.9302 percentage points；未执行假设检验 | 42/43/44 | 同 FORMAL-CLEAN-BASELINE-001 | 2026-08-11；有限三 seed 数值仿真证据 |
| FORMAL-CLEAN-PAIRED-002 | -1.3266666666666722 | robust D2NN minus baseline D2NN clean accuracy (percentage points) | 同 FORMAL-CLEAN-PAIRED-001 | 同 FORMAL-CLEAN-BASELINE-001 | 由 `baseline minus robust=+1.3267` 变换方向；95% Student-t CI -1.7499–-0.9034 percentage points；未执行假设检验 | 42/43/44 | 同 FORMAL-CLEAN-BASELINE-001 | 2026-08-11；用于描述 clean-accuracy trade-off，不预判扰动条件表现 |
| FORMAL-CLEAN-PAIRED-003 | 1.0266666666666684 | hybrid minus electronic clean accuracy (percentage points) | 同 FORMAL-CLEAN-PAIRED-001 | 同 FORMAL-CLEAN-BASELINE-001 | 按 training seed 配对；95% Student-t CI -0.5559–2.6092 percentage points；区间跨 0，未执行假设检验 | 42/43/44 | 同 FORMAL-CLEAN-BASELINE-001 | 2026-08-11；不得写成确定性优于 electronic |
| FORMAL-ROBUST-MIXED-001 | 83.36111111111111 | baseline D2NN mean accuracy under mixed stress (%) | `results/formal_mnist_v1/robustness_summary/method_summary.jsonl` | `python simulator/summarize_robustness.py --evaluations <9 protocol-bound robustness evaluation directories> --output-dir results/formal_mnist_v1/robustness_summary` | 14 条件/34 deployment draws；完整测试集；95% Student-t CI 82.2449%–84.4774%；summary manifest SHA-256 `085a8eb937f130a9ce30435c091283db04ea1d3e3152cfdd85cd2567bb62245c` | 训练 seed 42/43/44；draw seed 见 plan | Python 3.11.7 / PyTorch 2.10.0 / CPU；plan SHA-256 `de04e6f7e86b216cc406228c3f2180e03b8b47d17ba3c0b3ab63d1764b03fc71` | 2026-08-11；正式数值仿真 robustness 结果 |
| FORMAL-ROBUST-MIXED-002 | 86.12 | robust D2NN mean accuracy under mixed stress (%) | 同 FORMAL-ROBUST-MIXED-001 | 同 FORMAL-ROBUST-MIXED-001 | 95% Student-t CI 84.3047%–87.9353% | 训练 seed 42/43/44；draw seed 见 plan | 同 FORMAL-ROBUST-MIXED-001 | 2026-08-11；正式数值仿真 robustness 结果 |
| FORMAL-ROBUST-MIXED-003 | 93.07777777777777 | hybrid mean accuracy under mixed stress (%) | 同 FORMAL-ROBUST-MIXED-001 | 同 FORMAL-ROBUST-MIXED-001 | 95% Student-t CI 90.9532%–95.2024% | 训练 seed 42/43/44；draw seed 见 plan | 同 FORMAL-ROBUST-MIXED-001 | 2026-08-11；正式数值仿真 robustness 结果 |
| FORMAL-ROBUST-PAIRED-001 | 2.7588888888888875 | robust D2NN minus baseline D2NN accuracy under mixed stress (percentage points) | `results/formal_mnist_v1/robustness_summary/paired_comparisons.jsonl` | 同 FORMAL-ROBUST-MIXED-001 | 按 training seed 配对；95% Student-t CI 0.9613–4.5565 percentage points；未执行假设检验 | 42/43/44 | 同 FORMAL-ROBUST-MIXED-001 | 2026-08-11；deployment draws 先在 seed 内聚合 |
| FORMAL-ROBUST-PAIRED-002 | 9.716666666666665 | hybrid minus baseline D2NN accuracy under mixed stress (percentage points) | 同 FORMAL-ROBUST-PAIRED-001 | 同 FORMAL-ROBUST-MIXED-001 | 按 training seed 配对；95% Student-t CI 7.6156–11.8177 percentage points；未执行假设检验 | 42/43/44 | 同 FORMAL-ROBUST-MIXED-001 | 2026-08-11；有限三 seed 数值仿真证据 |
| FORMAL-ROBUST-PAIRED-003 | 2.2022222222222227 | robust D2NN minus baseline D2NN accuracy at 0.50 px lateral shift (percentage points) | 同 FORMAL-ROBUST-PAIRED-001 | 同 FORMAL-ROBUST-MIXED-001 | 95% Student-t CI 1.8244–2.5801 percentage points；未执行假设检验 | 42/43/44 | 同 FORMAL-ROBUST-MIXED-001 | 2026-08-11；正式数值仿真 robustness 结果 |
| FORMAL-ROBUST-PAIRED-004 | 17.91 | robust D2NN minus baseline D2NN accuracy at four phase levels (percentage points) | 同 FORMAL-ROBUST-PAIRED-001 | 同 FORMAL-ROBUST-MIXED-001 | 95% Student-t CI 3.0147–32.8053 percentage points；seed 间变异大；未执行假设检验 | 42/43/44 | 同 FORMAL-ROBUST-MIXED-001 | 2026-08-11；不得把宽区间写成稳定硬件容差 |
| FORMAL-ROBUST-PAIRED-005 | 9.756666666666673 | hybrid minus baseline D2NN accuracy at eight phase levels (percentage points) | 同 FORMAL-ROBUST-PAIRED-001 | 同 FORMAL-ROBUST-MIXED-001 | 95% Student-t CI 5.8497–13.6637 percentage points；未执行假设检验 | 42/43/44 | 同 FORMAL-ROBUST-MIXED-001 | 2026-08-11；正式数值仿真 robustness 结果 |
| FORMAL-ROBUST-RANK-001 | 13 | conditions in which hybrid has the highest mean accuracy among three optical-compatible methods (count out of 14) | `results/formal_mnist_v1/robustness_summary/method_summary.jsonl` | 同 FORMAL-ROBUST-MIXED-001 | 全部 14 个预注册条件；四级相位量化为唯一例外；排名基于均值，不等同于逐条件显著性结论 | 42/43/44 | 同 FORMAL-ROBUST-MIXED-001 | 2026-08-11；仅描述均值排序 |
| FORMAL-COMPLEXITY-001 | 12288 | baseline D2NN trainable parameters | `paper/figures/source_data_figure_3a.csv` | `python paper/figures/make_complexity_figure.py` | 从三个 protocol-bound clean evaluation manifest 校验跨 seed 一致性 | 42/43/44 | Python 3.11.7 / CPU | 2026-08-11；模型参数计数，不是硬件存储测量 |
| FORMAL-COMPLEXITY-002 | 12288 | robust D2NN trainable parameters | 同 FORMAL-COMPLEXITY-001 | 同 FORMAL-COMPLEXITY-001 | 同 FORMAL-COMPLEXITY-001 | 42/43/44 | 同 FORMAL-COMPLEXITY-001 | 2026-08-11 |
| FORMAL-COMPLEXITY-003 | 14698 | hybrid trainable parameters | 同 FORMAL-COMPLEXITY-001 | 同 FORMAL-COMPLEXITY-001 | 同 FORMAL-COMPLEXITY-001 | 42/43/44 | 同 FORMAL-COMPLEXITY-001 | 2026-08-11 |
| FORMAL-COMPLEXITY-004 | 14320 | electronic trainable parameters | 同 FORMAL-COMPLEXITY-001 | 同 FORMAL-COMPLEXITY-001 | 同 FORMAL-COMPLEXITY-001 | 42/43/44 | 同 FORMAL-COMPLEXITY-001 | 2026-08-11 |
| FORMAL-REPRESENTATION-001 | 10 | baseline D2NN detector-energy representation elements | 同 FORMAL-COMPLEXITY-001 | 同 FORMAL-COMPLEXITY-001 | feature schema `class_detector_energy:10` | 42/43/44 | 同 FORMAL-COMPLEXITY-001 | 2026-08-11；表示维度，不是传输字节或能耗 |
| FORMAL-REPRESENTATION-002 | 10 | robust D2NN detector-energy representation elements | 同 FORMAL-COMPLEXITY-001 | 同 FORMAL-COMPLEXITY-001 | feature schema `class_detector_energy:10` | 42/43/44 | 同 FORMAL-COMPLEXITY-001 | 2026-08-11；表示维度，不是传输字节或能耗 |
| FORMAL-REPRESENTATION-003 | 64 | hybrid pooled-intensity representation elements | 同 FORMAL-COMPLEXITY-001 | 同 FORMAL-COMPLEXITY-001 | feature schema `pooled_intensity:8x8` | 42/43/44 | 同 FORMAL-COMPLEXITY-001 | 2026-08-11；光学到电子边界的元素数 |
| FORMAL-REPRESENTATION-004 | 784 | electronic flattened-input representation elements | 同 FORMAL-COMPLEXITY-001 | 同 FORMAL-COMPLEXITY-001 | feature schema `flattened_input:1x28x28` | 42/43/44 | 同 FORMAL-COMPLEXITY-001 | 2026-08-11；输入元素数，不是光学到电子接口 |

## C2. 正式 MNIST v2 扩实验队列（外部评审意见 #1/#2/#3）

协议：`FORMAL_EXPERIMENT_PROTOCOL_V2.json`，SHA-256 `9da8b0e0a95c022c82d66b533737ff2ddb7c2832d4c0e2c2f6d05869f10472b6`，
`supersedes` v1。训练种子 42–46（n=5）；训练在云 GPU（RTX 4060 Ti）完成，
评估设备为 CUDA（CPU/GPU 指标逐位一致，见 `FORMAL_PROTOCOL_V2_DECISIONS.md`）。
鲁棒性方案 `plan_id=robustness-plan-62aae73d8284e7fd25a4`。

**v2 与 v1 不完全可比**：种子数由 3 增至 5，故下方数字与 C 表的 v1 数字不得混用。

| Metric ID | 数值 | 单位 | 数据文件 | 生成命令 | 配置 | 随机种子 | 环境 | 核验人/时间 |
|---|---:|---|---|---|---|---|---|---|
| FORMAL-V2-ROBUST-4LEVEL-QAT | 94.10 | mean accuracy at four phase levels (%) | `results/formal_mnist_v2/robustness_summary/method_summary.jsonl` | `python simulator/summarize_robustness.py --evaluations <20 v2 robustness dirs> --output-dir results/formal_mnist_v2/robustness_summary` | 冻结 4 级量化条件；95% Student-t CI 92.61%–95.58%；种子级值 94.29/94.88/92.02/94.35/94.94 | 训练 42–46 | GPU 训练 + CUDA 指标评估 | 2026-09-17；summary manifest SHA-256 `85f9ce1fd20e57167f3018c74864bc09d222baafbe923ab0c06a48cd0a336b0b` |
| FORMAL-V2-ROBUST-4LEVEL-HYBRID | 67.69 | mean accuracy at four phase levels (%) | 同上 | 同上 | 95% Student-t CI 50.13%–85.24%；种子级值 78.47/84.28/61.63/48.54/65.51 | 训练 42–46 | 同上 | 2026-09-17；非 QAT 混合模型在同条件严重退化且种子间极不稳定 |
| FORMAL-V2-ROBUST-4LEVEL-BASELINE | 59.51 | mean accuracy at four phase levels (%) | 同上 | 同上 | 95% Student-t CI 54.22%–64.79% | 训练 42–46 | 同上 | 2026-09-17 |
| FORMAL-V2-ROBUST-4LEVEL-ROBUST | 77.56 | mean accuracy at four phase levels (%) | 同上 | 同上 | 95% Student-t CI 72.20%–82.91% | 训练 42–46 | 同上 | 2026-09-17 |
| FORMAL-V2-PAIRED-QAT-MINUS-HYBRID | 26.41 | QAT hybrid minus hybrid at four phase levels (percentage points) | `results/formal_mnist_v2/robustness_summary/paired_comparisons.jsonl` | 同上 | 按训练种子配对；95% Student-t CI 9.26–43.56 percentage points；未执行假设检验 | 训练 42–46 | 同上 | 2026-09-17；量化感知训练修复了极端量化退化 |
| FORMAL-V2-PAIRED-QAT-MINUS-ROBUST | 16.54 | QAT hybrid minus robust D2NN at four phase levels (percentage points) | 同上 | 同上 | 95% Student-t CI 10.00–23.08 percentage points | 训练 42–46 | 同上 | 2026-09-17；区间不含 0 |
| FORMAL-V2-CLEAN-QAT | 61.74 | QAT hybrid mean clean accuracy (%) | `results/formal_mnist_v2/clean_summary_v2_all_methods/method_summary.jsonl` | `python simulator/summarize_clean.py --evaluations <25 v2 clean dirs> --output-dir results/formal_mnist_v2/clean_summary_v2_all_methods --tolerate-protocol-revisions` | 完整 10 000 测试集；95% Student-t CI 56.28%–67.21%；种子级值 64.14/68.34/57.84/59.09/59.31 | 训练 42–46 | GPU 训练；指标在 CUDA 与 CPU 上分别产出的记录已按设备一致性容忍并留痕 | 2026-09-17；summary manifest SHA-256 `01b368dddd768a04f188562c42c346112deff7ad037000139e5a85cac961300d` |
| FORMAL-V2-CLEAN-HYBRID | 95.68 | hybrid mean clean accuracy (%) | 同上 | 同上 | clean_continuous 条件 | 训练 42–46 | 同上 | 2026-09-17 |
| FORMAL-V2-CLEAN-BASELINE | 89.56 | baseline D2NN mean clean accuracy (%) | 同上 | 同上 | clean_continuous 条件 | 训练 42–46 | 同上 | 2026-09-17 |
| FORMAL-V2-CLEAN-ROBUST | 88.41 | robust D2NN mean clean accuracy (%) | 同上 | 同上 | clean_continuous 条件 | 训练 42–46 | 同上 | 2026-09-17 |
| FORMAL-V2-LENET5-CLEAN | 98.25 | LeNet-5 mean clean test accuracy across training seeds (%) | `results/formal_mnist_v2/clean/lenet5_seed*/clean_metrics.json` | `python simulator/evaluate_clean.py --checkpoint artifacts/formal_mnist_v2/best_mnist.formal_mnist_v2_lenet5_seed<42..46>.pth --method-id lenet5 --protocol FORMAL_EXPERIMENT_PROTOCOL_V2.json --metrics-device cuda --output-dir results/formal_mnist_v2/clean/lenet5_seed<seed>` | 完整 10 000 测试集；95% Student-t CI 97.91%–98.59%；种子级值 97.87/98.56/98.09/98.34/98.39 | 训练 42–46 | GPU 训练 + CUDA 指标评估 | 2026-09-17；34 622 参数；纯电子参考基线，不接受光学扰动 |
| FORMAL-V2-LENET5-PARAMS | 34622 | LeNet-5 trainable parameters | `artifacts/formal_mnist_v2/best_mnist.formal_mnist_v2_lenet5_seed42.json` | 训练 manifest 内嵌 | 2 卷积层（6/16 通道）+ 120 单元全连接 | 42–46 | 同上 | 2026-09-17；与混合模型 14 698 参数同量级，用于回应基线过弱质疑 |
| FORMAL-V2-DEVICE-PARITY | 0.00e+00 | worst CPU-vs-CUDA per-draw accuracy difference | `tmp/device_parity_probe.py` 输出 | 见 `FORMAL_PROTOCOL_V2_DECISIONS.md` 第三节 | 条件：clean、0.50 px 错位、4 级量化、混合强扰动；检查点：基线/混合/QAT 各 1 | — | CPU vs CUDA | 2026-09-17；CUDA 快 4.11×，故 v2 采用 GPU 指标评估 |

## C3. 正式 Fashion-MNIST v3 队列（外部评审意见 #4）

协议：`FORMAL_EXPERIMENT_PROTOCOL_V3.json`，`supersedes` v2；数据集 `fashion-mnist`，
训练种子 42–44（n=3），14 条件 / 34 draws，训练与评估均在云 GPU。
鲁棒性方案 `plan_id=robustness-plan-0a0a1feb6ace9c66a886`。

**v3 与 v1/v2 数据集不同，数字不可混用。**

| Metric ID | 数值 | 单位 | 数据文件 | 生成命令 | 配置 | 随机种子 | 环境 | 核验人/时间 |
|---|---:|---|---|---|---|---|---|---|
| FORMAL-V3-CLEAN-BASELINE | 78.09 | baseline D2NN 干净准确率均值（%） | `results/formal_fashion_mnist_v3/robustness_summary/method_summary.jsonl` | `python simulator/summarize_robustness.py --evaluations <9 v3 robustness dirs> --output-dir results/formal_fashion_mnist_v3/robustness_summary` | 95% Student-t CI 76.89–79.28；种子级值 78.28/77.54/78.44 | 训练 42–44 | GPU 训练 + CUDA 指标评估 | 2026-09-17 |
| FORMAL-V3-CLEAN-ROBUST | 76.37 | 鲁棒 D2NN 干净准确率均值（%） | 同上 | 同上 | 95% CI 75.66–77.08 | 42–44 | 同上 | 2026-09-17；第二数据集上仍未挽回干净代价 |
| FORMAL-V3-CLEAN-HYBRID | 81.88 | 混合模型干净准确率均值（%） | 同上 | 同上 | 95% CI 79.83–83.94 | 42–44 | 同上 | 2026-09-17 |
| FORMAL-V3-RANK | 13 | 混合模型取得最高平均准确率的条件数（共 14） | 同上 | 同上 | 唯一例外为四级相位量化，与 MNIST 队列一致 | 42–44 | 同上 | 2026-09-17；条件选择性模式在第二数据集复现 |
| FORMAL-V3-4LEVEL-BASELINE | 45.93 | 基准 D²NN 四级量化平均准确率（%） | 同上 | 同上 | 95% CI 15.35–76.51；种子级值 48.54/32.52/56.72，离散度大 | 42–44 | 同上 | 2026-09-17 |
| FORMAL-V3-4LEVEL-ROBUST | 67.59 | 鲁棒 D²NN 四级量化平均准确率（%） | 同上 | 同上 | 95% CI 62.14–73.04 | 42–44 | 同上 | 2026-09-17；四级量化在两个数据集上均为共同失效边界 |
| FORMAL-V3-4LEVEL-HYBRID | 51.10 | 混合模型四级量化平均准确率（%） | 同上 | 同上 | 95% CI 47.82–54.38 | 42–44 | 同上 | 2026-09-17 |
| FORMAL-V3-PAIRED-ROBUST-CLEAN | -1.72 | 鲁棒 D2NN 减基准 D2NN 干净准确率（百分点） | `results/formal_fashion_mnist_v3/robustness_summary/paired_comparisons.jsonl` | 同上 | 由 `baseline minus robust=+1.72` 变换方向；95% CI -3.33 至 -0.10 | 42–44 | 同上 | 2026-09-17；干净代价方向与 MNIST 一致 |

## C2-b. 量化感知训练的三条前向路径（外部评估意见 #4）

**问题**：QAT checkpoint 的验证/测试条件是 `clean_continuous`，因此此前引用的"干净准确率 61.74%"
测量的是**未量化的潜在参数**，而该配置不会被部署。评估方要求把前向路径分别命名并核查。

**核查方式**：
- P1（潜在连续测试）与 P2（四级标称测试）由 `simulator/probe_forward_paths.py` 实测
  （`results/formal_mnist_v2/forward_path_probe.json`，完整 10 000 测试集）。
- P3（四级受扰测试）由冻结的逐样本预测直接复算
  （`results/formal_mnist_v2/p3_four_level_disturbed.json`），未做新的前向计算。

| Metric ID | 数值 | 单位 | 数据文件 | 配置 | 种子 |
|---|---:|---|---|---|---|
| FORMAL-QAT-PATHS-001 | 61.74 | QAT 混合模型潜在连续测试准确率（%） | `results/formal_mnist_v2/forward_path_probe.json` | 学习连续相位，无量化、无扰动 | 42–46 |
| FORMAL-QAT-PATHS-002 | 94.10 | QAT 混合模型四级标称测试准确率（%） | 同上 | 由学习相位重建的四级掩模，无错位/噪声 | 42–46 |
| FORMAL-QAT-PATHS-003 | 94.10 | QAT 混合模型四级受扰测试准确率（%） | `results/formal_mnist_v2/p3_four_level_disturbed.json` | 四级掩模 + 部署采样；与 P2 逐种子相同 | 42–46 |
| FORMAL-QAT-PATHS-004 | 95.68 / 67.69 | 混合模型的 P1 / P2（%） | 同上两个文件 | 同上 | 42–46 |
| FORMAL-QAT-PATHS-005 | 89.56 / 59.51 | 基准 D²NN 的 P1 / P2（%） | 同上 | 同上 | 42–46 |

**结论**：P2 与 P3 在全部方法、全部种子上完全一致，说明本文的四级扰动仅改变相位取值，
错位/传播距离误差/探测器噪声在该条件下贡献为零。量化感知训练的 61.74% 对应其非部署配置；
在其部署配置下为 94.10%。因此"以干净准确率换取量化收益"的表述已被替换为
"不同部署假设下的最优选择不同"。

**实现细节**（评估方要求交代）：量化先对相位取 $2\pi$ 余数，再按步长 $\Delta=2\pi/4$ 四舍五入；
训练时使用直通估计（`d2nn.py` 中 `straight_through=self.training`），评估时关闭；
该模型的 checkpoint 按**连续相位**验证准确率选择，选择准则与部署条件不一致，已在正文披露。

## C2-c. 训练曲线与三 epoch 预算（外部评估意见 #7）

由 checkpoint manifest 的 `history` 字段提取，无需新训练。

| Metric ID | 数值 | 单位 | 数据文件 |
|---|---:|---|---|
| FORMAL-CURVES-001 | 3.71 | QAT 混合模型第三个 epoch 的验证准确率单轮增益（百分点） | `results/training_curves/training_curves_summary.json` |
| FORMAL-CURVES-002 | 0.98 | 鲁棒 D²NN 末期增益（百分点） | 同上 |
| FORMAL-CURVES-003 | -2.05 | 相位滤波 D²NN 末期增益（百分点，验证准确率单调下降） | 同上 |

多数方法在第三个 epoch 仍在上升，Fashion-MNIST 上三种方法末期增益为 0.59 至 1.41 个百分点。
因此全部比较限定为"三个 epoch 预算下"的结果，不能推断架构能力上界。

## C4. 正式 MNIST v4 消融与鲁棒方法对照（外部评审意见 #5/#6）

协议：`FORMAL_EXPERIMENT_PROTOCOL_V4.json`，`supersedes` v2；MNIST，训练种子 42–44（n=3），
14 条件 / 34 draws。新增方法：`phase_filtered_d2nn`（训练期高斯相位滤波，std=1.0 px）、
`hybrid_linear_head`（8×8 池化 + 单层线性读出）、`hybrid_pool4`（4×4 池化 + 32 单元 MLP）。

| Metric ID | 数值 | 单位 | 数据文件 | 生成命令 | 配置 | 随机种子 | 环境 | 核验人/时间 |
|---|---:|---|---|---|---|---|---|---|
| FORMAL-V4-LINEARHEAD-CLEAN | 96.60 | 线性头混合模型干净准确率均值（%） | `results/formal_mnist_v4/robustness_summary/method_summary.jsonl` | `python simulator/summarize_robustness.py --evaluations <9 v4 robustness dirs> --output-dir results/formal_mnist_v4/robustness_summary` | 95% CI 96.24–96.96；种子级值 96.75/96.59/96.46 | 训练 42–44 | 云 GPU 训练 + CUDA 指标评估 | 2026-09-17；全部 14 条件最高，参数少于原混合模型 |
| FORMAL-V4-POOL4-CLEAN | 96.04 | 4×4 池化混合模型干净准确率均值（%） | 同上 | 同上 | 95% CI 95.04–97.04 | 42–44 | 同上 | 2026-09-17；相对 8×8 池化仅低 0.56 pp |
| FORMAL-V4-PHASEFILTER-CLEAN | 78.01 | 相位滤波 D²NN 干净准确率均值（%） | 同上 | 同上 | 95% CI 76.83–79.19；种子级值 78.03/78.47/77.52 | 42–44 | 同上 | 2026-09-17 |
| FORMAL-V4-PHASEFILTER-COST | 11.63 | 相位滤波相对基准 D2NN 的干净准确率代价（百分点） | 同上 | 同上 | 同协议同种子的种子级配对差 | 42–44 | 同上 | 2026-09-18；代价远大于扰动感知训练的 1.33 pp |
| FORMAL-V4-LINEARHEAD-4LEVEL | 91.48 | 线性头混合模型四级量化平均准确率（%） | 同上 | 同上 | 95% CI 84.57–98.39 | 42–44 | 同上 | 2026-09-17 |
| FORMAL-V4-POOL4-4LEVEL | 82.87 | 4×4 池化混合模型四级量化平均准确率（%） | 同上 | 同上 | 95% CI 73.70–92.03 | 42–44 | 同上 | 2026-09-17 |
| FORMAL-V4-PHASEFILTER-4LEVEL | 60.20 | 相位滤波 D²NN 四级量化平均准确率（%） | 同上 | 同上 | 完整测试集 | 42–44 | 同上 | 2026-09-17；相位滤波未改善该条件 |

### C4-b. v4 同种子参照队列（统计口径修正）

**问题**：v4 队列最初只包含三个新方法，表 9 的参照列取自 v2 的五种子队列，
造成 n=3 与 n=5 的均值直接比较，违反本研究"不同种子集合的数字不可混用"的规定。

**修正**：`baseline_d2nn`、`robust_d2nn`、`hybrid` 的种子 42–44 检查点在 v4 协议下重新评估，
参照值改为同协议、同种子的配对结果。
鲁棒性方案 `plan_id=robustness-plan-3dbdfd82ebfa9d574902`。

| Metric ID | 数值 | 单位 | 数据文件 | 配置 | 随机种子 | 环境 |
|---|---:|---|---|---|---|---|
| FORMAL-V4-REF-BASELINE-CLEAN | 89.64 | 基准 D2NN 干净准确率（%） | `results/formal_mnist_v4/reference_robustness_summary/method_summary.jsonl` | 同协议 v4，种子 42–44；与 v1 队列同值 | 42–44 | 云 GPU + CUDA 指标 |
| FORMAL-V4-REF-HYBRID-CLEAN | 95.69 | 混合模型干净准确率（%） | 同上 | 同上 | 42–44 | 同上 |
| FORMAL-V4-REF-ROBUST-CLEAN | 88.31 | 鲁棒 D2NN 干净准确率（%） | 同上 | 同上 | 42–44 | 同上 |
| FORMAL-V4-PAIRED-LINEARHEAD-HYBRID | +0.91 | 线性头减混合模型干净准确率（百分点） | 同上 + `robustness_summary` | 同种子配对；95% CI -0.21 至 2.02，区间跨越 0 → 持平 | 42–44 | 同上 |
| FORMAL-V4-PAIRED-POOL4-HYBRID | +0.35 | 4×4 池化减混合模型干净准确率（百分点） | 同上 | 95% CI -1.56 至 2.25，区间跨越 0 → 持平 | 42–44 | 同上 |
| FORMAL-V4-PAIRED-PHASEFILTER-BASELINE | -11.63 | 相位滤波减基准 D2NN 干净准确率（百分点） | 同上 | 95% CI -13.20 至 -10.06，区间不含 0 | 42–44 | 同上 |

**v4 定性结论**：线性头在 14/14 条件取得最高平均准确率，4×4 池化仅低 0.56 个百分点，
故混合模型的收益来自"学习型读出"本身，而非电子头非线性或表示宽度。


## D. 当前图源数据登记

| 图源文件 | 行数 | SHA-256 | 状态与用途 |
|---|---:|---|---|
| `paper/figures/source_data_figure_2a.csv` | 42 | `93ff9c165b16f5b7520ec734dd668fdb8c59f7a64436cbe0fe32c0e0f990a479` | 已核验；图 2a 三方法 × 14 条件的准确率均值与区间 |
| `paper/figures/source_data_figure_2b.csv` | 28 | `7f5dd354dde9dacbfa7991e01d7bfa11153cedeee9c7b6035519a7339263f25b` | 已核验；图 2b 鲁棒和混合模型相对基准的配对差值 |
| `paper/figures/source_data_figure_3a.csv` | 4 | `e5a7cccf496c9cbf8779c53b17352698c6d113ba9416966db4f2416e05168302` | 已核验；图 3a 参数量和表示元素数 |
| `paper/figures/source_data_figure_3b.csv` | 12 | `3aa9dc38403ce82d736cbbbfd791c52f1c717aeefac5bb75240cfea80ab2304f` | 已核验；图 3b 种子级准确率、CPU 软件计时和计时重复数 |
| `paper/figures/source_data_figure_1a.csv` | 42 | `93ff9c165b16f5b7520ec734dd668fdb8c59f7a64436cbe0fe32c0e0f990a479` | 历史重复文件；与当前图 2a 哈希相同，不作为图 1 数据源 |
| `paper/figures/source_data_figure_1b.csv` | 28 | `7f5dd354dde9dacbfa7991e01d7bfa11153cedeee9c7b6035519a7339263f25b` | 历史重复文件；与当前图 2b 哈希相同，不作为图 1 数据源 |
