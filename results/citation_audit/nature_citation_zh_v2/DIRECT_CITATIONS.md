# Nature/CNS 引用支撑等级审计（nature-citation 2.0.0）

检索日期：2026-09-08
检索工具：`nature-citation` skill，脚本 `scripts/nature_citation.py`（Crossref REST API）
检索范围：`--scope nature`（Nature Portfolio 系列），`--from-year 2018 --to-year 2026`
分段输入：`tmp/nature_citation/claims.txt`（19 条可引用主张）
原始产物：`results/citation_audit/nature_citation_zh_v2/`

## 一、本文件的作用

本文件把该 skill 第 4 步（支撑等级评估）的结果固定下来，并为 `EVIDENCE_LEDGER.md` 的 B 表提供 DOI 对齐依据。
脚本默认把所有命中标记为 `metadata-only candidate`；本文件记录**逐条人工核验后的最终等级**。

## 二、命中结果的处置

脚本共返回 24 条唯一候选。按 skill 规则"标题相关不等于支撑"，其中 19 条被判定为**主题无关的误命中**，不予采用：

| 分段 | 误命中主题 | 与本文主张的关系 |
|---|---|---|
| S001 | 低算力边缘 SoC 上的目标检测基准 | 同属边缘推理但非光子/衍射计算 |
| S002 | 光子交织架构的空间-频率变换 | 架构不同，非衍射传播 |
| S007 | LLM 漏洞检测、生化通路拟合、LLM 鲁棒性 | 仅共享"perturbation/robustness"词面 |
| S010 | AR 显示用双层衍射波导 | 应用领域无关 |
| S011 | 波导布拉格光栅、多模传感内计算 | 非自由空间衍射网络 |
| S013 | 同上（片上光逻辑门、体全息） | 不支撑相干性建模边界主张 |
| S015 | 同上 | 不支撑归一化截止频率判据 |
| S017 | 散射储能、光子储备池 | 与权重分发架构不同 |
| S019 | 贝叶斯参数估计 | 完全无关 |

判定理由：这些论文的题名与检索词命中了 `diffractive`、`robustness`、`perturbation`、`photonic` 等术语，但其研究对象（器件类型、任务、证据类型）与对应主张不构成支撑关系。

## 三、技能自身判定"无范围内候选"的分段

以下 3 段在 Nature 系列内**找不到**可支撑文献，这是本项目的已知空白，需继续使用非 Nature 系列来源：

| 分段 | 主张 | 当前依据 |
|---|---|---|
| S012 | 相位受限量化感知训练直接优化离散相位 D²NN | `Wang2025PLQAT`（Applied Optics，非 Nature 系列） |
| S014 | 完全相干标量传播只是建模边界 | `Filipovich2024SpatialCoherence`（Optics Express，非 Nature 系列） |
| S018 | 数值仿真运行时间不能证明物理光学时延或能效 | 本文自有方法论主张，无需外部支撑 |

## 四、经核验后采用的候选（1 条）

### Hoshi et al. 2025 — 强支撑（robustness-related-work）

- **DOI：** [10.1038/s41598-024-82791-z](https://doi.org/10.1038/s41598-024-82791-z)
- **题名：** Wavefront-aberration-tolerant diffractive deep neural networks using volume holographic optical elements
- **期刊：** Scientific Reports（Nature Portfolio），2025
- **作者：** Ikuo Hoshi, Koki Wakunami, Yasuhiro Ichihashi, Ryutaro Oi
- **核验方式：** Crossref 元数据 + 摘要全文（非仅题名）
- **摘要要点：** vHOE 用于 D²NN 时存在多种制造误差叠加形成的未知波前像差；作者提出**在训练阶段使模型适应未知波前像差**的训练方法，报告分类准确率提升约 58 个百分点。
- **支撑等级：** **强支撑** —— 直接针对 D²NN 的制造/波前类扰动提出训练期自适应方法，属于本文 2.3 节所述"扰动感知训练"谱系。
- **可支持的论点：** 训练阶段纳入未知光学像差是已被公开报道的 D²NN 鲁棒化策略。
- **不能据此声称：** 其器件为体全息元件、扰动为波前像差，不能等同于本文的横向错位/相位噪声/量化网格；其 58 个百分点提升属于该文自有配置，不得写入本文结果。
- **插入位置：** `MANUSCRIPT_ZH.md` 第 1 节相关工作段，与 `Wang2025PhaseFilteredD2NN`、`Wang2026ParallelSubnetworkD2NN` 并列。

## 五、已采用但未再重复登记的候选

以下候选与 `references.bib` 现有条目为同一篇论文，本轮检索再次命中，仅作一致性确认：

| DOI | 现有 BibTeX key | 命中分段 |
|---|---|---|
| 10.1038/s41467-026-68470-9 | `Xu2026SharpnessAwarePNN` | S008 |
| 10.1038/s41598-018-30619-y | `Chang2018HybridOpticalElectronicCNN` | S009, S010, S013 |
| 10.1038/s41566-021-00796-w | `Zhou2021ReconfigurableDPU` | S011 |
| 10.1038/s41467-021-27774-8 | （EXT-025，背景） | S016 |

## 六、未采用的候选（需进一步核验）

### Zhou et al. 2025 — Hundred-layer photonic deep learning

- **DOI：** [10.1038/s41467-025-65356-0](https://doi.org/10.1038/s41467-025-65356-0)
- **期刊：** Nature Communications，2025
- **作者：** Tiankuang Zhou, Yizhou Jiang, Zhihao Xu, Zhiwei Xue, Lu Fang
- **核验状态：** **Crossref 未提供摘要**，仅核验到题名、期刊、作者与年份。
- **支撑等级：** `metadata-only candidate` —— 按 skill 规则，未检查摘要前**不得**作为任何主张的支撑引用。
- **处置：** 本轮不加入 `references.bib`。若后续需要引用其"光学网络深度可扩展至百层级"的背景主张，必须先取得并核验摘要或出版社页面。

## 七、本次审计的结论与缺口

1. 本文核心主张（条件选择性鲁棒性、混合读出、仿真/硬件证据边界）在 Nature 系列内的直接支撑文献稀缺；这符合该主题的社区分布——D²NN 鲁棒性工作主要发表在 Optics Express、Nanophotonics、Applied Optics、Laser & Photonics Reviews 等光学期刊。
2. 本轮新增 1 条经摘要核验的强支撑文献（Hoshi 2025），补上了"制造/波前类扰动 + 训练期自适应"这一谱系缺口。
3. 三条主张（S012、S014、S018）确认无 Nature 系列支撑，继续依赖非 Nature 来源或本文自有方法论论证。
4. 所有 19 条误命中均**未**进入 `references.bib`，避免以标题相关性冒充证据。

## 八、可复现命令

```bash
python <skills>/nature-citation/scripts/nature_citation.py \
  --claim-file tmp/nature_citation/claims.txt \
  --scope nature --from-year 2018 --to-year 2026 \
  --outdir results/citation_audit/nature_citation_zh_v2 \
  --format enw --with-artifacts --per-segment 3 --rows 40 --sleep 0.4 --max-retries 2
```

交互式浏览器：`results/citation_audit/nature_citation_zh_v2/references.html`
EndNote 导出：`results/citation_audit/nature_citation_zh_v2/references.enw`
