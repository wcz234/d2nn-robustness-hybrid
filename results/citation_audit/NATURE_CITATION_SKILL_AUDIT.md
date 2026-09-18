# `nature-citation` 实测审计

## 范围

- 日期：2026-07-25
- Skill：`nature-citation` 2.0.0
- 检索范围：CNS 及其子刊，2018—2026
- 目的：评估其在 D2NN、混合光电推理和边缘光子推理主题上的候选发现与书目导出能力。

## 结果

| 能力 | 结论 | 证据 |
|---|---|---|
| 已知 DOI 元数据与 ENW 导出 | 通过 | `nature_cns_known_dois/` 正确导出 D2NN、opticalCNN 与 Netcast 三条记录 |
| D2NN/混合光电候选发现 | 部分通过 | 找到 `10.1038/s41598-018-30619-y`，但没有优先返回更直接的 Science D2NN 奠基论文 |
| 边缘光子推理语义检索 | 失败 | 为相关主张返回火星内核新闻 `10.1038/d41586-021-00696-7` |
| 风险提示 | 部分通过 | 输出将所有搜索结果标为 `metadata-only candidate`，但仍生成了自动插入建议，人工筛选不可省略 |

## 使用边界

1. 本项目只把该 skill 用作已核验 DOI 的元数据整理和参考文献管理器导出工具。
2. 自动搜索结果不得直接进入 `references.bib`、正文或证据台账。
3. 每个候选必须通过摘要、出版社页面或全文核验，并与 `LITERATURE_MATRIX.md` 的支持边界对账。
4. `nature_cns_background/` 中的火星内核条目已明确判定为不相关，不得引用。
5. D2NN 的直接技术证据分布在 Science、IEEE、Optics Express、Applied Optics、Nanophotonics 等期刊；严格 CNS 白名单不能替代领域文献检索。
