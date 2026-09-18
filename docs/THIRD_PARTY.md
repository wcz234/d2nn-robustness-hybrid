# 第三方代码来源与使用边界

## nonlinear-d2nn

- Repository: https://github.com/yeungmkw/nonlinear-d2nn
- Pinned commit: `f1ae6e3e5076fcd2f60111ef4d42a2798d30ac8f`
- Local path: `third_party/nonlinear-d2nn`
- License: MIT
- Role: D2NN 数值传播、分类训练、检测区域读出与实验产物格式的参考基线。
- Verification: 在本机 Python 3.11 / PyTorch 2.10.0+cpu 下将 180 项核心测试按每组 20 项运行并全部通过；整文件两次触发 60 秒硬超时，未观察到断言失败。
- Boundary: 这是社区复现而非 Science 论文作者官方代码。README 和历史 artifacts 中的数字不能作为本论文结果；本项目必须重新训练并保存原始日志。

## TorchOptics

- Repository: https://github.com/MatthewFilipovich/torchoptics
- Pinned commit: `34fe9c40f874e53db225e703c5580befe8c523b5`
- Local path: `third_party/torchoptics`
- License: MIT
- Role: 使用独立实现交叉检查自由空间传播、平面几何与探测器计算。
- Verification: editable 安装到项目 `.venv` 后运行测试，`201 passed`；存在 53 条 Matplotlib/NumPy 兼容性警告，无测试失败。
- Boundary: TorchOptics 是通用可微傅里叶光学库，不是现成 D2NN 分类系统。

## Sharpness-Aware-Training（仅方法对照，未导入）

- Repository: https://github.com/cuhkhuangslab/Sharpness-Aware-Training
- Pinned commit: `9e52734f8e5829dad14fe0c1f943eb03ad00536b`
- Local path: 未检出；当前不执行或复用其代码。
- License: MIT（GitHub API 与仓库元数据核验，2026-07-28）。
- Role: 对照 Xu et al. 2026 的物理神经网络 sharpness-aware training；用于限制“通用扰动鲁棒训练”创新性表述。
- Verification: 已核对论文 DOI `10.1038/s41467-026-68470-9`、仓库题名/作者关联、默认分支提交和许可；未进行本地复现。
- Boundary: 该工作覆盖多类物理神经网络并含实体实验，其衍射光学示例不等于本文的无源多层 D2NN 评估协议；任何准确率、硬件鲁棒性或无需重训练结论都不能写成本项目结果。

## Normalized-Cutoff-Frequency robustness criterion（仅方法与数据对照，未导入）

- Repository: https://github.com/Rainbowseaaa/Robustness-Criterion-for-Optical-Diffractive-Neural-Networks-Using-Normalized-Cutoff-Frequency
- Pinned commit: `ee984268af4e8b78022198037797cf8c71a44ee4`
- Local path: 未检出；当前不执行或复用其代码。
- License: 未检测到许可证；默认保留所有权利。
- Role: 对照 Wang et al. 2026 的归一化截止频率鲁棒性判据，并为后续频域解释提供可核验开放材料。
- Verification: 已核对论文 DOI `10.1364/OE.601000`、仓库题名、README、默认分支固定提交和创建时间。
- Boundary: 本项目未实现或复现 NCF 判据；无许可证意味着不能复制代码。该仓库只支持相关工作定位，不能替代本项目的三 seed 扰动扫描。

## 复现原则

1. 第三方仓库保持固定提交，不把其历史仿真、实体实验、延迟或能耗数字写成本文结果。
2. 本项目的扩展代码放在独立源码目录中，避免修改第三方检出内容。
3. 对传播算子建立小规模数值一致性测试，并记录容差、采样参数和边界条件。
4. 若采用第三方实现片段，保留许可证声明并在代码与论文中注明来源。
5. 第三方代码仅支持数值仿真实现与交叉验证；不得据此声称真实边缘设备的延迟、吞吐、功耗或能效。
