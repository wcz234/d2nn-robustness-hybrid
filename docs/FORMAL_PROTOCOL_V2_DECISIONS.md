# 正式实验协议 v2：扩实验队列的设计决策记录

对应协议文件：`FORMAL_EXPERIMENT_PROTOCOL_V2.json`
生成脚本：`build_protocol_v2.py`
冻结日期：2026-09-17

## 一、为什么新建 v2 而不是修改 v1

v1（`FORMAL_EXPERIMENT_PROTOCOL.json`，SHA-256 `0c1d48d864fc02bbea64ae53c59d92f921cadf2edaee2cb6b73462a0bf386fe1`）是已冻结证据。
v2 以 `supersedes` 字段记录其与 v1 的溯源关系，并保证：

- 数据集、训练预算、评估运行时设置、14 个条件、统计口径与 v1 **逐项一致**；
- v1 队列的结果继续有效，不受影响；
- 唯一的新增是 `methods` 列表末尾追加 `lenet5`。

冻结后 v1 文件字节未变（复核见下）。

## 二、v2 的新增内容

### 1. LeNet-5 纯电子参考基线（对应评审意见 #3）

外部评审指出：784-18-10 MLP 仅 14 320 参数、MNIST 干净准确率 94.67%，作为电子参考偏弱，
"混合模型与纯电子基线表现相近"可能是基线选择造成的人为结论。

v2 新增 `lenet5` 方法：经典 LeNet-5 结构（2 卷积 + 3 全连接），隐藏层 120 单元，
34 622 个可训练参数，与混合模型 14 698 参数处于同一量级。

该方法是**纯电子**的，因此与 `electronic` 一样只进入干净条件评估，不进入光学扰动方案。

### 2. 量化感知训练对照（对应评审意见 #2）

新增命令行开关 `--train-phase-quantization-levels`：

- 训练阶段对相位执行均匀量化，使模型直接面向离散部署掩模优化；
- 验证与测试阶段保持干净连续相位，从而与既有方法的 clean 指标口径一致；
- 在冻结方案中复用现有的 `phase_quantization_16/8/4` 条件完成评估。

实现细节：`PerturbationConfig.train_phase_quantization_levels` 是**训练期专用**字段，
不进入 `serialized_perturbation_config` 的部署语义负载（该负载只含 5 个部署字段）。
训练期量化单独记录在 manifest 的
`training_perturbations.training_phase_quantization_levels` 与
`training_perturbations.quantization_aware_training` 字段中。

### 3. 训练种子扩展（对应评审意见 #1）

种子从 42/43/44 扩展到 **42–46（n=5）**，满足预注册协议中"正式比较目标 5 个种子"的要求。
42–44 与 v1 重叠，用于逐种子可比性核验；45–46 为新增独立训练种子。

## 三、设备决策与实测依据

### 训练：GPU

v2 队列在云 GPU（RTX 4060 Ti 16GB）上训练。**训练阶段没有 `--deterministic`**，原因已实测：

- `adaptive_avg_pool2d_backward` 与 `grid_sampler_2d_backward` 在 CUDA 上没有确定性实现，
  `torch.use_deterministic_algorithms(True)` 会直接抛错；
- 因此 GPU 训练的可复现性依赖显式种子，而非确定性算法开关；
- 未启用 `--allow-tf32`，矩阵乘保持 FP32。

设备与软件环境记录在每个 checkpoint 相邻 manifest 中。

**CPU/GPU 训练一致性已核验**：在相同种子与配置下，GPU 训练得到的干净准确率与冻结 v1 的
CPU 训练结果逐位相同：

| 方法 | 种子 | GPU (v2) | CPU (v1) | 差异 |
|---|---:|---:|---:|---:|
| baseline_d2nn | 42 | 89.26 | 89.26 | 0.00 |
| baseline_d2nn | 43 | 89.77 | 89.77 | 0.00 |
| baseline_d2nn | 44 | 89.88 | 89.88 | 0.00 |
| robust_d2nn | 42 | 88.13 | 88.13 | 0.00 |
| robust_d2nn | 43 | 88.35 | 88.35 | 0.00 |
| robust_d2nn | 44 | 88.45 | 88.45 | 0.00 |

这意味着 v2 与 v1 队列可以直接比较，不存在设备异质性混淆。

### 评估：光学鲁棒性评估使用 GPU，clean 评估保持 CPU

`device_parity_probe.py` 在基线、混合与 QAT 三个 checkpoint 上，对
`clean_continuous`、`lateral_shift_0p50px`、`phase_quantization_4`、`mixed_stress`
四个条件比较 CPU 与 CUDA 的逐 draw 指标：

```text
baseline_d2nn   cpu 2.41 s   gpu 0.75 s   3.22x   全部条件 delta = 0.00e+00
hybrid          cpu 2.05 s   gpu 0.46 s   4.45x   全部条件 delta = 0.00e+00
hybrid_qat      cpu 2.35 s   gpu 0.45 s   5.28x   全部条件 delta = 0.00e+00
合计            cpu 6.80 s   gpu 1.65 s   4.11x   逐位相同
```

结论：**CPU 与 CUDA 评估在准确率与 macro-F1 上逐位相同，CUDA 约快 4.1 倍**，
故 v2 的鲁棒性评估采用 GPU。

`evaluate_clean.py` 仍强制 CPU：其目的是产出固定软件环境下的 CPU 软件仿真开销记录，
属于被测量本身，不能改为 GPU。该限制由代码显式校验，v2 保持不变。

## 四、尚未纳入 v2 的评审意见

| 意见 | 状态 | 说明 |
|---|---|---|
| #1 训练种子扩展到 10 个 | 部分完成 | 当前 n=5，达到预注册最低要求；扩展到 42–51 为后续工作 |
| #4 Fashion-MNIST 冻结比较 | 未开始 | 需要在第二个数据集上重建整条流水线 |
| #5 相位滤波 / sharpness-aware 对照 | 未开始 | 需实现已有方法作为同协议对照 |
| #6 池化分辨率与电子头深度消融 | 未开始 | 需新增混合模型变体 |

## 五、v1 完整性复核

```text
FORMAL_EXPERIMENT_PROTOCOL.json  SHA-256
0c1d48d864fc02bbea64ae53c59d92f921cadf2edaee2cb6b73462a0bf386fe1
```

与 v1 冻结时登记的哈希一致，v2 的生成未修改 v1 的任何字节。
