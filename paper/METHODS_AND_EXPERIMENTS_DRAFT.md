# 方法与实验协议

> 正式结果已经冻结。本文所有实验均为公开 MNIST 数据上的数值仿真；结果章节只能引用 `results/formal_mnist_v1/` 中经过哈希核验的机器产物。

## 任务与比较对象

给定灰度输入图像 x in [0,1]^(1x28x28)，模型输出 10 个类别分数。我们在相同数据划分、训练预算和随机种子下比较四种方法：标准 D2NN（baseline_d2nn）、扰动感知 D2NN（robust_d2nn）、带电子分类头的混合模型（hybrid）和纯电子 MLP（electronic）。纯电子模型只参与 clean classification；它不进入 optical robustness plan。

## 标量衍射前端

光学模型先将输入幅度嵌入 64x64 采样平面，再以复振幅表示场。第 l 个相位层使用可学习相位 phi_l，其复透射率为 T_l = exp(i * phi_tilde_l)，其中 phi_tilde_l = Q(phi_l) + epsilon_l。epsilon_l 是相位噪声，Q 是可选的均匀相位量化；该顺序与正式评估代码的“先量化、后加噪”实现一致。

层间传播采用标量 Rayleigh-Sommerfeld 卷积。实现中的连续核为 h_d(dx,dy) = -i * (Delta^2/lambda) * (d/r^2) * exp(i*k*r)，其中 r = sqrt(dx^2 + dy^2 + d^2)，k = 2*pi/lambda，Delta 是采样间距。正式协议使用 FFT 卷积，波长 lambda = 0.75 mm，采样间距 Delta = 0.4 mm，名义层间距、输入距离和输出距离均为 30 mm，相位层数为 3，平面尺寸为 64x64。这些参数是数值仿真配置，不是制造系统标定值。

经过最后一层和输出传播后，输出强度为 I = |u|^2。十个固定探测区域形成类别能量 s_c = sum(M_c * I)，其中 M_c 是环形排列的第 c 个探测掩膜。纯 D2NN 将归一化能量的对数作为交叉熵 logits：z_c = log(s_c / (sum_j s_j + eps) + eps)。

## 混合与纯电子模型

Hybrid 模型从输出强度提取 8x8 自适应平均池化特征，并按每个样本的特征均值归一化。电子头为 64->32->10 的 MLP，中间层使用 GELU，输出 logits 经 softmax 得到类别分数。该设计把光学传播与低维电子分类明确分开，因而可以单独记录中间表示规模和电子参数量。

Electronic 模型直接处理原始 1x28x28 输入，结构为 Flatten、784->18 线性层、GELU 和 18->10 线性层。它作为不含光学前端的参数规模透明基线，并只提供 clean forward。

## 扰动模型与训练目标

每个部署 draw 独立采样横向错位和层间距误差。两者分别服从配置范围内的对称均匀分布。相位噪声和探测器噪声服从零均值高斯分布。相位量化使用 L >= 2 个均匀等级；相位层边界外的平移区域使用单位复透射率填充。探测器噪声按样本特征均值缩放后相加，并将能量截断到非负值。

分类训练使用 L = alpha * L_MSE + beta * L_CE + gamma * R_phase，其中 L_MSE 比较归一化类别能量与 one-hot 标签，L_CE 作用于模型 logits，R_phase 是相位水平与垂直方向的圆周差平方均值。正式协议固定 alpha = 1、beta = 0.1、gamma = 0.01。baseline 和 hybrid 使用 clean training；robust_d2nn 在训练中联合采样横向错位上限 0.5 pixel、层间距误差上限 1e-4 m、相位噪声标准差 0.05 rad 和相对探测器噪声标准差 0.01。electronic 不使用光学扰动。

## 预注册数据与训练

实验使用 torchvision MNIST 的 60,000 个训练样本和 10,000 个测试样本。每个训练 seed 通过固定 seed 的 `random_split` 将训练侧划分为 55,000 个训练样本和 5,000 个验证样本；测试集只在模型选择结束后使用。正式训练固定 3 个 epoch、batch size 128、学习率 0.01、DataLoader `num_workers=0`、确定性 PyTorch 算法和 seed 42、43、44。验证集按 clean accuracy、contrast 和较晚 epoch 的预先声明顺序选择 checkpoint。

## 冻结评估与统计单位

光学模型共享一个冻结 plan：14 个条件共 34 个 draw，覆盖 clean、横向错位、层间距误差、相位噪声、相对探测器噪声、相位量化和两种混合压力条件。所有 optical-compatible 方法使用完整 10,000 个测试样本；电子模型通过独立 clean-only evaluator 处理。plan identity 为 `robustness-plan-a8120eb01ea7093419fe`，并绑定冻结协议 SHA-256 `0c1d48d864fc02bbea64ae53c59d92f921cadf2edaee2cb6b73462a0bf386fe1`。

每个 draw 输出逐样本 prediction、整数混淆矩阵、accuracy、macro-F1、mean cross-entropy 和 mean contrast。macro-F1 对全部 10 个 score classes 计算，零分母类别的 F1 记为 0。部署 draw 先在同一个训练 seed 内求均值，再跨训练 seed 报告均值、标准差、每个 seed 的原始值和双侧 95% Student-t 区间。训练 seed 是唯一独立统计单位；测试样本、部署 draw 和计时 repeats 均不是独立重复。

统计分析不使用测试样本级显著性检验，也不对 14 个条件执行假设检验。每种方法的三次独立训练运行提供 seed-level estimate；跨方法差值按相同 training seed 配对。结果文件同时保留每个 seed 的原始值和区间。三 seed 的 Student-t 区间只作为有限重复仿真的边际不确定性描述，不提供跨条件同时覆盖，也无法在三个观测下可靠检验正态假设。由于未执行假设检验，本文不报告 p 值或多重比较校正。Python、PyTorch、SciPy 版本和 CPU 环境由每次 evaluation manifest 自动记录；没有生物学重复、实体样本随机化或盲法步骤。

## CPU 软件仿真计时边界

正式 evaluator 在固定 CPU 环境中将 PyTorch intra-op threads 设为 1，先执行 1 次完整测试集 warmup，再执行 5 次完整测试集 `forward_with_metrics`。计时包含 DataLoader 迭代和 forward 输出形状检查，不包含指标聚合和 artifact 写盘。该范围不同于冻结协议中用于选择 draw 数量的端到端 pilot 预算。报告每次 elapsed time 的中位数、四分位距和原始 repeats；这些数值只表示固定环境下的软件仿真开销，不表示真实光学硬件端到端延迟、边缘设备性能、功耗或能效。

## 可复现产物

每个正式训练运行保留 checkpoint、相邻训练 manifest、stdout/stderr 日志和 SHA-256。每个 clean 或 robustness evaluation 保留 evaluation manifest、逐样本 JSONL、汇总指标、timing（如适用）和源文件哈希。任何结果进入论文前，都必须在 `EVIDENCE_LEDGER.md` 中登记 Metric ID；中断运行和负结果保留在日志中，不补写缺失 manifest。
