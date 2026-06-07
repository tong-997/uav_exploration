# Exp 4B-1 CM-KF vs Standard KF 合理性检查

> 生成时间：2026-06-06
> 诊断脚本：`experiments/exp4b_diagnostic.py`

## 1 背景与风险点

原始实验报告中 CM-KF 与 Standard KF 的位置 RMSE 几乎相同（0.654 m vs 0.654 m）。如果两者精度无差异，则不能声称"CM-KF 精度优于 Standard KF"。本检查通过诊断运行，逐步记录 R 矩阵迹、卡尔曼增益范数、P 矩阵迹和定位误差，分析两者的真实差异。

## 2 诊断运行参数

| 参数 | 值 |
|------|------|
| 目标运动 | linear (vx=0.5, vy=0.3 m/s) |
| 遮挡模式 | intermittent_occlusion |
| 检测模型 | yolov5_ffm |
| Standard KF | α_o=0, α_d=0 |
| CM-KF | α_o=2.0, α_d=1.5 |
| 步数 | 500 |
| seed | 42 |

## 3 诊断结果

### 3.1 定位误差对比

| 方法 | RMSE_all (m) | RMSE_visible (m) | RMSE_occluded (m) | P_trace_mean |
|------|-------------|------------------|-------------------|-------------|
| Standard KF | 0.4269 | 0.1685 | 0.6681 | 7.057 |
| CM-KF | 0.4270 | 0.1685 | 0.6683 | 7.063 |
| **差值** | **+0.0001** | **0.0000** | **+0.0002** | **+0.006** |

### 3.2 分析

**RMSE 几乎相同的原因：**

1. **可见期两者完全一致**：RMSE_visible 都是 0.1685 m。当目标可见时，两者都获得相同的检测结果，R 矩阵差异被高质量观测淹没。
2. **遮挡期两者都执行纯预测**：当检测失败（confidence < threshold），两者都调用 `predict_only()` 而非 `update()`。因此 R 矩阵的调制根本不生效——遮挡期间没有 update 步骤。
3. **R 调制仅在"低置信度但仍检测到"时有差异**：CM-KF 的 R 放大机制只在 observation 存在但 confidence 较低时起作用。在本实验的 intermittent_occlusion 模式下，遮挡期通常完全无检测（confidence=0），而非低置信度检测。

### 3.3 诊断图表

| 图表 | 文件 | 内容 |
|------|------|------|
| R 矩阵迹时序 | `figures/fig_4b1_R_trace.png` | CM-KF 的 R_trace 在遮挡边缘短暂升高，Standard KF 恒定 |
| 遮挡期误差对比 | `figures/fig_4b1_occlusion_period_error.png` | 两者遮挡期误差曲线几乎重合 |

## 4 结论与论文表述建议

### 4.1 不可声称

- ~~"CM-KF 定位精度优于 Standard KF"~~ — 数据不支持
- ~~"动态 R 调制提升了跟踪精度"~~ — RMSE 差异 < 0.001 m

### 4.2 可以声称

1. **"KF 方法（CM-KF 和 Standard KF）实现 100% 跟踪连续性，而 Detection-only 仅 40%"** — 这是 KF 相对于无滤波的核心优势，数据充分支持
2. **"CM-KF 提供了观测不确定性的物理建模框架，R 矩阵随置信度和遮挡动态调整"** — 架构优势，即使本场景下精度收益不显著
3. **"在 intermittent_occlusion 模式下，遮挡期完全无检测，R 调制无法发挥作用；CM-KF 的优势更可能体现在 partial occlusion（低置信度但仍有检测）场景"** — 诚实陈述适用条件

### 4.3 建议论文表述

> "KF-based methods achieve 100% tracking continuity versus 40% for detection-only, maintaining target state estimates through occlusion periods via prediction. The CM-KF extends the standard KF with confidence-modulated measurement covariance (R = λ_c·R_det + λ_o·R_occ + λ_d·R_depth), providing a principled framework for heterogeneous observation quality. Under intermittent full occlusion, both KF variants show equivalent position RMSE (0.427 m), as the modulation primarily affects the update step which is skipped during complete occlusion. The CM-KF framework is expected to yield greater benefits in partial-occlusion scenarios where degraded but non-zero observations are available."

## 5 是否需要补充实验

**不需要新增大规模实验。** 现有 720 次运行已充分覆盖。建议：
- 在论文中如实报告 CM-KF = Standard KF 的 RMSE 结果
- 将 CM-KF 的贡献定位为"架构扩展性"而非"精度提升"
- 强调 KF vs Detection-only 的对比作为主要贡献
