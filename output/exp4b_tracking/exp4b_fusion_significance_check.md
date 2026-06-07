# Exp 4B-2 融合方法显著性检查

> 生成时间：2026-06-06
> 诊断脚本：`experiments/exp4b_diagnostic.py`

## 1 背景与风险点

原始报告中 info_filter 的 post_fusion_rmse = 0.131 m，naive_average = 0.132 m，差异仅 0.001 m。需检验：
1. 这一差异在多种子下是否稳定？
2. 各方法的 per-seed 分布是否重叠？
3. 能否声称"信息滤波显著优于其他方法"？

## 2 Per-Seed RMSE 统计

| 方法 | Mean RMSE (m) | Std (m) | Min | Max |
|------|-------------|---------|-----|-----|
| info_filter | 0.1040 | 0.0260 | 0.071 | 0.141 |
| single_best | 0.1055 | 0.0238 | 0.074 | 0.138 |
| naive_average | 0.1061 | 0.0262 | 0.072 | 0.143 |
| covariance_intersection | 0.1064 | 0.0238 | 0.075 | 0.141 |

### 2.1 观察

- **所有方法 mean RMSE 在 0.104 ~ 0.106 m 范围内**，差异 < 0.003 m
- **标准差 ~ 0.025 m**，远大于方法间差异（0.002 m）
- **Min/Max 范围高度重叠**：各方法最差种子 RMSE（~0.14 m）远大于方法间差异

### 2.2 诊断图表

| 图表 | 文件 | 内容 |
|------|------|------|
| Per-seed RMSE 柱状图 | `figures/fig_4b2_per_seed_rmse.png` | 5 个种子各方法 RMSE 对比，误差棒重叠 |
| 误差时间线 | `figures/fig_4b2_error_timeline.png` | 融合后误差随步数变化，各方法曲线交织 |

## 3 显著性分析

### 3.1 效应量评估

方法间最大差异 = 0.0024 m（info_filter vs covariance_intersection）
标准差均值 = 0.025 m
Cohen's d ≈ 0.0024 / 0.025 = **0.096**（极小效应）

### 3.2 样本量限制

- 仅 5 个种子（N=5），无法进行有统计力的假设检验
- 即使 t 检验 p 值偏低，也不具备实际意义（效应量太小）

## 4 结论与论文表述建议

### 4.1 不可声称

- ~~"信息滤波显著优于其他融合方法"~~ — 差异被种子间变异淹没
- ~~"协方差交叉表现最差"~~ — 仅差 0.002 m，无统计意义

### 4.2 可以声称

1. **"多机融合一致优于单机估计"** — single_best 的 cov_trace (0.499) 远大于任何融合方法 (~0.19)，不确定性减半是稳健的结论
2. **"四种融合方法在位置 RMSE 上表现相当（0.104 ± 0.026 m），但信息滤波在协方差迹上最优（0.188 vs 0.499）"** — 区分精度和不确定性估计
3. **"融合的主要收益在于降低状态估计不确定性（协方差迹从 0.499 降至 0.188），而非直接提升位置精度"** — 这是一个更准确的技术论断

### 4.3 建议论文表述

> "All four fusion methods achieve comparable position RMSE (0.104 ± 0.026 m, N=5 seeds), with inter-method differences (< 0.003 m) well within the seed-to-seed variability. The primary benefit of multi-UAV fusion lies in uncertainty reduction: the information filter achieves the lowest covariance trace (0.188), a 62% reduction from the single-best baseline (0.499), providing more reliable state estimates for downstream encirclement planning. Among fusion methods, the information filter is recommended for its natural handling of heterogeneous observation quality and computational simplicity."

## 5 是否需要补充实验

**不需要。** 现有 60 次融合运行已充分表明方法间差异不显著。增加种子数只会进一步确认这一结论。建议在论文中诚实报告各方法 RMSE 接近，将贡献聚焦于协方差缩减而非精度提升。
