# Exp 3 深度模式结果真实性检查

> 生成时间：2026-06-06

## 1 风险点

ResUNet-Depth 模式覆盖率仅 34.3%，远低于 Ray-cast 的 84.6%。如果在论文中声称"ResUNet 提升建图质量"，将与实验数据矛盾。需要明确 ResUNet-Depth 的真实表现和正确结论。

## 2 数据审查

| Depth Mode | Coverage | Map IoU | Occ Precision | Plan Success | Replan |
|------------|----------|---------|---------------|-------------|--------|
| GT-Depth | 90.1% | 1.000 | 1.000 | 100.0% | 36 |
| Ray-cast | 84.6% | 0.987 | 0.700 | 93.7% | 150 |
| Noisy-Depth | 26.1% | 0.711 | 0.092 | 65.6% | 776 |
| ResUNet-Depth | 34.3% | 0.712 | 0.075 | 53.0% | 728 |

### 2.1 ResUNet-Depth 的问题链

```
结构化深度误差 → 虚假占据标记 (Occ Precision = 0.075)
  → A* 路径被阻断 (Plan Success = 53%)
    → UAV 频繁重规划 (728 次)
      → 探索停滞 (Coverage = 34.3%)
```

### 2.2 ResUNet-Depth vs Noisy-Depth 对比

| 指标 | Noisy-Depth | ResUNet-Depth | 谁更差 |
|------|------------|---------------|--------|
| Coverage | 26.1% | 34.3% | Noisy |
| Map IoU | 0.711 | 0.712 | 相当 |
| Occ Precision | 0.092 | 0.075 | ResUNet |
| Occ Recall | 0.908 | 0.442 | ResUNet |
| Plan Success | 65.6% | 53.0% | ResUNet |
| Obs Safe | 47.7% | 28.1% | ResUNet |

**ResUNet-Depth 在 4/6 指标上劣于 Noisy-Depth。** 唯一"优势"是覆盖率略高（34.3% vs 26.1%），但这可能源于结构化误差分布不均匀，某些区域恰好误差较小。

## 3 结论审查

### 3.1 不可声称

- ~~"ResUNet 深度估计提升了建图质量"~~ — Occ Precision 最低（0.075）
- ~~"ResUNet-Depth 是可行的深度感知方案"~~ — 覆盖率仅 34.3%，规划成功率 53%
- ~~"深度学习方法优于传统方法"~~ — Ray-cast 在所有指标上远优于 ResUNet

### 3.2 可以声称

1. **"实验 3 量化了深度精度对下游探索的级联影响"** — 这是实验的核心价值
2. **"GT-Depth → Ray-cast → Noisy/ResUNet 形成了精度梯度，覆盖率从 90% 降至 26-34%"** — 数据支持
3. **"Ray-cast 作为系统默认模式，IoU = 0.987，覆盖率 84.6%，是可靠的感知基线"** — 正面结论
4. **"深度噪声通过'虚假障碍 → 路径阻断 → 探索停滞'的链条级联放大，Occ Precision 从 1.0 降至 0.075 导致覆盖率损失 >55%"** — 机理分析
5. **"ResUNet-Depth 的结构化误差（系统偏差 + 异常值）比随机高斯噪声对建图质量的破坏更严重（Occ Precision 0.075 < 0.092），因为结构化伪影容易形成大面积虚假障碍区域"** — 有价值的分析

### 3.3 建议论文表述

> "Experiment 3 quantifies the cascading impact of depth accuracy on exploration performance. The ray-cast baseline achieves Map IoU = 0.987 and 84.6% coverage, confirming its reliability as the system's default perception mode. Both Noisy-Depth and ResUNet-Depth suffer catastrophic coverage degradation (26–34%) due to false occupancy markings (Precision < 10%), which block A* path planning and halt exploration. Notably, ResUNet's structured errors (systematic bias + outliers) produce lower occupancy precision (0.075) than random Gaussian noise (0.092), as coherent depth artifacts create contiguous false obstacle regions that are more disruptive to path planning. This result underscores that for 2D grid-based exploration, depth sensing accuracy is a binding constraint — the exploration algorithm's efficiency is upper-bounded by the perception module's fidelity."

## 4 现有结论是否需要修改

查看 `实验3_深度输入模式分析.md` 的结论部分：

- 结论 1-5：准确，无需修改
- 结论 6（"高占据召回 ≠ 高建图质量"）：准确，是重要的 insight

**现有结论措辞已经诚实，未声称 ResUNet 优于 Ray-cast。** 无需修改。

## 5 论文定位建议

Exp 3 在论文中的角色：
- **不是**展示深度学习感知优势的实验
- **而是**量化感知精度对探索系统影响的灵敏度分析（sensitivity analysis）
- 核心贡献：证明"建图质量是覆盖率的瓶颈"，为未来高精度深度感知研究提供动机
