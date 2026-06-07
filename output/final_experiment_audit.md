# 实验审计终审报告

> 生成时间：2026-06-06
> 审计范围：Exp 1–6 + Exp 4B（共 1012 次运行）

## 1 审计摘要

| 审计项 | 状态 | 严重问题 | 文档 |
|--------|------|---------|------|
| (一) 完整性检查 | ✅ 通过 | 0 | — |
| (二) CM-KF 合理性 | ✅ 完成 | 1 | `exp4b_sanity_check.md` |
| (三) 融合显著性 | ✅ 完成 | 1 | `exp4b_fusion_significance_check.md` |
| (四) 围捕相位解释 | ✅ 完成 | 0 | `exp4b_phase_uniformity_explanation.md` |
| (五) 深度模式真实性 | ✅ 通过 | 0 | `depth_mode_truthfulness_check.md` |
| (六) 策略结论检查 | ✅ 完成 | 2 | `strategy_conclusion_check.md` |
| (七) 参数一致性 | ✅ 完成 | 1 | `parameter_consistency_check.md` |
| (八) 论文图表选择 | ✅ 完成 | 0 | `paper_figure_selection.md` |
| 接口说明 | ✅ 完成 | 0 | `实验4B_与主系统接口说明.md` |

**总计：5 个需要修正的问题，0 个需要新增实验的问题。**

## 2 三大风险点处理

### 风险点 1：CM-KF RMSE = Standard KF RMSE

**发现**：两者 RMSE 差异 < 0.001 m（0.4270 vs 0.4269），在 intermittent_occlusion 模式下无精度差异。

**原因**：遮挡期完全无检测（confidence=0），两者都执行 `predict_only()`，R 调制不生效。CM-KF 的优势仅在"低置信度但仍有检测"时体现。

**处理**：
- ❌ 删除声称"CM-KF 精度优于 Standard KF"
- ✅ 保留"KF vs Detection-only 的 100% vs 40% 跟踪连续性"作为核心贡献
- ✅ 将 CM-KF 定位为"架构扩展"而非"精度提升"
- 📊 已生成诊断图表：`fig_4b1_R_trace.png`、`fig_4b1_occlusion_period_error.png`

### 风险点 2：Greedy Success=100% > Ours=80%

**发现**：Greedy-Frontier 在成功率（100% vs 80%）和速度（384 步 vs 432 步）上均优于 Ours。

**原因**：Greedy 选择最近前沿，路径简单短距，A* 成功率高（Replan=45 vs 150）。Ours 的 Utility 评分选择远前沿增加了规划失败风险。

**处理**：
- ❌ 删除声称"本系统整体最优"
- ✅ 改为"Ours 在路径效率（457m 最短）和安全间距（15.5m 最大）上最优"
- ✅ 诚实承认"Greedy 在成功率和速度上更优"
- ✅ 修正 Voronoi 结论为"贡献有限，本场景下轻微负面"

### 风险点 3：ResUNet-Depth 表现差

**发现**：覆盖率 34.3%，Occ Precision 0.075（最低），Plan Success 53%。

**原因**：结构化深度误差产生大面积虚假障碍物，阻断 A* 路径规划。

**处理**：
- ❌ 删除任何声称"ResUNet 提升建图"
- ✅ 现有结论已诚实（"深度噪声严重影响建图质量"），无需修改
- ✅ 将 Exp 3 定位为"灵敏度分析"而非"深度学习优势展示"

## 3 已修正的参数错误

| 文档 | 位置 | 原值 | 修正值 |
|------|------|------|--------|
| `仿真实验总体报告.md` | 第 12 行 | FOV=120°, Range=10m | FOV=90°, Range=15m, 60 条射线 |

## 4 需要修正的结论（建议但未自动修改）

以下建议修正在各实验 MD 文档中手动确认：

### 4.1 Exp 2 策略对比

| 结论编号 | 原文 | 建议修正 |
|---------|------|---------|
| 4 | "Voronoi 影响有限但正面" | "Voronoi 影响有限，本场景下轻微负面（80% vs 90%）" |
| 6 | "系统各模块协同互补" | "完整系统在路径效率和安全间距上最优，但成功率低于贪心策略" |

### 4.2 Exp 4B 跟踪

| 项目 | 建议 |
|------|------|
| CM-KF 描述 | 不声称精度优势，改为"架构扩展，提供观测不确定性建模框架" |
| 融合描述 | 不声称"信息滤波显著优于其他"，改为"融合方法精度相当，主要收益在协方差缩减" |

## 5 实验完整性确认

### 5.1 文件完整性

| 检查项 | 状态 |
|--------|------|
| 所有 experiment MD 文档存在 | ✅ |
| 所有 figure 目录非空 | ✅ |
| raw 数据目录存在 | ✅ |
| experiment_results.md 包含所有实验 | ✅ |
| 仿真实验总体报告.md 包含所有实验 | ✅ |
| 实验任务书对照检查报告.md 已更新 | ✅ |

### 5.2 数据量统计

| 实验 | 运行次数 | 种子数 | 配置数 |
|------|---------|--------|--------|
| Exp 1 | 90 | 10 | 9 (3×3) |
| Exp 2 | 60 | 10 | 6 |
| Exp 3 | 40 | 10 | 4 |
| Exp 4 | 16 | 1 | 16 (4×4) |
| Exp 4B-1 | 720 | 5 | 144 (3×4×4×3) |
| Exp 4B-2 | 60 | 5 | 12 (3×4) |
| Exp 4B-3 | 15 | 5 | 3 |
| Exp 5 | 1 | 1 | 1 |
| Exp 6 | 10 | 10 | 1 |
| **合计** | **1012** | — | — |

### 5.3 图表统计

| 类别 | 数量 |
|------|------|
| 正文推荐 | 12 张 |
| 附录推荐 | ~40 张 |
| 审计诊断图（新增） | 6 张 |
| 总计 | ~58 张 |

## 6 论文关键数据速查表

论文中可直接引用的核心数据（均有实验支撑）：

| 结论 | 数据 | 来源 |
|------|------|------|
| 3 UAV vs 1 UAV 加速 | 301 步 vs 未完成（16% at 800步） | Exp 1 |
| 3 UAV vs 2 UAV 加速 | 42.9%（301 vs 527 步） | Exp 1 |
| 零碰撞 | 全部实验 0 collisions | Exp 2, 5 |
| 机间安全率 | 100%（分布式防撞） | Exp 5 |
| 任务均衡 | 路径长度差异 < 9%（139.5–152.0 m） | Exp 1 |
| 路径效率 | Ours 457m（最短） | Exp 2 |
| 障碍膨胀关键性 | w/o Inflation: 0% success, 12.4% safe | Exp 2 |
| 深度精度瓶颈 | GT 90.1% → Noisy 26.1%（3.5× 差距） | Exp 3 |
| 检测响应 | 2 步（0.4s）从发现到响应 | Exp 4 |
| KF 跟踪连续性 | KF 100% vs Detection-only 40% | Exp 4B-1 |
| 融合协方差缩减 | 0.188 vs 0.499（62% 降低） | Exp 4B-2 |
| 围捕编队时间 | 静态目标 19 步 / 3.8s | Exp 4B-3 |
| 多种子鲁棒性 | 覆盖率 79.1% ± 14.1% | Exp 6 |

## 7 最终结论

本审计确认：

1. **实验数据真实可靠**，1012 次运行覆盖了探索、策略、感知、检测、跟踪、融合、围捕等全部模块
2. **三个风险点已处理**，结论表述建议已给出，避免了过度声称
3. **一处参数错误已修正**（FOV/Range），其余参数一致
4. **不需要新增实验**，现有数据充分支撑论文结论
5. **图表选择方案已制定**，12 张正文 + ~40 张附录

---

*审计文档索引：*

| 文档 | 路径 |
|------|------|
| CM-KF 合理性 | `output/exp4b_tracking/exp4b_sanity_check.md` |
| 融合显著性 | `output/exp4b_tracking/exp4b_fusion_significance_check.md` |
| 围捕相位解释 | `output/exp4b_tracking/exp4b_phase_uniformity_explanation.md` |
| 系统接口说明 | `output/exp4b_tracking/实验4B_与主系统接口说明.md` |
| 深度模式真实性 | `output/exp3_depth/depth_mode_truthfulness_check.md` |
| 策略结论检查 | `output/exp2_strategy/strategy_conclusion_check.md` |
| 参数一致性 | `output/parameter_consistency_check.md` |
| 论文图表选择 | `output/paper_figure_selection.md` |
| 本审计报告 | `output/final_experiment_audit.md` |
