# 论文图表选择方案

> 生成时间：2026-06-06
> 目标：从 60+ 张图中选出 10–12 张用于论文正文，其余放入附录

## 1 选择原则

1. **每个实验至少 1 张**：确保完整覆盖
2. **优先选择"一图多信息"的综合图**：如 multi-panel 组图
3. **避免重复信息**：如覆盖率曲线和指标表信息重叠时只选一个
4. **关键结论必须有图支撑**：论文中引用的核心数据都能追溯到图表

## 2 推荐正文图表（12 张）

### 系统概览（2 张）

| 编号 | 来源 | 文件 | 内容 | 论文用途 |
|------|------|------|------|---------|
| Fig.1 | sim_demo | `7_pipeline.png` | EGO-Swarm vs Ours 系统流水线对比 | 系统架构介绍，展示创新点 |
| Fig.2 | sim_demo | `demo_snapshots.png` 或 `1_snapshot_step*.png` 拼图 | 探索过程关键时刻快照（6 帧） | 直观展示系统运行效果 |

### Exp 1: 单机 vs 多机（1 张）

| 编号 | 来源 | 文件 | 内容 | 论文用途 |
|------|------|------|------|---------|
| Fig.3 | output/figures | `fig1_scalability.png` | 1/2/3 UAV 覆盖率曲线 | 核心结论：多机加速 |

### Exp 2: 策略对比（2 张）

| 编号 | 来源 | 文件 | 内容 | 论文用途 |
|------|------|------|------|---------|
| Fig.4 | exp2 | `fig_coverage_curves.png` | 6 种方法覆盖率曲线 | 消融对比的核心可视化 |
| Fig.5 | exp2 | `fig_bar_charts.png` | 8 指标柱状图 | 定量对比各方法 |

### Exp 3: 深度模式（1 张）

| 编号 | 来源 | 文件 | 内容 | 论文用途 |
|------|------|------|------|---------|
| Fig.6 | exp3 | `fig_quality_bars.png` | IoU/Precision/Recall 柱状图 | 深度精度级联影响 |

### Exp 4: 目标检测（1 张）

| 编号 | 来源 | 文件 | 内容 | 论文用途 |
|------|------|------|------|---------|
| Fig.7 | exp4 | `fig_task_timeline.png` | 任务状态切换时序图 | 检测→确认→协同响应流程 |

### Exp 4B: 跟踪/融合/围捕（3 张）

| 编号 | 来源 | 文件 | 内容 | 论文用途 |
|------|------|------|------|---------|
| Fig.8 | exp4b | `fig_4b1_rmse_bars.png` | Detection-only vs KF vs CM-KF RMSE | 跟踪方法对比 |
| Fig.9 | exp4b | `fig_4b2_fusion_bars.png` | 4 种融合方法 RMSE + CovTrace | 多机融合效果 |
| Fig.10 | exp4b | `fig_enc_s350_snapshot.png` | 围捕编队快照（编队形成后） | 围捕效果展示 |

### Exp 5: 安全性（1 张）

| 编号 | 来源 | 文件 | 内容 | 论文用途 |
|------|------|------|------|---------|
| Fig.11 | exp5 | `fig_combined_safety.png` | 综合安全性多面板图 | 障碍物距离 + 机间距 |

### Exp 6: 鲁棒性（1 张）

| 编号 | 来源 | 文件 | 内容 | 论文用途 |
|------|------|------|------|---------|
| Fig.12 | exp6 | `fig_box_plots.png` | 多种子箱线图 | 鲁棒性统计 |

## 3 附录图表

以下图表放入论文附录（按实验分组）：

### 附录 A: 系统演示补充
- `sim_demo/8_comparison_table.png` — 功能对比表
- `sim_demo/9_final_map.png` — 高清最终地图

### 附录 B: 参数敏感性详情
- `exp1/fig_snapshots_n*_v*.png` — 各配置快照（12 张）
- `exp1/fig_path_length_bars.png` — 路径长度柱状图
- `exp1/fig_metrics_table.png` — 指标汇总表

### 附录 C: 策略对比详情
- `exp2/fig_final_maps.png` — 最终地图对比
- `exp2/fig_metrics_table.png` — 指标表

### 附录 D: 深度模式详情
- `exp3/fig_depth_grid_illustration.png` — 栅格对比
- `exp3/fig_final_map_comparison.png` — 最终地图
- `exp3/fig_coverage_curves.png` — 覆盖率曲线
- `exp3/fig_metrics_table.png` — 指标表

### 附录 E: 目标检测详情
- `exp4/fig_occlusion_illustration.png` — 遮挡等级示意
- `exp4/fig_localization_scatter.png` — 定位散点
- `exp4/fig_confidence_error_curves.png` — 置信度曲线
- `exp4/fig_cooperative_response.png` — 协同响应地图
- `exp4/fig_metrics_table.png` — 指标表

### 附录 F: 跟踪/融合/围捕详情
- `exp4b/fig_4b1_*_curves.png` — KF 跟踪曲线（3 张）
- `exp4b/fig_4b1_continuity.png` — 跟踪连续性对比
- `exp4b/fig_4b1_R_trace.png` — CM-KF R 矩阵迹（审计图）
- `exp4b/fig_4b1_occlusion_period_error.png` — 遮挡期误差（审计图）
- `exp4b/fig_4b2_per_seed_rmse.png` — 融合 per-seed（审计图）
- `exp4b/fig_4b2_error_timeline.png` — 融合误差时序（审计图）
- `exp4b/fig_4b3_phase_angles_over_time.png` — 围捕角度时序
- `exp4b/fig_4b3_distance_to_slot.png` — 槽位距离时序
- `exp4b/fig_4b3_fsm_timeline.png` — FSM 状态时间线
- `exp4b/fig_enc_s*.png` — 围捕快照（6 张，选 1 入正文已够）

### 附录 G: 安全性与鲁棒性详情
- `exp5/fig_inter_uav_pairwise.png` — 机间距成对时序
- `exp5/fig_per_uav_obs_dist.png` — 各机障碍距离
- `exp5/fig_safety_quadrant.png` — 安全象限图
- `exp5/fig_safety_table.png` — 安全指标表
- `exp6/fig_coverage_curves.png` — 多种子覆盖率曲线
- `exp6/fig_final_map_mosaic.png` — 地图拼接
- `exp6/fig_stats_table.png` — 统计表

## 4 图表格式建议

| 项目 | 建议 |
|------|------|
| 格式 | PDF（矢量图，已有 fig1/fig2 的 PDF 版本） |
| 分辨率 | ≥ 300 DPI（PNG 备选） |
| 字体大小 | 标签 ≥ 8pt，标题 ≥ 10pt |
| 配色 | 保持一致的色盘（当前已使用 `setup_style()` 统一） |
| 尺寸 | 单栏图 ≤ 3.5in 宽，双栏图 ≤ 7in 宽 |

## 5 缺失的 PDF 版本

当前仅 `fig1_scalability.pdf`、`fig2_safety.pdf`、`fig3_trajectory_map.pdf` 有 PDF 版本。如需投稿，需将所有 12 张正文图转为 PDF。这可以通过修改 `save_figure()` 函数统一输出 PDF 实现。
