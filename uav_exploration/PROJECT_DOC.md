# 多无人机协同探索系统 — 项目说明文档

> 最后更新: 2026-05-31 | 防撞架构: 分布式

## 1 项目概述

本项目实现了一个 **三机协同未知区域探索系统**，核心功能链路为：

```
前视深度感知 → 占据栅格构建 → 前沿检测 → Voronoi 区域分配 → A* 路径规划 → 分布式防撞 → GPS 航点记录
```

三架无人机从不同起始位置出发，各自携带前视深度传感器，边飞行边构建占据栅格地图，通过前沿探索（Frontier-based Exploration）策略选择下一个探索目标，利用 Voronoi 分区避免重复搜索，A* 规划安全路径，同时通过**分布式防撞机制**（各机独立决策, 无中心调度）保证多机间不碰撞。全程记录 GPS 航点，可导出为 JSON 文件。

系统支持两种运行模式：
- **仿真模式**（默认）：2D 射线投射模拟前视深度，用于算法验证和学术实验
- **真实模式**：接入 ONNX 双目深度估计模型（ResUNet），从真实图像推理深度

## 2 目录结构

```
uav_exploration/
├── config.py                        # 全局配置参数
├── run_exploration.py               # 主程序入口 (ExplorationSystem)
├── sim_demo.py                      # EGO-Swarm 风格仿真演示 (逐张出图)
├── requirements.txt                 # Python 依赖
│
├── simulation/                      # 仿真环境
│   └── sim_env.py                   #   2D 世界、障碍物、射线深度、运动学
│
├── perception/                      # 感知模块
│   ├── depth_estimator.py           #   深度估计 (仿真 / ONNX 推理)
│   └── occupancy_grid.py            #   占据栅格地图 (射线更新、融合、前沿检测)
│
├── planning/                        # 规划模块
│   ├── frontier_explorer.py         #   前沿探索目标选择
│   ├── path_planner.py              #   A* 路径规划 (障碍物膨胀)
│   └── waypoint_recorder.py         #   GPS 航点记录与导出
│
├── coordination/                    # 多机协调
│   ├── area_allocator.py            #   Voronoi 区域分配
│   └── deconfliction.py             #   分布式防撞 (ID优先级 + 轨迹广播 + 紧急避让)
│
├── docs/                            # 文档
│   └── comparison_ego_swarm.md      #   与 EGO-Planner-Swarm 对比分析
│
├── gen_figures.py                   # 学术实验图表生成 (4 张论文级图)
├── eval_experiments.py              # 完整实验评估脚本
│
└── output/                          # 运行输出
    ├── waypoints.json               #   航点数据
    ├── exploration_curve.png        #   探索进度曲线
    ├── final_map.png                #   最终地图
    ├── experiment_results.md        #   实验结果报告
    ├── figures/                     #   学术图表 (gen_figures.py 输出)
    │   ├── fig1_scalability.png     #     单机 vs 多机效率对比
    │   ├── fig2_safety.png          #     避障安全性分析 (4 子图)
    │   ├── fig3_trajectory.png      #     地图轨迹 + 每机指标
    │   └── fig4_robustness.png      #     多种子鲁棒性箱线图
    └── sim_demo/                    #   sim_demo 演示图 (每张独立)
        ├── 1_snapshot_step*.png     #     关键时刻地图快照 (6 张)
        ├── 2_coverage.png           #     覆盖率曲线
        ├── 3_safety.png             #     安全距离时序
        ├── 4_path_length.png        #     每机路径长度柱状图
        ├── 5_obs_clearance.png      #     每机障碍距离箱线图
        ├── 6_inter_uav.png          #     机间距时序
        ├── 7_pipeline.png           #     EGO-Swarm vs Ours 流水线对比
        ├── 8_comparison_table.png   #     功能与指标对比表
        └── 9_final_map.png          #     最终高清地图
```

## 3 核心算法

### 3.1 前视深度感知

每架无人机在正前方 **90° 视场角** 内发射 **60 条射线**，最远感知 **15 m**。

仿真模式下通过射线-圆相交检测（`sim_env.py: _ray_circle_intersect`）计算每条射线到最近障碍物的距离。真实模式下调用 disptools 的 `OnnxInference` 加载 ResUNet ONNX 模型，将双目图像推理为视差图，再经 `depth = fx × baseline / disp` 转换为深度。

```
输入: 无人机位置 + 朝向
输出: angles[60], depths[60]  (射线角度和对应深度)
```

**关键文件**: `simulation/sim_env.py` (第 63–88 行), `perception/depth_estimator.py`

### 3.2 占据栅格构建

100m × 100m 世界划分为 **200 × 200 栅格**（分辨率 0.5m），每个格子有三种状态：

| 值 | 含义 | 颜色 |
|----|------|------|
| 0 (UNKNOWN) | 未探索 | 浅蓝 |
| 1 (FREE) | 已探明无障碍 | 浅灰 |
| 2 (OCCUPIED) | 已检测到障碍 | 深灰 |

**射线更新**采用向量化实现（`occupancy_grid.py: update_from_rays`），核心思路：

1. 沿每条射线按 `0.8 × GRID_RES` 步长采样
2. 射线路径上的格子标记为 FREE
3. 射线终点（depth < SENSOR_RANGE）标记为 OCCUPIED

```python
# 向量化关键代码 — 一次性处理所有 60 条射线的所有采样点
all_px = drone_pos[0] + cos_a[:, None] * ts[None, :]   # (60, S)
all_py = drone_pos[1] + sin_a[:, None] * ts[None, :]   # (60, S)
mask = (ts[None, :] < depths[:, None]) & bounds_check
self.grid[iy[mask], ix[mask]] = FREE
```

相比逐射线 Python 循环，向量化实现速度提升约 **8.6 倍**（83ms → 9.7ms/步）。

**多机融合**：OCCUPIED 优先 > FREE > UNKNOWN，通过 `merge()` 方法合并多架无人机的局部地图为全局地图。

**关键文件**: `perception/occupancy_grid.py`

### 3.3 前沿检测

**前沿（Frontier）** 定义为 FREE 格子且四邻域内存在 UNKNOWN 格子的边界。检测流程：

1. 四方向膨胀 UNKNOWN 区域
2. 与 FREE 区域取交集得到前沿掩码
3. BFS 连通分量聚类
4. 过滤掉小于 `min_cluster_size`（默认 3）的簇
5. 返回每个簇的中心坐标和大小，按簇大小降序排列

**关键文件**: `perception/occupancy_grid.py` (第 70–120 行)

### 3.4 Voronoi 区域分配

为避免多架无人机重复搜索同一区域，系统根据当前无人机位置计算 **Voronoi 分区图**：

```
voronoi_map[iy, ix] = argmin_i  dist(cell, drone_i)
```

每个栅格被分配给距离最近的无人机。前沿检测后，每架无人机只从属于自己 Voronoi 区域的前沿中选择目标。若本区域无前沿，则回退到全局前沿。

**关键文件**: `coordination/area_allocator.py`

### 3.5 前沿目标选择

对每个候选前沿计算效用分数：

```
score = -w_dist × distance + w_size × log(1 + cluster_size) - penalty
```

- **距离代价**：越近越好（减少飞行时间）
- **大小奖励**：越大的前沿信息增益越高
- **冲突惩罚**：若目标距其他无人机已选目标 < 10m，施加惩罚 `50 × (1 - d/10)`

选择得分最高的前沿作为当前目标。

**关键文件**: `planning/frontier_explorer.py`

### 3.6 A* 路径规划

在占据栅格上执行 A* 搜索，支持 8 方向移动。关键设计：

1. **障碍物膨胀**：用 `scipy.ndimage.binary_dilation` 将 OCCUPIED 格子向外膨胀 `safety_margin` 格，构建代价地图
2. **起点/终点修正**：若起点或终点落在膨胀区内，自动搜索半径 10 格内最近可通行格子（`_find_nearest_free`）
3. **路径简化**：每隔 3 个格子取一个航点，减少冗余
4. **UNKNOWN 处理**：探索时将 UNKNOWN 视为 FREE（允许穿越未知区域）

**关键文件**: `planning/path_planner.py`

### 3.7 分布式防撞

采用去中心化架构，每架无人机仅依据本地通信范围内的邻居信息独立决策，无中心调度，类似 EGO-Swarm 的去中心化设计。

**第一层 — 轨迹广播 + 冲突检测**

- 每架无人机规划路径后广播自身轨迹（模拟 ROS Topic / 局域通信）
- 优先级规则：**ID 小的优先**（各机可独立计算, 无需协商, 保证一致性）
- 低优先级无人机检测到与高优先级邻居轨迹冲突时，插入 5 步原地等待

**第二层 — 反应式紧急避让**

- 每步各机独立检测通信范围内（80 m）邻居距离
- 若有邻居距离 < `SAFE_RADIUS × 1.5`，计算所有过近邻居的**合力排斥方向**
- 沿合力方向紧急偏移一步

**与集中式的区别**：
- 集中式需要全局位置信息和统一调度
- 分布式每机仅感知邻居，独立计算优先级和避让方向
- 优先级基于 ID（确定性规则），保证任意两机在无通信的情况下也能达成一致

**关键文件**: `coordination/deconfliction.py`

### 3.8 航点记录

每步记录无人机的 `(x, y, timestamp, heading, drone_id)`，距上一个航点 < 0.5m 时跳过。支持导出为 JSON 格式，以及计算每架无人机的总飞行距离。

**关键文件**: `planning/waypoint_recorder.py`

## 4 主循环流程

`ExplorationSystem.run_step()` 每步执行以下 7 个阶段：

```
┌──────────────────────────────────────────────────────────────┐
│ Step 1  感知       每架 UAV 前视射线投射 → 更新局部栅格        │
│ Step 2  融合       合并所有局部栅格 → 全局地图 → 同步回各机     │
│ Step 3  分区       根据当前位置计算 Voronoi 分区               │
│ Step 4  规划       前沿检测 → 目标选择 → A* 路径 (周期触发)    │
│ Step 5  轨迹广播   每机广播自身路径, 分布式冲突检测             │
│ Step 6  独立移动   各机独立: 检测邻居 → 紧急避让 / 沿路径前进  │
│ Step 7  记录       安全指标采集 + 航点记录                     │
└──────────────────────────────────────────────────────────────┘
```

终止条件：覆盖率 ≥ 90% 或步数 ≥ MAX_STEPS。

## 5 配置参数一览

| 类别 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| 世界 | `WORLD_SIZE` | 100.0 m | 正方形区域边长 |
| 世界 | `GRID_RES` | 0.5 m | 栅格分辨率 |
| 世界 | `GRID_N` | 200 | 栅格维度 (自动计算) |
| 无人机 | `N_DRONES` | 3 | 默认无人机数量 |
| 无人机 | `MAX_SPEED` | 3.0 m/s | 最大飞行速度 |
| 无人机 | `SAFE_RADIUS` | 3.0 m | 机间最小安全距离 |
| 无人机 | `COMM_RANGE` | 80.0 m | 通信距离 (分布式防撞感知范围) |
| 感知 | `SENSOR_FOV` | 90° | 前视视场角 |
| 感知 | `SENSOR_RANGE` | 15.0 m | 最大感知距离 |
| 感知 | `SENSOR_RAYS` | 60 | 射线数量 |
| 规划 | `REPLAN_INTERVAL` | 10 步 | 重规划周期 |
| 规划 | `FRONTIER_MIN_SIZE` | 3 格 | 最小前沿簇尺寸 |
| 仿真 | `DT` | 0.2 s | 仿真步长 |
| 仿真 | `MAX_STEPS` | 2000 | 最大仿真步数 |
| 仿真 | `N_OBSTACLES` | 12 | 随机障碍物数量 |
| 仿真 | `OBS_R_RANGE` | 1.5–4.0 m | 障碍物半径范围 |

**关键文件**: `config.py`

## 6 运行方式

### 6.1 安装依赖

```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib
```

### 6.2 批量仿真

```bash
python run_exploration.py                  # 默认 seed=42, 结果保存到 output/
python run_exploration.py --seed 123       # 指定随机种子
python run_exploration.py --animate        # 实时动画 (需要 GUI 环境)
```

输出文件：
- `output/final_map.png` — 最终地图截图
- `output/exploration_curve.png` — 覆盖率随步数变化曲线
- `output/waypoints.json` — 三架无人机的 GPS 航点数据

### 6.3 生成学术实验图表

```bash
python gen_figures.py       # 生成 4 张论文级图表 → output/figures/
python eval_experiments.py  # 更完整的实验评估 (含 LaTeX 表格)
```

### 6.4 sim_demo 演示 (EGO-Swarm 风格, 逐张出图)

```bash
python sim_demo.py              # 默认 seed=42, 输出 → output/sim_demo/
python sim_demo.py --seed 77    # 指定种子
```

输出 14 张独立图表 + EGO-Swarm 对比表。

## 7 安全指标体系

系统在运行过程中持续采集以下安全指标：

| 指标 | 计算方式 | 存储变量 |
|------|----------|----------|
| 障碍物避障率 | `P(min_obs_dist > SAFE_RADIUS)` | `min_obs_dist_log[i]` |
| 最小障碍距离 | 无人机表面到最近障碍物表面距离 | `min_obs_dist_log[i]` |
| 机间安全率 | `P(min_inter_dist > SAFE_RADIUS)` | `min_drone_dist_log` |
| 碰撞次数 | 障碍距离 < 0.5m 的步数 | `collision_count` |
| 重规划次数 | 触发路径重规划的总次数 | `replan_count` |
| 避障事件数 | 触发紧急避让的次数 (每机) | `avoidance_events[i]` |

## 8 实验结果 (最新指标)

### 8.1 基准实验 (seed=42, 3 UAVs, 分布式防撞)

| 指标 | 值 |
|------|------|
| 覆盖率 | **90.0%** |
| 总步数 | 301 |
| 碰撞次数 | **0** |
| 障碍物避障率 | 74.0% |
| 机间安全率 | **100.0%** |
| 最小机间距 | 19.63 m |
| 总路径长度 | 439.9 m |
| UAV-0 路径 | 139.5 m |
| UAV-1 路径 | 148.4 m |
| UAV-2 路径 | 152.0 m |
| 防撞架构 | 分布式 (ID 优先级) |

### 8.2 可扩展性对比

| 配置 | 达到 90% 的步数 | 800 步覆盖率 |
|------|-----------------|-------------|
| 1 UAV | 未达到 | 16% |
| 2 UAVs | 527 | 90% |
| 3 UAVs | **301** | **90%** |

### 8.3 多种子鲁棒性 (5 seeds)

| 统计量 | 覆盖率 | 避障率 | 步数 | 最小障碍距离 |
|--------|--------|--------|------|-------------|
| μ | 79.1% | 60.0% | 524 | 0.51 m |
| σ | 14.1% | 14.6% | 226 | 0.01 m |

### 8.4 学术图表说明

| 图表 | 文件 | 内容 |
|------|------|------|
| Fig 1 | `figures/fig1_scalability.png` | 1 vs 2 vs 3 UAV 覆盖率曲线 |
| Fig 2 | `figures/fig2_safety.png` | 避障安全性 4 子图 |
| Fig 3 | `figures/fig3_trajectory.png` | 地图轨迹 + 每机柱状图 |
| Fig 4 | `figures/fig4_robustness.png` | 多种子鲁棒性箱线图 |

### 8.5 sim_demo 演示图表 (每张独立)

| 文件 | 内容 |
|------|------|
| `sim_demo/1_snapshot_step*.png` | 6 个关键时刻地图快照 |
| `sim_demo/2_coverage.png` | 覆盖率曲线 |
| `sim_demo/3_safety.png` | 安全距离时序 |
| `sim_demo/4_path_length.png` | 每机路径长度 |
| `sim_demo/5_obs_clearance.png` | 障碍距离箱线图 |
| `sim_demo/6_inter_uav.png` | 机间距时序 |
| `sim_demo/7_pipeline.png` | EGO-Swarm vs Ours 流水线对比 |
| `sim_demo/8_comparison_table.png` | 功能与指标对比表 |
| `sim_demo/9_final_map.png` | 最终高清地图 |

## 9 与 EGO-Planner-Swarm 对比

| 维度 | EGO-Swarm | 本项目 |
|------|-----------|--------|
| 探索方式 | 人工指定目标 | 全自主前沿探索 |
| 深度感知 | CUDA / RealSense | ONNX ResUNet (14MB) |
| 建图 | 3D 体素 (0.1m) | 2D 栅格 (0.5m) |
| 区域分配 | 无 | Voronoi 分区 |
| 路径规划 | B-spline 梯度优化 | A* 栅格搜索 |
| 多机防撞 | 轨迹排斥项 (去中心化) | ID 优先级 + 轨迹广播 (分布式) |
| ROS 依赖 | 必须 | 无 |

详细对比文档见 `docs/comparison_ego_swarm.md`。

## 10 依赖关系图

```
config.py
   │
   ├──► simulation/sim_env.py          (读取全局参数, 构建仿真世界)
   │
   ├──► perception/
   │       ├── depth_estimator.py      (仿真/ONNX 深度估计)
   │       └── occupancy_grid.py       (栅格地图, 射线更新, 前沿检测)
   │
   ├──► planning/
   │       ├── frontier_explorer.py    (前沿目标选择)
   │       ├── path_planner.py         (A* 路径规划)
   │       └── waypoint_recorder.py    (航点记录)
   │
   ├──► coordination/
   │       ├── area_allocator.py       (Voronoi 区域分配)
   │       └── deconfliction.py        (分布式防撞: ID优先级 + 轨迹广播)
   │
   └──► run_exploration.py             (ExplorationSystem 主循环, 整合所有模块)
            │
            ├── sim_demo.py            (EGO-Swarm 风格演示, 逐张出图)
            ├── gen_figures.py         (学术实验图表生成)
            └── eval_experiments.py    (完整实验评估)
```

## 11 扩展接口

### 接入真实深度估计

修改 `config.py` 中的 `ONNX_PATH` 为 ONNX 模型路径即可切换到真实推理模式：

```python
ONNX_PATH = '/path/to/ResUNet_768x768.onnx'
DISPTOOLS_ROOT = '/home/R26062/disptool/offline_stereo/disptools'
```

### 修改无人机数量

`ExplorationSystem` 构造函数支持 `n_drones` 参数：

```python
system = ExplorationSystem(seed=42, n_drones=5)
```

注意：起始位置 `START_POSITIONS` 会循环复用，如增加到 4+ 架需要在 `config.py` 中补充起始位置。

### 修改环境参数

调整 `config.py` 中的参数可改变：
- 区域大小 (`WORLD_SIZE`)、障碍物数量和半径范围 (`N_OBSTACLES`, `OBS_R_RANGE`)
- 感知能力 (`SENSOR_FOV`, `SENSOR_RANGE`, `SENSOR_RAYS`)
- 安全约束 (`SAFE_RADIUS`, `MAX_SPEED`)
- 通信范围 (`COMM_RANGE`)，影响分布式防撞的邻居感知半径
