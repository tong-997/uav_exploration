# 本项目 vs EGO-Planner-Swarm 对比分析

## 1 项目定位

| | EGO-Planner-Swarm (ZJU-FAST-Lab) | 本项目 (uav_exploration) |
|---|---|---|
| **论文** | ICRA 2021, Zhou et al. | 毕设项目 |
| **目标** | 已知目标点的多机轨迹规划与避障飞行 | 未知区域的多机协同自主探索 |
| **核心问题** | "给定目标，如何安全快速地飞过去？" | "没有目标，去哪里探索、怎么分工？" |
| **语言** | C++ (81%) + CUDA + ROS | Python (100%) + NumPy |
| **维度** | 3D | 2D（可扩展 3D） |
| **部署平台** | ROS + Gazebo / 实机 | 独立 Python 脚本 |

**关系**：本项目可视为 EGO-Swarm 的**上层决策 + 感知前端替换**。EGO-Swarm 解决的是"轨迹优化 + 多机避撞"，本项目解决的是"去哪里 + 怎么分区 + 深度估计怎么来"。

## 2 系统架构对比

### EGO-Planner-Swarm 流水线

```
Stereo Camera / CUDA Ray-casting          ← 深度感知
        ↓
local_sensing → Depth Image / PointCloud  ← 传感器模拟
        ↓
grid_map → 3D Occupancy Grid (Voxel)      ← 环境建图
        ↓
EBK Search → 初始 B-spline 轨迹           ← 运动学搜索
        ↓
ESDF-Free Gradient Optimization           ← 轨迹优化 (核心)
  min  Js + λc·Jc + λd·Jd + λt·Jt
  s.t. B-spline control points
        ↓
Swarm Trajectory Broadcast                ← 多机协调
  (去中心化, 异步广播 B-spline)
        ↓
SO3 Controller / fake_drone               ← 底层控制
```

### 本项目 (uav_exploration) 流水线

```
ONNX ResUNet / 射线仿真                    ← 深度感知 (替换模块)
        ↓
OccupancyGrid.update_from_rays()          ← 2D 占据栅格建图
        ↓
Frontier Detection (BFS 连通分量)          ← 前沿检测 (EGO-Swarm 无此环节)
        ↓
VoronoiAllocator → 区域分配               ← 任务分配 (EGO-Swarm 无此环节)
        ↓
FrontierExplorer → 目标选择               ← 探索决策 (EGO-Swarm 无此环节)
        ↓
A* Path Planning (障碍膨胀)                ← 路径规划
        ↓
SpatioTemporalDeconfliction               ← 时空防撞
  (优先级 + 紧急避让)
        ↓
WaypointRecorder → GPS 航点               ← 航点输出
```

## 3 模块级对比

### 3.1 深度感知

| | EGO-Swarm | 本项目 |
|---|---|---|
| **仿真** | CUDA ray-casting 渲染深度图 / CPU 直接发点云 | 射线-圆相交检测 (2D) |
| **真实** | RealSense 等深度相机 | ONNX ResUNet 双目深度估计 |
| **输出** | 深度图 (640×480) 或 3D 点云 | 60 条射线深度值 |
| **帧率** | 30 Hz | 按仿真步进（DT=0.2s → 5Hz） |

**本项目的替换思路**：EGO-Swarm 的 `local_sensing` 包依赖 CUDA 或特定深度相机硬件。本项目用轻量 ONNX 模型（ResUNet, 14MB）替代，可在 RK3588 / NVIDIA Jetson 等嵌入式平台部署，不依赖 ROS。

### 3.2 环境建图

| | EGO-Swarm | 本项目 |
|---|---|---|
| **维度** | 3D 体素栅格 | 2D 占据栅格 |
| **分辨率** | 0.1 m | 0.5 m |
| **更新方式** | 深度图/点云 → 体素 raycasting | 向量化射线 → 栅格标记 |
| **融合** | ROS TF + 体素合并 | OCCUPIED > FREE > UNKNOWN |
| **多机融合** | 各机独立建图, 不融合 | 通信范围内全局融合 |

### 3.3 路径/轨迹规划

| | EGO-Swarm | 本项目 |
|---|---|---|
| **方法** | B-spline 梯度优化 | A* 栅格搜索 |
| **输出** | 连续时间 B-spline 轨迹 (位置+速度+加速度) | 离散航点序列 |
| **避障** | ESDF-free 梯度惩罚项 | 障碍物膨胀 + A* 绕行 |
| **动力学** | 速度/加速度约束嵌入优化 | 仅限速 (MAX_SPEED) |
| **计算时间** | ~1 ms (C++) | ~10 ms (Python, 含栅格更新) |
| **平滑性** | B-spline 保证 C2 连续 | 航点+路径简化 |

### 3.4 多机协调

| | EGO-Swarm | 本项目 |
|---|---|---|
| **架构** | 完全去中心化, 异步 | 集中式 (全局地图 + 全局分配) |
| **通信** | ROS Topic 广播 B-spline 轨迹 | 模拟通信 (直接共享栅格) |
| **防撞** | 轨迹优化中加入其他机轨迹的排斥项 | 优先级避让 + 紧急推开 |
| **任务分配** | 无 (用户指定目标点) | Voronoi 区域分配 + 前沿选择 |
| **探索策略** | 无 (不含自主探索) | Frontier-based Exploration |

### 3.5 探索决策 (本项目独有)

EGO-Swarm **不包含自主探索功能**，需要人工或上层系统指定目标点。本项目补充了完整的探索决策链：

```
前沿检测 → Voronoi 分区 → 效用评分 → 目标选择
```

这是本项目相对于 EGO-Swarm 最核心的**增量贡献**。

## 4 优劣对比

### EGO-Swarm 的优势

1. **轨迹质量**：B-spline 优化生成平滑、动力学可行的连续轨迹
2. **3D 规划**：完整三维空间规避
3. **实机验证**：ICRA 发表，有室内外飞行实验
4. **计算效率**：C++ 实现，单次规划 ~1ms
5. **去中心化**：无单点故障，可扩展至 15+ 架

### 本项目的优势

1. **自主探索**：不需要人工指定目标，自动发现未知区域
2. **深度估计替换**：ONNX 模型可部署到嵌入式端，不依赖 CUDA
3. **任务分配**：Voronoi 自动分区，避免重复搜索
4. **轻量部署**：纯 Python, 无 ROS 依赖，易于快速验证
5. **安全指标**：完整的避障率、碰撞率、机间安全率统计

### 各自的局限

| EGO-Swarm | 本项目 |
|---|---|
| 不含自主探索 | 2D, 非 3D |
| 依赖 ROS + CUDA | 无 B-spline 平滑轨迹 |
| 深度来源固定 (RealSense / CUDA) | 非去中心化 |
| 无前沿检测 / 区域分配 | 无动力学约束 |

## 5 替换方案：集成路线图

将本项目的深度估计模块接入 EGO-Swarm 的完整集成路线：

```
┌─────────────────────────────────────────────────────────┐
│                    集成后系统架构                         │
│                                                         │
│  ┌──────────────┐    替换     ┌───────────────────┐     │
│  │ local_sensing │  ──────►  │ ONNX DepthEstimator│     │
│  │ (CUDA/CPU)   │            │ (ResUNet 14MB)     │     │
│  └──────┬───────┘            └────────┬──────────┘     │
│         │                             │                 │
│         ▼                             ▼                 │
│  ┌──────────────┐            ┌───────────────────┐     │
│  │   grid_map    │            │  OccupancyGrid     │     │
│  │ (3D Voxel)   │  ◄─ 融合 ─ │  (2D → 升级 3D)   │     │
│  └──────┬───────┘            └───────────────────┘     │
│         │                                               │
│         ▼                    新增                        │
│  ┌──────────────┐    ┌───────────────────┐              │
│  │ EGO-Planner  │    │ FrontierExplorer  │ ← 探索决策   │
│  │ (B-spline)   │◄───│ VoronoiAllocator  │ ← 区域分配   │
│  └──────┬───────┘    └───────────────────┘              │
│         │                                               │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │ Swarm Deconf │  (保留 EGO-Swarm 原有防撞)            │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
```

### 集成步骤

| 步骤 | 内容 | 难度 |
|------|------|------|
| 1 | 将 `DepthEstimator` 封装为 ROS Node, 发布 `sensor_msgs/Image` | 低 |
| 2 | 替换 `local_sensing` 包, 保持 Topic 名称一致 | 低 |
| 3 | 将 `FrontierExplorer` + `VoronoiAllocator` 封装为 ROS Node, 发布目标点 | 中 |
| 4 | 目标点 → EGO-Planner 的 `/goal` Topic | 低 |
| 5 | OccupancyGrid 升级为 3D (2.5D 投影 或 完整体素) | 高 |
| 6 | 实机测试与参数调优 | 高 |

## 6 总结

| 维度 | EGO-Swarm | 本项目 | 集成后 |
|------|-----------|--------|--------|
| 深度感知 | CUDA / RealSense | ONNX ResUNet | ONNX ResUNet |
| 环境建图 | 3D 体素 | 2D 栅格 | 3D 体素 (ONNX 输入) |
| 探索决策 | 无 | Frontier + Voronoi | Frontier + Voronoi |
| 路径规划 | B-spline 优化 | A* | B-spline 优化 |
| 多机防撞 | 轨迹排斥项 | 优先级避让 | 轨迹排斥项 |
| 部署依赖 | ROS + CUDA | Python only | ROS + ONNX |
| 自主程度 | 需人工给目标 | 全自主探索 | 全自主探索 |

**核心结论**：本项目提供了 EGO-Swarm 所缺少的**自主探索决策层**（前沿检测 + Voronoi 分配）和**轻量深度感知前端**（ONNX 替代 CUDA），两者互补，集成后可构建完整的"感知→决策→规划→控制"自主探索系统。
