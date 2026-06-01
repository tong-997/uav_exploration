# uav_exploration_ros

Multi-UAV independent exploration with switchable stereo depth estimation.
Each drone independently builds its own occupancy grid via frontier-based exploration + A* planning.
Core contribution: ResUNet ONNX stereo depth estimation model, with system-level comparison experiment framework.

## Project Structure

```
uav_exploration_ros/
├── config/
│   ├── exploration.yaml          # 全局参数配置
│   └── exploration.rviz          # RViz 可视化配置
├── launch/
│   ├── gazebo_sim.launch         # 完整 Gazebo 仿真 (多机)
│   ├── depth_comparison.launch   # 深度估计对比实验 (单机)
│   ├── single_drone.launch       # 单机探索流水线
│   ├── sim_exploration.launch    # 探索流水线 (配合外部仿真)
│   ├── exploration_swarm.launch  # 多机探索 (include single_drone)
│   └── rviz.launch               # RViz 启动
├── scripts/
│   ├── depth_estimator_node.py   # 深度估计 (ResUNet / SGBM / GT)
│   ├── grid_map_node.py          # 深度 → 2D 占据栅格
│   ├── frontier_node.py          # 前沿检测 + 目标选择
│   ├── planner_node.py           # A* 路径规划 → pos_cmd
│   ├── fake_drone_node.py        # Gazebo 运动学控制器
│   ├── swarm_bridge_node.py      # 位姿中继 + 航迹 JSON 记录
│   ├── metrics_node.py           # 对比实验指标采集 → CSV
│   └── plot_comparison.py        # 读取 CSV 生成对比图表
├── models/
│   ├── ResUNet_768x768_simp_nohaze.onnx  # 深度估计模型
│   └── stereo_drone/model.sdf    # Gazebo 无人机模型 (双目+深度相机)
├── msg/
│   ├── PositionCommand.msg       # 位置指令 (兼容 ego-planner)
│   ├── Frontier.msg              # 前沿簇
│   └── FrontierArray.msg         # 前沿数组 + 探索进度
├── worlds/
│   └── exploration.world         # 100x100m 圆柱障碍物场景
├── package.xml
├── CMakeLists.txt
└── setup.py
```

## Dependencies

### System (Ubuntu 20.04 + ROS Noetic + Gazebo 11)

```bash
sudo apt update
sudo apt install ros-noetic-desktop-full
sudo apt install ros-noetic-cv-bridge ros-noetic-tf ros-noetic-tf2-ros \
                 ros-noetic-gazebo-ros ros-noetic-gazebo-msgs \
                 ros-noetic-message-generation ros-noetic-message-runtime \
                 ros-noetic-std-msgs ros-noetic-geometry-msgs \
                 ros-noetic-nav-msgs ros-noetic-sensor-msgs
```

### Python

```bash
pip install numpy opencv-python onnxruntime scipy matplotlib

# GPU 推理 (可选, 需 CUDA):
# pip install onnxruntime-gpu
```

## Build

```bash
# 1. 创建工作空间 (如果还没有)
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# 2. 将本包放入 src (拷贝或软链接)
cp -r /path/to/uav_exploration_ros .
# 或: ln -s /path/to/uav_exploration_ros .

# 3. 编译
cd ~/catkin_ws
catkin_make

# 4. 加载环境 (建议加入 ~/.bashrc)
source devel/setup.bash
```

## Quick Start — Gazebo 仿真

### 多机一键启动

```bash
roslaunch uav_exploration_ros gazebo_sim.launch
```

默认 3 架无人机, 固定高度 2m, 独立探索. 可选参数:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_drones` | 3 | 无人机数量 (1~5) |
| `gui` | true | 是否显示 Gazebo GUI |
| `rviz` | true | 是否启动 RViz |

```bash
roslaunch uav_exploration_ros gazebo_sim.launch n_drones:=5 gui:=true rviz:=true
```

### 单机启动 (不含 Gazebo)

```bash
roslaunch uav_exploration_ros single_drone.launch drone_id:=0
```

### 实时查看相机画面

```bash
# 左相机原始画面
rqt_image_view /uav0/stereo/left/image_raw

# 深度估计结果
rqt_image_view /uav0/depth_estimator/depth

# 不确定度
rqt_image_view /uav0/depth_estimator/uncertainty
```

## Depth Estimation Comparison Experiments

核心实验: 同一 Gazebo 场景 + 同一起点, 仅切换深度估计方法, 对比探索效果和深度精度.

### 三种方法

| `depth_method` | 说明 | 需要 |
|----------------|------|------|
| `resunet` | ResUNet ONNX 推理 (提出方法) | `onnxruntime` + 双目图像 |
| `sgbm` | OpenCV StereoSGBM (传统基线) | 双目图像 |
| `gt` | Gazebo 深度相机真值 (性能上界) | Gazebo depth camera plugin |

### Step 1: 跑实验

每次跑一种方法, 探索一段时间后 Ctrl+C 停止. 指标自动保存到 `/tmp/uav_exploration/`.

```bash
# 实验 1: ResUNet (提出方法)
roslaunch uav_exploration_ros depth_comparison.launch depth_method:=resunet
# 等待探索... Ctrl+C 停止

# 实验 2: OpenCV SGBM (传统基线)
roslaunch uav_exploration_ros depth_comparison.launch depth_method:=sgbm
# 等待探索... Ctrl+C 停止

# 实验 3: Gazebo 真值深度 (性能上界)
roslaunch uav_exploration_ros depth_comparison.launch depth_method:=gt
# 等待探索... Ctrl+C 停止
```

`depth_comparison.launch` 可选参数:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `depth_method` | `resunet` | 深度估计方法 |
| `gui` | true | Gazebo GUI |
| `rviz` | true | RViz 可视化 |

### Step 2: 生成对比图

```bash
python scripts/plot_comparison.py
```

可选参数:
```bash
python scripts/plot_comparison.py \
  --input_dir /tmp/uav_exploration \
  --output_dir /tmp/uav_exploration \
  --drone_id 0
```

输出文件:
- `/tmp/uav_exploration/depth_comparison.png` — 四合一对比图 (MAE, RMSE, 覆盖率, 效率)
- 终端打印指标汇总表

### 输出文件说明

实验结束后 `/tmp/uav_exploration/` 下会生成:

```
/tmp/uav_exploration/
├── metrics_uav0_resunet.csv    # ResUNet 指标时序
├── metrics_uav0_sgbm.csv       # SGBM 指标时序
├── metrics_uav0_gt.csv         # GT 指标时序
├── waypoints_uav0.json         # 航迹记录
└── depth_comparison.png        # 对比图 (plot_comparison.py 生成)
```

### 指标说明

| 指标 | CSV 字段 | 说明 |
|------|----------|------|
| 深度 MAE | `depth_mae` | 与真值的平均绝对误差 (m) |
| 深度 RMSE | `depth_rmse` | 与真值的均方根误差 (m) |
| delta < 1m | `delta_1m` | 误差 < 1m 的像素比例 (0~1) |
| delta < 0.5m | `delta_05m` | 误差 < 0.5m 的像素比例 (0~1) |
| 探索覆盖率 | `explored_ratio` | 已知区域 / 总区域 (0~1) |
| 累计距离 | `cum_dist_m` | 累计飞行距离 (m) |

## Node Pipeline

每架无人机启动 5 个节点, 在 `/uavN/` 命名空间下独立运行:

```
stereo/left/image_raw ──┐
                        ├─→ depth_estimator ──→ depth ──→ grid_map ──→ global_grid ──→ frontier_explorer ──→ exploration_goal ──→ planner ──→ pos_cmd ──→ fake_drone
stereo/right/image_raw ─┘                                    ↑                              ↑                                       ↑                        │
                                                          pose ←─────────────────── swarm_bridge ←──────────────────────────── pose ←────────────────────── pose
```

对比实验额外运行 `metrics_node`, 同时订阅估计深度和真值深度计算指标.

## Config Reference

所有参数集中在 `config/exploration.yaml`:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **世界** | | |
| `world_size` | 100.0 m | 场景尺寸 |
| `grid_resolution` | 0.5 m/cell | 栅格分辨率 |
| **无人机** | | |
| `n_drones` | 3 | 无人机数量 |
| `max_speed` | 3.0 m/s | 最大飞行速度 |
| **深度估计** | | |
| `depth_method` | `resunet` | 方法 (`resunet` / `sgbm` / `gt`) |
| `onnx_model_path` | `models/ResUNet_...onnx` | ONNX 模型路径 (相对包目录) |
| `onnx_device` | `cpu` | 推理设备 (`cpu` / `cuda`) |
| `baseline` | 0.09 m | 双目基线 |
| `fx` | 368.92 px | 焦距 |
| `depth_max` | 15.0 m | 最大有效深度 |
| `depth_min` | 0.3 m | 最小有效深度 |
| `input_h/w` | 768 | ONNX 输入分辨率 |
| `sgbm_num_disparities` | 128 | SGBM 视差搜索范围 |
| `sgbm_block_size` | 9 | SGBM 块大小 |
| **栅格** | | |
| `sensor_fov_deg` | 90.0 deg | 前视视场角 |
| `sensor_range` | 15.0 m | 最大感知距离 |
| **探索** | | |
| `frontier_min_size` | 3 | 最小前沿簇大小 |
| `utility_weight_dist` | 1.0 | 距离权重 |
| `utility_weight_size` | 0.5 | 前沿大小权重 |
| `exploration_done_ratio` | 0.90 | 探索完成阈值 |
| **规划** | | |
| `safety_margin` | 2 cells | 障碍膨胀格数 |
| `path_simplify_step` | 3 | 路径简化步长 |
| **频率** | | |
| `depth_rate` | 10.0 Hz | 深度估计频率 |
| `grid_rate` | 5.0 Hz | 栅格更新频率 |
| `frontier_rate` | 2.0 Hz | 前沿检测频率 |
| `planner_rate` | 5.0 Hz | 规划控制频率 |

## Gazebo World

`worlds/exploration.world`: 100m x 100m 围墙区域 + 12 个随机圆柱障碍物 (seed=42, 半径 1.6~3.8m, 高 5m).

## UAV Model

`models/stereo_drone/model.sdf`:
- 四旋翼外形, 质量 1.5kg
- 双目相机: 左右各一, 基线 0.09m, 384x384@10Hz, FOV 90deg
- 深度相机: 与左相机同位置, 提供 Gazebo 真值深度 (对比实验用)
- 话题: `/uavN/stereo/{left,right}/image_raw`, `/uavN/depth_camera/depth/image_raw`
