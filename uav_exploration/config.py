"""全局配置"""
import numpy as np

# ---- 世界 ----
WORLD_SIZE = 100.0        # 米
GRID_RES   = 0.5          # 栅格分辨率 m/cell
GRID_N     = int(WORLD_SIZE / GRID_RES)

# ---- 无人机 ----
N_DRONES      = 3
MAX_SPEED     = 3.0       # m/s
SAFE_RADIUS   = 3.0       # 机间最小安全距离 m
COMM_RANGE    = 80.0      # 通信距离 m

# ---- 前视感知 ----
SENSOR_FOV    = np.deg2rad(90)   # 前视视场角
SENSOR_RANGE  = 15.0             # 最大感知距离 m
SENSOR_RAYS   = 60               # 射线数量

# ---- ONNX 深度估计 ----
ONNX_PATH     = None      # 设为 onnx 路径启用真实推理
DISPTOOLS_ROOT = '/home/R26062/disptool/offline_stereo/disptools'
BASELINE      = 0.09      # 双目基线 m
FX            = 368.92    # 焦距 px

# ---- 规划 ----
REPLAN_INTERVAL = 10      # 每 N 步重规划
FRONTIER_MIN_SIZE = 3     # 最小前沿簇尺寸

# ---- 仿真 ----
DT            = 0.2       # 仿真步长 s
MAX_STEPS     = 2000      # 最大仿真步数
N_OBSTACLES   = 12        # 随机障碍物数量
OBS_R_RANGE   = (1.5, 4.0)  # 障碍物半径范围

# ---- 栅格标记 ----
UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2

# ---- 起始位置 ----
START_POSITIONS = np.array([
    [5.0,  5.0],
    [5.0, 50.0],
    [5.0, 95.0],
])
START_HEADINGS = np.array([0.0, 0.0, 0.0])  # 朝 +x 方向
