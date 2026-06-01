"""
2D 仿真环境: 随机障碍物 + 无人机运动学 + 前视深度模拟
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import *


@dataclass
class Obstacle:
    x: float
    y: float
    r: float


@dataclass
class DroneState:
    pos: np.ndarray          # (2,) 位置 m
    heading: float           # 朝向 rad
    speed: float = 0.0
    waypoints: list = field(default_factory=list)   # 已走过的 GPS 航点
    path: list = field(default_factory=list)         # 当前规划路径 [(x,y), ...]
    target_frontier: np.ndarray = None               # 当前目标前沿点


class SimEnvironment:
    """2D 仿真世界"""

    def __init__(self, seed=42, n_drones=None):
        self._n = n_drones or N_DRONES
        rng = np.random.RandomState(seed)
        self.obstacles: List[Obstacle] = []
        self._generate_obstacles(rng)
        self.drones: List[DroneState] = []
        for i in range(self._n):
            self.drones.append(DroneState(
                pos=START_POSITIONS[i % len(START_POSITIONS)].copy(),
                heading=START_HEADINGS[i % len(START_HEADINGS)],
            ))
        self.step_count = 0

    def _generate_obstacles(self, rng):
        margin = 8.0
        for _ in range(N_OBSTACLES):
            for _try in range(50):
                r = rng.uniform(*OBS_R_RANGE)
                x = rng.uniform(margin + r, WORLD_SIZE - margin - r)
                y = rng.uniform(margin + r, WORLD_SIZE - margin - r)
                # 不要挡住起点
                too_close = False
                for sp in START_POSITIONS:
                    if np.hypot(x - sp[0], y - sp[1]) < r + 8.0:
                        too_close = True
                        break
                if not too_close:
                    self.obstacles.append(Obstacle(x, y, r))
                    break

    # ---- 前视深度模拟 (射线投射) ---- #
    def sense_depth(self, drone_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回:
          angles : (N_RAYS,)  射线角度 (相对正前方)
          depths : (N_RAYS,)  每条射线的距离, 无碰撞返回 SENSOR_RANGE
        """
        d = self.drones[drone_id]
        half_fov = SENSOR_FOV / 2
        angles = np.linspace(-half_fov, half_fov, SENSOR_RAYS)
        depths = np.full(SENSOR_RAYS, SENSOR_RANGE)

        for i, a in enumerate(angles):
            ray_angle = d.heading + a
            dx = np.cos(ray_angle)
            dy = np.sin(ray_angle)
            # 检测所有障碍物
            for obs in self.obstacles:
                dist = self._ray_circle_intersect(
                    d.pos[0], d.pos[1], dx, dy, obs.x, obs.y, obs.r)
                if dist is not None and dist < depths[i]:
                    depths[i] = dist
            # 检测世界边界
            for t_bound in self._ray_boundary_intersect(d.pos[0], d.pos[1], dx, dy):
                if 0 < t_bound < depths[i]:
                    depths[i] = t_bound

        return angles, depths

    @staticmethod
    def _ray_circle_intersect(ox, oy, dx, dy, cx, cy, cr):
        """射线与圆的交点距离"""
        fx, fy = ox - cx, oy - cy
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - cr * cr
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        disc_sqrt = np.sqrt(disc)
        t1 = (-b - disc_sqrt) / (2 * a)
        t2 = (-b + disc_sqrt) / (2 * a)
        if t1 > 0.1:
            return t1
        if t2 > 0.1:
            return t2
        return None

    @staticmethod
    def _ray_boundary_intersect(ox, oy, dx, dy):
        """射线与世界边界的交点"""
        hits = []
        if dx != 0:
            t = (0 - ox) / dx
            if t > 0.1:
                hits.append(t)
            t = (WORLD_SIZE - ox) / dx
            if t > 0.1:
                hits.append(t)
        if dy != 0:
            t = (0 - oy) / dy
            if t > 0.1:
                hits.append(t)
            t = (WORLD_SIZE - oy) / dy
            if t > 0.1:
                hits.append(t)
        return hits

    # ---- 运动学 ---- #
    def move_drone(self, drone_id: int, target: np.ndarray):
        """向目标点移动一步, 返回是否到达"""
        d = self.drones[drone_id]
        diff = target - d.pos
        dist = np.linalg.norm(diff)
        if dist < GRID_RES:
            return True
        direction = diff / dist
        step = min(MAX_SPEED * DT, dist)
        new_pos = d.pos + direction * step
        # 碰撞检测
        if not self._collides(new_pos, drone_id):
            d.pos = new_pos
            d.heading = np.arctan2(direction[1], direction[0])
            d.speed = step / DT
        d.waypoints.append(d.pos.copy())
        return dist < GRID_RES * 2

    def _collides(self, pos, drone_id):
        """检测碰撞: 障碍物 + 世界边界"""
        if pos[0] < 0.5 or pos[0] > WORLD_SIZE - 0.5:
            return True
        if pos[1] < 0.5 or pos[1] > WORLD_SIZE - 0.5:
            return True
        for obs in self.obstacles:
            if np.hypot(pos[0] - obs.x, pos[1] - obs.y) < obs.r + 0.5:
                return True
        return False

    def get_ground_truth_grid(self) -> np.ndarray:
        """生成真值障碍栅格 (用于可视化对比)"""
        grid = np.full((GRID_N, GRID_N), FREE, dtype=np.int8)
        xs = np.arange(GRID_N) * GRID_RES + GRID_RES / 2
        XX, YY = np.meshgrid(xs, xs)
        for obs in self.obstacles:
            mask = np.sqrt((XX - obs.x)**2 + (YY - obs.y)**2) < obs.r
            grid[mask] = OCCUPIED
        # 边界
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = OCCUPIED
        return grid
