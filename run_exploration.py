"""
三机协同探索仿真主程序
============================
三架无人机从不同位置出发, 分区搜索未知区域:
  1. 前视深度感知 → 实时构建占据栅格
  2. 前沿探索 + A* 避障路径规划
  3. Voronoi 区域分配 + 时空防撞
  4. GPS 航点记录

运行:
  python run_exploration.py                  # 保存结果到 output/
  python run_exploration.py --animate        # 实时动画
  python run_exploration.py --seed 123       # 指定随机种子
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from simulation.sim_env import SimEnvironment
from perception.occupancy_grid import OccupancyGrid
from planning.path_planner import AStarPlanner
from planning.frontier_explorer import FrontierExplorer
from planning.waypoint_recorder import WaypointRecorder
from coordination.area_allocator import VoronoiAllocator
from coordination.deconfliction import DistributedDeconfliction

COLORS = ['#29b6f6', '#66bb6a', '#ffa726']
DRONE_NAMES = ['UAV-0', 'UAV-1', 'UAV-2']


class ExplorationSystem:
    """多机协同探索系统 — 支持参数化运行"""

    def __init__(self, seed=42, n_drones=None, config=None):
        if config is not None:
            self.n_drones = config.n_drones
            self._max_speed = config.max_speed
            self._max_steps = config.max_steps
            self._coverage_threshold = config.coverage_threshold
            self._use_voronoi = config.use_voronoi
            self._use_deconflict = config.use_deconflict
            self._use_inflation = config.use_obstacle_inflation
            self._frontier_strategy = config.frontier_strategy
            self._depth_mode = config.depth_mode
            self._depth_noise_std = config.depth_noise_std
            self._enable_targets = config.enable_targets
            self._rng = np.random.RandomState(seed)
        else:
            self.n_drones = n_drones or N_DRONES
            self._max_speed = MAX_SPEED
            self._max_steps = MAX_STEPS
            self._coverage_threshold = 0.90
            self._use_voronoi = True
            self._use_deconflict = True
            self._use_inflation = True
            self._frontier_strategy = 'utility'
            self._depth_mode = 'raycast'
            self._depth_noise_std = 0.5
            self._enable_targets = False
            self._rng = np.random.RandomState(seed)

        self.env = SimEnvironment(seed=seed, n_drones=self.n_drones,
                                  max_speed=self._max_speed)
        self.grids = [OccupancyGrid() for _ in range(self.n_drones)]
        self.global_grid = OccupancyGrid()
        safety_margin = 1 if self._use_inflation else 0
        self.planner = AStarPlanner(safety_margin=safety_margin)
        self.explorer = FrontierExplorer()
        self.allocator = VoronoiAllocator()
        self.deconflict = DistributedDeconfliction()
        self.recorder = WaypointRecorder(n_drones=self.n_drones)

        self.step = 0
        self.exploration_log = []
        self._path_follow_idx = [0] * self.n_drones

        self.min_obs_dist_log = [[] for _ in range(self.n_drones)]
        self.min_drone_dist_log = []
        self.pairwise_drone_dist_log = {}
        for i in range(self.n_drones):
            for j in range(i + 1, self.n_drones):
                self.pairwise_drone_dist_log[(i, j)] = []
        self.avoidance_events = [0] * self.n_drones
        self.collision_count = 0
        self.replan_count = 0
        self.plan_success_count = 0
        self.plan_total_count = 0
        self._per_drone_explored = [set() for _ in range(self.n_drones)]

        self._gt_grid = None
        if self._depth_mode == 'gt':
            self._gt_grid = self.env.get_ground_truth_grid()

        self.targets = []
        self.target_detection_log = []
        self._cooperative_override = False

    def run_step(self):
        """执行一步仿真"""
        drone_positions = [d.pos.copy() for d in self.env.drones]

        # ---- 1. 感知 ---- #
        for i in range(self.n_drones):
            d = self.env.drones[i]
            if self._depth_mode == 'gt':
                self._stamp_gt_in_range(i)
            else:
                angles, depths = self.env.sense_depth(i)
                if self._depth_mode == 'noisy':
                    depths = self._apply_noisy_depth(depths)
                elif self._depth_mode == 'resunet':
                    depths = self._apply_resunet_depth(depths)
                self.grids[i].update_from_rays(d.pos, d.heading, angles, depths)
            prev_unknown = self.grids[i].grid == UNKNOWN
            new_known = ~prev_unknown
            known_cells = set(zip(*np.where(new_known)))
            self._per_drone_explored[i] |= known_cells

        # ---- 2. 融合 ---- #
        self.global_grid = self.grids[0].copy()
        for i in range(1, self.n_drones):
            self.global_grid.merge(self.grids[i])
        for i in range(self.n_drones):
            self.grids[i] = self.global_grid.copy()

        # ---- 3. 区域分配 ---- #
        if self._use_voronoi:
            self.allocator.compute_voronoi(drone_positions)

        # ---- 4. 前沿探索 + 路径规划 ---- #
        needs_replan = self.step % REPLAN_INTERVAL == 0
        for i in range(self.n_drones):
            d = self.env.drones[i]
            if not d.path or self._path_follow_idx[i] >= len(d.path):
                needs_replan = True
        if needs_replan and not self._cooperative_override:
            self._replan(drone_positions)
            self.replan_count += 1

        # ---- 5. 防撞 ---- #
        if self._use_deconflict:
            for i in range(self.n_drones):
                self.deconflict.broadcast_trajectory(i, self.env.drones[i].path)

        # ---- 6. 执行移动 ---- #
        for i in range(self.n_drones):
            d = self.env.drones[i]

            if self._use_deconflict:
                local_conflicts = self.deconflict.check_proximity(
                    i, d.pos, drone_positions)
                if local_conflicts:
                    for _, nb, _ in local_conflicts:
                        self.avoidance_events[i] += 1
                    avoid_dir = self.deconflict.local_emergency_avoidance(
                        i, d.pos, drone_positions)
                    if avoid_dir is not None:
                        emergency_target = d.pos + avoid_dir * self._max_speed * DT
                        self.env.move_drone(i, emergency_target)
                        self.recorder.record(i, d.pos, d.heading, self.step * DT)
                        continue

            if d.path and self._path_follow_idx[i] < len(d.path):
                target = d.path[self._path_follow_idx[i]]
                arrived = self.env.move_drone(i, target)
                if arrived:
                    self._path_follow_idx[i] += 1
            elif d.target_frontier is not None:
                self.env.move_drone(i, d.target_frontier)

            self.recorder.record(i, d.pos, d.heading, self.step * DT)

        # ---- 7. 目标检测 ---- #
        if self._enable_targets and self.targets:
            self._run_target_detection(drone_positions)

        # ---- 8. 指标采集 ---- #
        for i in range(self.n_drones):
            d = self.env.drones[i]
            min_d = min(
                (np.hypot(d.pos[0] - o.x, d.pos[1] - o.y) - o.r
                 for o in self.env.obstacles),
                default=999.0
            )
            self.min_obs_dist_log[i].append(min_d)
            if min_d < 0.5:
                self.collision_count += 1

        dists_ij = []
        for i in range(self.n_drones):
            for j in range(i + 1, self.n_drones):
                d_ij = np.linalg.norm(
                    self.env.drones[i].pos - self.env.drones[j].pos)
                dists_ij.append(d_ij)
                self.pairwise_drone_dist_log[(i, j)].append(d_ij)
        self.min_drone_dist_log.append(min(dists_ij) if dists_ij else 999.0)

        ratio = self.global_grid.explored_ratio
        self.exploration_log.append((self.step, ratio))
        self.step += 1
        return ratio

    def _stamp_gt_in_range(self, drone_id):
        """GT-Depth: 直接将传感器范围内的真值写入栅格"""
        d = self.env.drones[drone_id]
        cy, cx = int(d.pos[1] / GRID_RES), int(d.pos[0] / GRID_RES)
        r_cells = int(SENSOR_RANGE / GRID_RES)
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < GRID_N and 0 <= nx < GRID_N:
                    if dy * dy + dx * dx <= r_cells * r_cells:
                        half_fov = SENSOR_FOV / 2
                        cell_pos = np.array([nx * GRID_RES + GRID_RES / 2,
                                             ny * GRID_RES + GRID_RES / 2])
                        diff = cell_pos - d.pos
                        angle = np.arctan2(diff[1], diff[0]) - d.heading
                        angle = (angle + np.pi) % (2 * np.pi) - np.pi
                        if abs(angle) <= half_fov:
                            self.grids[drone_id].grid[ny, nx] = self._gt_grid[ny, nx]

    def _apply_noisy_depth(self, depths):
        noisy = depths + self._rng.normal(0, self._depth_noise_std, size=depths.shape)
        return np.clip(noisy, 0.5, SENSOR_RANGE)

    def _apply_resunet_depth(self, depths):
        n = len(depths)
        bias = self._rng.uniform(-0.05, 0.05, size=n)
        result = depths * (1.0 + bias)
        for k in range(1, n):
            if abs(depths[k] - depths[k - 1]) > 2.0:
                result[k] += self._rng.normal(0, 1.0)
        outlier_mask = self._rng.random(n) < 0.05
        result[outlier_mask] = SENSOR_RANGE
        return np.clip(result, 0.5, SENSOR_RANGE)

    def _run_target_detection(self, drone_positions):
        pass

    def _replan(self, drone_positions):
        """重新规划所有无人机的路径"""
        all_frontiers = self.global_grid.get_frontiers(FRONTIER_MIN_SIZE)

        for i in range(self.n_drones):
            if self._use_voronoi:
                my_frontiers = self.allocator.filter_frontiers_for_drone(all_frontiers, i)
                if not my_frontiers:
                    my_frontiers = all_frontiers
            else:
                my_frontiers = all_frontiers

            other_targets = [self.env.drones[j].target_frontier
                             for j in range(self.n_drones) if j != i]

            if self._frontier_strategy == 'greedy':
                target = self._select_greedy(my_frontiers, drone_positions[i])
            elif self._frontier_strategy == 'random':
                target = self._select_random(my_frontiers, drone_positions[i])
            else:
                target = self.explorer.select_frontier(
                    my_frontiers, drone_positions[i], other_targets)

            self.env.drones[i].target_frontier = target

            if target is not None:
                self.plan_total_count += 1
                path = self.planner.plan(
                    self.global_grid.grid, drone_positions[i], target,
                    treat_unknown_as='free')
                if path:
                    self.plan_success_count += 1
                    if self._use_deconflict:
                        path = self.deconflict.check_and_resolve(
                            i, drone_positions[i], path, drone_positions)
                    self.env.drones[i].path = path
                    if self._use_deconflict:
                        self.deconflict.broadcast_trajectory(i, path)
                    self._path_follow_idx[i] = 0
                else:
                    self.env.drones[i].path = []
            else:
                self.env.drones[i].path = []

    @staticmethod
    def _select_greedy(frontiers, drone_pos):
        if not frontiers:
            return None
        best_dist = np.inf
        best_pos = None
        for iy, ix, size in frontiers:
            pos = np.array([ix * GRID_RES + GRID_RES / 2,
                            iy * GRID_RES + GRID_RES / 2])
            d = np.linalg.norm(pos - drone_pos)
            if d < best_dist and d > 1.0:
                best_dist = d
                best_pos = pos
        return best_pos

    def _select_random(self, frontiers, drone_pos):
        if not frontiers:
            return None
        valid = [(iy, ix, s) for iy, ix, s in frontiers
                 if np.linalg.norm(np.array([ix * GRID_RES + GRID_RES / 2,
                                             iy * GRID_RES + GRID_RES / 2]) - drone_pos) > 1.0]
        if not valid:
            return None
        iy, ix, _ = valid[self._rng.randint(len(valid))]
        return np.array([ix * GRID_RES + GRID_RES / 2,
                         iy * GRID_RES + GRID_RES / 2])

    def is_done(self):
        return (self.global_grid.explored_ratio > self._coverage_threshold or
                self.step >= self._max_steps)


# =========================================================================== #
#  可视化
# =========================================================================== #

def render_frame(ax, system: ExplorationSystem):
    ax.clear()
    env = system.env
    grid = system.global_grid.grid

    # 栅格地图
    display = np.ones((GRID_N, GRID_N, 3))
    display[grid == FREE] = [0.95, 0.95, 0.95]       # 浅灰 = 已知空闲
    display[grid == OCCUPIED] = [0.2, 0.2, 0.2]       # 深灰 = 障碍
    display[grid == UNKNOWN] = [0.75, 0.85, 0.95]     # 浅蓝 = 未知
    ax.imshow(display, origin='lower',
              extent=[0, WORLD_SIZE, 0, WORLD_SIZE])

    # 真实障碍物轮廓
    for obs in env.obstacles:
        circle = plt.Circle((obs.x, obs.y), obs.r,
                             fill=False, color='red', linewidth=0.8, linestyle='--')
        ax.add_patch(circle)

    # 无人机
    for i in range(N_DRONES):
        d = env.drones[i]
        ax.plot(d.pos[0], d.pos[1], 'o', color=COLORS[i], markersize=8,
                markeredgecolor='black', markeredgewidth=0.8)
        # 朝向
        dx = np.cos(d.heading) * 2.5
        dy = np.sin(d.heading) * 2.5
        ax.arrow(d.pos[0], d.pos[1], dx, dy,
                 head_width=0.8, head_length=0.5, fc=COLORS[i], ec=COLORS[i])
        # 路径
        if d.path and len(d.path) > 1:
            px = [p[0] for p in d.path]
            py = [p[1] for p in d.path]
            ax.plot(px, py, '--', color=COLORS[i], linewidth=0.8, alpha=0.6)
        # 目标
        if d.target_frontier is not None:
            ax.plot(d.target_frontier[0], d.target_frontier[1], '*',
                    color=COLORS[i], markersize=12)
        # 轨迹
        wps = d.waypoints
        if len(wps) > 1:
            wx = [w[0] for w in wps[-200:]]
            wy = [w[1] for w in wps[-200:]]
            ax.plot(wx, wy, '-', color=COLORS[i], linewidth=0.5, alpha=0.3)

    # 信息
    ratio = system.global_grid.explored_ratio
    ax.set_title(f'Step {system.step}  |  Explored: {ratio:.1%}', fontsize=11)
    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # 图例
    patches = [mpatches.Patch(color=COLORS[i], label=DRONE_NAMES[i])
               for i in range(N_DRONES)]
    patches.append(mpatches.Patch(color=[0.75, 0.85, 0.95], label='Unknown'))
    patches.append(mpatches.Patch(color=[0.2, 0.2, 0.2], label='Obstacle'))
    ax.legend(handles=patches, loc='upper right', fontsize=7)


def save_exploration_curve(log, output_dir):
    steps = [x[0] for x in log]
    ratios = [x[1] for x in log]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, [r * 100 for r in ratios], 'b-', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Explored (%)')
    ax.set_title('Exploration Progress')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exploration_curve.png'), dpi=150)
    plt.close(fig)


# =========================================================================== #
#  主入口
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--animate', action='store_true', help='实时动画')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='output')
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), args.output)
    os.makedirs(output_dir, exist_ok=True)

    system = ExplorationSystem(seed=args.seed)
    print(f'[Exploration] {N_DRONES} drones, world={WORLD_SIZE}m, '
          f'obstacles={len(system.env.obstacles)}, seed={args.seed}')

    if args.animate:
        # ---- 实时动画模式 ---- #
        fig, ax = plt.subplots(figsize=(9, 9))

        def update(frame):
            if not system.is_done():
                ratio = system.run_step()
                if system.step % 50 == 0:
                    print(f'  step={system.step}, explored={ratio:.1%}')
            render_frame(ax, system)

        anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
        plt.show()
    else:
        # ---- 批量运行模式 ---- #
        while not system.is_done():
            ratio = system.run_step()
            if system.step % 100 == 0:
                print(f'  step={system.step}, explored={ratio:.1%}')

        print(f'\n[Done] Total steps: {system.step}, '
              f'explored: {system.global_grid.explored_ratio:.1%}')

        # 保存最终地图
        fig, ax = plt.subplots(figsize=(9, 9))
        render_frame(ax, system)
        fig.savefig(os.path.join(output_dir, 'final_map.png'), dpi=150)
        plt.close(fig)

        # 保存探索曲线
        save_exploration_curve(system.exploration_log, output_dir)

        # 保存航点
        system.recorder.export_json(os.path.join(output_dir, 'waypoints.json'))

        # 打印统计
        for i in range(N_DRONES):
            dist = system.recorder.get_total_distance(i)
            n_wp = len(system.recorder.waypoints[i])
            print(f'  {DRONE_NAMES[i]}: distance={dist:.1f}m, waypoints={n_wp}')

        print(f'\nOutput saved to: {output_dir}/')


if __name__ == '__main__':
    main()
