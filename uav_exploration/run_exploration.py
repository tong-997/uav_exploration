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
    """三机协同探索系统"""

    def __init__(self, seed=42, n_drones=None):
        self.n_drones = n_drones or N_DRONES
        self.env = SimEnvironment(seed=seed, n_drones=self.n_drones)
        self.grids = [OccupancyGrid() for _ in range(self.n_drones)]
        self.global_grid = OccupancyGrid()
        self.planner = AStarPlanner(safety_margin=1)
        self.explorer = FrontierExplorer()
        self.allocator = VoronoiAllocator()
        self.deconflict = DistributedDeconfliction()
        self.recorder = WaypointRecorder(n_drones=self.n_drones)

        self.step = 0
        self.exploration_log = []  # (step, explored_ratio)
        self._path_follow_idx = [0] * self.n_drones

        # ---- 避障与安全指标 ---- #
        self.min_obs_dist_log = [[] for _ in range(self.n_drones)]   # 每步到最近障碍距离
        self.min_drone_dist_log = []                             # 每步最近机间距
        self.avoidance_events = [0] * self.n_drones                  # 避障机动次数
        self.collision_count = 0                                 # 碰撞次数 (应为 0)
        self.replan_count = 0                                    # 重规划次数

    def run_step(self):
        """执行一步仿真"""
        drone_positions = [d.pos.copy() for d in self.env.drones]

        # ---- 1. 感知: 前视深度 → 更新栅格 ---- #
        for i in range(self.n_drones):
            d = self.env.drones[i]
            angles, depths = self.env.sense_depth(i)
            self.grids[i].update_from_rays(d.pos, d.heading, angles, depths)

        # ---- 2. 融合: 合并所有无人机栅格 → 全局地图 ---- #
        self.global_grid = self.grids[0].copy()
        for i in range(1, self.n_drones):
            self.global_grid.merge(self.grids[i])
        # 同步回各无人机 (模拟通信)
        for i in range(self.n_drones):
            self.grids[i] = self.global_grid.copy()

        # ---- 3. 区域分配 ---- #
        self.allocator.compute_voronoi(drone_positions)

        # ---- 4. 前沿探索 + 路径规划 ---- #
        needs_replan = self.step % REPLAN_INTERVAL == 0
        for i in range(self.n_drones):
            d = self.env.drones[i]
            # 路径走完或无路径时也触发重规划
            if not d.path or self._path_follow_idx[i] >= len(d.path):
                needs_replan = True
        if needs_replan:
            self._replan(drone_positions)
            self.replan_count += 1

        # ---- 5. 分布式防撞: 每机独立决策 ---- #
        # 每机广播自身轨迹 (模拟局域通信)
        for i in range(self.n_drones):
            self.deconflict.broadcast_trajectory(i, self.env.drones[i].path)

        # ---- 6. 执行移动 (各机独立) ---- #
        for i in range(self.n_drones):
            d = self.env.drones[i]

            # 本机独立检测邻居过近
            local_conflicts = self.deconflict.check_proximity(
                i, d.pos, drone_positions)
            if local_conflicts:
                for _, nb, _ in local_conflicts:
                    self.avoidance_events[i] += 1
                avoid_dir = self.deconflict.local_emergency_avoidance(
                    i, d.pos, drone_positions)
                if avoid_dir is not None:
                    emergency_target = d.pos + avoid_dir * MAX_SPEED * DT
                    self.env.move_drone(i, emergency_target)
                    continue

            # 沿路径移动
            if d.path and self._path_follow_idx[i] < len(d.path):
                target = d.path[self._path_follow_idx[i]]
                arrived = self.env.move_drone(i, target)
                if arrived:
                    self._path_follow_idx[i] += 1
            elif d.target_frontier is not None:
                self.env.move_drone(i, d.target_frontier)

            # 记录航点
            self.recorder.record(i, d.pos, d.heading, self.step * DT)

        # ---- 7. 指标采集 ---- #
        for i in range(self.n_drones):
            d = self.env.drones[i]
            # 到最近障碍的距离
            min_d = min(
                (np.hypot(d.pos[0] - o.x, d.pos[1] - o.y) - o.r
                 for o in self.env.obstacles),
                default=999.0
            )
            self.min_obs_dist_log[i].append(min_d)
            if min_d < 0.5:
                self.collision_count += 1

        # 最近机间距
        dists_ij = []
        for i in range(self.n_drones):
            for j in range(i + 1, self.n_drones):
                dists_ij.append(np.linalg.norm(
                    self.env.drones[i].pos - self.env.drones[j].pos))
        self.min_drone_dist_log.append(min(dists_ij) if dists_ij else 999.0)

        ratio = self.global_grid.explored_ratio
        self.exploration_log.append((self.step, ratio))
        self.step += 1
        return ratio

    def _replan(self, drone_positions):
        """重新规划所有无人机的路径"""
        all_frontiers = self.global_grid.get_frontiers(FRONTIER_MIN_SIZE)

        for i in range(self.n_drones):
            # Voronoi 过滤
            my_frontiers = self.allocator.filter_frontiers_for_drone(all_frontiers, i)
            if not my_frontiers:
                my_frontiers = all_frontiers  # fallback: 不限制区域

            # 选择目标前沿
            other_targets = [self.env.drones[j].target_frontier
                             for j in range(self.n_drones) if j != i]
            target = self.explorer.select_frontier(
                my_frontiers, drone_positions[i], other_targets)

            self.env.drones[i].target_frontier = target

            if target is not None:
                path = self.planner.plan(
                    self.global_grid.grid, drone_positions[i], target,
                    treat_unknown_as='free')
                # 分布式防撞: 本机独立检查与邻居广播轨迹的冲突
                if path:
                    path = self.deconflict.check_and_resolve(
                        i, drone_positions[i], path, drone_positions)
                self.env.drones[i].path = path
                self.deconflict.broadcast_trajectory(i, path)
                self._path_follow_idx[i] = 0
            else:
                self.env.drones[i].path = []

    def is_done(self):
        return (self.global_grid.explored_ratio > 0.90 or
                self.step >= MAX_STEPS)


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
