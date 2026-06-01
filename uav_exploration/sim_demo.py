"""
sim_demo.py — EGO-Swarm 风格协同探索仿真演示
=================================================
每张图单独输出, 分布式防撞, 3 机协同探索

输出 output/sim_demo/:
  1_snapshot_stepXXX.png   — 各关键时刻地图快照 (每步一张)
  2_coverage.png           — 覆盖率曲线
  3_safety.png             — 安全距离时序
  4_path_length.png        — 每机路径长度柱状图
  5_obs_clearance.png      — 每机障碍距离箱线图
  6_inter_uav.png          — 机间距时序
  7_pipeline.png           — 流水线对比图
  8_comparison_table.png   — EGO-Swarm vs Ours 对比表
  9_final_map.png          — 最终高清地图

用法:
  python sim_demo.py
  python sim_demo.py --seed 77
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from run_exploration import ExplorationSystem

OUT = os.path.join(os.path.dirname(__file__), 'output', 'sim_demo')
os.makedirs(OUT, exist_ok=True)

DC = ['#29b6f6', '#66bb6a', '#ffa726']
SNAP_STEPS = [0, 50, 100, 200, 300]


# ================================================================ #
#  渲染工具
# ================================================================ #

def render_map(ax, system, step_label=None, show_fov=True, show_trail=True):
    env = system.env
    g = system.global_grid.grid
    n = system.n_drones

    dp = np.ones((GRID_N, GRID_N, 3))
    dp[g == FREE] = [0.95, 0.95, 0.95]
    dp[g == OCCUPIED] = [0.20, 0.20, 0.20]
    dp[g == UNKNOWN] = [0.82, 0.88, 0.95]
    ax.imshow(dp, origin='lower', extent=[0, WORLD_SIZE, 0, WORLD_SIZE])

    for obs in env.obstacles:
        ax.add_patch(plt.Circle((obs.x, obs.y), obs.r,
                                fill=True, fc='#EF5350', alpha=0.25,
                                ec='#C62828', lw=1.0))

    for i in range(n):
        d = env.drones[i]
        ax.plot(d.pos[0], d.pos[1], 'o', color=DC[i], ms=10,
                mec='white', mew=1.5, zorder=10)
        hdx = np.cos(d.heading) * 3.0
        hdy = np.sin(d.heading) * 3.0
        ax.annotate('', xy=(d.pos[0]+hdx, d.pos[1]+hdy),
                    xytext=(d.pos[0], d.pos[1]),
                    arrowprops=dict(arrowstyle='->', color=DC[i], lw=2))

        if show_fov:
            fov_half = SENSOR_FOV / 2
            angles_fan = np.linspace(d.heading - fov_half,
                                     d.heading + fov_half, 30)
            fan_x = [d.pos[0]] + [d.pos[0] + np.cos(a)*SENSOR_RANGE
                                   for a in angles_fan] + [d.pos[0]]
            fan_y = [d.pos[1]] + [d.pos[1] + np.sin(a)*SENSOR_RANGE
                                   for a in angles_fan] + [d.pos[1]]
            ax.fill(fan_x, fan_y, color=DC[i], alpha=0.06)

        if d.path and len(d.path) > 1:
            px = [p[0] for p in d.path]
            py = [p[1] for p in d.path]
            ax.plot(px, py, '-', color=DC[i], lw=1.5, alpha=0.7)

        if d.target_frontier is not None:
            ax.plot(d.target_frontier[0], d.target_frontier[1], '*',
                    color=DC[i], ms=14, mec='k', mew=0.5, zorder=8)

        if show_trail and len(d.waypoints) > 1:
            trail = d.waypoints[-300:]
            ax.plot([w[0] for w in trail], [w[1] for w in trail],
                    '-', color=DC[i], lw=0.6, alpha=0.25)

    ratio = system.global_grid.explored_ratio
    label = step_label or f'Step {system.step}'
    ax.set_title(f'{label}  |  Coverage: {ratio:.1%}', fontsize=12, fontweight='bold')
    ax.set_xlim(0, WORLD_SIZE); ax.set_ylim(0, WORLD_SIZE)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')


# ================================================================ #
#  运行仿真
# ================================================================ #

def run_demo(seed=42, max_steps=400):
    system = ExplorationSystem(seed=seed, n_drones=3)
    n = system.n_drones

    snapshots = {}
    coverage_log = []
    obs_dist_log = []
    inter_dist_log = []

    print(f'[sim_demo] {n} UAVs, {len(system.env.obstacles)} obstacles, seed={seed}')
    print(f'[sim_demo] Distributed deconfliction (ID-based priority)')
    print()

    while not system.is_done() and system.step < max_steps:
        if system.step in SNAP_STEPS:
            snapshots[system.step] = snapshot_state(system)
            print(f'  [snap] step={system.step}, coverage={system.global_grid.explored_ratio:.1%}')

        ratio = system.run_step()
        coverage_log.append((system.step, ratio))
        all_obs = [system.min_obs_dist_log[i][-1] for i in range(n)]
        obs_dist_log.append(min(all_obs))
        inter_dist_log.append(system.min_drone_dist_log[-1])

    snapshots[system.step] = snapshot_state(system)
    print(f'  [final] step={system.step}, coverage={system.global_grid.explored_ratio:.1%}')

    return system, snapshots, coverage_log, obs_dist_log, inter_dist_log


def snapshot_state(system):
    return {
        'grid': system.global_grid.grid.copy(),
        'drones': [(d.pos.copy(), d.heading, list(d.path),
                    d.target_frontier.copy() if d.target_frontier is not None else None,
                    list(d.waypoints))
                   for d in system.env.drones],
        'obstacles': [(o.x, o.y, o.r) for o in system.env.obstacles],
        'ratio': system.global_grid.explored_ratio,
    }


# ================================================================ #
#  逐张生成图
# ================================================================ #

def fig1_snapshots(snapshots):
    """每个关键步骤单独一张地图快照"""
    keys = sorted(snapshots.keys())
    for step_key in keys:
        snap = snapshots[step_key]
        fig, ax = plt.subplots(figsize=(7, 7))
        g = snap['grid']
        dp = np.ones((GRID_N, GRID_N, 3))
        dp[g == FREE] = [0.95, 0.95, 0.95]
        dp[g == OCCUPIED] = [0.20, 0.20, 0.20]
        dp[g == UNKNOWN] = [0.82, 0.88, 0.95]
        ax.imshow(dp, origin='lower', extent=[0, WORLD_SIZE, 0, WORLD_SIZE])

        for ox, oy, orr in snap['obstacles']:
            ax.add_patch(plt.Circle((ox, oy), orr, fill=True,
                                    fc='#EF5350', alpha=0.25, ec='#C62828', lw=0.8))

        for i, (pos, heading, path, target, wps) in enumerate(snap['drones']):
            ax.plot(pos[0], pos[1], 'o', color=DC[i], ms=10,
                    mec='white', mew=1.5, zorder=10)
            hdx = np.cos(heading) * 3.0
            hdy = np.sin(heading) * 3.0
            ax.annotate('', xy=(pos[0]+hdx, pos[1]+hdy), xytext=(pos[0], pos[1]),
                        arrowprops=dict(arrowstyle='->', color=DC[i], lw=2))
            if path and len(path) > 1:
                ax.plot([p[0] for p in path], [p[1] for p in path],
                        '-', color=DC[i], lw=1.5, alpha=0.6)
            if target is not None:
                ax.plot(target[0], target[1], '*', color=DC[i], ms=14,
                        mec='k', mew=0.5, zorder=8)
            if len(wps) > 1:
                trail = wps[-200:]
                ax.plot([w[0] for w in trail], [w[1] for w in trail],
                        '-', color=DC[i], lw=0.5, alpha=0.2)

        ax.set_title(f'Step {step_key}  |  Coverage: {snap["ratio"]:.1%}',
                     fontsize=13, fontweight='bold')
        ax.set_xlim(0, WORLD_SIZE); ax.set_ylim(0, WORLD_SIZE)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        patches = [mpatches.Patch(color=DC[i], label=f'UAV-{i}') for i in range(3)]
        ax.legend(handles=patches, fontsize=9, loc='upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'1_snapshot_step{step_key:03d}.png'), dpi=200)
        plt.close(fig)
    print(f'  Saved {len(keys)} snapshot images')


def fig2_coverage(coverage_log):
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = [x[0] for x in coverage_log]
    vals = [x[1] * 100 for x in coverage_log]
    ax.plot(steps, vals, color='#2196F3', lw=2.5)
    ax.fill_between(steps, vals, alpha=0.12, color='#2196F3')
    ax.axhline(90, color='gray', ls='--', lw=1, alpha=0.6)
    ax.text(10, 91.5, '90% threshold', fontsize=9, color='gray')
    ax.set(xlabel='Step', ylabel='Coverage (%)', xlim=(0, None), ylim=(0, 100))
    ax.set_title('Exploration Coverage', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '2_coverage.png'), dpi=200)
    plt.close(fig)
    print('  Saved 2_coverage.png')


def fig3_safety(obs_dist_log, inter_dist_log):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(obs_dist_log, color='#FF9800', lw=0.7, alpha=0.8, label='Min Obstacle Dist')
    ax.plot(inter_dist_log, color='#7E57C2', lw=0.7, alpha=0.8, label='Min Inter-UAV Dist')
    ax.axhline(SAFE_RADIUS, color='red', ls='--', lw=1.5, label=f'Safe Radius ({SAFE_RADIUS}m)')
    ax.set(xlabel='Step', ylabel='Distance (m)', ylim=(0, None))
    ax.set_title('Safety Distances Over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '3_safety.png'), dpi=200)
    plt.close(fig)
    print('  Saved 3_safety.png')


def fig4_path_length(system):
    n = system.n_drones
    dists = [system.recorder.get_total_distance(i) for i in range(n)]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([f'UAV-{i}' for i in range(n)], dists,
                  color=DC[:n], alpha=0.85, edgecolor='white', lw=2, width=0.5)
    for bar, d in zip(bars, dists):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{d:.0f} m', ha='center', fontsize=11, fontweight='bold')
    ax.set(ylabel='Path Length (m)')
    ax.set_title('Per-UAV Path Length', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '4_path_length.png'), dpi=200)
    plt.close(fig)
    print('  Saved 4_path_length.png')


def fig5_obs_clearance(system):
    n = system.n_drones
    data = [system.min_obs_dist_log[i] for i in range(n)]
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(data, tick_labels=[f'UAV-{i}' for i in range(n)],
                    patch_artist=True, widths=0.5)
    for patch, c in zip(bp['boxes'], DC[:n]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.axhline(SAFE_RADIUS, color='red', ls='--', lw=1.2,
               label=f'Safe Radius ({SAFE_RADIUS}m)')
    ax.set(ylabel='Min Obstacle Dist (m)')
    ax.set_title('Obstacle Clearance Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '5_obs_clearance.png'), dpi=200)
    plt.close(fig)
    print('  Saved 5_obs_clearance.png')


def fig6_inter_uav(system):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(system.min_drone_dist_log, color='#7E57C2', lw=1)
    ax.axhline(SAFE_RADIUS, color='red', ls='--', lw=1.5,
               label=f'Safe Radius ({SAFE_RADIUS}m)')
    inter_safe = np.mean(np.array(system.min_drone_dist_log) > SAFE_RADIUS) * 100
    ax.set(xlabel='Step', ylabel='Min Inter-UAV Distance (m)', ylim=(0, None))
    ax.set_title(f'Inter-UAV Separation (safe rate: {inter_safe:.1f}%)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '6_inter_uav.png'), dpi=200)
    plt.close(fig)
    print('  Saved 6_inter_uav.png')


def fig7_pipeline():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis('off')

    ego_steps = [
        ('Stereo Camera\n/ CUDA', '#BBDEFB'),
        ('Depth Image\n/ PointCloud', '#C8E6C9'),
        ('3D Voxel\nGrid Map', '#FFF9C4'),
        ('B-spline\nTrajectory Opt', '#FFCCBC'),
        ('Swarm Traj\nBroadcast', '#E1BEE7'),
        ('SO3 Controller\n/ fake_drone', '#B2DFDB'),
    ]
    our_steps = [
        ('ONNX ResUNet\nDepth Est.', '#BBDEFB'),
        ('Ray-cast\n→ 2D Grid', '#C8E6C9'),
        ('Frontier\nDetection', '#FFF9C4'),
        ('Voronoi\nAllocation', '#FFE0B2'),
        ('A* Path\nPlanning', '#FFCCBC'),
        ('Distributed\nDeconflict', '#E1BEE7'),
        ('GPS Waypoint\nRecording', '#B2DFDB'),
    ]

    y_ego, y_our = 5.0, 1.5
    bw, bh, gap = 1.8, 1.1, 0.35

    ax.text(-0.5, y_ego + 0.15, 'EGO-Swarm', fontsize=13, fontweight='bold',
            va='center', ha='right', color='#1565C0')
    for idx, (label, color) in enumerate(ego_steps):
        x = idx * (bw + gap)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y_ego - bh/2), bw, bh, boxstyle='round,pad=0.1',
            facecolor=color, edgecolor='#616161', lw=1.2))
        ax.text(x + bw/2, y_ego, label, ha='center', va='center',
                fontsize=8, fontweight='bold')
        if idx > 0:
            ax.annotate('', xy=(x-0.02, y_ego), xytext=(x-gap+0.02, y_ego),
                        arrowprops=dict(arrowstyle='->', color='#424242', lw=1.5))

    ax.text(-0.5, y_our + 0.15, 'Ours', fontsize=13, fontweight='bold',
            va='center', ha='right', color='#2E7D32')
    for idx, (label, color) in enumerate(our_steps):
        x = idx * (bw + gap)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y_our - bh/2), bw, bh, boxstyle='round,pad=0.1',
            facecolor=color, edgecolor='#616161', lw=1.2))
        ax.text(x + bw/2, y_our, label, ha='center', va='center',
                fontsize=8, fontweight='bold')
        if idx > 0:
            ax.annotate('', xy=(x-0.02, y_our), xytext=(x-gap+0.02, y_our),
                        arrowprops=dict(arrowstyle='->', color='#424242', lw=1.5))

    for ego_i, our_i in [(0, 0), (1, 1)]:
        ex = ego_i * (bw + gap) + bw / 2
        ox = our_i * (bw + gap) + bw / 2
        ax.annotate('', xy=(ox, y_our + bh/2 + 0.05),
                    xytext=(ex, y_ego - bh/2 - 0.05),
                    arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.8, ls='--'))
        mid_y = (y_ego + y_our) / 2
        ax.text(ex + 0.15, mid_y, 'REPLACE', fontsize=7, color='#F44336',
                fontweight='bold', rotation=90, ha='left', va='center')

    for our_i in [2, 3]:
        ox = our_i * (bw + gap) + bw / 2
        ax.text(ox, y_our + bh/2 + 0.15, 'NEW', fontsize=8, color='#2E7D32',
                fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='#C8E6C9', ec='#2E7D32', lw=1))

    ax.set_xlim(-1.5, max(len(ego_steps), len(our_steps)) * (bw + gap))
    ax.set_ylim(0, 7)
    ax.set_title('Pipeline Comparison: EGO-Swarm vs Ours', fontsize=14,
                 fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '7_pipeline.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved 7_pipeline.png')


def fig8_comparison_table(system):
    n = system.n_drones
    all_obs = np.concatenate(system.min_obs_dist_log)
    avoid_rate = np.mean(all_obs > SAFE_RADIUS) * 100
    inter_safe = np.mean(np.array(system.min_drone_dist_log) > SAFE_RADIUS) * 100
    total_dist = sum(system.recorder.get_total_distance(i) for i in range(n))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axis('off')

    rows = [
        ['Exploration', 'Manual Goal', 'Autonomous Frontier'],
        ['Depth Sensor', 'CUDA / RealSense', 'ONNX ResUNet (14MB)'],
        ['Mapping', '3D Voxel (0.1m)', '2D Grid (0.5m)'],
        ['Area Allocation', '—', 'Voronoi Partition'],
        ['Path Planning', 'B-spline Optimization', 'A* Grid Search'],
        ['Deconfliction', 'Traj Repulsion\n(decentralized)', 'ID-Priority + Avoidance\n(distributed)'],
        ['ROS Dependency', 'Required', 'None'],
        ['', '', ''],
        ['Coverage', '—', f'{system.global_grid.explored_ratio*100:.1f}%'],
        ['Total Steps', '—', f'{system.step}'],
        ['Collisions', '—', f'{system.collision_count}'],
        ['Obstacle Avoid Rate', '—', f'{avoid_rate:.1f}%'],
        ['Inter-UAV Safe Rate', '—', f'{inter_safe:.1f}%'],
        ['Total Path Length', '—', f'{total_dist:.0f} m'],
    ]

    table = ax.table(cellText=rows, colLabels=['Module', 'EGO-Swarm', 'Ours (uav_exploration)'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#BDBDBD')
        if r == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', weight='bold')
        elif r == 8:
            cell.set_facecolor('#F5F5F5')
            cell.set_text_props(weight='bold')
        elif 1 <= r <= 7:
            if c == 2:
                cell.set_facecolor('#E8F5E9')
        elif r >= 9:
            if c == 2:
                cell.set_facecolor('#E3F2FD')

    ax.set_title('EGO-Swarm vs Ours — Feature & Metrics Comparison',
                 fontsize=13, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '8_comparison_table.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved 8_comparison_table.png')


def fig9_final_map(system):
    fig, ax = plt.subplots(figsize=(9, 9))
    render_map(ax, system, step_label='Final Result', show_fov=True)
    patches = [mpatches.Patch(color=DC[i], label=f'UAV-{i}') for i in range(system.n_drones)]
    patches += [mpatches.Patch(color=[0.82, 0.88, 0.95], label='Unknown'),
                mpatches.Patch(color=[0.20, 0.20, 0.20], label='Obstacle')]
    ax.legend(handles=patches, fontsize=10, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '9_final_map.png'), dpi=250)
    plt.close(fig)
    print('  Saved 9_final_map.png')


# ================================================================ #
#  主函数
# ================================================================ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print('=' * 60)
    print('  Multi-UAV Collaborative Exploration — sim_demo')
    print('  3 UAVs | Distributed Deconfliction | Frontier Exploration')
    print('=' * 60)
    print()

    system, snapshots, cov_log, obs_log, inter_log = run_demo(args.seed)
    print()
    print('Generating figures (one by one)...')

    fig1_snapshots(snapshots)
    fig2_coverage(cov_log)
    fig3_safety(obs_log, inter_log)
    fig4_path_length(system)
    fig5_obs_clearance(system)
    fig6_inter_uav(system)
    fig7_pipeline()
    fig8_comparison_table(system)
    fig9_final_map(system)

    # 汇总
    n = system.n_drones
    all_obs = np.concatenate(system.min_obs_dist_log)
    print()
    print('=' * 60)
    print('  Results Summary')
    print('=' * 60)
    print(f'  Drones:         {n}')
    print(f'  Deconfliction:  Distributed (ID-priority)')
    print(f'  Coverage:       {system.global_grid.explored_ratio:.1%}')
    print(f'  Steps:          {system.step}')
    print(f'  Collisions:     {system.collision_count}')
    print(f'  Avoid Rate:     {np.mean(all_obs > SAFE_RADIUS)*100:.1f}%')
    print(f'  Inter-UAV Safe: {np.mean(np.array(system.min_drone_dist_log) > SAFE_RADIUS)*100:.1f}%')
    for i in range(n):
        d = system.recorder.get_total_distance(i)
        print(f'  UAV-{i} path:    {d:.1f} m')
    print(f'\n  Output → {OUT}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
