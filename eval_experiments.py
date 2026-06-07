"""
学术实验评估脚本
==============================
实验 1: 单机 vs 多机探索效率对比 (轻量: 2 seeds × 800 steps)
实验 2: 避障安全性指标 (单次 seed=42)
实验 3: 最终地图 + 轨迹图 (单次 seed=42)
实验 4: 多种子鲁棒性统计 (5 seeds)

输出: output/figures/ 下的论文级图表
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from run_exploration import ExplorationSystem

FIG_DIR = os.path.join(os.path.dirname(__file__), 'output', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

C1, C2, C3 = '#2196F3', '#4CAF50', '#FF9800'
C_SINGLE = '#9E9E9E'
DRONE_COLORS = [C1, C2, C3]
RUN_STEPS = 800


def run_experiment(n_drones, seed, max_steps=RUN_STEPS):
    system = ExplorationSystem(seed=seed, n_drones=n_drones)
    while not system.is_done() and system.step < max_steps:
        system.run_step()
    return system


# =========================================================================== #
#  实验 1: 1 vs 2 vs 3 探索效率
# =========================================================================== #
def experiment_1():
    print('=== Exp1: Scalability ===')
    configs = [(1, C_SINGLE, '1 UAV'), (2, '#7E57C2', '2 UAVs'), (3, C1, '3 UAVs')]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n, color, label in configs:
        print(f'  n={n} seed=42...')
        sys = run_experiment(n, 42)
        steps = [x[0] for x in sys.exploration_log]
        vals = [x[1] * 100 for x in sys.exploration_log]
        ax.plot(steps, vals, color=color, linewidth=2, label=label)
        print(f'    → {sys.step} steps, {sys.global_grid.explored_ratio:.1%}')

    ax.axhline(90, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(30, 91.5, '90% threshold', fontsize=8, color='gray')
    ax.set_xlabel('Simulation Step', fontsize=11)
    ax.set_ylabel('Exploration Coverage (%)', fontsize=11)
    ax.set_title('Exploration Efficiency: Single vs. Multi-UAV', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(0, RUN_STEPS)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1_scalability.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig1')


# =========================================================================== #
#  实验 2: 避障安全性
# =========================================================================== #
def experiment_2(system):
    print('\n=== Exp2: Safety ===')
    n = system.n_drones

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # (a) 障碍距离时序
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(n):
        ax1.plot(system.min_obs_dist_log[i], color=DRONE_COLORS[i],
                 linewidth=0.5, alpha=0.7, label=f'UAV-{i}')
    ax1.axhline(SAFE_RADIUS, color='red', linestyle='--', linewidth=1.2,
                label=f'Safety margin ({SAFE_RADIUS}m)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Min Distance to Obstacle (m)')
    ax1.set_title('(a) Obstacle Clearance over Time', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.2)

    # (b) 距离直方图
    ax2 = fig.add_subplot(gs[0, 1])
    all_dists = np.concatenate(system.min_obs_dist_log)
    bins = np.linspace(0, min(all_dists.max(), 40), 50)
    ax2.hist(all_dists, bins=bins, color=C1, alpha=0.7, edgecolor='white', linewidth=0.5)
    ax2.axvline(SAFE_RADIUS, color='red', linestyle='--', linewidth=1.5)
    pct_safe = np.mean(all_dists > SAFE_RADIUS) * 100
    ax2.set_xlabel('Min Distance to Nearest Obstacle (m)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'(b) Clearance Distribution (safe rate: {pct_safe:.1f}%)', fontsize=11)
    ax2.grid(True, alpha=0.2)

    # (c) 机间距
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(system.min_drone_dist_log, color='#7E57C2', linewidth=0.8)
    ax3.axhline(SAFE_RADIUS, color='red', linestyle='--', linewidth=1.2,
                label=f'Inter-UAV safety ({SAFE_RADIUS}m)')
    pct_inter = np.mean(np.array(system.min_drone_dist_log) > SAFE_RADIUS) * 100
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Min Inter-UAV Distance (m)')
    ax3.set_title(f'(c) Inter-UAV Separation (safe rate: {pct_inter:.1f}%)', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.2)

    # (d) 汇总表
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    min_obs = min(min(d) for d in system.min_obs_dist_log)
    avg_obs = np.mean(all_dists)
    min_inter = min(system.min_drone_dist_log)
    avg_inter = np.mean(system.min_drone_dist_log)
    total_dist = sum(system.recorder.get_total_distance(i) for i in range(n))
    coverage = system.global_grid.explored_ratio * 100

    rows = [
        ['Coverage (%)', f'{coverage:.1f}'],
        ['Total Steps', f'{system.step}'],
        ['Total Path Length (m)', f'{total_dist:.1f}'],
        ['Collisions', f'{system.collision_count}'],
        ['Obstacle Avoid Rate (%)', f'{pct_safe:.1f}'],
        ['Min Obstacle Dist (m)', f'{min_obs:.2f}'],
        ['Avg Obstacle Dist (m)', f'{avg_obs:.2f}'],
        ['Min Inter-UAV Dist (m)', f'{min_inter:.2f}'],
        ['Avg Inter-UAV Dist (m)', f'{avg_inter:.2f}'],
        ['Inter-UAV Safe Rate (%)', f'{pct_inter:.1f}'],
        ['Replan Count', f'{system.replan_count}'],
        ['Avoidance Events', f'{sum(system.avoidance_events)}'],
    ]
    table = ax4.table(cellText=rows, colLabels=['Metric', 'Value'],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#E3F2FD')
            cell.set_text_props(weight='bold')
        cell.set_edgecolor('#BDBDBD')
    ax4.set_title('(d) Performance Summary', fontsize=11, pad=15)

    fig.savefig(os.path.join(FIG_DIR, 'fig2_safety.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig2')


# =========================================================================== #
#  实验 3: 地图 + 轨迹 + 柱状图
# =========================================================================== #
def experiment_3(system):
    print('\n=== Exp3: Trajectory Map ===')
    n = system.n_drones

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    # (a) 地图
    ax = axes[0]
    grid = system.global_grid.grid
    display = np.ones((GRID_N, GRID_N, 3))
    display[grid == FREE] = [0.96, 0.96, 0.96]
    display[grid == OCCUPIED] = [0.25, 0.25, 0.25]
    display[grid == UNKNOWN] = [0.82, 0.89, 0.96]
    ax.imshow(display, origin='lower', extent=[0, WORLD_SIZE, 0, WORLD_SIZE])

    for obs in system.env.obstacles:
        circle = plt.Circle((obs.x, obs.y), obs.r, fill=True,
                             facecolor='#EF5350', alpha=0.3,
                             edgecolor='#C62828', linewidth=1.2)
        ax.add_patch(circle)

    for i in range(n):
        wps = system.env.drones[i].waypoints
        if len(wps) > 1:
            wx, wy = zip(*wps)
            ax.plot(wx, wy, '-', color=DRONE_COLORS[i], linewidth=1.0,
                    alpha=0.6, label=f'UAV-{i}')
            ax.plot(wx[0], wy[0], 's', color=DRONE_COLORS[i], markersize=9,
                    markeredgecolor='black', markeredgewidth=0.8, zorder=5)
            ax.plot(wx[-1], wy[-1], '*', color=DRONE_COLORS[i], markersize=12,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=5)

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('(a) Exploration Map & UAV Trajectories', fontsize=12)
    ax.set_xlim(0, WORLD_SIZE); ax.set_ylim(0, WORLD_SIZE)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, loc='upper right')

    # (b) 柱状图
    ax2 = axes[1]
    bar_w = 0.22
    x_pos = np.arange(n)
    dists = [system.recorder.get_total_distance(i) for i in range(n)]
    wps = [len(system.recorder.waypoints[i]) for i in range(n)]
    avg_obs = [np.mean(system.min_obs_dist_log[i]) for i in range(n)]

    b1 = ax2.bar(x_pos - bar_w, dists, bar_w, label='Path Length (m)', color=C1, alpha=0.8)
    b2 = ax2.bar(x_pos, wps, bar_w, label='Waypoints', color=C2, alpha=0.8)
    b3 = ax2.bar(x_pos + bar_w, [d*10 for d in avg_obs], bar_w,
                 label='Avg Obs Dist (x10 m)', color=C3, alpha=0.8)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f'{h:.0f}', xy=(bar.get_x()+bar.get_width()/2, h),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', fontsize=7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'UAV-{i}' for i in range(n)])
    ax2.set_ylabel('Value', fontsize=11)
    ax2.set_title('(b) Per-UAV Performance', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig3_trajectory.png'), dpi=300)
    plt.close(fig)
    print('  Saved fig3')


# =========================================================================== #
#  实验 4: 多种子鲁棒性 (5 seeds)
# =========================================================================== #
def experiment_4():
    print('\n=== Exp4: Robustness ===')
    seeds = [42, 77, 123, 200, 256]
    records = []

    for s in seeds:
        print(f'  seed={s}...')
        sys = run_experiment(3, s, max_steps=RUN_STEPS)
        n = sys.n_drones
        all_obs = np.concatenate(sys.min_obs_dist_log)
        rec = {
            'seed': s,
            'steps': sys.step,
            'coverage': sys.global_grid.explored_ratio * 100,
            'collisions': sys.collision_count,
            'avoid_rate': np.mean(all_obs > SAFE_RADIUS) * 100,
            'min_obs': min(min(d) for d in sys.min_obs_dist_log),
            'total_dist': sum(sys.recorder.get_total_distance(i) for i in range(n)),
        }
        records.append(rec)
        print(f'    cov={rec["coverage"]:.1f}% avoid={rec["avoid_rate"]:.1f}%')

    # 箱线图
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    items = [
        ([r['coverage'] for r in records], 'Coverage (%)', C1),
        ([r['avoid_rate'] for r in records], 'Avoid Rate (%)', C2),
        ([r['steps'] for r in records], 'Steps', C3),
        ([r['min_obs'] for r in records], 'Min Obs Dist (m)', '#7E57C2'),
    ]
    for ax, (data, ylabel, color) in zip(axes, items):
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.6)
        bp['medians'][0].set_color('black')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticklabels([f'n=3\n({len(records)} seeds)'])
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_title(f'$\\mu$={np.mean(data):.1f}, $\\sigma$={np.std(data):.1f}',
                     fontsize=9)

    fig.suptitle(f'Robustness ({len(records)} random seeds)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_robustness.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print('  Saved fig4')

    # LaTeX 表
    print('\n  LaTeX Table:')
    print('  \\begin{tabular}{lcccc}')
    print('  \\toprule')
    print('  Seed & Coverage(\\%) & Avoid Rate(\\%) & Steps & Min Obs Dist(m) \\\\')
    print('  \\midrule')
    for r in records:
        print(f'  {r["seed"]} & {r["coverage"]:.1f} & {r["avoid_rate"]:.1f} '
              f'& {r["steps"]} & {r["min_obs"]:.2f} \\\\')
    print('  \\bottomrule')
    print('  \\end{tabular}')


# =========================================================================== #
if __name__ == '__main__':
    # 先跑一次 3 机基准 (复用于 exp2, exp3)
    print('Running baseline (3 drones, seed=42)...')
    baseline = run_experiment(3, 42)
    print(f'  → {baseline.step} steps, {baseline.global_grid.explored_ratio:.1%}\n')

    experiment_2(baseline)
    experiment_3(baseline)
    experiment_1()
    experiment_4()
    print(f'\nAll figures → {FIG_DIR}/')
