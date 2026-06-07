"""
实验5: 避障安全性与机间防撞
- 3 UAV, seed=42, 运行至覆盖率>=90%
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.run_config import RunConfig, run_parametric, save_summary, save_raw_log
from experiments.plotting.common import (setup_style, save_figure, render_metrics_table,
                                          save_table_csv, COLORS_DRONE)
from config import SAFE_RADIUS

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'exp5_safety')


def run_exp5():
    setup_style()
    raw_dir = os.path.join(OUT_DIR, 'raw')
    fig_dir = os.path.join(OUT_DIR, 'figures')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print('=== Experiment 5: Safety Analysis ===')
    config = RunConfig(n_drones=3, max_speed=3.0, seed=42,
                       max_steps=2000, coverage_threshold=0.90)
    results, system = run_parametric(config, verbose=True)
    save_summary(results, raw_dir, 'safety_s42')
    save_raw_log(system, raw_dir, 'safety_s42')
    n = system.n_drones

    # ---- Fig 1: 三对机间距曲线 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    pair_labels = []
    for idx, ((i, j), dists) in enumerate(system.pairwise_drone_dist_log.items()):
        ax.plot(dists, color=COLORS_DRONE[idx % len(COLORS_DRONE)],
                lw=0.7, alpha=0.8, label=f'UAV-{i} ↔ UAV-{j}')
        pair_labels.append(f'UAV-{i}↔{j}')
    ax.axhline(SAFE_RADIUS, color='red', ls='--', lw=1.5,
               label=f'Safe radius ({SAFE_RADIUS}m)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Inter-UAV Distance (m)')
    ax.set_title('Inter-UAV Distance Over Time')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)
    save_figure(fig, fig_dir, 'fig_inter_uav_pairwise.png')

    # ---- Fig 2: 障碍距离 + 机间距 双轴 ----
    fig, ax1 = plt.subplots(figsize=(10, 5))
    min_obs_all = [min(system.min_obs_dist_log[i][s] for i in range(n))
                   for s in range(len(system.exploration_log))]
    ax1.plot(min_obs_all, color='#FF9800', lw=0.7, alpha=0.8, label='Min Obstacle Dist')
    ax1.axhline(SAFE_RADIUS, color='red', ls='--', lw=1.0)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Min Obstacle Distance (m)', color='#FF9800')
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis='y', labelcolor='#FF9800')

    ax2 = ax1.twinx()
    ax2.plot(system.min_drone_dist_log, color='#7E57C2', lw=0.7, alpha=0.8,
             label='Min Inter-UAV Dist')
    ax2.axhline(SAFE_RADIUS, color='#7E57C2', ls=':', lw=1.0)
    ax2.set_ylabel('Min Inter-UAV Distance (m)', color='#7E57C2')
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='y', labelcolor='#7E57C2')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax1.set_title('Obstacle & Inter-UAV Safety Distances')
    ax1.grid(True, alpha=0.2)
    save_figure(fig, fig_dir, 'fig_combined_safety.png')

    # ---- Fig 3: 各UAV障碍距离子图 ----
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for i in range(n):
        ax = axes[i]
        ax.plot(system.min_obs_dist_log[i], color=COLORS_DRONE[i], lw=0.6)
        ax.axhline(SAFE_RADIUS, color='red', ls='--', lw=1.0)
        pct = np.mean(np.array(system.min_obs_dist_log[i]) > SAFE_RADIUS) * 100
        ax.set_title(f'UAV-{i} (safe: {pct:.1f}%)')
        ax.set_xlabel('Step')
        if i == 0:
            ax.set_ylabel('Min Obstacle Distance (m)')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)
    fig.suptitle('Per-UAV Obstacle Clearance', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_per_uav_obs_dist.png')

    # ---- Fig 4: 安全四象限图 ----
    fig, ax = plt.subplots(figsize=(8, 7))
    n_steps = len(system.exploration_log)
    min_obs_step = []
    min_inter_step = []
    for s in range(n_steps):
        obs_d = min(system.min_obs_dist_log[i][s] for i in range(n))
        inter_d = system.min_drone_dist_log[s]
        min_obs_step.append(obs_d)
        min_inter_step.append(inter_d)

    sc = ax.scatter(min_obs_step, min_inter_step,
                    c=range(n_steps), cmap='viridis', s=3, alpha=0.5)
    ax.axvline(SAFE_RADIUS, color='red', ls='--', lw=1.5)
    ax.axhline(SAFE_RADIUS, color='red', ls='--', lw=1.5)

    obs_arr = np.array(min_obs_step)
    inter_arr = np.array(min_inter_step)
    q1 = np.mean((obs_arr >= SAFE_RADIUS) & (inter_arr >= SAFE_RADIUS)) * 100
    q2 = np.mean((obs_arr < SAFE_RADIUS) & (inter_arr >= SAFE_RADIUS)) * 100
    q3 = np.mean((obs_arr < SAFE_RADIUS) & (inter_arr < SAFE_RADIUS)) * 100
    q4 = np.mean((obs_arr >= SAFE_RADIUS) & (inter_arr < SAFE_RADIUS)) * 100

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(SAFE_RADIUS + (xlim[1]-SAFE_RADIUS)*0.5, SAFE_RADIUS + (ylim[1]-SAFE_RADIUS)*0.5,
            f'Safe\n{q1:.1f}%', ha='center', va='center', fontsize=14,
            fontweight='bold', color='#4CAF50', alpha=0.7)
    ax.text(SAFE_RADIUS * 0.5, SAFE_RADIUS + (ylim[1]-SAFE_RADIUS)*0.5,
            f'Obs Risk\n{q2:.1f}%', ha='center', va='center', fontsize=12,
            fontweight='bold', color='#FF9800', alpha=0.7)
    ax.text(SAFE_RADIUS * 0.5, SAFE_RADIUS * 0.5,
            f'Both Risk\n{q3:.1f}%', ha='center', va='center', fontsize=12,
            fontweight='bold', color='#F44336', alpha=0.7)
    ax.text(SAFE_RADIUS + (xlim[1]-SAFE_RADIUS)*0.5, SAFE_RADIUS * 0.5,
            f'UAV Risk\n{q4:.1f}%', ha='center', va='center', fontsize=12,
            fontweight='bold', color='#9C27B0', alpha=0.7)

    cb = plt.colorbar(sc, ax=ax, label='Step')
    ax.set_xlabel('Min Obstacle Distance (m)')
    ax.set_ylabel('Min Inter-UAV Distance (m)')
    ax.set_title('Safety Quadrant Analysis')
    ax.grid(True, alpha=0.2)
    save_figure(fig, fig_dir, 'fig_safety_quadrant.png')

    # ---- Fig 5: 安全指标表 ----
    all_obs = np.concatenate(system.min_obs_dist_log)
    fig, ax = plt.subplots(figsize=(8, 4))
    rows = [
        ['Collision Count', f'{system.collision_count}'],
        ['Obstacle Safe Rate (%)', f'{np.mean(all_obs > SAFE_RADIUS)*100:.1f}'],
        ['Min Obstacle Dist (m)', f'{np.min(all_obs):.2f}'],
        ['Avg Obstacle Dist (m)', f'{np.mean(all_obs):.2f}'],
        ['Min Inter-UAV Dist (m)', f'{np.min(inter_arr):.2f}'],
        ['Avg Inter-UAV Dist (m)', f'{np.mean(inter_arr):.2f}'],
        ['Inter-UAV Safe Rate (%)', f'{np.mean(inter_arr > SAFE_RADIUS)*100:.1f}'],
        ['Replan Count', f'{system.replan_count}'],
        ['Avoidance Events', f'{sum(system.avoidance_events)}'],
    ]
    render_metrics_table(ax, rows, ['Metric', 'Value'], 'Safety Metrics Summary')
    save_figure(fig, fig_dir, 'fig_safety_table.png')

    col_labels = ['Metric', 'Value']
    save_table_csv(rows, col_labels, raw_dir, 'table_safety.csv')

    print(f'  Saved to {OUT_DIR}/')
    return results


if __name__ == '__main__':
    run_exp5()
