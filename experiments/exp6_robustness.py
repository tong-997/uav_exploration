"""
实验6: 多随机种子鲁棒性
- 10 seeds, 3 UAVs, speed 3, max_steps=800
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.run_config import RunConfig, run_parametric, save_summary
from experiments.plotting.common import (setup_style, save_figure, render_grid_on_ax,
                                          render_metrics_table, save_table_csv, COLORS_DRONE)
from config import SAFE_RADIUS

SEEDS = [42, 77, 123, 200, 256, 314, 512, 666, 777, 999]
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'exp6_robustness')


def run_exp6():
    setup_style()
    raw_dir = os.path.join(OUT_DIR, 'raw')
    fig_dir = os.path.join(OUT_DIR, 'figures')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print('=== Experiment 6: Multi-Seed Robustness ===')
    all_results = []
    all_systems = []

    for seed in SEEDS:
        print(f'  seed={seed}...')
        config = RunConfig(n_drones=3, max_speed=3.0, seed=seed,
                           max_steps=800, coverage_threshold=0.90)
        results, system = run_parametric(config, verbose=False)
        save_summary(results, raw_dir, f'robustness_s{seed}')
        all_results.append(results)
        all_systems.append(system)
        print(f'    coverage={results["final_coverage"]:.1%}, steps={results["final_step"]}')

    # ---- Fig 1: 箱线图 ----
    coverages = [r['final_coverage'] * 100 for r in all_results]
    steps_list = [r['steps_to_threshold'] for r in all_results]
    obs_safe = [r['obstacle_safe_rate'] * 100 for r in all_results]
    min_obs = [r['min_obstacle_distance'] for r in all_results]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    items = [
        (coverages, 'Coverage (%)', '#2196F3'),
        (steps_list, 'Steps to 90%', '#4CAF50'),
        (obs_safe, 'Obstacle Safe Rate (%)', '#FF9800'),
        (min_obs, 'Min Obstacle Dist (m)', '#9C27B0'),
    ]
    for ax, (data, ylabel, color) in zip(axes, items):
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.6)
        bp['medians'][0].set_color('black')
        ax.set_ylabel(ylabel)
        ax.set_xticklabels([f'n=3\n({len(SEEDS)} seeds)'])
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_title(f'$\\mu$={np.mean(data):.1f}, $\\sigma$={np.std(data):.1f}', fontsize=9)
    fig.suptitle(f'Robustness Analysis ({len(SEEDS)} Random Seeds)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_box_plots.png')

    # ---- Fig 2: 最终地图拼图 ----
    nrow, ncol = 2, 5
    fig, axes = plt.subplots(nrow, ncol, figsize=(25, 10))
    for idx, (seed, system) in enumerate(zip(SEEDS, all_systems)):
        r, c = idx // ncol, idx % ncol
        ax = axes[r][c]
        render_grid_on_ax(ax, system.global_grid.grid,
                         obstacles=[(o.x, o.y, o.r) for o in system.env.obstacles],
                         title=f'Seed={seed}\nCov={system.global_grid.explored_ratio:.1%}')
    fig.suptitle('Final Maps Across Seeds', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_final_map_mosaic.png')

    # ---- Fig 3: 覆盖率曲线 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    all_curves = []
    for idx, (seed, res) in enumerate(zip(SEEDS, all_results)):
        steps = [x[0] for x in res['exploration_log']]
        vals = [x[1] * 100 for x in res['exploration_log']]
        ax.plot(steps, vals, lw=0.5, alpha=0.4, color='#2196F3')
        all_curves.append((steps, vals))

    max_len = max(len(c[0]) for c in all_curves)
    interp_vals = np.zeros((len(SEEDS), max_len))
    interp_vals[:] = np.nan
    for idx, (steps, vals) in enumerate(all_curves):
        interp_vals[idx, :len(vals)] = vals
        if len(vals) < max_len:
            interp_vals[idx, len(vals):] = vals[-1]
    mean_curve = np.nanmean(interp_vals, axis=0)
    std_curve = np.nanstd(interp_vals, axis=0)
    x_axis = range(max_len)
    ax.plot(x_axis, mean_curve, color='#1565C0', lw=2.5, label='Mean')
    ax.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve,
                    alpha=0.15, color='#2196F3')
    ax.axhline(90, color='gray', ls='--', lw=1.0, alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Coverage Across Seeds (mean ± std)')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 800)
    ax.grid(True, alpha=0.2)
    save_figure(fig, fig_dir, 'fig_coverage_curves.png')

    # ---- Fig 4: 统计表 ----
    fig, ax = plt.subplots(figsize=(14, 6))
    col_labels = ['Seed', 'Coverage(%)', 'Steps', 'Collisions', 'Obs Safe(%)',
                  'Min Obs(m)', 'Inter Safe(%)', 'Path(m)', 'Load CV']
    rows = []
    for seed, r in zip(SEEDS, all_results):
        rows.append([
            str(seed),
            f'{r["final_coverage"]*100:.1f}',
            str(r['steps_to_threshold']),
            str(r['collision_count']),
            f'{r["obstacle_safe_rate"]*100:.1f}',
            f'{r["min_obstacle_distance"]:.2f}',
            f'{r["inter_uav_safe_rate"]*100:.1f}',
            f'{r["total_path_length"]:.0f}',
            f'{r["load_balance_cv"]:.3f}',
        ])
    rows.append([
        'Mean±Std',
        f'{np.mean(coverages):.1f}±{np.std(coverages):.1f}',
        f'{np.mean(steps_list):.0f}±{np.std(steps_list):.0f}',
        f'{np.mean([r["collision_count"] for r in all_results]):.1f}',
        f'{np.mean(obs_safe):.1f}±{np.std(obs_safe):.1f}',
        f'{np.mean(min_obs):.2f}±{np.std(min_obs):.2f}',
        f'{np.mean([r["inter_uav_safe_rate"]*100 for r in all_results]):.1f}',
        f'{np.mean([r["total_path_length"] for r in all_results]):.0f}',
        f'{np.mean([r["load_balance_cv"] for r in all_results]):.3f}',
    ])
    render_metrics_table(ax, rows, col_labels, 'Robustness Statistics')
    save_figure(fig, fig_dir, 'fig_stats_table.png')
    save_table_csv(rows, col_labels, raw_dir, 'table_robustness.csv')

    print(f'  Saved to {OUT_DIR}/')
    return all_results


if __name__ == '__main__':
    run_exp6()
