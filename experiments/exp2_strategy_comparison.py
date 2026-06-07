"""
实验2: 协同探索与规划策略对比
- 6种方法, 3 UAVs, speed 3, 10 seeds
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.run_config import RunConfig, run_parametric, save_summary
from experiments.plotting.common import (setup_style, save_figure, render_grid_on_ax,
                                          render_metrics_table, save_table_csv, COLORS_METHOD)
from config import SAFE_RADIUS

SEEDS = [42, 77, 123, 200, 256, 314, 512, 666, 777, 999]
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'exp2_strategy')

METHODS = {
    'Ours': dict(frontier_strategy='utility', use_voronoi=True, use_deconflict=True, use_obstacle_inflation=True),
    'Greedy-Frontier': dict(frontier_strategy='greedy', use_voronoi=False, use_deconflict=True, use_obstacle_inflation=True),
    'Random-Frontier': dict(frontier_strategy='random', use_voronoi=False, use_deconflict=True, use_obstacle_inflation=True),
    'w/o Voronoi': dict(frontier_strategy='utility', use_voronoi=False, use_deconflict=True, use_obstacle_inflation=True),
    'w/o Deconflict': dict(frontier_strategy='utility', use_voronoi=True, use_deconflict=False, use_obstacle_inflation=True),
    'w/o Inflation': dict(frontier_strategy='utility', use_voronoi=True, use_deconflict=True, use_obstacle_inflation=False),
}


def run_exp2():
    setup_style()
    raw_dir = os.path.join(OUT_DIR, 'raw')
    fig_dir = os.path.join(OUT_DIR, 'figures')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print('=== Experiment 2: Strategy Comparison ===')
    all_data = {}

    for method_name, method_cfg in METHODS.items():
        all_data[method_name] = []
        for seed in SEEDS:
            snap = [0, 100, 200, 300] if seed == 42 else None
            config = RunConfig(
                n_drones=3, max_speed=3.0, seed=seed,
                max_steps=800, coverage_threshold=0.90,
                **method_cfg
            )
            results, system = run_parametric(config, snapshot_steps=snap)
            save_summary(results, raw_dir, f'{method_name.replace(" ", "_").replace("/", "")}_s{seed}')
            all_data[method_name].append(results)
            print(f'  {method_name:20s} s={seed} → cov={results["final_coverage"]:.1%} '
                  f'steps={results["final_step"]} coll={results["collision_count"]}')

    method_names = list(METHODS.keys())
    n_methods = len(method_names)

    # ---- Fig 1: 各方法探索快照 (seed=42) ----
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    for idx, method_name in enumerate(method_names):
        r, c = idx // 3, idx % 3
        ax = axes[r][c]
        res42 = [r2 for r2 in all_data[method_name] if r2['config']['seed'] == 42]
        if res42 and 'snapshots' in res42[0] and res42[0]['snapshots']:
            snaps = res42[0]['snapshots']
            last_key = max(snaps.keys())
            snap = snaps[last_key]
            render_grid_on_ax(ax, snap['grid'], obstacles=snap['obstacles'],
                             drones=snap['drones'],
                             title=f'{method_name}\nCov={snap["ratio"]:.1%}')
        else:
            ax.set_title(method_name)
            ax.text(0.5, 0.5, 'No snapshot', transform=ax.transAxes,
                    ha='center', va='center')
    fig.suptitle('Strategy Comparison: Final Maps (seed=42)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_final_maps.png')

    # ---- Fig 2: 指标柱状图 ----
    metric_keys = [
        ('steps_to_threshold', 'Steps to 90%'),
        ('final_coverage', 'Coverage (ratio)'),
        ('overlap_rate', 'Overlap Rate'),
        ('total_path_length', 'Total Path (m)'),
        ('collision_count', 'Collisions'),
        ('inter_uav_safe_rate', 'Inter-UAV Safe Rate'),
        ('obstacle_safe_rate', 'Obstacle Safe Rate'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    success_rate = []
    for mn in method_names:
        sr = np.mean([1 if r['success'] else 0 for r in all_data[mn]]) * 100
        success_rate.append(sr)

    ax = axes[0]
    bars = ax.bar(range(n_methods), success_rate,
                  color=COLORS_METHOD[:n_methods], alpha=0.8)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(method_names, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate')
    ax.grid(True, alpha=0.2, axis='y')
    for bar, v in zip(bars, success_rate):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{v:.0f}%', ha='center', fontsize=7)

    for fig_idx, (mkey, mtitle) in enumerate(metric_keys):
        ax = axes[fig_idx + 1]
        means = []
        stds = []
        for mn in method_names:
            vals = [r[mkey] for r in all_data[mn]]
            if mkey in ('final_coverage', 'overlap_rate', 'inter_uav_safe_rate', 'obstacle_safe_rate'):
                vals = [v * 100 for v in vals]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        x = range(n_methods)
        bars = ax.bar(x, means, yerr=stds, color=COLORS_METHOD[:n_methods],
                      alpha=0.8, capsize=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels(method_names, fontsize=7, rotation=30, ha='right')
        unit = '(%)' if mkey in ('final_coverage', 'overlap_rate', 'inter_uav_safe_rate', 'obstacle_safe_rate') else ''
        ax.set_ylabel(f'{mtitle} {unit}')
        ax.set_title(mtitle)
        ax.grid(True, alpha=0.2, axis='y')
        if fig_idx + 2 < len(axes):
            pass

    if len(metric_keys) + 1 < len(axes):
        axes[-1].axis('off')
    fig.suptitle('Strategy Comparison: Metrics', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_bar_charts.png')

    # ---- Fig 3: 指标均值表 ----
    fig, ax = plt.subplots(figsize=(18, 6))
    col_labels = ['Method', 'Steps', 'Success(%)', 'Coverage(%)', 'Overlap(%)',
                  'Path(m)', 'Collisions', 'Inter Safe(%)', 'Obs Safe(%)']
    rows = []
    for mn in method_names:
        rl = all_data[mn]
        rows.append([
            mn,
            f'{np.mean([r["steps_to_threshold"] for r in rl]):.0f}',
            f'{np.mean([1 if r["success"] else 0 for r in rl])*100:.0f}',
            f'{np.mean([r["final_coverage"]*100 for r in rl]):.1f}',
            f'{np.mean([r["overlap_rate"]*100 for r in rl]):.1f}',
            f'{np.mean([r["total_path_length"] for r in rl]):.0f}',
            f'{np.mean([r["collision_count"] for r in rl]):.1f}',
            f'{np.mean([r["inter_uav_safe_rate"]*100 for r in rl]):.1f}',
            f'{np.mean([r["obstacle_safe_rate"]*100 for r in rl]):.1f}',
        ])
    render_metrics_table(ax, rows, col_labels, 'Strategy Comparison: Mean Metrics')
    save_figure(fig, fig_dir, 'fig_metrics_table.png')
    save_table_csv(rows, col_labels, raw_dir, 'table_metrics.csv')

    print(f'  Saved to {OUT_DIR}/')
    return all_data


if __name__ == '__main__':
    run_exp2()
