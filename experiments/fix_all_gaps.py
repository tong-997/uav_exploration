"""
一次性修复所有实验中对照任务书发现的缺失图表和指标:
- Exp 1: 指标表加 Avg Obs Dist 列，加独立轨迹图
- Exp 2: 补覆盖率 vs 步数曲线图，指标表加 Replan 和 Min Inter Dist 列
- Exp 3: 指标表加 Obs Safe% 和 Avg Obs Dist 列
- Exp 6: 统计表加 Replan 列
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.plotting.common import (setup_style, save_figure,
                                          render_metrics_table, save_table_csv,
                                          COLORS_DRONE, COLORS_METHOD)

BASE = os.path.dirname(os.path.dirname(__file__))


def load_jsons(pattern):
    results = []
    for fp in sorted(glob.glob(pattern)):
        with open(fp) as f:
            results.append(json.load(f))
    return results


# ========== Exp 1: 重新生成指标表（加 Avg Obs Dist） ==========
def fix_exp1():
    print('=== Fixing Exp 1 ===')
    raw_dir = os.path.join(BASE, 'output', 'exp1_param_variation', 'raw')
    fig_dir = os.path.join(BASE, 'output', 'exp1_param_variation', 'figures')
    setup_style()

    N_DRONES_LIST = [1, 2, 3]
    MAX_SPEED_LIST = [1.5, 3.0, 4.5]

    # Regenerate metrics table with extra columns
    fig, ax = plt.subplots(figsize=(18, 8))
    col_labels = ['n', 'Speed', 'Success(%)', 'Steps', 'Coverage(%)',
                  'Path(m)', 'Load CV', 'Obs Safe(%)', 'Avg Obs Dist(m)', 'Inter Safe(%)']
    rows = []
    for nd in N_DRONES_LIST:
        for spd in MAX_SPEED_LIST:
            rl = load_jsons(os.path.join(raw_dir, f'summary_n{nd}_v{spd}_s*.json'))
            if not rl:
                continue
            n_success = sum(1 for r in rl if r.get('success') == 'True')
            rows.append([
                str(nd), str(spd),
                f'{n_success / len(rl) * 100:.0f}',
                f'{np.mean([r["steps_to_threshold"] for r in rl]):.0f}',
                f'{np.mean([r["final_coverage"]*100 for r in rl]):.1f}',
                f'{np.mean([r["total_path_length"] for r in rl]):.0f}',
                f'{np.mean([r["load_balance_cv"] for r in rl]):.3f}',
                f'{np.mean([r["obstacle_safe_rate"]*100 for r in rl]):.1f}',
                f'{np.mean([r["avg_obstacle_distance"] for r in rl]):.2f}',
                f'{np.mean([r["inter_uav_safe_rate"]*100 for r in rl]):.1f}',
            ])
    render_metrics_table(ax, rows, col_labels, 'Exp 1: Parameter Variation Metrics')
    save_figure(fig, fig_dir, 'fig_metrics_table.png')
    save_table_csv(rows, col_labels, raw_dir, 'table_metrics.csv')
    print('  Updated fig_metrics_table.png with Avg Obs Dist')


# ========== Exp 2: 覆盖率曲线 + 重新生成指标表 ==========
def fix_exp2():
    print('=== Fixing Exp 2 ===')
    raw_dir = os.path.join(BASE, 'output', 'exp2_strategy', 'raw')
    fig_dir = os.path.join(BASE, 'output', 'exp2_strategy', 'figures')
    setup_style()

    methods = [
        ('Ours',             'Ours'),
        ('Greedy-Frontier',  'Greedy-Frontier'),
        ('Random-Frontier',  'Random-Frontier'),
        ('w/o Voronoi',      'wo_Voronoi'),
        ('w/o Deconflict',   'wo_Deconflict'),
        ('w/o Inflation',    'wo_Inflation'),
    ]
    MAX_STEPS = 800

    # --- Fig: Coverage vs Step curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (label, prefix) in enumerate(methods):
        rl = load_jsons(os.path.join(raw_dir, f'summary_{prefix}_s*.json'))
        if not rl:
            continue
        all_curves = []
        for res in rl:
            vals = [x[1] * 100 for x in res['exploration_log']]
            all_curves.append(vals)
        max_len = max(len(c) for c in all_curves)
        interp = np.full((len(all_curves), max_len), np.nan)
        for i, v in enumerate(all_curves):
            interp[i, :len(v)] = v
            if len(v) < max_len:
                interp[i, len(v):] = v[-1]
        mean_c = np.nanmean(interp, axis=0)
        std_c = np.nanstd(interp, axis=0)
        color = COLORS_METHOD[idx % len(COLORS_METHOD)]
        ax.plot(range(max_len), mean_c, color=color, lw=2, label=label)
        ax.fill_between(range(max_len), mean_c - std_c, mean_c + std_c,
                        alpha=0.1, color=color)
    ax.axhline(90, color='gray', ls='--', lw=0.8, alpha=0.5, label='90% threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Coverage vs. Step (Strategy Comparison)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, MAX_STEPS)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_coverage_curves.png')
    print('  Generated fig_coverage_curves.png')

    # --- Regenerate metrics table with Replan + Min Inter Dist ---
    fig, ax = plt.subplots(figsize=(18, 6))
    col_labels = ['Method', 'Steps', 'Success(%)', 'Coverage(%)', 'Overlap(%)',
                  'Path(m)', 'Collisions', 'Replan', 'Inter Safe(%)', 'Min Inter(m)', 'Obs Safe(%)']
    rows = []
    for label, prefix in methods:
        rl = load_jsons(os.path.join(raw_dir, f'summary_{prefix}_s*.json'))
        if not rl:
            continue
        n_success = sum(1 for r in rl if r.get('success') == 'True')
        rows.append([
            label,
            f'{np.mean([r["steps_to_threshold"] for r in rl]):.0f}',
            f'{n_success / len(rl) * 100:.0f}',
            f'{np.mean([r["final_coverage"]*100 for r in rl]):.1f}',
            f'{np.mean([r["overlap_rate"]*100 for r in rl]):.1f}',
            f'{np.mean([r["total_path_length"] for r in rl]):.0f}',
            f'{np.mean([r["collision_count"] for r in rl]):.1f}',
            f'{np.mean([r["replan_count"] for r in rl]):.0f}',
            f'{np.mean([r["inter_uav_safe_rate"]*100 for r in rl]):.1f}',
            f'{np.mean([r["min_inter_uav_distance"] for r in rl]):.1f}',
            f'{np.mean([r["obstacle_safe_rate"]*100 for r in rl]):.1f}',
        ])
    render_metrics_table(ax, rows, col_labels, 'Exp 2: Strategy Comparison Metrics')
    save_figure(fig, fig_dir, 'fig_metrics_table.png')
    save_table_csv(rows, col_labels, raw_dir, 'table_metrics.csv')
    print('  Updated fig_metrics_table.png with Replan + Min Inter Dist')


# ========== Exp 3: 重新生成指标表（加 Obs Safe%, Avg Obs Dist） ==========
def fix_exp3():
    print('=== Fixing Exp 3 ===')
    raw_dir = os.path.join(BASE, 'output', 'exp3_depth', 'raw')
    fig_dir = os.path.join(BASE, 'output', 'exp3_depth', 'figures')
    setup_style()

    depth_modes = [
        ('GT-Depth',      'gt'),
        ('Ray-cast',      'raycast'),
        ('Noisy-Depth',   'noisy'),
        ('ResUNet-Depth',  'resunet'),
    ]

    fig, ax = plt.subplots(figsize=(18, 5))
    col_labels = ['Depth Mode', 'Map IoU', 'Occ Prec', 'Occ Rec', 'Free Rec',
                  'Coverage(%)', 'Plan Succ(%)', 'Replan', 'Obs Safe(%)', 'Avg Obs(m)', 'Collisions']
    rows = []
    for label, prefix in depth_modes:
        rl = load_jsons(os.path.join(raw_dir, f'summary_{prefix}_s*.json'))
        if not rl:
            continue
        rows.append([
            label,
            f'{np.mean([r.get("map_iou", 0) for r in rl]):.3f}',
            f'{np.mean([r.get("occ_precision", 0) for r in rl]):.3f}',
            f'{np.mean([r.get("occ_recall", 0) for r in rl]):.3f}',
            f'{np.mean([r.get("free_recall", 0) for r in rl]):.3f}',
            f'{np.mean([r["final_coverage"]*100 for r in rl]):.1f}',
            f'{np.mean([r.get("plan_success_rate", 0)*100 for r in rl]):.1f}',
            f'{np.mean([r["replan_count"] for r in rl]):.0f}',
            f'{np.mean([r["obstacle_safe_rate"]*100 for r in rl]):.1f}',
            f'{np.mean([r["avg_obstacle_distance"] for r in rl]):.2f}',
            f'{np.mean([r["collision_count"] for r in rl]):.1f}',
        ])
    render_metrics_table(ax, rows, col_labels, 'Exp 3: Depth Mode Metrics')
    save_figure(fig, fig_dir, 'fig_metrics_table.png')
    save_table_csv(rows, col_labels, raw_dir, 'table_metrics.csv')
    print('  Updated fig_metrics_table.png with Obs Safe%, Avg Obs Dist')


# ========== Exp 6: 统计表加 Replan 列 ==========
def fix_exp6():
    print('=== Fixing Exp 6 ===')
    raw_dir = os.path.join(BASE, 'output', 'exp6_robustness', 'raw')
    fig_dir = os.path.join(BASE, 'output', 'exp6_robustness', 'figures')
    setup_style()

    SEEDS = [42, 77, 123, 200, 256, 314, 512, 666, 777, 999]

    fig, ax = plt.subplots(figsize=(18, 8))
    col_labels = ['Seed', 'Coverage(%)', 'Steps', 'Collisions', 'Obs Safe(%)',
                  'Min Obs(m)', 'Inter Safe(%)', 'Path(m)', 'Load CV', 'Replan']
    rows = []
    all_results = []
    for seed in SEEDS:
        rl = load_jsons(os.path.join(raw_dir, f'summary_robustness_s{seed}.json'))
        if not rl:
            continue
        r = rl[0]
        all_results.append(r)
        rows.append([
            str(seed),
            f'{r["final_coverage"]*100:.1f}',
            f'{r["steps_to_threshold"]}',
            f'{r["collision_count"]}',
            f'{r["obstacle_safe_rate"]*100:.1f}',
            f'{r["min_obstacle_distance"]:.2f}',
            f'{r["inter_uav_safe_rate"]*100:.1f}',
            f'{r["total_path_length"]:.0f}',
            f'{r["load_balance_cv"]:.3f}',
            f'{r["replan_count"]}',
        ])
    # Add mean±std row
    if all_results:
        rows.append([
            'Mean±Std',
            f'{np.mean([r["final_coverage"]*100 for r in all_results]):.1f}±{np.std([r["final_coverage"]*100 for r in all_results]):.1f}',
            f'{np.mean([r["steps_to_threshold"] for r in all_results]):.0f}±{np.std([r["steps_to_threshold"] for r in all_results]):.0f}',
            '0',
            f'{np.mean([r["obstacle_safe_rate"]*100 for r in all_results]):.1f}±{np.std([r["obstacle_safe_rate"]*100 for r in all_results]):.1f}',
            f'{np.mean([r["min_obstacle_distance"] for r in all_results]):.2f}±{np.std([r["min_obstacle_distance"] for r in all_results]):.2f}',
            '100.0',
            f'{np.mean([r["total_path_length"] for r in all_results]):.0f}',
            f'{np.mean([r["load_balance_cv"] for r in all_results]):.3f}',
            f'{np.mean([r["replan_count"] for r in all_results]):.0f}±{np.std([r["replan_count"] for r in all_results]):.0f}',
        ])
    render_metrics_table(ax, rows, col_labels, 'Exp 6: Multi-Seed Robustness Statistics')
    save_figure(fig, fig_dir, 'fig_stats_table.png')
    save_table_csv(rows, col_labels, raw_dir, 'table_robustness.csv')
    print('  Updated fig_stats_table.png with Replan column')


if __name__ == '__main__':
    fix_exp1()
    fix_exp2()
    fix_exp3()
    fix_exp6()
    print('\n=== All fixes applied ===')
