"""
实验3: 深度输入模式对局部建图质量
- 4 modes: GT-Depth, Ray-cast, Noisy-Depth, ResUNet-Depth
- 3 UAVs, speed 3, 10 seeds
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.run_config import RunConfig, run_parametric, save_summary
from experiments.plotting.common import (setup_style, save_figure, render_grid_on_ax,
                                          render_metrics_table, save_table_csv, COLORS_DEPTH)
from config import SAFE_RADIUS, GRID_N, FREE, OCCUPIED, UNKNOWN

SEEDS = [42, 77, 123, 200, 256, 314, 512, 666, 777, 999]
DEPTH_MODES = ['gt', 'raycast', 'noisy', 'resunet']
DEPTH_LABELS = {'gt': 'GT-Depth', 'raycast': 'Ray-cast', 'noisy': 'Noisy-Depth', 'resunet': 'ResUNet-Depth'}
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'exp3_depth')


def compute_map_quality(system):
    gt_grid = system.env.get_ground_truth_grid()
    pred = system.global_grid.grid
    known_mask = pred != UNKNOWN

    gt_occ = gt_grid == OCCUPIED
    gt_free = gt_grid == FREE
    pred_occ = pred == OCCUPIED
    pred_free = pred == FREE

    known_correct = np.sum((pred[known_mask] == gt_grid[known_mask]))
    known_total = np.sum(known_mask)
    iou = known_correct / max(known_total, 1)

    pred_occ_known = pred_occ & known_mask
    occ_tp = np.sum(pred_occ_known & gt_occ)
    occ_precision = occ_tp / max(np.sum(pred_occ_known), 1)
    occ_recall = occ_tp / max(np.sum(gt_occ & known_mask), 1)

    pred_free_known = pred_free & known_mask
    free_tp = np.sum(pred_free_known & gt_free)
    free_recall = free_tp / max(np.sum(gt_free & known_mask), 1)

    return {
        'map_iou': float(iou),
        'occ_precision': float(occ_precision),
        'occ_recall': float(occ_recall),
        'free_recall': float(free_recall),
    }


def run_exp3():
    setup_style()
    raw_dir = os.path.join(OUT_DIR, 'raw')
    fig_dir = os.path.join(OUT_DIR, 'figures')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print('=== Experiment 3: Depth Input Modes ===')
    all_data = {}

    for mode in DEPTH_MODES:
        all_data[mode] = []
        for seed in SEEDS:
            snap = [100] if seed == 42 else None
            config = RunConfig(
                n_drones=3, max_speed=3.0, seed=seed,
                max_steps=800, coverage_threshold=0.90,
                depth_mode=mode, depth_noise_std=0.5
            )
            results, system = run_parametric(config, snapshot_steps=snap)

            quality = compute_map_quality(system)
            results.update(quality)
            results['plan_success_rate'] = system.plan_success_count / max(system.plan_total_count, 1)

            save_summary(results, raw_dir, f'{mode}_s{seed}')
            all_data[mode].append(results)
            print(f'  {DEPTH_LABELS[mode]:15s} s={seed} → cov={results["final_coverage"]:.1%} '
                  f'IoU={quality["map_iou"]:.3f}')

    # ---- Fig 1: 深度图→栅格示意 (seed=42, step=100) ----
    fig, axes = plt.subplots(2, len(DEPTH_MODES), figsize=(5 * len(DEPTH_MODES), 10))
    for idx, mode in enumerate(DEPTH_MODES):
        res42 = [r for r in all_data[mode] if r['config']['seed'] == 42]
        ax_top = axes[0][idx]
        ax_bot = axes[1][idx]

        if res42 and 'snapshots' in res42[0] and res42[0]['snapshots']:
            snaps = res42[0]['snapshots']
            snap_key = min(snaps.keys())
            snap = snaps[snap_key]
            render_grid_on_ax(ax_top, snap['grid'], obstacles=snap['obstacles'],
                             title=f'{DEPTH_LABELS[mode]}\nStep {snap_key}')
        else:
            ax_top.set_title(f'{DEPTH_LABELS[mode]}\n(no snapshot)')
            ax_top.axis('off')

        config = RunConfig(n_drones=3, max_speed=3.0, seed=42,
                           max_steps=800, depth_mode=mode)
        _, sys42 = run_parametric(config)
        render_grid_on_ax(ax_bot, sys42.global_grid.grid,
                         obstacles=[(o.x, o.y, o.r) for o in sys42.env.obstacles],
                         title=f'Final Map\nCov={sys42.global_grid.explored_ratio:.1%}')

    fig.suptitle('Depth Mode → Occupancy Grid', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_depth_grid_illustration.png')

    # ---- Fig 2: 最终地图对比 (seed=42) ----
    fig, axes = plt.subplots(1, len(DEPTH_MODES), figsize=(6 * len(DEPTH_MODES), 6))
    for idx, mode in enumerate(DEPTH_MODES):
        config = RunConfig(n_drones=3, max_speed=3.0, seed=42,
                           max_steps=800, depth_mode=mode)
        _, sys42 = run_parametric(config)
        ax = axes[idx]
        render_grid_on_ax(ax, sys42.global_grid.grid,
                         obstacles=[(o.x, o.y, o.r) for o in sys42.env.obstacles],
                         title=f'{DEPTH_LABELS[mode]}\nCov={sys42.global_grid.explored_ratio:.1%}')
    fig.suptitle('Final Map Comparison (seed=42)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_final_map_comparison.png')

    # ---- Fig 3: 建图质量柱状图 ----
    quality_keys = [
        ('map_iou', 'Map IoU'),
        ('occ_precision', 'Obstacle Precision'),
        ('occ_recall', 'Obstacle Recall'),
        ('free_recall', 'Free Recall'),
    ]
    fig, axes = plt.subplots(1, len(quality_keys), figsize=(4 * len(quality_keys), 5))
    for ax, (qk, qt) in zip(axes, quality_keys):
        means = []
        stds = []
        for mode in DEPTH_MODES:
            vals = [r[qk] for r in all_data[mode]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        bars = ax.bar(range(len(DEPTH_MODES)), means, yerr=stds,
                      color=COLORS_DEPTH, alpha=0.8, capsize=3)
        ax.set_xticks(range(len(DEPTH_MODES)))
        ax.set_xticklabels([DEPTH_LABELS[m] for m in DEPTH_MODES], fontsize=8, rotation=20, ha='right')
        ax.set_ylabel(qt)
        ax.set_title(qt)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2, axis='y')
    fig.suptitle('Mapping Quality by Depth Mode', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_quality_bars.png')

    # ---- Fig 4: 覆盖率曲线 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, mode in enumerate(DEPTH_MODES):
        all_curves = []
        for res in all_data[mode]:
            vals = [x[1] * 100 for x in res['exploration_log']]
            all_curves.append(vals)
        max_len = max(len(c) for c in all_curves)
        interp = np.full((len(all_curves), max_len), np.nan)
        for ii, v in enumerate(all_curves):
            interp[ii, :len(v)] = v
            if len(v) < max_len:
                interp[ii, len(v):] = v[-1]
        mean_c = np.nanmean(interp, axis=0)
        ax.plot(range(max_len), mean_c, color=COLORS_DEPTH[idx], lw=2,
                label=DEPTH_LABELS[mode])
    ax.axhline(90, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Coverage by Depth Mode')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 800)
    ax.grid(True, alpha=0.2)
    save_figure(fig, fig_dir, 'fig_coverage_curves.png')

    # ---- Fig 5: 指标表 ----
    fig, ax = plt.subplots(figsize=(16, 5))
    col_labels = ['Depth Mode', 'Map IoU', 'Occ Prec', 'Occ Rec', 'Free Rec',
                  'Coverage(%)', 'Plan Succ(%)', 'Collisions', 'Replan']
    rows = []
    for mode in DEPTH_MODES:
        rl = all_data[mode]
        rows.append([
            DEPTH_LABELS[mode],
            f'{np.mean([r["map_iou"] for r in rl]):.3f}',
            f'{np.mean([r["occ_precision"] for r in rl]):.3f}',
            f'{np.mean([r["occ_recall"] for r in rl]):.3f}',
            f'{np.mean([r["free_recall"] for r in rl]):.3f}',
            f'{np.mean([r["final_coverage"]*100 for r in rl]):.1f}',
            f'{np.mean([r["plan_success_rate"]*100 for r in rl]):.1f}',
            f'{np.mean([r["collision_count"] for r in rl]):.1f}',
            f'{np.mean([r["replan_count"] for r in rl]):.0f}',
        ])
    render_metrics_table(ax, rows, col_labels, 'Depth Mode: Mean Metrics')
    save_figure(fig, fig_dir, 'fig_metrics_table.png')
    save_table_csv(rows, col_labels, raw_dir, 'table_metrics.csv')

    print(f'  Saved to {OUT_DIR}/')
    return all_data


if __name__ == '__main__':
    run_exp3()
