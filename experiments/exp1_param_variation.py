"""
实验1: UAV数量与速度参数变化
- n_drones: 1, 2, 3
- max_speed: 1.5, 3.0, 4.5
- 10 seeds, max_steps=800, coverage_threshold=90%
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

N_DRONES_LIST = [1, 2, 3]
MAX_SPEED_LIST = [1.5, 3.0, 4.5]
SEEDS = [42, 77, 123, 200, 256, 314, 512, 666, 777, 999]
MAX_STEPS = 800
SNAP_STEPS = [0, 50, 100, 200, 300, 301]
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'exp1_param_variation')


def run_exp1():
    setup_style()
    raw_dir = os.path.join(OUT_DIR, 'raw')
    fig_dir = os.path.join(OUT_DIR, 'figures')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print('=== Experiment 1: UAV Count & Speed Parameter Variation ===')
    all_data = {}

    for nd in N_DRONES_LIST:
        for spd in MAX_SPEED_LIST:
            key = (nd, spd)
            all_data[key] = []
            for seed in SEEDS:
                snap = SNAP_STEPS if seed == 42 else None
                config = RunConfig(n_drones=nd, max_speed=spd, seed=seed,
                                   max_steps=MAX_STEPS, coverage_threshold=0.90)
                results, system = run_parametric(config, snapshot_steps=snap)
                save_summary(results, raw_dir, f'n{nd}_v{spd}_s{seed}')
                all_data[key].append(results)
                print(f'  n={nd} v={spd} s={seed} → cov={results["final_coverage"]:.1%} '
                      f'steps={results["final_step"]}')

    # ---- Fig 1: 覆盖率曲线 3x3 子图 ----
    fig, axes = plt.subplots(len(N_DRONES_LIST), len(MAX_SPEED_LIST),
                             figsize=(5 * len(MAX_SPEED_LIST), 4 * len(N_DRONES_LIST)),
                             sharex=True, sharey=True)
    for ri, nd in enumerate(N_DRONES_LIST):
        for ci, spd in enumerate(MAX_SPEED_LIST):
            ax = axes[ri][ci] if len(N_DRONES_LIST) > 1 else axes[ci]
            results_list = all_data[(nd, spd)]
            all_curves = []
            for res in results_list:
                steps = [x[0] for x in res['exploration_log']]
                vals = [x[1] * 100 for x in res['exploration_log']]
                ax.plot(steps, vals, lw=0.3, alpha=0.3, color='#2196F3')
                all_curves.append(vals)
            max_len = max(len(c) for c in all_curves)
            interp = np.full((len(all_curves), max_len), np.nan)
            for idx, v in enumerate(all_curves):
                interp[idx, :len(v)] = v
                if len(v) < max_len:
                    interp[idx, len(v):] = v[-1]
            mean_c = np.nanmean(interp, axis=0)
            std_c = np.nanstd(interp, axis=0)
            ax.plot(range(max_len), mean_c, color='#1565C0', lw=2)
            ax.fill_between(range(max_len), mean_c - std_c, mean_c + std_c,
                            alpha=0.15, color='#2196F3')
            ax.axhline(90, color='gray', ls='--', lw=0.8, alpha=0.5)
            ax.set_title(f'n={nd}, v={spd}m/s', fontsize=10)
            ax.set_ylim(0, 100)
            ax.set_xlim(0, MAX_STEPS)
            ax.grid(True, alpha=0.2)
            if ri == len(N_DRONES_LIST) - 1:
                ax.set_xlabel('Step')
            if ci == 0:
                ax.set_ylabel('Coverage (%)')
    fig.suptitle('Coverage vs. Step (UAV count × Speed)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_coverage_curves.png')

    # ---- Fig 2: 快照拼图 (seed=42) ----
    for nd in N_DRONES_LIST:
        for spd in MAX_SPEED_LIST:
            res42 = [r for r in all_data[(nd, spd)] if r['config']['seed'] == 42]
            if not res42 or 'snapshots' not in res42[0] or not res42[0]['snapshots']:
                continue
            snaps = res42[0]['snapshots']
            snap_keys = sorted(snaps.keys())
            ncols = min(len(snap_keys), 3)
            nrows = (len(snap_keys) + ncols - 1) // ncols
            fig, axes_g = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows))
            if nrows == 1 and ncols == 1:
                axes_g = np.array([[axes_g]])
            elif nrows == 1:
                axes_g = axes_g[np.newaxis, :]
            elif ncols == 1:
                axes_g = axes_g[:, np.newaxis]
            for idx, step_k in enumerate(snap_keys):
                r, c = idx // ncols, idx % ncols
                ax = axes_g[r][c]
                snap = snaps[step_k]
                render_grid_on_ax(ax, snap['grid'], obstacles=snap['obstacles'],
                                 drones=snap['drones'],
                                 title=f'Step {step_k} | Cov={snap["ratio"]:.1%}')
            for idx in range(len(snap_keys), nrows * ncols):
                r, c = idx // ncols, idx % ncols
                axes_g[r][c].axis('off')
            fig.suptitle(f'Snapshots: n={nd}, v={spd}m/s, seed=42',
                         fontsize=13, fontweight='bold')
            fig.tight_layout()
            save_figure(fig, fig_dir, f'fig_snapshots_n{nd}_v{spd}.png')

    # ---- Fig 3: 路径长度柱状图 ----
    fig, ax = plt.subplots(figsize=(12, 5))
    group_labels = []
    x_pos = 0
    x_ticks = []
    bar_w = 0.25
    for nd in N_DRONES_LIST:
        for spd in MAX_SPEED_LIST:
            results_list = all_data[(nd, spd)]
            mean_paths = np.zeros(nd)
            std_paths = np.zeros(nd)
            for i in range(nd):
                pl = [r['path_lengths'][i] for r in results_list if len(r['path_lengths']) > i]
                mean_paths[i] = np.mean(pl)
                std_paths[i] = np.std(pl)
            for i in range(nd):
                ax.bar(x_pos + i * bar_w, mean_paths[i], bar_w * 0.9,
                       yerr=std_paths[i], color=COLORS_DRONE[i % len(COLORS_DRONE)],
                       alpha=0.8, capsize=3, label=f'UAV-{i}' if x_pos == 0 else '')
            x_ticks.append(x_pos + (nd - 1) * bar_w / 2)
            group_labels.append(f'n={nd}\nv={spd}')
            x_pos += max(nd, 1) * bar_w + 0.5
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_ylabel('Path Length (m)')
    ax.set_title('Per-UAV Path Length')
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS_DRONE[i]) for i in range(3)]
    ax.legend(handles, [f'UAV-{i}' for i in range(3)], fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')
    save_figure(fig, fig_dir, 'fig_path_length_bars.png')

    # ---- Fig 4: 指标均值表 ----
    fig, ax = plt.subplots(figsize=(16, 8))
    col_labels = ['n_drones', 'Speed(m/s)', 'Steps(mean)', 'Coverage(%)',
                  'Load CV', 'Obs Safe(%)', 'Inter Safe(%)']
    rows = []
    for nd in N_DRONES_LIST:
        for spd in MAX_SPEED_LIST:
            rl = all_data[(nd, spd)]
            rows.append([
                str(nd), str(spd),
                f'{np.mean([r["steps_to_threshold"] for r in rl]):.0f}',
                f'{np.mean([r["final_coverage"]*100 for r in rl]):.1f}',
                f'{np.mean([r["load_balance_cv"] for r in rl]):.3f}',
                f'{np.mean([r["obstacle_safe_rate"]*100 for r in rl]):.1f}',
                f'{np.mean([r["inter_uav_safe_rate"]*100 for r in rl]):.1f}',
            ])
    render_metrics_table(ax, rows, col_labels, 'Parameter Variation: Mean Metrics')
    save_figure(fig, fig_dir, 'fig_metrics_table.png')
    save_table_csv(rows, col_labels, raw_dir, 'table_metrics.csv')

    print(f'  Saved to {OUT_DIR}/')
    return all_data


if __name__ == '__main__':
    run_exp1()
