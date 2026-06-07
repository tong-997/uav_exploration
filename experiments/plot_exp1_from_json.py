"""
Generate Exp 1 figures from existing JSON summaries (no re-running simulations).
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.plotting.common import (setup_style, save_figure, render_grid_on_ax,
                                          render_metrics_table, save_table_csv, COLORS_DRONE)

N_DRONES_LIST = [1, 2, 3]
MAX_SPEED_LIST = [1.5, 3.0, 4.5]
MAX_STEPS = 800
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'exp1_param_variation')


def load_all_data():
    raw_dir = os.path.join(OUT_DIR, 'raw')
    all_data = {}
    for nd in N_DRONES_LIST:
        for spd in MAX_SPEED_LIST:
            key = (nd, spd)
            all_data[key] = []
            pattern = os.path.join(raw_dir, f'summary_n{nd}_v{spd}_s*.json')
            for fp in sorted(glob.glob(pattern)):
                with open(fp) as f:
                    all_data[key].append(json.load(f))
            print(f'  Loaded {len(all_data[key])} runs for n={nd}, v={spd}')
    return all_data


def plot_all(all_data):
    setup_style()
    fig_dir = os.path.join(OUT_DIR, 'figures')
    raw_dir = os.path.join(OUT_DIR, 'raw')
    os.makedirs(fig_dir, exist_ok=True)

    # ---- Fig 1: Coverage curves 3x3 ----
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
            if all_curves:
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
    fig.suptitle('Coverage vs. Step (UAV count x Speed)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_coverage_curves.png')
    print('  Saved fig_coverage_curves.png')

    # ---- Fig 2: Snapshots (seed=42, if available) ----
    for nd in N_DRONES_LIST:
        for spd in MAX_SPEED_LIST:
            res42 = [r for r in all_data[(nd, spd)] if r['config']['seed'] == 42]
            if not res42 or 'snapshots' not in res42[0] or not res42[0]['snapshots']:
                continue
            snaps = res42[0]['snapshots']
            snap_keys = sorted(snaps.keys(), key=lambda x: int(x))
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
                render_grid_on_ax(ax, np.array(snap['grid']), obstacles=snap['obstacles'],
                                 drones=snap['drones'],
                                 title=f'Step {step_k} | Cov={snap["ratio"]:.1%}')
            for idx in range(len(snap_keys), nrows * ncols):
                r, c = idx // ncols, idx % ncols
                axes_g[r][c].axis('off')
            fig.suptitle(f'Snapshots: n={nd}, v={spd}m/s, seed=42',
                         fontsize=13, fontweight='bold')
            fig.tight_layout()
            save_figure(fig, fig_dir, f'fig_snapshots_n{nd}_v{spd}.png')
            print(f'  Saved fig_snapshots_n{nd}_v{spd}.png')

    # ---- Fig 3: Path length bars ----
    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = 0
    x_ticks = []
    group_labels = []
    bar_w = 0.25
    for nd in N_DRONES_LIST:
        for spd in MAX_SPEED_LIST:
            results_list = all_data[(nd, spd)]
            if not results_list:
                continue
            mean_paths = np.zeros(nd)
            std_paths = np.zeros(nd)
            for i in range(nd):
                pl = [r['path_lengths'][i] for r in results_list if len(r['path_lengths']) > i]
                if pl:
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
    print('  Saved fig_path_length_bars.png')

    # ---- Fig 4: Metrics table ----
    fig, ax = plt.subplots(figsize=(16, 8))
    col_labels = ['n_drones', 'Speed(m/s)', 'Steps(mean)', 'Success(%)', 'Coverage(%)',
                  'Path(m)', 'Load CV', 'Obs Safe(%)', 'Inter Safe(%)']
    rows = []
    for nd in N_DRONES_LIST:
        for spd in MAX_SPEED_LIST:
            rl = all_data[(nd, spd)]
            if not rl:
                continue
            n_success = sum(1 for r in rl if r.get('success') == 'True')
            rows.append([
                str(nd), str(spd),
                f'{np.mean([r["steps_to_threshold"] for r in rl]):.0f}',
                f'{n_success / len(rl) * 100:.0f}',
                f'{np.mean([r["final_coverage"]*100 for r in rl]):.1f}',
                f'{np.mean([r["total_path_length"] for r in rl]):.0f}',
                f'{np.mean([r["load_balance_cv"] for r in rl]):.3f}',
                f'{np.mean([r["obstacle_safe_rate"]*100 for r in rl]):.1f}',
                f'{np.mean([r["inter_uav_safe_rate"]*100 for r in rl]):.1f}',
            ])
    render_metrics_table(ax, rows, col_labels, 'Parameter Variation: Mean Metrics')
    save_figure(fig, fig_dir, 'fig_metrics_table.png')
    save_table_csv(rows, col_labels, raw_dir, 'table_metrics.csv')
    print('  Saved fig_metrics_table.png and table_metrics.csv')


if __name__ == '__main__':
    print('Loading JSON summaries...')
    all_data = load_all_data()
    print('Generating figures...')
    plot_all(all_data)
    print('Done!')
