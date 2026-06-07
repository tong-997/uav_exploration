"""
实验4: 遮挡目标检测与协同响应
- 遮挡等级: none, light, medium, heavy
- 检测模式: yolov5_ffm, yolov5s, oracle, noisy_oracle
- 3 UAVs, seed=42
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.run_config import RunConfig, run_parametric, save_summary
from experiments.plotting.common import (setup_style, save_figure, render_grid_on_ax,
                                          render_metrics_table, save_table_csv,
                                          COLORS_DRONE, COLORS_METHOD)
from config import SAFE_RADIUS, SENSOR_RANGE, SENSOR_FOV, WORLD_SIZE

OCCLUSION_LEVELS = ['none', 'light', 'medium', 'heavy']
DETECTION_MODELS = ['yolov5_ffm', 'yolov5s', 'oracle', 'noisy_oracle']
MODEL_LABELS = {
    'yolov5_ffm': 'YOLOv5-FFM', 'yolov5s': 'YOLOv5s',
    'oracle': 'Oracle', 'noisy_oracle': 'Noisy-Oracle'
}
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'exp4_detection')


def run_exp4():
    setup_style()
    raw_dir = os.path.join(OUT_DIR, 'raw')
    fig_dir = os.path.join(OUT_DIR, 'figures')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print('=== Experiment 4: Occlusion Target Detection ===')
    all_data = {}

    for occ in OCCLUSION_LEVELS:
        for model in DETECTION_MODELS:
            key = (occ, model)
            config = RunConfig(
                n_drones=3, max_speed=3.0, seed=42,
                max_steps=800, coverage_threshold=0.90,
                enable_targets=True,
                occlusion_level=occ,
                detection_model=model,
                confirmation_frames=3,
            )
            results, system = run_parametric(config, snapshot_steps=[0, 100, 200, 300], verbose=False)

            target_metrics = _extract_target_metrics(system)
            results.update(target_metrics)

            save_summary(results, raw_dir, f'{occ}_{model}_s42')
            all_data[key] = {'results': results, 'system': system, 'targets': system.targets}
            det_str = 'Y' if target_metrics.get('target_detected', False) else 'N'
            print(f'  occ={occ:6s} model={model:12s} → det={det_str} '
                  f'first={target_metrics.get("first_detection_step", -1)} '
                  f'confirm={target_metrics.get("confirmation_step", -1)}')

    # ---- Fig 1: 不同遮挡等级检测示意 (oracle, seed=42) ----
    fig, axes = plt.subplots(1, len(OCCLUSION_LEVELS), figsize=(6 * len(OCCLUSION_LEVELS), 6))
    for idx, occ in enumerate(OCCLUSION_LEVELS):
        ax = axes[idx]
        data = all_data[(occ, 'oracle')]
        sys_ref = data['system']
        snaps = data['results'].get('snapshots', {})
        last_snap_key = max(snaps.keys()) if snaps else None
        if last_snap_key and snaps[last_snap_key]:
            snap = snaps[last_snap_key]
            render_grid_on_ax(ax, snap['grid'], obstacles=snap['obstacles'],
                             drones=snap['drones'], title=f'Occlusion: {occ}')
        else:
            render_grid_on_ax(ax, sys_ref.global_grid.grid,
                             obstacles=[(o.x, o.y, o.r) for o in sys_ref.env.obstacles],
                             title=f'Occlusion: {occ}')
        for t in data['targets']:
            ax.plot(t.pos[0], t.pos[1], 'P', color='red', ms=15,
                    mec='darkred', mew=2, zorder=20)
            if t.estimated_pos is not None:
                ax.plot(t.estimated_pos[0], t.estimated_pos[1], 'X',
                        color='lime', ms=12, mec='darkgreen', mew=2, zorder=19)
    fig.suptitle('Detection Under Different Occlusion Levels (Oracle)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_occlusion_illustration.png')

    # ---- Fig 2: 目标定位散点 ----
    fig, axes = plt.subplots(1, len(DETECTION_MODELS), figsize=(5 * len(DETECTION_MODELS), 5))
    for idx, model in enumerate(DETECTION_MODELS):
        ax = axes[idx]
        for occ_idx, occ in enumerate(OCCLUSION_LEVELS):
            data = all_data[(occ, model)]
            for t in data['targets']:
                if t.detection_history:
                    errs = [np.linalg.norm(h['error']) for h in t.detection_history]
                    dists = [h['distance'] for h in t.detection_history]
                    ax.scatter(dists, errs, s=10, alpha=0.5,
                               color=COLORS_METHOD[occ_idx], label=occ if idx == 0 else '')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Localization Error (m)')
        ax.set_title(MODEL_LABELS[model])
        ax.grid(True, alpha=0.2)
    axes[0].legend(fontsize=8, title='Occlusion')
    fig.suptitle('Localization Error vs. Distance', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_localization_scatter.png')

    # ---- Fig 3: 置信度与定位误差随步数变化 ----
    fig, axes = plt.subplots(2, len(DETECTION_MODELS), figsize=(5 * len(DETECTION_MODELS), 8))
    for idx, model in enumerate(DETECTION_MODELS):
        data = all_data[('medium', model)]
        ax_conf = axes[0][idx]
        ax_err = axes[1][idx]
        for t in data['targets']:
            if t.detection_history:
                steps = [h['step'] for h in t.detection_history]
                confs = [h['confidence'] for h in t.detection_history]
                errs = [np.linalg.norm(h['error']) for h in t.detection_history]
                ax_conf.plot(steps, confs, 'o-', ms=3, lw=0.8, color='#2196F3')
                ax_err.plot(steps, errs, 'o-', ms=3, lw=0.8, color='#FF9800')
        ax_conf.set_title(f'{MODEL_LABELS[model]}')
        ax_conf.set_ylabel('Confidence')
        ax_conf.set_ylim(0, 1.1)
        ax_conf.grid(True, alpha=0.2)
        ax_err.set_xlabel('Step')
        ax_err.set_ylabel('Loc. Error (m)')
        ax_err.grid(True, alpha=0.2)
    fig.suptitle('Confidence & Error Over Time (medium occlusion)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_confidence_error_curves.png')

    # ---- Fig 4: 任务状态时序图 ----
    fig, ax = plt.subplots(figsize=(12, 4))
    data = all_data[('medium', 'yolov5_ffm')]
    sys_ref = data['system']
    n_steps = sys_ref.step
    for i in range(sys_ref.n_drones):
        y = i
        ax.barh(y, n_steps, left=0, height=0.6, color=COLORS_DRONE[i], alpha=0.3, label=f'UAV-{i} explore')
        for t in data['targets']:
            if t.first_detected_step >= 0:
                ax.barh(y, 3, left=t.first_detected_step, height=0.6, color='#FF9800', alpha=0.8)
            if t.confirmed_at_step >= 0:
                ax.barh(y, n_steps - t.confirmed_at_step, left=t.confirmed_at_step,
                        height=0.6, color='#F44336', alpha=0.5)
    ax.set_yticks(range(sys_ref.n_drones))
    ax.set_yticklabels([f'UAV-{i}' for i in range(sys_ref.n_drones)])
    ax.set_xlabel('Step')
    ax.set_title('Task State Timeline (medium, YOLOv5-FFM)')
    patches = [mpatches.Patch(color=COLORS_DRONE[0], alpha=0.3, label='Exploring'),
               mpatches.Patch(color='#FF9800', alpha=0.8, label='First Detection'),
               mpatches.Patch(color='#F44336', alpha=0.5, label='Cooperative Response')]
    ax.legend(handles=patches, fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2, axis='x')
    save_figure(fig, fig_dir, 'fig_task_timeline.png')

    # ---- Fig 5: 协同响应过程图 ----
    fig, ax = plt.subplots(figsize=(8, 8))
    data = all_data[('medium', 'yolov5_ffm')]
    sys_ref = data['system']
    render_grid_on_ax(ax, sys_ref.global_grid.grid,
                     obstacles=[(o.x, o.y, o.r) for o in sys_ref.env.obstacles],
                     title='Cooperative Response (medium, YOLOv5-FFM)', show_fov=True)
    for t in data['targets']:
        ax.plot(t.pos[0], t.pos[1], 'P', color='red', ms=18, mec='darkred', mew=2, zorder=20)
        ax.annotate('Target', xy=(t.pos[0], t.pos[1]), fontsize=9,
                    fontweight='bold', color='darkred',
                    xytext=(t.pos[0] + 3, t.pos[1] + 3))
        if t.estimated_pos is not None:
            ax.plot(t.estimated_pos[0], t.estimated_pos[1], 'X',
                    color='lime', ms=14, mec='darkgreen', mew=2, zorder=19)
    for i in range(sys_ref.n_drones):
        d = sys_ref.env.drones[i]
        ax.plot(d.pos[0], d.pos[1], 'o', color=COLORS_DRONE[i], ms=12,
                mec='white', mew=2, zorder=15)
        wps = d.waypoints[-100:]
        if len(wps) > 1:
            ax.plot([w[0] for w in wps], [w[1] for w in wps],
                    '-', color=COLORS_DRONE[i], lw=1.5, alpha=0.5)
    save_figure(fig, fig_dir, 'fig_cooperative_response.png')

    # ---- Fig 6: 指标表 ----
    fig, ax = plt.subplots(figsize=(18, 8))
    col_labels = ['Occlusion', 'Model', 'Detected', 'First Step', 'Confirm Step',
                  'Avg Conf', 'Loc Error(m)', 'Response Time']
    rows = []
    for occ in OCCLUSION_LEVELS:
        for model in DETECTION_MODELS:
            r = all_data[(occ, model)]['results']
            rows.append([
                occ, MODEL_LABELS[model],
                'Yes' if r.get('target_detected', False) else 'No',
                str(r.get('first_detection_step', '-')),
                str(r.get('confirmation_step', '-')),
                f'{r.get("avg_confidence", 0):.3f}',
                f'{r.get("avg_localization_error", 0):.3f}',
                str(r.get('cooperative_response_time', '-')),
            ])
    render_metrics_table(ax, rows, col_labels, 'Target Detection: Metrics')
    save_figure(fig, fig_dir, 'fig_metrics_table.png')
    save_table_csv(rows, col_labels, raw_dir, 'table_metrics.csv')

    print(f'  Saved to {OUT_DIR}/')
    return all_data


def _extract_target_metrics(system):
    metrics = {
        'target_detected': False,
        'first_detection_step': -1,
        'confirmation_step': -1,
        'avg_confidence': 0.0,
        'avg_localization_error': 0.0,
        'cooperative_response_time': -1,
    }
    if not system.targets:
        return metrics

    t = system.targets[0]
    if t.detection_history:
        metrics['target_detected'] = True
        metrics['first_detection_step'] = t.first_detected_step
        confs = [h['confidence'] for h in t.detection_history]
        errs = [np.linalg.norm(h['error']) for h in t.detection_history]
        metrics['avg_confidence'] = float(np.mean(confs))
        metrics['avg_localization_error'] = float(np.mean(errs))

    if t.is_confirmed:
        metrics['confirmation_step'] = t.confirmed_at_step
        metrics['cooperative_response_time'] = system.step - t.confirmed_at_step

    return metrics


if __name__ == '__main__':
    run_exp4()
