"""
Exp 4B 绘图: 跟踪曲线、RMSE 对比、融合散点、围捕快照、FSM 时序、指标表
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from experiments.plotting.common import (setup_style, save_figure,
                                          render_metrics_table, save_table_csv,
                                          COLORS_DRONE, COLORS_METHOD)

FSM_COLORS = {
    'SEARCH': '#90CAF9',
    'CONFIRM_TARGET': '#FFF176',
    'TRACK_TARGET': '#66BB6A',
    'OCCLUDED_TRACK': '#FF9800',
    'REACQUIRE_TARGET': '#CE93D8',
    'CONVERGE_TO_TARGET': '#4DD0E1',
    'MULTI_TRACK': '#26A69A',
    'ENCIRCLE_TARGET': '#EF5350',
    'LOST_TARGET': '#757575',
    'FINISH': '#1B5E20',
}


def plot_tracking_curves(log, fig_dir, prefix='tracking'):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    steps = range(len(log['true_pos']))
    true_x = [p[0] for p in log['true_pos']]
    true_y = [p[1] for p in log['true_pos']]
    est_x = [p[0] if not np.isnan(p[0]) else np.nan for p in log['est_pos']]
    est_y = [p[1] if not np.isnan(p[1]) else np.nan for p in log['est_pos']]

    axes[0].plot(steps, true_x, 'b-', lw=1.5, label='True X')
    axes[0].plot(steps, est_x, 'r--', lw=1, label='Est X')
    axes[0].plot(steps, true_y, 'g-', lw=1.5, label='True Y')
    axes[0].plot(steps, est_y, 'm--', lw=1, label='Est Y')
    axes[0].set_ylabel('Position (m)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)
    axes[0].set_title('Position Tracking', fontweight='bold')

    axes[1].plot(steps, log['error'], 'k-', lw=1)
    axes[1].set_ylabel('Error (m)')
    axes[1].grid(True, alpha=0.2)
    axes[1].set_title('Position Error', fontweight='bold')

    axes[2].plot(steps, log['confidence'], '#2196F3', lw=1, label='Confidence')
    ax2r = axes[2].twinx()
    detected_float = [1.0 if d else 0.0 for d in log['detected']]
    ax2r.fill_between(steps, detected_float, alpha=0.15, color='green',
                       label='Detected')
    axes[2].set_ylabel('Confidence')
    ax2r.set_ylabel('Detected')
    axes[2].set_xlabel('Step')
    axes[2].grid(True, alpha=0.2)
    axes[2].set_title('Detection Confidence', fontweight='bold')

    fig.tight_layout()
    save_figure(fig, fig_dir, f'fig_{prefix}_curves.png')


def plot_rmse_bars(summary, fig_dir):
    methods = ['detection_only', 'standard_kf', 'cm_kf']
    method_labels = ['Detection-only', 'Standard KF', 'CM-KF']
    motions = ['static', 'linear', 'random_turn']
    occlusions = ['no_occlusion', 'intermittent_occlusion',
                   'full_occlusion_interval']

    fig, axes = plt.subplots(1, len(occlusions), figsize=(5 * len(occlusions), 5),
                              sharey=True)
    bar_w = 0.25
    for oi, occ in enumerate(occlusions):
        ax = axes[oi]
        x = np.arange(len(motions))
        for mi, method in enumerate(methods):
            vals = []
            for motion in motions:
                key = (motion, occ, method)
                vals.append(summary.get(key, {}).get('rmse', np.nan))
            ax.bar(x + mi * bar_w, vals, bar_w, label=method_labels[mi],
                   color=COLORS_METHOD[mi])
        ax.set_xticks(x + bar_w)
        ax.set_xticklabels(motions, fontsize=8)
        ax.set_title(occ.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('RMSE (m)')
        ax.grid(True, alpha=0.2, axis='y')
        if oi == 0:
            ax.legend(fontsize=7)
    fig.suptitle('Tracking RMSE by Motion & Occlusion (YOLOv5-FFM)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_4b1_rmse_bars.png')


def plot_track_continuity(summary, fig_dir):
    methods = ['detection_only', 'standard_kf', 'cm_kf']
    method_labels = ['Detection-only', 'Standard KF', 'CM-KF']
    occlusions = ['no_occlusion', 'partial_occlusion',
                   'intermittent_occlusion', 'full_occlusion_interval']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(occlusions))
    bar_w = 0.25
    for mi, method in enumerate(methods):
        vals = []
        for occ in occlusions:
            key = ('linear', occ, method)
            vals.append(summary.get(key, {}).get('track_rate', 0) * 100)
        ax.bar(x + mi * bar_w, vals, bar_w, label=method_labels[mi],
               color=COLORS_METHOD[mi])
    ax.set_xticks(x + bar_w)
    ax.set_xticklabels([o.replace('_', '\n') for o in occlusions], fontsize=8)
    ax.set_ylabel('Tracking Continuity (%)')
    ax.set_title('Tracking Continuity: linear motion, YOLOv5-FFM',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_4b1_continuity.png')


def plot_fusion_bars(fusion_summary, fig_dir):
    methods = ['single_best', 'naive_average', 'info_filter',
               'covariance_intersection']
    labels = ['Single Best', 'Naive Avg', 'Info Filter', 'Cov Intersect']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # RMSE
    pre_vals = [fusion_summary.get(m, {}).get('pre_fusion_rmse', 0) for m in methods]
    post_vals = [fusion_summary.get(m, {}).get('post_fusion_rmse', 0) for m in methods]
    x = np.arange(len(methods))
    axes[0].bar(x - 0.15, pre_vals, 0.3, label='Pre-fusion', color='#90CAF9')
    axes[0].bar(x + 0.15, post_vals, 0.3, label='Post-fusion', color='#1565C0')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel('RMSE (m)')
    axes[0].set_title('Fusion RMSE', fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2, axis='y')

    # Cov trace
    trace_vals = [fusion_summary.get(m, {}).get('cov_trace', 0) for m in methods]
    axes[1].bar(x, trace_vals, 0.5, color=COLORS_METHOD[:4])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel('Covariance Trace')
    axes[1].set_title('Fused Covariance Trace', fontweight='bold')
    axes[1].grid(True, alpha=0.2, axis='y')

    fig.suptitle('Multi-UAV Fusion Comparison', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, fig_dir, 'fig_4b2_fusion_bars.png')


def plot_encirclement_snapshot(snapshot, fig_dir, prefix='enc'):
    fig, ax = plt.subplots(figsize=(8, 8))
    from config import WORLD_SIZE

    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.set_aspect('equal')

    if 'obstacles' in snapshot:
        for o in snapshot['obstacles']:
            ox = o.x if hasattr(o, 'x') else o[0]
            oy = o.y if hasattr(o, 'y') else o[1]
            r = o.r if hasattr(o, 'r') else o[2]
            ax.add_patch(plt.Circle((ox, oy), r, fill=True,
                                     fc='#EF5350', alpha=0.25, ec='#C62828', lw=0.8))

    tp = snapshot['target_pos']
    ax.plot(tp[0], tp[1], 'P', color='red', ms=15, mec='darkred', mew=2, zorder=20)

    if 'target_est' in snapshot and snapshot['target_est'] is not None:
        te = snapshot['target_est']
        ax.plot(te[0], te[1], 'X', color='lime', ms=12, mec='darkgreen',
                mew=2, zorder=19)

    if 'r_enc' in snapshot:
        ax.add_patch(plt.Circle(tp, snapshot['r_enc'], fill=False,
                                 color='gray', ls='--', lw=1))

    if 'slots' in snapshot:
        for j, s in enumerate(snapshot['slots']):
            if s is not None:
                ax.plot(s[0], s[1], 's', color=COLORS_DRONE[j], ms=10,
                        mec='black', mew=1, zorder=15, alpha=0.5)

    for i, pos in enumerate(snapshot['uav_positions']):
        ax.plot(pos[0], pos[1], 'o', color=COLORS_DRONE[i], ms=12,
                mec='white', mew=2, zorder=16)
        ax.annotate(f'UAV-{i}', (pos[0] + 1, pos[1] + 1), fontsize=8)

    ax.set_title(f'Encirclement (step={snapshot.get("step", "?")})',
                 fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    fig.tight_layout()
    save_figure(fig, fig_dir, f'fig_{prefix}_snapshot.png')


def plot_fsm_timeline(fsm_histories, max_steps, fig_dir, prefix='fsm'):
    n_uavs = len(fsm_histories)
    fig, ax = plt.subplots(figsize=(14, 3 + n_uavs * 0.8))

    for i, history in enumerate(fsm_histories):
        segments = []
        prev_step = 0
        prev_state = 'SEARCH'
        for step, from_s, to_s in history:
            segments.append((prev_step, step - prev_step, prev_state))
            prev_step = step
            prev_state = to_s
        segments.append((prev_step, max_steps - prev_step, prev_state))

        for start, width, state in segments:
            color = FSM_COLORS.get(state, '#BDBDBD')
            ax.barh(i, width, left=start, height=0.6, color=color, alpha=0.8)

    ax.set_yticks(range(n_uavs))
    ax.set_yticklabels([f'UAV-{i}' for i in range(n_uavs)])
    ax.set_xlabel('Step')
    ax.set_title('FSM State Timeline', fontweight='bold')
    ax.set_xlim(0, max_steps)

    patches = [mpatches.Patch(color=c, label=s, alpha=0.8)
               for s, c in FSM_COLORS.items() if s != 'FINISH']
    ax.legend(handles=patches, fontsize=6, loc='upper right', ncol=3)
    ax.grid(True, alpha=0.2, axis='x')
    fig.tight_layout()
    save_figure(fig, fig_dir, f'fig_{prefix}_timeline.png')


def plot_4b1_metrics_table(summary, fig_dir, raw_dir):
    col_labels = ['Motion', 'Occlusion', 'Method', 'RMSE(m)',
                  'Track Rate(%)', 'Reassoc Rate(%)']
    rows = []
    for (motion, occ, method), v in sorted(summary.items()):
        rows.append([
            motion, occ.replace('_', ' '),
            method.replace('_', ' ').title(),
            f'{v.get("rmse", 0):.3f}',
            f'{v.get("track_rate", 0) * 100:.1f}',
            f'{v.get("reassoc_rate", 0) * 100:.1f}',
        ])
    fig, ax = plt.subplots(figsize=(18, max(6, len(rows) * 0.3 + 2)))
    render_metrics_table(ax, rows, col_labels,
                         'Exp 4B-1: Single-UAV Tracking Metrics')
    save_figure(fig, fig_dir, 'fig_4b1_metrics_table.png')
    save_table_csv(rows, col_labels, raw_dir, 'table_4b1_metrics.csv')


def plot_4b2_metrics_table(fusion_summary, fig_dir, raw_dir):
    col_labels = ['Fusion Method', 'Pre RMSE(m)', 'Post RMSE(m)',
                  'Cov Trace', 'Max Eigenvalue']
    rows = []
    for method in ['single_best', 'naive_average', 'info_filter',
                   'covariance_intersection']:
        v = fusion_summary.get(method, {})
        rows.append([
            method.replace('_', ' ').title(),
            f'{v.get("pre_fusion_rmse", 0):.3f}',
            f'{v.get("post_fusion_rmse", 0):.3f}',
            f'{v.get("cov_trace", 0):.4f}',
            f'{v.get("max_eigenvalue", 0):.4f}',
        ])
    fig, ax = plt.subplots(figsize=(14, 4))
    render_metrics_table(ax, rows, col_labels,
                         'Exp 4B-2: Multi-UAV Fusion Metrics')
    save_figure(fig, fig_dir, 'fig_4b2_metrics_table.png')
    save_table_csv(rows, col_labels, raw_dir, 'table_4b2_metrics.csv')


def plot_4b3_metrics_table(enc_summary, fig_dir, raw_dir):
    col_labels = ['Motion', 'Formation Time', 'Phase Uniformity',
                  'Min Inter(m)', 'Track RMSE(m)', 'Encircle Success']
    rows = []
    for motion, v in sorted(enc_summary.items()):
        rows.append([
            motion,
            f'{v.get("formation_time", -1):.0f}',
            f'{v.get("phase_uniformity", 0):.3f}',
            f'{v.get("min_inter_distance", 0):.1f}',
            f'{v.get("track_rmse", 0):.3f}',
            'Yes' if v.get('encirclement_success', False) else 'No',
        ])
    fig, ax = plt.subplots(figsize=(14, 4))
    render_metrics_table(ax, rows, col_labels,
                         'Exp 4B-3: Encirclement Metrics')
    save_figure(fig, fig_dir, 'fig_4b3_metrics_table.png')
    save_table_csv(rows, col_labels, raw_dir, 'table_4b3_metrics.csv')
