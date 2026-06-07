"""
共享绘图工具
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import GRID_N, GRID_RES, WORLD_SIZE, FREE, OCCUPIED, UNKNOWN, SENSOR_FOV, SENSOR_RANGE

COLORS_DRONE = ['#29b6f6', '#66bb6a', '#ffa726', '#ef5350', '#ab47bc']
COLORS_METHOD = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#795548']
COLORS_DEPTH = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']


def setup_style():
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def render_grid_on_ax(ax, grid, obstacles=None, drones=None, title=None, show_fov=False):
    dp = np.ones((GRID_N, GRID_N, 3))
    dp[grid == FREE] = [0.95, 0.95, 0.95]
    dp[grid == OCCUPIED] = [0.20, 0.20, 0.20]
    dp[grid == UNKNOWN] = [0.82, 0.88, 0.95]
    ax.imshow(dp, origin='lower', extent=[0, WORLD_SIZE, 0, WORLD_SIZE])

    if obstacles:
        for ox, oy, r in obstacles:
            ax.add_patch(plt.Circle((ox, oy), r, fill=True,
                                    fc='#EF5350', alpha=0.25, ec='#C62828', lw=0.8))

    if drones:
        n = len(drones)
        for i, (pos, heading, path, target, wps) in enumerate(drones):
            c = COLORS_DRONE[i % len(COLORS_DRONE)]
            ax.plot(pos[0], pos[1], 'o', color=c, ms=8, mec='white', mew=1.2, zorder=10)
            hdx = np.cos(heading) * 2.5
            hdy = np.sin(heading) * 2.5
            ax.annotate('', xy=(pos[0]+hdx, pos[1]+hdy), xytext=(pos[0], pos[1]),
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.5))
            if show_fov:
                half = SENSOR_FOV / 2
                angles = np.linspace(heading - half, heading + half, 20)
                fx = [pos[0]] + [pos[0] + np.cos(a)*SENSOR_RANGE for a in angles] + [pos[0]]
                fy = [pos[1]] + [pos[1] + np.sin(a)*SENSOR_RANGE for a in angles] + [pos[1]]
                ax.fill(fx, fy, color=c, alpha=0.05)
            if path and len(path) > 1:
                ax.plot([p[0] for p in path], [p[1] for p in path],
                        '-', color=c, lw=1.0, alpha=0.6)
            if target is not None:
                ax.plot(target[0], target[1], '*', color=c, ms=10, mec='k', mew=0.4, zorder=8)
            if len(wps) > 1:
                trail = wps[-200:]
                ax.plot([w[0] for w in trail], [w[1] for w in trail],
                        '-', color=c, lw=0.4, alpha=0.2)

    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')


def save_figure(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def render_metrics_table(ax, rows, col_labels, title=None):
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#BDBDBD')
        if r == 0:
            cell.set_facecolor('#E3F2FD')
            cell.set_text_props(weight='bold')
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=15)


def save_table_csv(rows, col_labels, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    import csv
    path = os.path.join(output_dir, filename)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(col_labels)
        w.writerows(rows)
    return path
