"""生成论文图表"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from run_exploration import ExplorationSystem
from config import *

FIG = os.path.join(os.path.dirname(__file__), 'output', 'figures')
os.makedirs(FIG, exist_ok=True)
C1, C2, C3 = '#2196F3', '#4CAF50', '#FF9800'
DC = [C1, C2, C3]
MAX_S = 800

def run(nd, seed):
    s = ExplorationSystem(seed=seed, n_drones=nd)
    while not s.is_done() and s.step < MAX_S:
        s.run_step()
    return s

# ===== Baseline =====
print('Baseline 3-drone...')
b = run(3, 42)
n = b.n_drones
print(f'  {b.step} steps, {b.global_grid.explored_ratio:.1%}')

# ===== Fig1: Scalability =====
print('Fig1...')
fig, ax = plt.subplots(figsize=(7, 4.5))
for nd, c, lb in [(1, '#9E9E9E', '1 UAV'), (2, '#7E57C2', '2 UAVs'), (3, C1, '3 UAVs')]:
    print(f'  n={nd}...', end='')
    s = run(nd, 42)
    ax.plot([x[0] for x in s.exploration_log],
            [x[1]*100 for x in s.exploration_log],
            color=c, linewidth=2, label=lb)
    print(f' {s.step}steps {s.global_grid.explored_ratio:.0%}')
ax.axhline(90, color='gray', ls='--', lw=0.8, alpha=0.5)
ax.set(xlabel='Step', ylabel='Coverage (%)', xlim=(0, MAX_S), ylim=(0, 100))
ax.set_title('Exploration Efficiency: Single vs. Multi-UAV')
ax.legend(loc='lower right'); ax.grid(True, alpha=0.2)
fig.tight_layout(); fig.savefig(f'{FIG}/fig1_scalability.png', dpi=300); plt.close()
print('  saved')

# ===== Fig2: Safety =====
print('Fig2...')
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
for i in range(n):
    ax1.plot(b.min_obs_dist_log[i], color=DC[i], lw=0.5, alpha=0.7, label=f'UAV-{i}')
ax1.axhline(SAFE_RADIUS, color='red', ls='--', lw=1.2, label=f'Safety ({SAFE_RADIUS}m)')
ax1.set(xlabel='Step', ylabel='Min Dist to Obstacle (m)', ylim=(0, None))
ax1.set_title('(a) Obstacle Clearance'); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.2)

ax2 = fig.add_subplot(gs[0, 1])
ad = np.concatenate(b.min_obs_dist_log)
ax2.hist(ad, bins=50, color=C1, alpha=0.7, edgecolor='white', lw=0.5)
ax2.axvline(SAFE_RADIUS, color='red', ls='--', lw=1.5)
ps = np.mean(ad > SAFE_RADIUS) * 100
ax2.set(xlabel='Distance (m)', ylabel='Frequency')
ax2.set_title(f'(b) Clearance Distribution (safe: {ps:.1f}%)'); ax2.grid(True, alpha=0.2)

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(b.min_drone_dist_log, color='#7E57C2', lw=0.8)
ax3.axhline(SAFE_RADIUS, color='red', ls='--', lw=1.2, label=f'Safety ({SAFE_RADIUS}m)')
pi = np.mean(np.array(b.min_drone_dist_log) > SAFE_RADIUS) * 100
ax3.set(xlabel='Step', ylabel='Min Inter-UAV Dist (m)', ylim=(0, None))
ax3.set_title(f'(c) Inter-UAV Separation (safe: {pi:.1f}%)'); ax3.legend(fontsize=9); ax3.grid(True, alpha=0.2)

ax4 = fig.add_subplot(gs[1, 1]); ax4.axis('off')
mo = min(min(d) for d in b.min_obs_dist_log)
ao = np.mean(ad)
mi = min(b.min_drone_dist_log); ai = np.mean(b.min_drone_dist_log)
td = sum(b.recorder.get_total_distance(i) for i in range(n))
cv = b.global_grid.explored_ratio * 100
rows = [
    ['Coverage (%)', f'{cv:.1f}'], ['Steps', f'{b.step}'],
    ['Path Length (m)', f'{td:.1f}'], ['Collisions', f'{b.collision_count}'],
    ['Obs Avoid Rate (%)', f'{ps:.1f}'], ['Min Obs Dist (m)', f'{mo:.2f}'],
    ['Avg Obs Dist (m)', f'{ao:.2f}'], ['Min Inter-UAV (m)', f'{mi:.2f}'],
    ['Avg Inter-UAV (m)', f'{ai:.2f}'], ['Inter-UAV Safe (%)', f'{pi:.1f}'],
    ['Replans', f'{b.replan_count}'], ['Avoidance Events', f'{sum(b.avoidance_events)}'],
]
t = ax4.table(cellText=rows, colLabels=['Metric', 'Value'], loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1.0, 1.45)
for (r, c), cell in t.get_celld().items():
    if r == 0: cell.set_facecolor('#E3F2FD'); cell.set_text_props(weight='bold')
    cell.set_edgecolor('#BDBDBD')
ax4.set_title('(d) Summary', fontsize=11, pad=15)
fig.savefig(f'{FIG}/fig2_safety.png', dpi=300); plt.close()
print('  saved')

# ===== Fig3: Map + Bar =====
print('Fig3...')
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
ax = axes[0]; g = b.global_grid.grid
dp = np.ones((GRID_N, GRID_N, 3))
dp[g == FREE] = [.96,.96,.96]; dp[g == OCCUPIED] = [.25,.25,.25]; dp[g == UNKNOWN] = [.82,.89,.96]
ax.imshow(dp, origin='lower', extent=[0, WORLD_SIZE, 0, WORLD_SIZE])
for o in b.env.obstacles:
    ax.add_patch(plt.Circle((o.x, o.y), o.r, fill=True, fc='#EF5350', alpha=0.3, ec='#C62828', lw=1.2))
for i in range(n):
    w = b.env.drones[i].waypoints
    if len(w) > 1:
        wx, wy = zip(*w)
        ax.plot(wx, wy, '-', color=DC[i], lw=1, alpha=0.6, label=f'UAV-{i}')
        ax.plot(wx[0], wy[0], 's', color=DC[i], ms=9, mec='k', mew=0.8, zorder=5)
        ax.plot(wx[-1], wy[-1], '*', color=DC[i], ms=12, mec='k', mew=0.5, zorder=5)
ax.set(xlabel='X (m)', ylabel='Y (m)', xlim=(0, WORLD_SIZE), ylim=(0, WORLD_SIZE), aspect='equal')
ax.set_title('(a) Exploration Map & Trajectories'); ax.legend(fontsize=9)
a2 = axes[1]; bw = 0.22; xp = np.arange(n)
dv = [b.recorder.get_total_distance(i) for i in range(n)]
wv = [len(b.recorder.waypoints[i]) for i in range(n)]
ov = [np.mean(b.min_obs_dist_log[i]) for i in range(n)]
b1 = a2.bar(xp-bw, dv, bw, label='Path (m)', color=C1, alpha=.8)
b2 = a2.bar(xp, wv, bw, label='Waypoints', color=C2, alpha=.8)
b3 = a2.bar(xp+bw, [d*10 for d in ov], bw, label='AvgObsDist(x10m)', color=C3, alpha=.8)
for bs in [b1,b2,b3]:
    for br in bs:
        a2.annotate(f'{br.get_height():.0f}', xy=(br.get_x()+br.get_width()/2, br.get_height()),
                    xytext=(0,3), textcoords='offset points', ha='center', fontsize=7)
a2.set_xticks(xp); a2.set_xticklabels([f'UAV-{i}' for i in range(n)])
a2.set(ylabel='Value'); a2.set_title('(b) Per-UAV Metrics'); a2.legend(fontsize=9); a2.grid(True, alpha=.2, axis='y')
fig.tight_layout(); fig.savefig(f'{FIG}/fig3_trajectory.png', dpi=300); plt.close()
print('  saved')

# ===== Fig4: Robustness =====
print('Fig4...')
recs = []
for sd in [42, 77, 123, 200, 256]:
    s = run(3, sd)
    a = np.concatenate(s.min_obs_dist_log)
    recs.append({
        'cov': s.global_grid.explored_ratio*100,
        'ar': np.mean(a > SAFE_RADIUS)*100,
        'st': s.step,
        'mo': min(min(d) for d in s.min_obs_dist_log),
    })
    print(f'  seed={sd}: cov={recs[-1]["cov"]:.0f}% ar={recs[-1]["ar"]:.0f}%')
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, (k, yl, c) in zip(axes, [('cov','Coverage(%)',C1),('ar','AvoidRate(%)',C2),('st','Steps',C3),('mo','MinObsDist(m)','#7E57C2')]):
    d = [r[k] for r in recs]
    bp = ax.boxplot(d, patch_artist=True, widths=0.5)
    bp['boxes'][0].set(facecolor=c, alpha=0.6); bp['medians'][0].set_color('k')
    ax.set_ylabel(yl); ax.set_xticklabels(['3 UAVs\n(5 seeds)']); ax.grid(True, alpha=.2, axis='y')
    ax.set_title(f'$\\mu$={np.mean(d):.1f} $\\sigma$={np.std(d):.1f}', fontsize=9)
fig.suptitle('Robustness (5 seeds)', fontsize=13, y=1.02)
fig.tight_layout(); fig.savefig(f'{FIG}/fig4_robustness.png', dpi=300, bbox_inches='tight'); plt.close()
print('  saved')

print(f'\nAll figures → {FIG}/')
