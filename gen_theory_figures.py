"""
生成 3.5 理论章节所需的全部插图
输出到 output/theory_figures/
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Circle, Wedge
from matplotlib.lines import Line2D
from scipy.ndimage import binary_dilation, distance_transform_edt

OUT = os.path.join(os.path.dirname(__file__), 'output', 'theory_figures')
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Fig 1: System Architecture Pipeline (3.5.1)
# ============================================================
def fig1_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis('off')

    modules = [
        ('Stereo Depth\nEstimation\n(ONNX ResUNet)', '#BBDEFB', '3.5.2'),
        ('Ray-based\nOccupancy Grid\nMapping', '#C8E6C9', '3.5.3'),
        ('Frontier\nDetection\n(BFS)', '#FFF9C4', '3.5.4'),
        ('Voronoi\nArea Allocation', '#FFE0B2', '3.5.5'),
        ("A* Path\nPlanning\n(Inflation)", '#FFCCBC', '3.5.6'),
        ('Distributed\nDeconfliction\n(ID-Priority)', '#E1BEE7', '3.5.7'),
        ('Waypoint\nRecording\n& Interface', '#B2DFDB', '3.5.8'),
    ]

    bw, bh, gap = 1.6, 1.3, 0.25
    y0 = 2.0

    for idx, (label, color, sec) in enumerate(modules):
        x = idx * (bw + gap)
        box = FancyBboxPatch((x, y0 - bh/2), bw, bh,
                             boxstyle='round,pad=0.12',
                             facecolor=color, edgecolor='#424242', lw=1.5)
        ax.add_patch(box)
        ax.text(x + bw/2, y0 + 0.05, label, ha='center', va='center',
                fontsize=8, fontweight='bold')
        ax.text(x + bw/2, y0 - bh/2 + 0.12, f'({sec})',
                ha='center', va='bottom', fontsize=7, color='#616161')
        if idx > 0:
            ax.annotate('', xy=(x - 0.02, y0),
                        xytext=(x - gap + 0.02, y0),
                        arrowprops=dict(arrowstyle='->', color='#424242', lw=1.8))

    # feedback loop
    x_last = (len(modules)-1) * (bw + gap) + bw/2
    x_first = bw/2
    ax.annotate('', xy=(x_first, y0 - bh/2 - 0.25),
                xytext=(x_last, y0 - bh/2 - 0.25),
                arrowprops=dict(arrowstyle='<-', color='#F44336', lw=1.5, ls='--'))
    ax.text((x_first + x_last)/2, y0 - bh/2 - 0.45,
            'Replan every N steps (closed-loop)',
            ha='center', fontsize=8, color='#F44336', style='italic')

    # Input/output labels
    ax.annotate('Stereo\nImages', xy=(0, y0 + bh/2 + 0.05),
                fontsize=8, ha='center', va='bottom', color='#1565C0',
                fontweight='bold')
    ax.annotate('', xy=(bw/2, y0 + bh/2), xytext=(bw/2, y0 + bh/2 + 0.5),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

    x_out = (len(modules)-1) * (bw + gap) + bw/2
    ax.annotate('GPS\nWaypoints', xy=(x_out, y0 + bh/2 + 0.05),
                fontsize=8, ha='center', va='bottom', color='#2E7D32',
                fontweight='bold')
    ax.annotate('', xy=(x_out, y0 + bh/2), xytext=(x_out, y0 + bh/2 + 0.5),
                arrowprops=dict(arrowstyle='<-', color='#2E7D32', lw=1.5))

    total_w = len(modules) * (bw + gap)
    ax.set_xlim(-1.0, total_w + 0.5)
    ax.set_ylim(y0 - bh/2 - 1.0, y0 + bh/2 + 1.2)
    ax.set_title('Multi-UAV Collaborative Exploration System Architecture',
                 fontsize=13, fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_system_architecture.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig_system_architecture.png')


# ============================================================
# Fig 2: Depth Estimation Pipeline (3.5.2)
# ============================================================
def fig2_depth_pipeline():
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    # (a) Stereo pair
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.add_patch(FancyBboxPatch((0.5, 4.2), 4, 3, boxstyle='round,pad=0.1',
                                fc='#E3F2FD', ec='#1565C0', lw=1.5))
    ax.text(2.5, 5.7, 'Left Image', ha='center', fontsize=9, fontweight='bold')
    ax.add_patch(FancyBboxPatch((5.5, 4.2), 4, 3, boxstyle='round,pad=0.1',
                                fc='#E8F5E9', ec='#2E7D32', lw=1.5))
    ax.text(7.5, 5.7, 'Right Image', ha='center', fontsize=9, fontweight='bold')
    ax.annotate('', xy=(7.5, 4.0), xytext=(2.5, 4.0),
                arrowprops=dict(arrowstyle='<->', color='#F44336', lw=1.5))
    ax.text(5.0, 3.5, f'baseline b={0.09}m', ha='center', fontsize=8, color='#F44336')
    ax.set_title('(a) Stereo Pair', fontsize=10, fontweight='bold')
    ax.axis('off')

    # (b) ONNX Inference
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.add_patch(FancyBboxPatch((1, 2.5), 8, 4, boxstyle='round,pad=0.2',
                                fc='#FFF3E0', ec='#E65100', lw=2))
    ax.text(5, 5.5, 'ONNX ResUNet', ha='center', fontsize=10, fontweight='bold', color='#E65100')
    ax.text(5, 4.5, '(14MB, ERP mode)', ha='center', fontsize=8, color='#BF360C')
    ax.text(5, 3.2, r'$d(u,v) \rightarrow$ Disparity Map', ha='center', fontsize=9)
    ax.set_title('(b) Network Inference', fontsize=10, fontweight='bold')
    ax.axis('off')

    # (c) Disparity to depth
    ax = axes[2]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.text(5, 6.5, 'Disparity to Depth', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 5.0, r'$Z = \frac{f_x \cdot b}{d(u,v)}$', ha='center', fontsize=14)
    ax.text(5, 3.5, f'$f_x = 368.92$ px\n$b = 0.09$ m', ha='center', fontsize=9, color='#424242')
    ax.set_title('(c) Depth Conversion', fontsize=10, fontweight='bold')
    ax.axis('off')

    # (d) Local point cloud
    ax = axes[3]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.text(5, 6.5, 'Local Point Cloud', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 5.0, r'$x_i = Z_i \cos\theta_i$', ha='center', fontsize=11)
    ax.text(5, 4.0, r'$y_i = Z_i \sin\theta_i$', ha='center', fontsize=11)
    ax.text(5, 2.8, r'$\mathbf{p}_w = R(\psi)\mathbf{p}_l + \mathbf{t}$',
            ha='center', fontsize=11, color='#1565C0')
    ax.set_title('(d) Coordinate Transform', fontsize=10, fontweight='bold')
    ax.axis('off')

    fig.suptitle('Stereo Depth Estimation to Local Point Cloud Pipeline',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_depth_pipeline.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig_depth_pipeline.png')


# ============================================================
# Fig 3: Ray-casting Occupancy Grid (3.5.3)
# ============================================================
def fig3_ray_casting():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # (a) Ray casting illustration
    ax = axes[0]
    grid_n = 20
    ax.set_xlim(-0.5, grid_n - 0.5)
    ax.set_ylim(-0.5, grid_n - 0.5)

    for i in range(grid_n + 1):
        ax.axhline(i - 0.5, color='#E0E0E0', lw=0.5)
        ax.axvline(i - 0.5, color='#E0E0E0', lw=0.5)

    drone_pos = np.array([3.0, 3.0])
    heading = np.deg2rad(30)
    fov = np.deg2rad(90)
    n_rays = 7
    sensor_range = 14.0

    obs_center = np.array([12.0, 10.0])
    obs_r = 2.5

    free_cells = set()
    occ_cells = set()

    angles = np.linspace(-fov/2, fov/2, n_rays)
    for a in angles:
        ray_angle = heading + a
        dx, dy = np.cos(ray_angle), np.sin(ray_angle)
        # check intersection with obstacle
        fx, fy = drone_pos[0] - obs_center[0], drone_pos[1] - obs_center[1]
        aa = dx*dx + dy*dy
        bb = 2*(fx*dx + fy*dy)
        cc = fx*fx + fy*fy - obs_r*obs_r
        disc = bb*bb - 4*aa*cc
        hit_dist = sensor_range
        if disc >= 0:
            t1 = (-bb - np.sqrt(disc)) / (2*aa)
            if t1 > 0.3:
                hit_dist = min(hit_dist, t1)

        # draw ray
        end_x = drone_pos[0] + dx * hit_dist
        end_y = drone_pos[1] + dy * hit_dist
        ax.plot([drone_pos[0], end_x], [drone_pos[1], end_y],
                '-', color='#FFA726', lw=0.8, alpha=0.6)

        # mark free cells along ray
        step = 0.8
        for t in np.arange(0, hit_dist, step):
            px = drone_pos[0] + dx * t
            py = drone_pos[1] + dy * t
            ix, iy = int(px), int(py)
            if 0 <= ix < grid_n and 0 <= iy < grid_n:
                free_cells.add((ix, iy))

        # mark occupied at hit
        if hit_dist < sensor_range - 0.5:
            ix, iy = int(end_x), int(end_y)
            if 0 <= ix < grid_n and 0 <= iy < grid_n:
                occ_cells.add((ix, iy))
                free_cells.discard((ix, iy))

    # Draw cells
    for ix, iy in free_cells:
        ax.add_patch(plt.Rectangle((ix-0.5, iy-0.5), 1, 1,
                                   fc='#E8F5E9', ec='none'))
    for ix, iy in occ_cells:
        ax.add_patch(plt.Rectangle((ix-0.5, iy-0.5), 1, 1,
                                   fc='#424242', ec='none'))

    # Unknown cells (not touched)
    all_touched = free_cells | occ_cells
    for ix in range(grid_n):
        for iy in range(grid_n):
            if (ix, iy) not in all_touched:
                ax.add_patch(plt.Rectangle((ix-0.5, iy-0.5), 1, 1,
                                           fc='#E3F2FD', ec='none'))

    # Re-draw grid lines on top
    for i in range(grid_n + 1):
        ax.axhline(i - 0.5, color='#BDBDBD', lw=0.3)
        ax.axvline(i - 0.5, color='#BDBDBD', lw=0.3)

    # Obstacle
    circle = plt.Circle(obs_center, obs_r, fc='#EF5350', alpha=0.4, ec='#C62828', lw=1.5)
    ax.add_patch(circle)

    # Drone
    ax.plot(drone_pos[0], drone_pos[1], 'o', color='#29b6f6', ms=12,
            mec='white', mew=2, zorder=10)
    ax.annotate('UAV', xy=(drone_pos[0], drone_pos[1]+1), fontsize=9,
                ha='center', fontweight='bold', color='#0277BD')

    # FOV arc
    fov_angles = np.linspace(heading - fov/2, heading + fov/2, 50)
    arc_r = 4
    arc_x = drone_pos[0] + arc_r * np.cos(fov_angles)
    arc_y = drone_pos[1] + arc_r * np.sin(fov_angles)
    ax.plot(arc_x, arc_y, '--', color='#29b6f6', lw=1.5, alpha=0.7)

    # Legend
    patches = [
        mpatches.Patch(fc='#E8F5E9', ec='#BDBDBD', label='FREE'),
        mpatches.Patch(fc='#424242', label='OCCUPIED'),
        mpatches.Patch(fc='#E3F2FD', ec='#BDBDBD', label='UNKNOWN'),
        Line2D([0],[0], color='#FFA726', lw=1.5, label='Depth Ray'),
    ]
    ax.legend(handles=patches, fontsize=8, loc='upper right')
    ax.set_title('(a) Ray-casting Grid Update', fontsize=11, fontweight='bold')
    ax.set_xlabel('Grid X'); ax.set_ylabel('Grid Y')
    ax.set_aspect('equal')

    # (b) Merge priority
    ax = axes[1]
    ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)

    ax.text(5, 7.2, 'Multi-Robot Grid Merging', fontsize=12, fontweight='bold', ha='center')

    merge_rules = [
        ('OCCUPIED', '#424242', 'white', 'Highest priority (always overwrites)'),
        ('FREE', '#E8F5E9', 'black', 'Overwrites UNKNOWN only'),
        ('UNKNOWN', '#E3F2FD', 'black', 'Default / lowest priority'),
    ]
    for i, (label, color, tc, desc) in enumerate(merge_rules):
        y = 5.5 - i * 1.8
        ax.add_patch(FancyBboxPatch((0.5, y - 0.4), 2.5, 0.8,
                                    boxstyle='round,pad=0.1', fc=color, ec='#616161', lw=1.2))
        ax.text(1.75, y, label, ha='center', va='center', fontsize=9,
                fontweight='bold', color=tc)
        ax.text(3.5, y, f'← {desc}', va='center', fontsize=8, color='#424242')
        if i < 2:
            ax.annotate('', xy=(1.75, y - 0.5), xytext=(1.75, y - 0.9),
                        arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5))

    ax.text(5, 1.0, r'$g_{\mathrm{fused}}(i,j) = \max\{g_1(i,j),\, g_2(i,j),\, \ldots\}$',
            ha='center', fontsize=11, color='#1565C0',
            bbox=dict(boxstyle='round,pad=0.3', fc='#E3F2FD', ec='#1565C0'))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_ray_casting.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig_ray_casting.png')


# ============================================================
# Fig 4: Frontier Detection (3.5.4)
# ============================================================
def fig4_frontier_detection():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Create a realistic grid
    np.random.seed(42)
    N = 30
    grid = np.zeros((N, N), dtype=int)  # UNKNOWN=0

    # explored region (FREE=1)
    for iy in range(N):
        for ix in range(N):
            if (ix - 5)**2 + (iy - 5)**2 < 144:
                grid[iy, ix] = 1
            if (ix - 20)**2 + (iy - 20)**2 < 16:
                grid[iy, ix] = 2  # obstacle

    # (a) Frontier cells
    ax = axes[0]
    display = np.ones((N, N, 3))
    display[grid == 1] = [0.92, 0.95, 0.92]
    display[grid == 2] = [0.2, 0.2, 0.2]
    display[grid == 0] = [0.82, 0.88, 0.95]

    # Find frontiers
    free_mask = grid == 1
    unknown_mask = grid == 0
    expanded = np.zeros_like(unknown_mask)
    expanded[1:, :] |= unknown_mask[:-1, :]
    expanded[:-1, :] |= unknown_mask[1:, :]
    expanded[:, 1:] |= unknown_mask[:, :-1]
    expanded[:, :-1] |= unknown_mask[:, 1:]
    frontier_mask = free_mask & expanded

    # Mark frontiers in red
    display[frontier_mask] = [1.0, 0.3, 0.3]

    ax.imshow(display, origin='lower', extent=[0, N, 0, N])
    for i in range(N + 1):
        ax.axhline(i, color='#BDBDBD', lw=0.2)
        ax.axvline(i, color='#BDBDBD', lw=0.2)

    ax.plot(5, 5, 'o', color='#29b6f6', ms=10, mec='white', mew=1.5, zorder=10)
    ax.annotate('UAV', xy=(5, 6.5), fontsize=9, ha='center', fontweight='bold', color='#0277BD')

    patches = [
        mpatches.Patch(fc=[0.92, 0.95, 0.92], ec='#BDBDBD', label='FREE'),
        mpatches.Patch(fc=[0.82, 0.88, 0.95], ec='#BDBDBD', label='UNKNOWN'),
        mpatches.Patch(fc=[0.2, 0.2, 0.2], label='OCCUPIED'),
        mpatches.Patch(fc=[1.0, 0.3, 0.3], label='Frontier Cell'),
    ]
    ax.legend(handles=patches, fontsize=8, loc='upper right')
    ax.set_title('(a) Frontier Cell Detection', fontsize=11, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y')

    # (b) BFS clustering + utility scoring
    ax = axes[1]
    ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.text(5, 7.5, 'Frontier Selection Utility', fontsize=12, fontweight='bold', ha='center')

    ax.text(5, 6.2, r'$U(f) = -w_d \cdot \|\mathbf{p}_{uav} - \mathbf{c}_f\|'
                     r' + w_s \cdot \ln(1 + |f|)$',
            ha='center', fontsize=12, color='#1565C0',
            bbox=dict(boxstyle='round,pad=0.3', fc='#E3F2FD', ec='#1565C0'))

    labels = [
        (r'$w_d$: distance weight (= 1.0)', 4.8),
        (r'$w_s$: size weight (= 0.5)', 4.2),
        (r'$\mathbf{c}_f$: frontier cluster center (BFS)', 3.6),
        (r'$|f|$: frontier cluster size (cells)', 3.0),
    ]
    for txt, y in labels:
        ax.text(1.5, y, txt, fontsize=9, va='center')

    ax.text(5, 2.0, 'Overlap Penalty:', fontsize=10, fontweight='bold', ha='center')
    ax.text(5, 1.2, r"$U(f) -= 50 \cdot \max(0,\, 1 - d_{other}/10)$",
            ha='center', fontsize=11, color='#E65100',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3E0', ec='#E65100'))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_frontier_detection.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig_frontier_detection.png')


# ============================================================
# Fig 5: Voronoi Partition (3.5.5)
# ============================================================
def fig5_voronoi_partition():
    fig, ax = plt.subplots(figsize=(7, 7))

    N = 100
    drone_positions = np.array([[15, 15], [15, 50], [15, 85]])
    colors_fill = ['#BBDEFB', '#C8E6C9', '#FFE0B2']
    colors_drone = ['#29b6f6', '#66bb6a', '#ffa726']

    xs = np.arange(N) + 0.5
    XX, YY = np.meshgrid(xs, xs)

    dist_maps = np.zeros((3, N, N))
    for i, pos in enumerate(drone_positions):
        dist_maps[i] = np.sqrt((XX - pos[0])**2 + (YY - pos[1])**2)
    voronoi = np.argmin(dist_maps, axis=0)

    display = np.ones((N, N, 3))
    for i in range(3):
        r, g, b = matplotlib.colors.to_rgb(colors_fill[i])
        mask = voronoi == i
        display[mask] = [r, g, b]

    ax.imshow(display, origin='lower', extent=[0, N, 0, N])

    # obstacles
    obs_positions = [(40, 30, 4), (60, 60, 3), (75, 25, 3.5), (30, 70, 2.5)]
    for ox, oy, orr in obs_positions:
        ax.add_patch(Circle((ox, oy), orr, fc='#EF5350', alpha=0.3, ec='#C62828', lw=1))

    # frontier points (simulated)
    np.random.seed(7)
    frontiers = [(45, 20), (55, 45), (70, 75), (25, 60), (80, 50), (35, 85)]
    for fx, fy in frontiers:
        ax.plot(fx, fy, 's', color='#F44336', ms=7, mec='#B71C1C', mew=1, zorder=5)

    # Voronoi boundaries (approximate)
    # Draw drones
    for i, pos in enumerate(drone_positions):
        ax.plot(pos[0], pos[1], 'o', color=colors_drone[i], ms=14,
                mec='white', mew=2, zorder=10)
        ax.annotate(f'UAV-{i}', xy=(pos[0] + 2, pos[1] + 2), fontsize=10,
                    fontweight='bold', color=colors_drone[i])

    patches = [
        mpatches.Patch(fc=colors_fill[i], label=f'UAV-{i} Region') for i in range(3)
    ]
    patches.append(Line2D([0],[0], marker='s', color='w', markerfacecolor='#F44336',
                          ms=8, label='Frontier'))
    patches.append(Circle((0,0), 0.1, fc='#EF5350', alpha=0.3, label='Obstacle'))
    ax.legend(handles=patches, fontsize=9, loc='upper right')
    ax.set_title('Voronoi Partition for Multi-UAV Task Allocation', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_xlim(0, N); ax.set_ylim(0, N)
    ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_voronoi_partition.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig_voronoi_partition.png')


# ============================================================
# Fig 6: A* Path Planning with Inflation (3.5.6)
# ============================================================
def fig6_astar_planning():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    N = 40
    grid = np.ones((N, N), dtype=np.float64)  # 1 = free

    # obstacles
    obs_list = [(15, 20, 3), (25, 12, 2.5), (30, 30, 3.5), (10, 32, 2)]
    blocked = np.zeros((N, N), dtype=bool)
    xs = np.arange(N) + 0.5
    XX, YY = np.meshgrid(xs, xs)
    for ox, oy, orr in obs_list:
        mask = np.sqrt((XX - ox)**2 + (YY - oy)**2) < orr
        blocked[mask] = True

    # (a) Original grid + inflation
    ax = axes[0]
    display = np.ones((N, N, 3))
    display[blocked] = [0.2, 0.2, 0.2]

    # Inflation
    margin = 2
    struct = np.ones((2*margin+1, 2*margin+1))
    inflated = binary_dilation(blocked, structure=struct)
    inflated_only = inflated & ~blocked
    display[inflated_only] = [1.0, 0.85, 0.7]

    ax.imshow(display, origin='lower', extent=[0, N, 0, N])

    # Start and goal
    start = np.array([3, 3])
    goal = np.array([37, 37])
    ax.plot(start[0], start[1], 's', color='#2196F3', ms=12, mec='white', mew=2, zorder=10)
    ax.plot(goal[0], goal[1], '*', color='#4CAF50', ms=16, mec='white', mew=1.5, zorder=10)
    ax.annotate('Start', xy=(start[0]+1.5, start[1]), fontsize=9, fontweight='bold', color='#1565C0')
    ax.annotate('Goal', xy=(goal[0]+1.5, goal[1]), fontsize=9, fontweight='bold', color='#2E7D32')

    patches = [
        mpatches.Patch(fc=[0.2, 0.2, 0.2], label='Obstacle'),
        mpatches.Patch(fc=[1.0, 0.85, 0.7], label=f'Inflation (margin={margin})'),
        mpatches.Patch(fc='white', ec='#BDBDBD', label='Free Space'),
    ]
    ax.legend(handles=patches, fontsize=8, loc='upper left')
    ax.set_title('(a) Obstacle Inflation', fontsize=11, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_aspect('equal')

    # (b) A* path
    ax = axes[1]
    ax.imshow(display, origin='lower', extent=[0, N, 0, N])

    # Simulate A* path (manually crafted realistic path)
    path = np.array([
        [3, 3], [5, 5], [7, 7], [9, 10], [10, 13], [10, 16],
        [11, 19], [13, 22], [15, 25], [17, 27], [19, 28],
        [21, 29], [23, 30], [25, 32], [27, 34], [29, 35],
        [31, 36], [33, 37], [35, 37], [37, 37]
    ], dtype=float)

    ax.plot(path[:, 0], path[:, 1], '-', color='#F44336', lw=2.5, zorder=8)
    ax.plot(path[::3, 0], path[::3, 1], 'o', color='#F44336', ms=4, zorder=9)
    ax.plot(start[0], start[1], 's', color='#2196F3', ms=12, mec='white', mew=2, zorder=10)
    ax.plot(goal[0], goal[1], '*', color='#4CAF50', ms=16, mec='white', mew=1.5, zorder=10)

    patches = [
        Line2D([0],[0], color='#F44336', lw=2.5, label='A* Path'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='#2196F3', ms=10, label='Start'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor='#4CAF50', ms=12, label='Goal'),
    ]
    ax.legend(handles=patches, fontsize=8, loc='upper left')
    ax.set_title('(b) A* Path (8-connectivity)', fontsize=11, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_astar_planning.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig_astar_planning.png')


# ============================================================
# Fig 7: Distributed Deconfliction (3.5.7)
# ============================================================
def fig7_deconfliction():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) Priority-based trajectory deconfliction
    ax = axes[0]
    ax.set_xlim(0, 20); ax.set_ylim(0, 15)

    # UAV-0 (higher priority)
    uav0_path = np.array([[3, 4], [6, 6], [9, 8], [12, 9], [15, 9]])
    ax.plot(uav0_path[:, 0], uav0_path[:, 1], '-o', color='#29b6f6', lw=2.5,
            ms=6, mec='white', mew=1.5, label='UAV-0 (ID=0, high priority)')

    # UAV-1 (lower priority, conflict)
    uav1_path_orig = np.array([[3, 12], [6, 10], [9, 8.5], [12, 7], [15, 5]])
    ax.plot(uav1_path_orig[:, 0], uav1_path_orig[:, 1], '--', color='#ffa726',
            lw=1.5, alpha=0.5)

    # UAV-1 resolved (wait + detour)
    uav1_path_resolved = np.array([[3, 12], [3, 12], [3, 12], [6, 10],
                                    [9, 8.5], [12, 7], [15, 5]])
    ax.plot(uav1_path_resolved[2:, 0], uav1_path_resolved[2:, 1], '-s',
            color='#ffa726', lw=2.5, ms=6, mec='white', mew=1.5,
            label='UAV-1 (ID=1, yield & wait)')

    # Conflict zone
    conflict_circle = plt.Circle((9, 8.2), 3, fc='#FFCDD2', ec='#F44336',
                                  lw=1.5, ls='--', alpha=0.3)
    ax.add_patch(conflict_circle)
    ax.text(9, 11.5, f'Conflict Zone\n(d < {3.0*1.5:.1f}m)', ha='center',
            fontsize=8, color='#F44336', fontweight='bold')

    # Wait annotation
    ax.annotate('WAIT\n(5 steps)', xy=(3, 12), fontsize=8, ha='center',
                va='bottom', color='#E65100', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='#FFF3E0', ec='#E65100'))

    ax.set_title('(a) ID-Priority Trajectory Deconfliction', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)

    # (b) Emergency avoidance
    ax = axes[1]
    ax.set_xlim(0, 14); ax.set_ylim(0, 14)

    # Two UAVs too close
    pos0 = np.array([6.0, 7.0])
    pos1 = np.array([8.0, 7.5])

    ax.plot(pos0[0], pos0[1], 'o', color='#29b6f6', ms=14, mec='white', mew=2, zorder=10)
    ax.plot(pos1[0], pos1[1], 'o', color='#ffa726', ms=14, mec='white', mew=2, zorder=10)
    ax.text(pos0[0]-0.5, pos0[1]+1, 'UAV-0', fontsize=9, fontweight='bold', color='#0277BD')
    ax.text(pos1[0]+0.5, pos1[1]+1, 'UAV-1', fontsize=9, fontweight='bold', color='#E65100')

    # Distance line
    ax.plot([pos0[0], pos1[0]], [pos0[1], pos1[1]], ':', color='#F44336', lw=2)
    mid = (pos0 + pos1) / 2
    d = np.linalg.norm(pos1 - pos0)
    ax.text(mid[0], mid[1] - 0.6, f'd = {d:.1f}m < {3.0}m', ha='center',
            fontsize=9, color='#F44336', fontweight='bold')

    # Repulsion forces
    diff0 = pos0 - pos1
    diff0_n = diff0 / np.linalg.norm(diff0)
    diff1 = pos1 - pos0
    diff1_n = diff1 / np.linalg.norm(diff1)

    ax.annotate('', xy=(pos0[0] + diff0_n[0]*3, pos0[1] + diff0_n[1]*3),
                xytext=(pos0[0], pos0[1]),
                arrowprops=dict(arrowstyle='->', color='#29b6f6', lw=2.5))
    ax.annotate('', xy=(pos1[0] + diff1_n[0]*3, pos1[1] + diff1_n[1]*3),
                xytext=(pos1[0], pos1[1]),
                arrowprops=dict(arrowstyle='->', color='#ffa726', lw=2.5))

    ax.text(pos0[0] + diff0_n[0]*3.5, pos0[1] + diff0_n[1]*3.5,
            r'$\mathbf{F}_{rep}$', fontsize=11, color='#0277BD', fontweight='bold')
    ax.text(pos1[0] + diff1_n[0]*3.5, pos1[1] + diff1_n[1]*3.5,
            r'$\mathbf{F}_{rep}$', fontsize=11, color='#E65100', fontweight='bold')

    # Safe radius
    ax.add_patch(plt.Circle(pos0, 3.0, fc='none', ec='#29b6f6', lw=1, ls='--', alpha=0.5))
    ax.add_patch(plt.Circle(pos1, 3.0, fc='none', ec='#ffa726', lw=1, ls='--', alpha=0.5))

    # Formula
    ax.text(7, 1.5, r'$\mathbf{F}_{rep} = \sum_{j \in \mathcal{N}_i}'
                    r'\frac{\mathbf{p}_i - \mathbf{p}_j}{\|\mathbf{p}_i - \mathbf{p}_j\|^2}$',
            ha='center', fontsize=12, color='#424242',
            bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5', ec='#616161'))

    ax.set_title('(b) Emergency Repulsion Avoidance', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_deconfliction.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig_deconfliction.png')


# ============================================================
# Fig 8: Waypoint Interface (3.5.8)
# ============================================================
def fig8_waypoint_interface():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.axis('off')
    ax.set_xlim(0, 20); ax.set_ylim(0, 11)

    # ---- Phase 1: Exploration ----
    ax.text(4.5, 10.3, 'Phase 1: Collaborative Exploration',
            fontsize=12, fontweight='bold', ha='center', color='#1565C0')

    # 3 recon UAVs
    for i, (y, c, label) in enumerate([
        (8.8, '#29b6f6', 'Recon UAV-0'),
        (7.6, '#66bb6a', 'Recon UAV-1'),
        (6.4, '#ffa726', 'Recon UAV-2'),
    ]):
        ax.add_patch(FancyBboxPatch((0.3, y-0.35), 2.8, 0.7,
                     boxstyle='round,pad=0.1', fc=c, ec='white', lw=1.5, alpha=0.85))
        ax.text(1.7, y, label, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    # Arrow -> Waypoint Recording
    ax.annotate('', xy=(3.8, 7.6), xytext=(3.2, 7.6),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=1.8))

    # Waypoint Recorder box
    ax.add_patch(FancyBboxPatch((3.8, 6.1), 3.0, 3.2, boxstyle='round,pad=0.15',
                                fc='#FFF3E0', ec='#E65100', lw=2))
    ax.text(5.3, 8.7, 'Waypoint', ha='center', fontsize=10, fontweight='bold', color='#E65100')
    ax.text(5.3, 8.1, 'Recorder', ha='center', fontsize=10, fontweight='bold', color='#E65100')
    ax.text(5.3, 7.3, '(x, y, t, hdg, id)', ha='center', fontsize=8, fontfamily='monospace', color='#BF360C')
    ax.text(5.3, 6.6, 'filter: d > 0.5m', ha='center', fontsize=8, color='#616161')

    # Strike UAV waiting at base
    ax.add_patch(FancyBboxPatch((0.3, 4.7), 2.8, 0.9, boxstyle='round,pad=0.1',
                                fc='#FFCDD2', ec='#C62828', lw=2))
    ax.text(1.7, 5.15, 'Strike UAV', ha='center', fontsize=9, fontweight='bold', color='#B71C1C')
    ax.text(1.7, 4.85, '(no sensor, waiting)', ha='center', fontsize=7, color='#C62828')

    # ---- Phase 2: Target Detected ----
    ax.text(13.5, 10.3, 'Phase 2: Target Detected & Strike',
            fontsize=12, fontweight='bold', ha='center', color='#C62828')

    # Target detected event
    ax.add_patch(FancyBboxPatch((7.8, 7.8), 3.4, 1.5, boxstyle='round,pad=0.15',
                                fc='#FFEBEE', ec='#F44336', lw=2.5))
    ax.text(9.5, 8.9, 'TARGET', ha='center', fontsize=11, fontweight='bold', color='#D32F2F')
    ax.text(9.5, 8.2, 'DETECTED!', ha='center', fontsize=11, fontweight='bold', color='#D32F2F')

    ax.annotate('', xy=(7.7, 8.5), xytext=(6.9, 7.8),
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=2))

    # Three actions from target detection
    # Action 1: Discoverer -> tracking
    ax.add_patch(FancyBboxPatch((12.0, 9.0), 3.5, 1.0, boxstyle='round,pad=0.1',
                                fc='#E8F5E9', ec='#2E7D32', lw=1.5))
    ax.text(13.75, 9.7, '1. Discoverer', ha='center', fontsize=9, fontweight='bold', color='#2E7D32')
    ax.text(13.75, 9.25, 'Track target + occlusion', ha='center', fontsize=7.5, color='#424242')
    ax.annotate('', xy=(11.9, 9.3), xytext=(11.3, 8.8),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.8))

    # Action 2: Broadcast -> other recon UAVs encircle
    ax.add_patch(FancyBboxPatch((12.0, 7.5), 3.5, 1.2, boxstyle='round,pad=0.1',
                                fc='#E3F2FD', ec='#1565C0', lw=1.5))
    ax.text(13.75, 8.35, '2. Broadcast target pos', ha='center', fontsize=9, fontweight='bold', color='#1565C0')
    ax.text(13.75, 7.85, 'Other recon UAVs', ha='center', fontsize=7.5, color='#424242')
    ax.text(13.75, 7.55, 'converge & encircle', ha='center', fontsize=7.5, color='#424242')
    ax.annotate('', xy=(11.9, 8.0), xytext=(11.3, 8.4),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.8))

    # Action 3: Send waypoints -> strike UAV
    ax.add_patch(FancyBboxPatch((12.0, 5.5), 3.5, 1.7, boxstyle='round,pad=0.1',
                                fc='#F3E5F5', ec='#7B1FA2', lw=2))
    ax.text(13.75, 6.85, '3. Send waypoints', ha='center', fontsize=9, fontweight='bold', color='#7B1FA2')
    ax.text(13.75, 6.35, 'to Strike UAV', ha='center', fontsize=9, fontweight='bold', color='#7B1FA2')
    ax.text(13.75, 5.85, 'verified safe corridor', ha='center', fontsize=7.5, color='#616161')

    # Arrow from waypoint recorder to action 3
    ax.annotate('', xy=(11.9, 6.3), xytext=(6.9, 6.8),
                arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=2, ls='--'))
    ax.text(9.4, 6.2, 'waypoints', fontsize=8, color='#7B1FA2', fontweight='bold',
            rotation=3, ha='center')

    # Strike UAV follows waypoints
    ax.add_patch(FancyBboxPatch((16.2, 5.5), 3.3, 1.7, boxstyle='round,pad=0.1',
                                fc='#FFCDD2', ec='#C62828', lw=2))
    ax.text(17.85, 6.85, 'Strike UAV', ha='center', fontsize=10, fontweight='bold', color='#B71C1C')
    ax.text(17.85, 6.3, 'Follow waypoints', ha='center', fontsize=8, color='#424242')
    ax.text(17.85, 5.85, 'No sensor needed!', ha='center', fontsize=8, fontweight='bold', color='#C62828')

    ax.annotate('', xy=(16.1, 6.3), xytext=(15.6, 6.3),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.5))

    # Final: precision strike
    ax.add_patch(FancyBboxPatch((16.2, 3.8), 3.3, 1.2, boxstyle='round,pad=0.1',
                                fc='#FF8A80', ec='#D50000', lw=2))
    ax.text(17.85, 4.6, 'Precision Strike', ha='center', fontsize=10, fontweight='bold', color='white')
    ax.text(17.85, 4.1, 'at encircled target', ha='center', fontsize=8, color='white')

    ax.annotate('', xy=(17.85, 3.7), xytext=(17.85, 5.4),
                arrowprops=dict(arrowstyle='->', color='#D50000', lw=2.5))

    ax.set_title('Reconnaissance-Strike Cooperative Workflow & Waypoint Interface',
                 fontsize=14, fontweight='bold', pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_waypoint_interface.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] fig_waypoint_interface.png')


# ============================================================
#  Main
# ============================================================
if __name__ == '__main__':
    print('Generating theory chapter figures...')
    fig1_system_architecture()
    fig2_depth_pipeline()
    fig3_ray_casting()
    fig4_frontier_detection()
    fig5_voronoi_partition()
    fig6_astar_planning()
    fig7_deconfliction()
    fig8_waypoint_interface()
    print(f'\nAll figures saved to: {OUT}/')
