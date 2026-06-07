"""
RunConfig + 参数化运行器 + 数据持久化
"""
import os
import sys
import json
import csv
import dataclasses
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@dataclasses.dataclass
class RunConfig:
    n_drones: int = 3
    max_speed: float = 3.0
    max_steps: int = 2000
    seed: int = 42
    coverage_threshold: float = 0.90

    use_voronoi: bool = True
    use_deconflict: bool = True
    use_obstacle_inflation: bool = True
    frontier_strategy: str = "utility"

    depth_mode: str = "raycast"
    depth_noise_std: float = 0.5

    enable_targets: bool = False
    occlusion_level: str = "none"
    detection_model: str = "oracle"
    confirmation_frames: int = 3


def run_parametric(config, snapshot_steps=None, verbose=False):
    from run_exploration import ExplorationSystem
    from config import SAFE_RADIUS

    system = ExplorationSystem(seed=config.seed, config=config)

    if config.enable_targets:
        from simulation.targets import setup_targets
        setup_targets(system, config)

    snapshots = {}
    if snapshot_steps is None:
        snapshot_steps = set()
    else:
        snapshot_steps = set(snapshot_steps)

    while not system.is_done():
        if system.step in snapshot_steps:
            snapshots[system.step] = _capture_snapshot(system)
        ratio = system.run_step()
        if verbose and system.step % 100 == 0:
            print(f'  step={system.step}, coverage={ratio:.1%}')

    if system.step in snapshot_steps:
        snapshots[system.step] = _capture_snapshot(system)

    results = extract_results(system, config)
    results['snapshots'] = snapshots
    return results, system


def _capture_snapshot(system):
    from config import GRID_N
    return {
        'grid': system.global_grid.grid.copy(),
        'drones': [(d.pos.copy(), d.heading,
                     list(d.path) if d.path else [],
                     d.target_frontier.copy() if d.target_frontier is not None else None,
                     list(d.waypoints))
                    for d in system.env.drones],
        'obstacles': [(o.x, o.y, o.r) for o in system.env.obstacles],
        'ratio': system.global_grid.explored_ratio,
        'step': system.step,
    }


def extract_results(system, config):
    from config import SAFE_RADIUS
    n = system.n_drones
    all_obs = np.concatenate(system.min_obs_dist_log) if any(system.min_obs_dist_log) else np.array([999.0])
    inter_arr = np.array(system.min_drone_dist_log) if system.min_drone_dist_log else np.array([999.0])
    path_lengths = [system.recorder.get_total_distance(i) for i in range(n)]

    steps_to_threshold = system.step
    for s, r in system.exploration_log:
        if r >= config.coverage_threshold:
            steps_to_threshold = s + 1
            break

    total_explored = set()
    for s in system._per_drone_explored:
        total_explored |= s
    per_drone_counts = [len(s) for s in system._per_drone_explored]
    sum_counts = sum(per_drone_counts)
    overlap_rate = 1.0 - len(total_explored) / max(sum_counts, 1)

    load_balance_cv = 0.0
    if len(path_lengths) > 1 and np.mean(path_lengths) > 0:
        load_balance_cv = np.std(path_lengths) / np.mean(path_lengths)

    plan_success_rate = (system.plan_success_count / max(system.plan_total_count, 1))

    return {
        'config': dataclasses.asdict(config),
        'final_step': system.step,
        'final_coverage': system.global_grid.explored_ratio,
        'steps_to_threshold': steps_to_threshold,
        'success': system.global_grid.explored_ratio >= config.coverage_threshold,
        'collision_count': system.collision_count,
        'obstacle_safe_rate': float(np.mean(all_obs > SAFE_RADIUS)),
        'inter_uav_safe_rate': float(np.mean(inter_arr > SAFE_RADIUS)),
        'min_obstacle_distance': float(np.min(all_obs)),
        'avg_obstacle_distance': float(np.mean(all_obs)),
        'min_inter_uav_distance': float(np.min(inter_arr)),
        'avg_inter_uav_distance': float(np.mean(inter_arr)),
        'path_lengths': path_lengths,
        'total_path_length': sum(path_lengths),
        'load_balance_cv': load_balance_cv,
        'overlap_rate': overlap_rate,
        'replan_count': system.replan_count,
        'plan_success_rate': plan_success_rate,
        'avoidance_events': sum(system.avoidance_events),
        'exploration_log': system.exploration_log,
    }


def save_raw_log(system, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'raw_log_{prefix}.csv')
    n = system.n_drones

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['step', 'time']
        for i in range(n):
            header += [f'uav{i}_x', f'uav{i}_y', f'uav{i}_yaw', f'uav{i}_state',
                       f'uav{i}_min_obs_dist', f'uav{i}_path_length']
        header += ['coverage', 'min_inter_uav_dist', 'replan_count', 'collision_flag']
        writer.writerow(header)

        from config import DT
        n_steps = len(system.exploration_log)
        cum_path = [0.0] * n
        for s_idx in range(n_steps):
            step_num, coverage = system.exploration_log[s_idx]
            row = [step_num, step_num * DT]
            for i in range(n):
                wps = system.recorder.waypoints[i]
                if s_idx < len(wps):
                    wp = wps[s_idx] if s_idx < len(wps) else wps[-1]
                    row += [f'{wp.x:.2f}', f'{wp.y:.2f}', f'{wp.heading:.3f}', 'explore']
                else:
                    d = system.env.drones[i]
                    row += [f'{d.pos[0]:.2f}', f'{d.pos[1]:.2f}', f'{d.heading:.3f}', 'explore']
                obs_d = system.min_obs_dist_log[i][s_idx] if s_idx < len(system.min_obs_dist_log[i]) else 999.0
                row += [f'{obs_d:.2f}', f'{cum_path[i]:.1f}']
            row += [f'{coverage:.4f}']
            inter_d = system.min_drone_dist_log[s_idx] if s_idx < len(system.min_drone_dist_log) else 999.0
            row += [f'{inter_d:.2f}', system.replan_count, 0]
            writer.writerow(row)

    return filepath


def save_summary(results, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'summary_{prefix}.json')
    serializable = {}
    for k, v in results.items():
        if k == 'snapshots':
            continue
        if k == 'exploration_log':
            serializable[k] = [(int(s), float(r)) for s, r in v]
        elif isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v)
        elif isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, list) and v and isinstance(v[0], (np.floating, np.integer)):
            serializable[k] = [float(x) for x in v]
        else:
            serializable[k] = v
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    return filepath
