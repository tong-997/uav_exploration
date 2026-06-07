"""
目标检测仿真: 目标放置、遮挡计算、检测模型、多帧确认、协同响应
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SENSOR_FOV, SENSOR_RANGE, GRID_RES


OCCLUSION_PRESETS = {
    'none':   {'n_occluders': 0,  'r_range': (0, 0)},
    'light':  {'n_occluders': 2,  'r_range': (1.0, 2.0)},
    'medium': {'n_occluders': 5,  'r_range': (1.5, 3.0)},
    'heavy':  {'n_occluders': 10, 'r_range': (2.0, 4.0)},
}


@dataclass
class Target:
    pos: np.ndarray
    target_id: int
    is_confirmed: bool = False
    confirmation_count: int = 0
    confirmed_at_step: int = -1
    estimated_pos: np.ndarray = None
    first_detected_step: int = -1
    detection_history: list = field(default_factory=list)
    _consecutive: int = 0


def place_targets(env, n_targets=1, seed=42):
    rng = np.random.RandomState(seed + 1000)
    from config import WORLD_SIZE
    targets = []
    for tid in range(n_targets):
        for _ in range(200):
            x = rng.uniform(50, WORLD_SIZE - 10)
            y = rng.uniform(10, WORLD_SIZE - 10)
            too_close = False
            for o in env.obstacles:
                if np.hypot(x - o.x, y - o.y) < o.r + 3.0:
                    too_close = True
                    break
            for t in targets:
                if np.hypot(x - t.pos[0], y - t.pos[1]) < 10.0:
                    too_close = True
                    break
            if not too_close:
                targets.append(Target(pos=np.array([x, y]), target_id=tid))
                break
    return targets


def add_occluders(env, occlusion_level, targets, seed=42):
    preset = OCCLUSION_PRESETS.get(occlusion_level, OCCLUSION_PRESETS['none'])
    if preset['n_occluders'] == 0:
        return
    from simulation.sim_env import Obstacle
    from config import WORLD_SIZE, START_POSITIONS
    rng = np.random.RandomState(seed + 2000)
    for _ in range(preset['n_occluders']):
        for _try in range(100):
            r = rng.uniform(*preset['r_range'])
            x = rng.uniform(20, WORLD_SIZE - 10)
            y = rng.uniform(5, WORLD_SIZE - 5)
            ok = True
            for sp in START_POSITIONS:
                if np.hypot(x - sp[0], y - sp[1]) < r + 8.0:
                    ok = False
                    break
            if ok:
                env.obstacles.append(Obstacle(x, y, r))
                break


def compute_occlusion_ratio(drone_pos, target_pos, obstacles):
    diff = target_pos - drone_pos
    dist = np.linalg.norm(diff)
    if dist < 0.1:
        return 0.0
    direction = diff / dist
    blocked_frac = 0.0
    for o in obstacles:
        ox, oy, r = o.x, o.y, o.r
        to_center = np.array([ox, oy]) - drone_pos
        proj = np.dot(to_center, direction)
        if proj < 0 or proj > dist:
            continue
        perp = np.abs(np.cross(direction, to_center))
        if perp < r:
            angular_size = np.arctan2(r, max(proj, 0.1))
            blocked_frac += angular_size / (np.pi / 4)
    return min(blocked_frac, 1.0)


def detect_target(drone_pos, drone_heading, target_pos, obstacles,
                  model_type='oracle', rng=None):
    if rng is None:
        rng = np.random.RandomState()

    diff = target_pos - drone_pos
    dist = np.linalg.norm(diff)
    if dist > SENSOR_RANGE or dist < 0.1:
        return False, 0.0, np.zeros(2)

    angle = np.arctan2(diff[1], diff[0]) - drone_heading
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    if abs(angle) > SENSOR_FOV / 2:
        return False, 0.0, np.zeros(2)

    occ = compute_occlusion_ratio(drone_pos, target_pos, obstacles)

    if model_type == 'oracle':
        if occ >= 0.95:
            return False, 0.0, np.zeros(2)
        return True, 1.0, np.zeros(2)

    elif model_type == 'noisy_oracle':
        if occ >= 0.95:
            return False, 0.0, np.zeros(2)
        conf = 0.8 + rng.uniform(0, 0.2)
        err = rng.normal(0, 0.3, size=2)
        return True, conf, err

    elif model_type == 'yolov5s':
        conf = (1.0 - dist / SENSOR_RANGE) * (1.0 - occ) * 0.85 + rng.normal(0, 0.05)
        conf = np.clip(conf, 0, 1)
        if conf < 0.5:
            return False, conf, np.zeros(2)
        err = rng.normal(0, dist * 0.03, size=2)
        return True, conf, err

    elif model_type == 'yolov5_ffm':
        conf = (1.0 - dist / SENSOR_RANGE) * (1.0 - 0.5 * occ) * 0.92 + rng.normal(0, 0.03)
        conf = np.clip(conf, 0, 1)
        if conf < 0.5:
            return False, conf, np.zeros(2)
        err = rng.normal(0, dist * 0.02, size=2)
        return True, conf, err

    return False, 0.0, np.zeros(2)


def setup_targets(system, config):
    from simulation.targets import place_targets, add_occluders
    system.targets = place_targets(system.env, n_targets=1, seed=config.seed)
    add_occluders(system.env, config.occlusion_level, system.targets, seed=config.seed)

    system._detection_model = config.detection_model
    system._confirmation_frames = config.confirmation_frames
    system._det_rng = np.random.RandomState(config.seed + 3000)

    original_run_target = system._run_target_detection

    def _run_target_detection_impl(drone_positions):
        for t in system.targets:
            if t.is_confirmed:
                continue
            detected_this_step = False
            step_errors = []
            step_confs = []
            for i in range(system.n_drones):
                d = system.env.drones[i]
                det, conf, err = detect_target(
                    d.pos, d.heading, t.pos, system.env.obstacles,
                    model_type=system._detection_model,
                    rng=system._det_rng)
                if det:
                    detected_this_step = True
                    step_errors.append(err)
                    step_confs.append(conf)
                    t.detection_history.append({
                        'step': system.step, 'drone_id': i,
                        'confidence': float(conf),
                        'error': err.tolist(),
                        'distance': float(np.linalg.norm(d.pos - t.pos)),
                    })
                    if t.first_detected_step < 0:
                        t.first_detected_step = system.step

            if detected_this_step:
                t._consecutive += 1
            else:
                t._consecutive = 0

            if t._consecutive >= system._confirmation_frames and not t.is_confirmed:
                t.is_confirmed = True
                t.confirmed_at_step = system.step
                recent = t.detection_history[-system._confirmation_frames:]
                mean_err = np.mean([e['error'] for e in recent], axis=0)
                t.estimated_pos = t.pos + mean_err

                for i in range(system.n_drones):
                    path = system.planner.plan(
                        system.global_grid.grid, system.env.drones[i].pos,
                        t.estimated_pos, treat_unknown_as='free')
                    system.env.drones[i].path = path if path else []
                    system.env.drones[i].target_frontier = t.estimated_pos.copy()
                    system._path_follow_idx[i] = 0
                system._cooperative_override = True

        if system._cooperative_override:
            all_arrived = True
            for t in system.targets:
                if t.is_confirmed and t.estimated_pos is not None:
                    for i in range(system.n_drones):
                        if np.linalg.norm(system.env.drones[i].pos - t.estimated_pos) > 3.0:
                            all_arrived = False
            if all_arrived:
                system._cooperative_override = False

    system._run_target_detection = _run_target_detection_impl
