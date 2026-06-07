"""
实验 4B: 遮挡目标连续跟踪与多机联合围捕
- 4B-1: 单机遮挡目标连续跟踪 (motion × occlusion × detector × method × seed)
- 4B-2: 多机联合状态融合 (occlusion × fusion_method × seed)
- 4B-3: 协同围捕 (motion × seed, full pipeline)
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import DT, MAX_SPEED, SENSOR_RANGE, WORLD_SIZE, SAFE_RADIUS
from simulation.sim_env import SimEnvironment
from simulation.target_motion import TargetMotionModel
from simulation.occlusion_schedule import OcclusionSchedule
from tracking.visual_observation import generate_observation
from tracking.confidence_modulated_kf import ConfidenceModulatedKalmanFilter
from tracking.reassociation import ReassociationModule
from tracking.multi_uav_fusion import MultiUAVFusion
from tracking.encirclement_slots import EncirclementSlots
from coordination.target_broadcast import TargetBroadcast
from coordination.target_tracking_fsm import TargetTrackingFSM
from planning.path_planner import AStarPlanner
from experiments.plotting.common import setup_style
from experiments.plotting.plot_exp4b import (
    plot_tracking_curves, plot_rmse_bars, plot_track_continuity,
    plot_fusion_bars, plot_encirclement_snapshot, plot_fsm_timeline,
    plot_4b1_metrics_table, plot_4b2_metrics_table, plot_4b3_metrics_table,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       'output', 'exp4b_tracking')

# =====================================================================
#  Exp 4B-1: Single-UAV Tracking
# =====================================================================

def _run_single_tracking(motion, occ_mode, detector, method, seed,
                         max_steps=500):
    rng = np.random.RandomState(seed)
    env = SimEnvironment(seed=seed, n_drones=1)

    target = TargetMotionModel(
        init_pos=np.array([60.0, 50.0]),
        init_vel=np.array([0.5, 0.3]) if motion != 'static' else None,
        mode=motion, speed=0.8, seed=seed + 100)
    schedule = OcclusionSchedule(mode=occ_mode)

    uav_pos = np.array([54.0, 50.0])
    uav_heading = 0.0

    kf = None
    reassoc = ReassociationModule()
    if method == 'cm_kf':
        kf = ConfidenceModulatedKalmanFilter(dt=DT)
    elif method == 'standard_kf':
        kf = ConfidenceModulatedKalmanFilter(
            dt=DT, alpha_o=0.0, alpha_d=0.0)

    log = {'true_pos': [], 'est_pos': [], 'error': [],
           'confidence': [], 'track_state': [], 'detected': []}
    initialized = False

    for step in range(max_steps):
        target.step()
        true_pos = target.get_pos()

        diff = true_pos - uav_pos
        dist = np.linalg.norm(diff)
        desired_dist = 6.0
        if dist > desired_dist + 1.0:
            direction = diff / max(dist, 0.01)
            move = min(MAX_SPEED * DT, dist - desired_dist)
            new_pos = uav_pos + direction * move
            if 1.0 < new_pos[0] < WORLD_SIZE - 1.0 and \
               1.0 < new_pos[1] < WORLD_SIZE - 1.0:
                uav_pos = new_pos
            uav_heading = np.arctan2(direction[1], direction[0])
        elif dist > 0.1:
            uav_heading = np.arctan2(diff[1], diff[0])

        obs = generate_observation(
            uav_pos, uav_heading, true_pos, env.obstacles,
            detector, schedule, step, 0, rng)

        est_pos = None
        track_state = 'N/A'

        if method == 'detection_only':
            if obs.detected:
                est_pos = obs.target_meas_pos.copy()
        else:
            if obs.detected and not initialized:
                kf.init_state(obs.target_meas_pos)
                initialized = True

            if initialized:
                kf.predict()
                if obs.detected:
                    if kf.track_state in ('LOST', 'OCCLUDED_TRACK'):
                        ok, _ = reassoc.check_reacquisition(
                            kf, obs.target_meas_pos, step)
                        if ok or kf.track_state == 'OCCLUDED_TRACK':
                            kf.update(obs.target_meas_pos, obs.confidence,
                                      1.0 - obs.visibility,
                                      obs.depth_uncertainty,
                                      obs.measurement_cov)
                        else:
                            kf.predict_only()
                    else:
                        kf.update(obs.target_meas_pos, obs.confidence,
                                  1.0 - obs.visibility,
                                  obs.depth_uncertainty,
                                  obs.measurement_cov)
                else:
                    if kf.track_state != 'LOST':
                        reassoc.mark_lost(step)
                    kf.predict_only()
                est_pos = kf.get_position()
                track_state = kf.track_state

        log['true_pos'].append(true_pos.copy())
        if est_pos is not None:
            log['est_pos'].append(est_pos.copy())
            log['error'].append(float(np.linalg.norm(est_pos - true_pos)))
        else:
            log['est_pos'].append(np.array([np.nan, np.nan]))
            log['error'].append(np.nan)
        log['confidence'].append(obs.confidence)
        log['track_state'].append(track_state)
        log['detected'].append(obs.detected)

    errors = np.array(log['error'])
    valid = ~np.isnan(errors)
    rmse = float(np.sqrt(np.nanmean(errors[valid] ** 2))) if valid.any() else float('nan')
    track_rate = float(valid.sum()) / max_steps

    occ_steps = [i for i in range(max_steps)
                 if schedule.get_level(i) != 'none']
    vis_steps = [i for i in range(max_steps)
                 if schedule.get_level(i) == 'none']
    occ_errors = [errors[i] for i in occ_steps if not np.isnan(errors[i])]
    vis_errors = [errors[i] for i in vis_steps if not np.isnan(errors[i])]
    occ_rmse = float(np.sqrt(np.mean(np.array(occ_errors) ** 2))) \
        if occ_errors else float('nan')
    vis_rmse = float(np.sqrt(np.mean(np.array(vis_errors) ** 2))) \
        if vis_errors else float('nan')

    return {
        'rmse': rmse,
        'visible_rmse': vis_rmse,
        'occlusion_rmse': occ_rmse,
        'track_rate': track_rate,
        'reassoc_rate': reassoc.success_rate,
        'mean_reacq_time': reassoc.mean_reacquisition_time,
        'lost_count': reassoc.get_metrics()['lost_count'],
        'log': log,
    }


def run_exp4b1():
    print('=== Exp 4B-1: Single-UAV Tracking ===')
    raw_dir = os.path.join(OUT_DIR, 'raw')
    fig_dir = os.path.join(OUT_DIR, 'figures')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    motions = ['static', 'linear', 'random_turn']
    occlusions = ['no_occlusion', 'partial_occlusion',
                   'intermittent_occlusion', 'full_occlusion_interval']
    detectors = ['oracle', 'noisy_oracle', 'yolov5s', 'yolov5_ffm']
    methods = ['detection_only', 'standard_kf', 'cm_kf']
    seeds = [42, 43, 44, 45, 46]

    summary = {}
    count = 0
    total = len(motions) * len(occlusions) * len(detectors) * len(methods) * len(seeds)

    for motion in motions:
        for occ in occlusions:
            for detector in detectors:
                for method in methods:
                    seed_results = []
                    for seed in seeds:
                        r = _run_single_tracking(motion, occ, detector,
                                                  method, seed)
                        seed_results.append(r)
                        count += 1

                    avg = {
                        'rmse': float(np.nanmean([r['rmse'] for r in seed_results])),
                        'visible_rmse': float(np.nanmean([r['visible_rmse'] for r in seed_results])),
                        'occlusion_rmse': float(np.nanmean([r['occlusion_rmse'] for r in seed_results])),
                        'track_rate': float(np.mean([r['track_rate'] for r in seed_results])),
                        'reassoc_rate': float(np.mean([r['reassoc_rate'] for r in seed_results])),
                        'lost_count': float(np.mean([r['lost_count'] for r in seed_results])),
                    }
                    summary[(motion, occ, method)] = avg

                    if count % 60 == 0 or count == total:
                        print(f'  Progress: {count}/{total}')

    # Save one representative tracking curve (linear, intermittent, yolov5_ffm)
    for method in methods:
        rep = _run_single_tracking('linear', 'intermittent_occlusion',
                                    'yolov5_ffm', method, 42)
        plot_tracking_curves(rep['log'], fig_dir,
                              prefix=f'4b1_{method}')

    # Filter summary to yolov5_ffm detector for main plots
    ffm_summary = {k: v for k, v in summary.items()}
    plot_rmse_bars(ffm_summary, fig_dir)
    plot_track_continuity(ffm_summary, fig_dir)
    plot_4b1_metrics_table(ffm_summary, fig_dir, raw_dir)

    # Save JSON
    json_summary = {}
    for k, v in summary.items():
        json_summary[f'{k[0]}_{k[1]}_{k[2]}'] = v
    with open(os.path.join(raw_dir, 'summary_4b1.json'), 'w') as f:
        json.dump(json_summary, f, indent=2)

    print(f'  4B-1 complete: {count} runs')
    return summary


# =====================================================================
#  Exp 4B-2: Multi-UAV Fusion
# =====================================================================

def _run_multi_fusion(occ_mode, fusion_method, seed, max_steps=500):
    rng = np.random.RandomState(seed)
    env = SimEnvironment(seed=seed, n_drones=3)

    target = TargetMotionModel(
        init_pos=np.array([60.0, 50.0]),
        init_vel=np.array([0.3, 0.1]),
        mode='linear', speed=0.5, seed=seed + 200)
    schedule = OcclusionSchedule(mode=occ_mode)

    follow_dist = 6.0
    angles_offset = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    target_pos_init = np.array([60.0, 50.0])
    uav_positions = [
        target_pos_init + follow_dist * np.array(
            [np.cos(a), np.sin(a)]) for a in angles_offset
    ]
    uav_headings = [float(np.arctan2(target_pos_init[1] - p[1],
                                      target_pos_init[0] - p[0]))
                    for p in uav_positions]

    kfs = [ConfidenceModulatedKalmanFilter(dt=DT) for _ in range(3)]
    fusion = MultiUAVFusion(method=fusion_method)

    initialized = [False] * 3
    fusion_metrics_list = []

    for step in range(max_steps):
        target.step()
        true_pos = target.get_pos()

        estimates = []
        for i in range(3):
            diff = true_pos - uav_positions[i]
            dist = np.linalg.norm(diff)
            if dist > 0.1:
                uav_headings[i] = np.arctan2(diff[1], diff[0])
                if dist > follow_dist:
                    move = min(MAX_SPEED * DT, dist - follow_dist + 0.5)
                    uav_positions[i] = uav_positions[i] + (diff / dist) * move

            obs = generate_observation(
                uav_positions[i], uav_headings[i], true_pos,
                env.obstacles, 'yolov5_ffm', schedule, step, i, rng)

            if obs.detected and not initialized[i]:
                kfs[i].init_state(obs.target_meas_pos)
                initialized[i] = True

            if initialized[i]:
                kfs[i].predict()
                if obs.detected:
                    kfs[i].update(obs.target_meas_pos, obs.confidence,
                                  1.0 - obs.visibility,
                                  obs.depth_uncertainty,
                                  obs.measurement_cov)
                else:
                    kfs[i].predict_only()

                if kfs[i].track_state in ('TRACKED', 'OCCLUDED_TRACK'):
                    estimates.append({
                        'x': kfs[i].get_position(),
                        'P': kfs[i].get_position_cov(),
                        'confidence': kfs[i].confidence,
                        'uav_id': i,
                    })

        if len(estimates) >= 2:
            m = fusion.fuse_with_metrics(estimates, true_pos)
            fusion_metrics_list.append(m)

    if fusion_metrics_list:
        avg = {k: float(np.nanmean([m[k] for m in fusion_metrics_list]))
               for k in fusion_metrics_list[0]}
    else:
        avg = {'pre_fusion_rmse': float('nan'),
               'post_fusion_rmse': float('nan'),
               'cov_trace': float('nan'),
               'max_eigenvalue': float('nan')}
    return avg


def run_exp4b2():
    print('=== Exp 4B-2: Multi-UAV Fusion ===')
    raw_dir = os.path.join(OUT_DIR, 'raw')
    fig_dir = os.path.join(OUT_DIR, 'figures')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    occlusions = ['no_occlusion', 'partial_occlusion',
                   'full_occlusion_interval']
    fusion_methods = ['single_best', 'naive_average',
                       'info_filter', 'covariance_intersection']
    seeds = [42, 43, 44, 45, 46]

    fusion_summary = {}
    count = 0

    for fm in fusion_methods:
        seed_results = []
        for occ in occlusions:
            for seed in seeds:
                r = _run_multi_fusion(occ, fm, seed)
                seed_results.append(r)
                count += 1
        avg = {k: float(np.nanmean([r[k] for r in seed_results]))
               for k in seed_results[0]}
        fusion_summary[fm] = avg
        print(f'  {fm}: post_rmse={avg["post_fusion_rmse"]:.3f}')

    plot_fusion_bars(fusion_summary, fig_dir)
    plot_4b2_metrics_table(fusion_summary, fig_dir, raw_dir)

    with open(os.path.join(raw_dir, 'summary_4b2.json'), 'w') as f:
        json.dump(fusion_summary, f, indent=2)

    print(f'  4B-2 complete: {count} runs')
    return fusion_summary


# =====================================================================
#  Exp 4B-3: Encirclement
# =====================================================================

def _run_encirclement(motion, seed, max_steps=600):
    rng = np.random.RandomState(seed)
    env = SimEnvironment(seed=seed, n_drones=3)

    target = TargetMotionModel(
        init_pos=np.array([65.0, 50.0]),
        init_vel=np.array([0.3, 0.2]) if motion != 'static' else None,
        mode=motion, speed=0.6, seed=seed + 400)
    schedule = OcclusionSchedule(mode='intermittent_occlusion')

    kfs = [ConfidenceModulatedKalmanFilter(dt=DT) for _ in range(3)]
    fsms = [TargetTrackingFSM(uav_id=i) for i in range(3)]
    broadcast = TargetBroadcast(seed=seed + 500)
    fusion = MultiUAVFusion(method='info_filter')
    enc = EncirclementSlots(n_slots=3, r_min=8.0, k_p=2.0)
    reassocs = [ReassociationModule() for _ in range(3)]

    target_pos_init = np.array([65.0, 50.0])
    follow_dist_init = 7.0
    angles_init = [np.pi, np.pi * 2 / 3, -np.pi * 2 / 3]
    uav_positions = [
        target_pos_init + follow_dist_init * np.array(
            [np.cos(a), np.sin(a)]) for a in angles_init
    ]
    uav_headings = [float(np.arctan2(target_pos_init[1] - p[1],
                                      target_pos_init[0] - p[0]))
                    for p in uav_positions]
    initialized = [False] * 3

    slots = [None, None, None]
    r_enc = 0.0
    formation_step = -1
    errors = []
    snapshots = {}

    for step in range(max_steps):
        target.step()
        true_pos = target.get_pos()

        estimates = []
        n_tracking = sum(1 for kf in kfs
                          if kf.initialized and kf.track_state != 'LOST')

        for i in range(3):
            diff = true_pos - uav_positions[i]
            dist = np.linalg.norm(diff)
            if dist > 0.1:
                uav_headings[i] = np.arctan2(diff[1], diff[0])

            obs = generate_observation(
                uav_positions[i], uav_headings[i], true_pos,
                env.obstacles, 'yolov5_ffm', schedule, step, i, rng)

            if obs.detected and not initialized[i]:
                kfs[i].init_state(obs.target_meas_pos)
                initialized[i] = True

            if initialized[i]:
                kfs[i].predict()
                if obs.detected:
                    if kfs[i].track_state in ('LOST', 'OCCLUDED_TRACK'):
                        ok, _ = reassocs[i].check_reacquisition(
                            kfs[i], obs.target_meas_pos, step)
                        if ok or kfs[i].track_state == 'OCCLUDED_TRACK':
                            kfs[i].update(obs.target_meas_pos, obs.confidence,
                                          1.0 - obs.visibility,
                                          obs.depth_uncertainty,
                                          obs.measurement_cov)
                    else:
                        kfs[i].update(obs.target_meas_pos, obs.confidence,
                                      1.0 - obs.visibility,
                                      obs.depth_uncertainty,
                                      obs.measurement_cov)
                else:
                    kfs[i].predict_only()

                if kfs[i].track_state in ('TRACKED', 'OCCLUDED_TRACK'):
                    estimates.append({
                        'x': kfs[i].get_position(),
                        'P': kfs[i].get_position_cov(),
                        'confidence': kfs[i].confidence,
                        'uav_id': i,
                    })

                broadcast.broadcast(i, uav_positions[i], step,
                                     kfs[i].get_state_dict(),
                                     uav_positions)

        # Receive broadcasts
        for i in range(3):
            packets = broadcast.receive(i, step)
            if packets and not initialized[i]:
                p = packets[-1]
                pos_prop, _ = TargetBroadcast.propagate_to_current(p, step)
                kfs[i].init_state(pos_prop)
                initialized[i] = True
                fsms[i].receive_broadcast(step)

        # Fusion
        x_fused, P_fused = None, None
        if len(estimates) >= 2:
            x_fused, P_fused = fusion.fuse(estimates)
        elif len(estimates) == 1:
            x_fused = estimates[0]['x']
            P_fused = estimates[0]['P']

        if x_fused is not None:
            P_xy = P_fused if P_fused.shape == (2, 2) else P_fused[:2, :2]
            slots, r_enc = enc.compute_slots(
                x_fused, P_xy, env.obstacles)
            err = float(np.linalg.norm(x_fused - true_pos))
            errors.append(err)

        # FSM + move
        at_slots = [False] * 3
        for i in range(3):
            if slots[i] is not None:
                at_slots[i] = np.linalg.norm(
                    uav_positions[i] - slots[i]) < 5.0
        all_at = all(at_slots) and all(s is not None for s in slots)

        for i in range(3):
            fsms[i].transition(
                step, kfs[i].initialized and kfs[i].track_state == 'TRACKED',
                kfs[i].track_state if kfs[i].initialized else 'INIT',
                n_tracking, at_slots[i], all_at)

            target_wp = fsms[i].get_action(
                x_fused, slots[i] if slots[i] is not None else None,
                uav_positions[i])

            if target_wp is not None:
                diff = target_wp - uav_positions[i]
                dist = np.linalg.norm(diff)
                if dist > 0.5:
                    direction = diff / dist
                    move = min(MAX_SPEED * DT, dist)
                    new_pos = uav_positions[i] + direction * move
                    collides = False
                    for o in env.obstacles:
                        if np.hypot(new_pos[0] - o.x,
                                     new_pos[1] - o.y) < o.r + 0.5:
                            collides = True
                            break
                    if not collides and \
                       0.5 < new_pos[0] < WORLD_SIZE - 0.5 and \
                       0.5 < new_pos[1] < WORLD_SIZE - 0.5:
                        uav_positions[i] = new_pos
                    uav_headings[i] = np.arctan2(direction[1], direction[0])

        if all_at and formation_step < 0:
            formation_step = step

        if step in (50, 150, 250, 350, 450, 550):
            snapshots[step] = {
                'step': step,
                'target_pos': true_pos.copy(),
                'target_est': x_fused.copy() if x_fused is not None else None,
                'uav_positions': [p.copy() for p in uav_positions],
                'slots': [s.copy() if s is not None else None for s in slots],
                'r_enc': r_enc,
                'obstacles': env.obstacles,
            }

    # Metrics
    enc_metrics = EncirclementSlots.compute_metrics(
        uav_positions, slots, true_pos)
    track_rmse = float(np.sqrt(np.mean(np.array(errors) ** 2))) \
        if errors else float('nan')

    return {
        'formation_time': formation_step if formation_step > 0 else max_steps,
        'phase_uniformity': enc_metrics['phase_uniformity'],
        'min_inter_distance': enc_metrics['min_inter_uav_distance'],
        'mean_enc_radius': enc_metrics['mean_encirclement_radius'],
        'slot_valid_rate': enc_metrics['slot_valid_rate'],
        'track_rmse': track_rmse,
        'encirclement_success': formation_step > 0,
        'fsm_histories': [f.state_history for f in fsms],
        'snapshots': snapshots,
    }


def run_exp4b3():
    print('=== Exp 4B-3: Encirclement ===')
    raw_dir = os.path.join(OUT_DIR, 'raw')
    fig_dir = os.path.join(OUT_DIR, 'figures')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    motions = ['static', 'linear', 'random_turn']
    seeds = [42, 43, 44, 45, 46]

    enc_summary = {}
    for motion in motions:
        seed_results = []
        for seed in seeds:
            r = _run_encirclement(motion, seed)
            seed_results.append(r)
        avg = {
            'formation_time': float(np.mean([r['formation_time'] for r in seed_results])),
            'phase_uniformity': float(np.mean([r['phase_uniformity'] for r in seed_results])),
            'min_inter_distance': float(np.mean([r['min_inter_distance'] for r in seed_results])),
            'track_rmse': float(np.nanmean([r['track_rmse'] for r in seed_results])),
            'encirclement_success': any(r['encirclement_success'] for r in seed_results),
        }
        enc_summary[motion] = avg
        print(f'  {motion}: form_time={avg["formation_time"]:.0f}, '
              f'phase={avg["phase_uniformity"]:.3f}')

    # Plot representative run snapshots and FSM timeline
    rep = _run_encirclement('linear', 42)
    for step_key, snap in rep['snapshots'].items():
        plot_encirclement_snapshot(snap, fig_dir, prefix=f'enc_s{step_key}')

    plot_fsm_timeline(rep['fsm_histories'], 600, fig_dir, prefix='4b3_fsm')
    plot_4b3_metrics_table(enc_summary, fig_dir, raw_dir)

    with open(os.path.join(raw_dir, 'summary_4b3.json'), 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items()
                        if not isinstance(vv, list)}
                    for k, v in enc_summary.items()}, f, indent=2)

    print(f'  4B-3 complete: {len(motions) * len(seeds)} runs')
    return enc_summary


# =====================================================================
#  Main
# =====================================================================

def run_exp4b():
    setup_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    r1 = run_exp4b1()
    r2 = run_exp4b2()
    r3 = run_exp4b3()

    elapsed = time.time() - t0
    print(f'\n=== Exp 4B complete in {elapsed:.1f}s ===')
    print(f'  Output → {OUT_DIR}/')
    return r1, r2, r3


if __name__ == '__main__':
    run_exp4b()
