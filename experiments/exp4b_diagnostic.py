"""
Exp 4B 诊断脚本: CM-KF vs Standard KF 详细对比 + 融合显著性 + 围捕相位分析
生成补充图表和审计文档
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import DT, MAX_SPEED, SENSOR_RANGE, WORLD_SIZE
from simulation.sim_env import SimEnvironment
from simulation.target_motion import TargetMotionModel
from simulation.occlusion_schedule import OcclusionSchedule
from tracking.visual_observation import generate_observation
from tracking.confidence_modulated_kf import ConfidenceModulatedKalmanFilter
from tracking.reassociation import ReassociationModule
from tracking.multi_uav_fusion import MultiUAVFusion
from tracking.encirclement_slots import EncirclementSlots
from coordination.target_tracking_fsm import TargetTrackingFSM
from coordination.target_broadcast import TargetBroadcast
from experiments.plotting.common import setup_style, save_figure

setup_style()
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'exp4b_tracking')
FIG = os.path.join(OUT, 'figures')
RAW = os.path.join(OUT, 'raw')
os.makedirs(FIG, exist_ok=True)
os.makedirs(RAW, exist_ok=True)


def run_single_diagnostic(method_name, alpha_o, alpha_d, seed=42):
    rng = np.random.RandomState(seed)
    env = SimEnvironment(seed=seed, n_drones=1)
    target = TargetMotionModel(
        init_pos=np.array([60.0, 50.0]),
        init_vel=np.array([0.5, 0.3]),
        mode='linear', speed=0.8, seed=seed + 100)
    schedule = OcclusionSchedule(mode='intermittent_occlusion')
    kf = ConfidenceModulatedKalmanFilter(dt=DT, alpha_o=alpha_o, alpha_d=alpha_d)
    reassoc = ReassociationModule()
    uav_pos = np.array([54.0, 50.0])
    initialized = False

    log = {
        'R_trace': [], 'K_norm': [], 'P_trace': [],
        'error': [], 'confidence': [], 'occ_level': [],
        'detected': [], 'step': [],
    }

    for step in range(500):
        target.step()
        true_pos = target.get_pos()
        diff = true_pos - uav_pos
        dist = np.linalg.norm(diff)
        if dist > 7.0:
            direction = diff / max(dist, 0.01)
            move = min(MAX_SPEED * DT, dist - 6.0)
            new_pos = uav_pos + direction * move
            if 1.0 < new_pos[0] < 99.0 and 1.0 < new_pos[1] < 99.0:
                uav_pos = new_pos
            uav_heading = np.arctan2(direction[1], direction[0])
        elif dist > 0.1:
            uav_heading = np.arctan2(diff[1], diff[0])
        else:
            uav_heading = 0.0

        obs = generate_observation(uav_pos, uav_heading, true_pos,
                                   env.obstacles, 'yolov5_ffm', schedule, step, 0, rng)

        if obs.detected and not initialized:
            kf.init_state(obs.target_meas_pos)
            initialized = True

        R_trace_val = np.nan
        K_norm_val = np.nan

        if initialized:
            kf.predict()
            if obs.detected:
                if kf.track_state in ('LOST', 'OCCLUDED_TRACK'):
                    ok, _ = reassoc.check_reacquisition(kf, obs.target_meas_pos, step)
                    if ok or kf.track_state == 'OCCLUDED_TRACK':
                        eps = 1e-6
                        lc = 1.0 / (obs.confidence ** 2 + eps)
                        lo = 1.0 + alpha_o * (1.0 - obs.visibility)
                        ld = 1.0 + alpha_d * obs.depth_uncertainty ** 2
                        R = lc * obs.measurement_cov + lo * np.eye(2) * (2.0 * (1.0 - obs.visibility)) ** 2 + ld * np.eye(2) * obs.depth_uncertainty ** 2
                        R_trace_val = float(np.trace(R))
                        S = kf.H @ kf.P_pred @ kf.H.T + R
                        K = kf.P_pred @ kf.H.T @ np.linalg.inv(S)
                        K_norm_val = float(np.linalg.norm(K))
                        kf.update(obs.target_meas_pos, obs.confidence,
                                  1.0 - obs.visibility, obs.depth_uncertainty,
                                  obs.measurement_cov)
                    else:
                        kf.predict_only()
                else:
                    eps = 1e-6
                    lc = 1.0 / (obs.confidence ** 2 + eps)
                    lo = 1.0 + alpha_o * (1.0 - obs.visibility)
                    ld = 1.0 + alpha_d * obs.depth_uncertainty ** 2
                    R = lc * obs.measurement_cov + lo * np.eye(2) * (2.0 * (1.0 - obs.visibility)) ** 2 + ld * np.eye(2) * obs.depth_uncertainty ** 2
                    R_trace_val = float(np.trace(R))
                    S = kf.H @ kf.P_pred @ kf.H.T + R
                    K = kf.P_pred @ kf.H.T @ np.linalg.inv(S)
                    K_norm_val = float(np.linalg.norm(K))
                    kf.update(obs.target_meas_pos, obs.confidence,
                              1.0 - obs.visibility, obs.depth_uncertainty,
                              obs.measurement_cov)
            else:
                if kf.track_state != 'LOST':
                    reassoc.mark_lost(step)
                kf.predict_only()

            est = kf.get_position()
            err = float(np.linalg.norm(est - true_pos))
        else:
            err = np.nan

        log['R_trace'].append(R_trace_val)
        log['K_norm'].append(K_norm_val)
        log['P_trace'].append(float(np.trace(kf.P[:2, :2])) if initialized else np.nan)
        log['error'].append(err)
        log['confidence'].append(obs.confidence)
        log['occ_level'].append(schedule.get_level(step))
        log['detected'].append(obs.detected)
        log['step'].append(step)

    return log


def plot_diagnostic_comparison():
    print('Generating CM-KF vs Standard KF diagnostic plots...')
    log_std = run_single_diagnostic('standard_kf', 0.0, 0.0)
    log_cm = run_single_diagnostic('cm_kf', 2.0, 1.5)

    steps = log_std['step']
    occ_mask = [l != 'none' for l in log_std['occ_level']]

    # R trace
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    ax.plot(steps, log_std['R_trace'], 'b-', lw=0.8, label='Standard KF', alpha=0.7)
    ax.plot(steps, log_cm['R_trace'], 'r-', lw=0.8, label='CM-KF', alpha=0.7)
    for i, is_occ in enumerate(occ_mask):
        if is_occ:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='orange')
    ax.set_ylabel('R Trace')
    ax.set_title('Measurement Noise R Trace Comparison', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # K gain norm
    ax = axes[1]
    ax.plot(steps, log_std['K_norm'], 'b-', lw=0.8, label='Standard KF', alpha=0.7)
    ax.plot(steps, log_cm['K_norm'], 'r-', lw=0.8, label='CM-KF', alpha=0.7)
    for i, is_occ in enumerate(occ_mask):
        if is_occ:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='orange')
    ax.set_ylabel('||K||')
    ax.set_title('Kalman Gain Norm Comparison', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # P trace
    ax = axes[2]
    ax.plot(steps, log_std['P_trace'], 'b-', lw=0.8, label='Standard KF', alpha=0.7)
    ax.plot(steps, log_cm['P_trace'], 'r-', lw=0.8, label='CM-KF', alpha=0.7)
    for i, is_occ in enumerate(occ_mask):
        if is_occ:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='orange')
    ax.set_ylabel('P Trace (pos)')
    ax.set_xlabel('Step')
    ax.set_title('State Covariance P Trace Comparison', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save_figure(fig, FIG, 'fig_4b1_R_trace.png')

    # Occlusion period error comparison
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax = axes2[0]
    ax.plot(steps, log_std['error'], 'b-', lw=0.8, label='Standard KF', alpha=0.7)
    ax.plot(steps, log_cm['error'], 'r-', lw=0.8, label='CM-KF', alpha=0.7)
    for i, is_occ in enumerate(occ_mask):
        if is_occ:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='orange')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Error: Standard KF vs CM-KF', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes2[1]
    det_std = [1.0 if d else 0.0 for d in log_std['detected']]
    ax.fill_between(steps, det_std, alpha=0.3, color='green', label='Detected')
    occ_float = [1.0 if o else 0.0 for o in occ_mask]
    ax.fill_between(steps, occ_float, alpha=0.3, color='orange', label='Occluded')
    ax.set_ylabel('Status')
    ax.set_xlabel('Step')
    ax.set_title('Detection & Occlusion Status', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig2.tight_layout()
    save_figure(fig2, FIG, 'fig_4b1_occlusion_period_error.png')

    # Compute detailed metrics
    err_std = np.array(log_std['error'])
    err_cm = np.array(log_cm['error'])
    occ_arr = np.array(occ_mask)
    valid_std = ~np.isnan(err_std)
    valid_cm = ~np.isnan(err_cm)

    metrics = {}
    for name, err, valid in [('standard_kf', err_std, valid_std), ('cm_kf', err_cm, valid_cm)]:
        vis_idx = valid & ~occ_arr
        occ_idx = valid & occ_arr
        metrics[name] = {
            'rmse_all': float(np.sqrt(np.mean(err[valid] ** 2))),
            'rmse_visible': float(np.sqrt(np.mean(err[vis_idx] ** 2))) if vis_idx.any() else float('nan'),
            'rmse_occlusion': float(np.sqrt(np.mean(err[occ_idx] ** 2))) if occ_idx.any() else float('nan'),
            'mean_P_trace': float(np.nanmean(log_std['P_trace'] if name == 'standard_kf' else log_cm['P_trace'])),
            'n_visible': int(vis_idx.sum()),
            'n_occlusion': int(occ_idx.sum()),
        }
    return metrics


def plot_fusion_per_seed():
    print('Generating fusion per-seed and timeline plots...')
    from config import DT, MAX_SPEED

    occlusions = ['no_occlusion', 'partial_occlusion', 'full_occlusion_interval']
    fusion_methods = ['single_best', 'naive_average', 'info_filter', 'covariance_intersection']
    labels = ['Single Best', 'Naive Avg', 'Info Filter', 'Cov Intersect']
    seeds = [42, 43, 44, 45, 46]

    # Per-seed RMSE
    per_seed = {fm: [] for fm in fusion_methods}
    for fm in fusion_methods:
        for seed in seeds:
            seed_rmses = []
            for occ in occlusions:
                rng = np.random.RandomState(seed)
                env = SimEnvironment(seed=seed, n_drones=3)
                target = TargetMotionModel(
                    init_pos=np.array([60.0, 50.0]),
                    init_vel=np.array([0.3, 0.1]),
                    mode='linear', speed=0.5, seed=seed + 200)
                schedule = OcclusionSchedule(mode=occ)
                follow_dist = 6.0
                angles_offset = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
                tp_init = np.array([60.0, 50.0])
                uav_positions = [tp_init + follow_dist * np.array([np.cos(a), np.sin(a)]) for a in angles_offset]
                uav_headings = [float(np.arctan2(tp_init[1] - p[1], tp_init[0] - p[0])) for p in uav_positions]
                kfs = [ConfidenceModulatedKalmanFilter(dt=DT) for _ in range(3)]
                fusion = MultiUAVFusion(method=fm)
                initialized = [False] * 3
                step_errors = []

                for step in range(300):
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
                        obs = generate_observation(uav_positions[i], uav_headings[i], true_pos,
                                                   env.obstacles, 'yolov5_ffm', schedule, step, i, rng)
                        if obs.detected and not initialized[i]:
                            kfs[i].init_state(obs.target_meas_pos)
                            initialized[i] = True
                        if initialized[i]:
                            kfs[i].predict()
                            if obs.detected:
                                kfs[i].update(obs.target_meas_pos, obs.confidence,
                                              1.0 - obs.visibility, obs.depth_uncertainty,
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
                        step_errors.append(m['post_fusion_rmse'])

                if step_errors:
                    seed_rmses.append(float(np.mean(step_errors)))
            per_seed[fm].append(float(np.mean(seed_rmses)) if seed_rmses else float('nan'))

    # Per-seed bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(seeds))
    bar_w = 0.2
    colors = ['#90CAF9', '#42A5F5', '#1565C0', '#0D47A1']
    for mi, fm in enumerate(fusion_methods):
        ax.bar(x + mi * bar_w, per_seed[fm], bar_w, label=labels[mi], color=colors[mi])
    ax.set_xticks(x + 1.5 * bar_w)
    ax.set_xticklabels([f'Seed {s}' for s in seeds], fontsize=9)
    ax.set_ylabel('Mean Post-Fusion RMSE (m)')
    ax.set_title('Fusion RMSE per Seed (averaged over occlusion modes)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    save_figure(fig, FIG, 'fig_4b2_per_seed_rmse.png')

    # Error timeline (single representative run)
    rng = np.random.RandomState(42)
    env = SimEnvironment(seed=42, n_drones=3)
    target = TargetMotionModel(init_pos=np.array([60.0, 50.0]),
                               init_vel=np.array([0.3, 0.1]),
                               mode='linear', speed=0.5, seed=242)
    schedule = OcclusionSchedule(mode='intermittent_occlusion')
    tp_init = np.array([60.0, 50.0])
    follow_dist = 6.0
    angles_offset = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    uav_positions = [tp_init + follow_dist * np.array([np.cos(a), np.sin(a)]) for a in angles_offset]
    uav_headings = [float(np.arctan2(tp_init[1] - p[1], tp_init[0] - p[0])) for p in uav_positions]

    timelines = {fm: [] for fm in fusion_methods}
    for fm in fusion_methods:
        rng2 = np.random.RandomState(42)
        target2 = TargetMotionModel(init_pos=np.array([60.0, 50.0]),
                                    init_vel=np.array([0.3, 0.1]),
                                    mode='linear', speed=0.5, seed=242)
        kfs = [ConfidenceModulatedKalmanFilter(dt=DT) for _ in range(3)]
        fusion = MultiUAVFusion(method=fm)
        initialized2 = [False] * 3
        uav_p = [tp_init + follow_dist * np.array([np.cos(a), np.sin(a)]) for a in angles_offset]
        uav_h = [float(np.arctan2(tp_init[1] - p[1], tp_init[0] - p[0])) for p in uav_p]

        for step in range(300):
            target2.step()
            true_pos = target2.get_pos()
            estimates = []
            for i in range(3):
                diff = true_pos - uav_p[i]
                dist = np.linalg.norm(diff)
                if dist > 0.1:
                    uav_h[i] = np.arctan2(diff[1], diff[0])
                    if dist > follow_dist:
                        move = min(MAX_SPEED * DT, dist - follow_dist + 0.5)
                        uav_p[i] = uav_p[i] + (diff / dist) * move
                obs = generate_observation(uav_p[i], uav_h[i], true_pos,
                                           env.obstacles, 'yolov5_ffm', schedule, step, i, rng2)
                if obs.detected and not initialized2[i]:
                    kfs[i].init_state(obs.target_meas_pos)
                    initialized2[i] = True
                if initialized2[i]:
                    kfs[i].predict()
                    if obs.detected:
                        kfs[i].update(obs.target_meas_pos, obs.confidence,
                                      1.0 - obs.visibility, obs.depth_uncertainty,
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
                x_f, _ = fusion.fuse(estimates)
                timelines[fm].append(float(np.linalg.norm(x_f - true_pos)))
            else:
                timelines[fm].append(np.nan)

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    for mi, fm in enumerate(fusion_methods):
        ax2.plot(range(len(timelines[fm])), timelines[fm], lw=0.8, label=labels[mi], color=colors[mi])
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Fusion Error (m)')
    ax2.set_title('Fusion Error Timeline (intermittent occlusion, seed=42)', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)
    fig2.tight_layout()
    save_figure(fig2, FIG, 'fig_4b2_error_timeline.png')

    # Compute std for significance
    stats = {}
    for fm in fusion_methods:
        vals = per_seed[fm]
        stats[fm] = {'mean': float(np.nanmean(vals)), 'std': float(np.nanstd(vals)),
                     'values': vals}
    return stats


def plot_phase_analysis():
    print('Generating phase uniformity analysis...')
    from coordination.target_tracking_fsm import TargetTrackingFSM
    from coordination.target_broadcast import TargetBroadcast

    target_pos_init = np.array([65.0, 50.0])
    follow_dist_init = 7.0
    angles_init = [np.pi, np.pi * 2 / 3, -np.pi * 2 / 3]

    motions = ['static', 'linear', 'random_turn']
    motion_logs = {}

    for motion in motions:
        rng = np.random.RandomState(42)
        env = SimEnvironment(seed=42, n_drones=3)
        target = TargetMotionModel(
            init_pos=np.array([65.0, 50.0]),
            init_vel=np.array([0.3, 0.2]) if motion != 'static' else None,
            mode=motion, speed=0.6, seed=442)
        schedule = OcclusionSchedule(mode='intermittent_occlusion')
        kfs = [ConfidenceModulatedKalmanFilter(dt=DT) for _ in range(3)]
        fsms = [TargetTrackingFSM(uav_id=i) for i in range(3)]
        broadcast = TargetBroadcast(seed=542)
        fusion = MultiUAVFusion(method='info_filter')
        enc = EncirclementSlots(n_slots=3, r_min=8.0, k_p=2.0)
        reassocs = [ReassociationModule() for _ in range(3)]

        uav_positions = [target_pos_init + follow_dist_init * np.array([np.cos(a), np.sin(a)]) for a in angles_init]
        uav_headings = [float(np.arctan2(target_pos_init[1] - p[1], target_pos_init[0] - p[0])) for p in uav_positions]
        initialized = [False] * 3
        slots = [None, None, None]
        r_enc = 0.0

        phase_log = []
        dist_to_slot_log = []
        angle_log = []

        for step in range(600):
            target.step()
            true_pos = target.get_pos()
            estimates = []
            n_tracking = sum(1 for kf in kfs if kf.initialized and kf.track_state != 'LOST')

            for i in range(3):
                diff = true_pos - uav_positions[i]
                dist = np.linalg.norm(diff)
                if dist > 0.1:
                    uav_headings[i] = np.arctan2(diff[1], diff[0])
                obs = generate_observation(uav_positions[i], uav_headings[i], true_pos,
                                           env.obstacles, 'yolov5_ffm', schedule, step, i, rng)
                if obs.detected and not initialized[i]:
                    kfs[i].init_state(obs.target_meas_pos)
                    initialized[i] = True
                if initialized[i]:
                    kfs[i].predict()
                    if obs.detected:
                        if kfs[i].track_state in ('LOST', 'OCCLUDED_TRACK'):
                            ok, _ = reassocs[i].check_reacquisition(kfs[i], obs.target_meas_pos, step)
                            if ok or kfs[i].track_state == 'OCCLUDED_TRACK':
                                kfs[i].update(obs.target_meas_pos, obs.confidence,
                                              1.0 - obs.visibility, obs.depth_uncertainty,
                                              obs.measurement_cov)
                        else:
                            kfs[i].update(obs.target_meas_pos, obs.confidence,
                                          1.0 - obs.visibility, obs.depth_uncertainty,
                                          obs.measurement_cov)
                    else:
                        kfs[i].predict_only()
                    if kfs[i].track_state in ('TRACKED', 'OCCLUDED_TRACK'):
                        estimates.append({
                            'x': kfs[i].get_position(), 'P': kfs[i].get_position_cov(),
                            'confidence': kfs[i].confidence, 'uav_id': i,
                        })
                    broadcast.broadcast(i, uav_positions[i], step, kfs[i].get_state_dict(), uav_positions)

            for i in range(3):
                packets = broadcast.receive(i, step)
                if packets and not initialized[i]:
                    p = packets[-1]
                    pos_prop, _ = TargetBroadcast.propagate_to_current(p, step)
                    kfs[i].init_state(pos_prop)
                    initialized[i] = True
                    fsms[i].receive_broadcast(step)

            x_fused, P_fused = None, None
            if len(estimates) >= 2:
                x_fused, P_fused = fusion.fuse(estimates)
            elif len(estimates) == 1:
                x_fused = estimates[0]['x']
                P_fused = estimates[0]['P']

            if x_fused is not None:
                P_xy = P_fused if P_fused.shape == (2, 2) else P_fused[:2, :2]
                slots, r_enc = enc.compute_slots(x_fused, P_xy, env.obstacles)

            at_slots = [False] * 3
            for i in range(3):
                if slots[i] is not None:
                    at_slots[i] = np.linalg.norm(uav_positions[i] - slots[i]) < 5.0
            all_at = all(at_slots) and all(s is not None for s in slots)

            for i in range(3):
                fsms[i].transition(step, kfs[i].initialized and kfs[i].track_state == 'TRACKED',
                                   kfs[i].track_state if kfs[i].initialized else 'INIT',
                                   n_tracking, at_slots[i], all_at)
                target_wp = fsms[i].get_action(x_fused, slots[i] if slots[i] is not None else None, uav_positions[i])
                if target_wp is not None:
                    diff = target_wp - uav_positions[i]
                    dist = np.linalg.norm(diff)
                    if dist > 0.5:
                        direction = diff / dist
                        move = min(MAX_SPEED * DT, dist)
                        new_pos = uav_positions[i] + direction * move
                        collides = False
                        for o in env.obstacles:
                            if np.hypot(new_pos[0] - o.x, new_pos[1] - o.y) < o.r + 0.5:
                                collides = True
                                break
                        if not collides and 0.5 < new_pos[0] < WORLD_SIZE - 0.5 and 0.5 < new_pos[1] < WORLD_SIZE - 0.5:
                            uav_positions[i] = new_pos
                        uav_headings[i] = np.arctan2(direction[1], direction[0])

            # Log angles and distances
            angles = []
            dists_to_slot = []
            for i in range(3):
                diff_t = uav_positions[i] - true_pos
                angles.append(np.arctan2(diff_t[1], diff_t[0]))
                if slots[i] is not None:
                    dists_to_slot.append(float(np.linalg.norm(uav_positions[i] - slots[i])))
                else:
                    dists_to_slot.append(np.nan)

            angles_sorted = sorted(angles)
            deltas = []
            for idx in range(3):
                j = (idx + 1) % 3
                d = angles_sorted[j] - angles_sorted[idx]
                if d < 0:
                    d += 2 * np.pi
                deltas.append(d)
            ideal = 2 * np.pi / 3
            phase_u = float(np.std([d - ideal for d in deltas]))

            phase_log.append(phase_u)
            dist_to_slot_log.append(dists_to_slot)
            angle_log.append(angles)

        motion_logs[motion] = {
            'phase': phase_log,
            'dist_to_slot': dist_to_slot_log,
            'angles': angle_log,
        }

    # Plot phase angles over time
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for mi, motion in enumerate(motions):
        ax = axes[mi]
        ax.plot(range(600), motion_logs[motion]['phase'], 'k-', lw=0.8)
        ax.set_ylabel('Phase Uniformity')
        ax.set_title(f'{motion}: Phase Uniformity over Time', fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color='green', ls='--', lw=0.5, alpha=0.5)
    axes[2].set_xlabel('Step')
    fig.tight_layout()
    save_figure(fig, FIG, 'fig_4b3_phase_angles_over_time.png')

    # Plot distance to slot
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    colors_d = ['#E53935', '#43A047', '#1E88E5']
    for mi, motion in enumerate(motions):
        ax = axes2[mi]
        dists = motion_logs[motion]['dist_to_slot']
        for i in range(3):
            d_i = [d[i] for d in dists]
            ax.plot(range(600), d_i, lw=0.8, color=colors_d[i], label=f'UAV-{i}', alpha=0.7)
        ax.set_ylabel('Distance to Slot (m)')
        ax.set_title(f'{motion}: Distance to Assigned Slot', fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
    axes2[2].set_xlabel('Step')
    fig2.tight_layout()
    save_figure(fig2, FIG, 'fig_4b3_distance_to_slot.png')

    return motion_logs


if __name__ == '__main__':
    kf_metrics = plot_diagnostic_comparison()
    fusion_stats = plot_fusion_per_seed()
    phase_logs = plot_phase_analysis()

    print('\n=== KF Diagnostic Results ===')
    for name, m in kf_metrics.items():
        print(f'  {name}: rmse_all={m["rmse_all"]:.4f}, rmse_vis={m["rmse_visible"]:.4f}, '
              f'rmse_occ={m["rmse_occlusion"]:.4f}, P_trace={m["mean_P_trace"]:.4f}')
    print('\n=== Fusion Significance ===')
    for fm, s in fusion_stats.items():
        print(f'  {fm}: mean={s["mean"]:.4f} ± {s["std"]:.4f}')

    print('\nDiagnostic plots saved to', FIG)
