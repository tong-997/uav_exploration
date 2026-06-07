"""
视觉观测模型: 包装已有 detect_target() 并添加遮挡调度和测量协方差
"""
import numpy as np
from dataclasses import dataclass
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SENSOR_FOV, SENSOR_RANGE
from simulation.targets import detect_target, compute_occlusion_ratio


@dataclass
class VisualObservation:
    detected: bool
    uav_id: int
    step: int
    target_true_pos: np.ndarray
    target_meas_pos: np.ndarray  # None if not detected
    confidence: float
    visibility: float
    occlusion_level: str
    depth: float
    depth_uncertainty: float
    measurement_cov: np.ndarray  # 2x2
    detector_mode: str


def generate_observation(uav_pos, uav_heading, target_pos, obstacles,
                         detector_mode, occlusion_schedule, step, uav_id,
                         rng):
    diff = target_pos - uav_pos
    depth = float(np.linalg.norm(diff))

    occ_level = occlusion_schedule.get_level(step)
    sched_occ = occlusion_schedule.get_occlusion_value(step)

    if sched_occ >= 1.0:
        depth_unc = 0.02 * depth + 0.1
        r_val = (0.5 + depth * 0.03) ** 2
        return VisualObservation(
            detected=False, uav_id=uav_id, step=step,
            target_true_pos=target_pos.copy(),
            target_meas_pos=None,
            confidence=0.0, visibility=0.0,
            occlusion_level=occ_level, depth=depth,
            depth_uncertainty=depth_unc,
            measurement_cov=np.diag([r_val, r_val]),
            detector_mode=detector_mode,
        )

    detected, conf, err = detect_target(
        uav_pos, uav_heading, target_pos, obstacles,
        model_type=detector_mode, rng=rng)

    if sched_occ > 0 and detected:
        conf *= (1.0 - sched_occ)
        if detector_mode in ('yolov5s', 'yolov5_ffm') and conf < 0.5:
            detected = False
        err *= (1.0 + sched_occ * 2.0)

    base_occ = compute_occlusion_ratio(uav_pos, target_pos, obstacles)
    effective_occ = min(base_occ + sched_occ, 1.0)
    visibility = 1.0 - effective_occ

    depth_unc = 0.02 * depth + 0.1 + sched_occ * 0.5
    r_val = (0.5 + depth * 0.03) ** 2
    meas_cov = np.diag([r_val, r_val])

    meas_pos = None
    if detected:
        meas_pos = target_pos + err

    return VisualObservation(
        detected=detected, uav_id=uav_id, step=step,
        target_true_pos=target_pos.copy(),
        target_meas_pos=meas_pos,
        confidence=float(conf) if detected else 0.0,
        visibility=float(visibility),
        occlusion_level=occ_level, depth=depth,
        depth_uncertainty=float(depth_unc),
        measurement_cov=meas_cov,
        detector_mode=detector_mode,
    )
