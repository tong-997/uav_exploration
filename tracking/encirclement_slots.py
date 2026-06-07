"""
围捕槽位生成: 根据目标估计位置和不确定度分配围捕位置
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import WORLD_SIZE


class EncirclementSlots:

    def __init__(self, n_slots=3, r_min=8.0, k_p=2.0):
        self.n_slots = n_slots
        self.r_min = r_min
        self.k_p = k_p

    def compute_slots(self, target_pos, P_xy, obstacles=None):
        eigs = np.linalg.eigvalsh(P_xy)
        lambda_max = max(eigs.max(), 1e-6)
        r_enc = self.r_min + self.k_p * np.sqrt(lambda_max)

        slots = []
        for j in range(self.n_slots):
            theta = 2.0 * np.pi * j / self.n_slots
            slot = self._find_valid_slot(target_pos, r_enc, theta, obstacles)
            slots.append(slot)
        return slots, r_enc

    def _find_valid_slot(self, target_pos, r_enc, theta, obstacles):
        margin = 2.0
        for delta_deg in range(0, 180, 5):
            for sign in [0, 1, -1]:
                t = theta + np.deg2rad(delta_deg * sign)
                pos = target_pos + r_enc * np.array([np.cos(t), np.sin(t)])
                if pos[0] < margin or pos[0] > WORLD_SIZE - margin:
                    continue
                if pos[1] < margin or pos[1] > WORLD_SIZE - margin:
                    continue
                if obstacles and self._collides_obstacle(pos, obstacles):
                    continue
                return pos
        return target_pos + r_enc * np.array([np.cos(theta), np.sin(theta)])

    @staticmethod
    def _collides_obstacle(pos, obstacles):
        for o in obstacles:
            ox = o.x if hasattr(o, 'x') else o[0]
            oy = o.y if hasattr(o, 'y') else o[1]
            r = o.r if hasattr(o, 'r') else o[2]
            if np.hypot(pos[0] - ox, pos[1] - oy) < r + 1.5:
                return True
        return False

    @staticmethod
    def compute_metrics(uav_positions, slot_positions, target_pos):
        n = len(uav_positions)
        angles = []
        dists_to_target = []
        for pos in uav_positions:
            diff = pos - target_pos
            angles.append(np.arctan2(diff[1], diff[0]))
            dists_to_target.append(np.linalg.norm(diff))

        angles_sorted = sorted(angles)
        deltas = []
        for i in range(len(angles_sorted)):
            j = (i + 1) % len(angles_sorted)
            d = angles_sorted[j] - angles_sorted[i]
            if d < 0:
                d += 2 * np.pi
            deltas.append(d)
        ideal = 2.0 * np.pi / n
        phase_uniformity = float(np.std([d - ideal for d in deltas]))

        inter_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                inter_dists.append(np.linalg.norm(
                    np.asarray(uav_positions[i]) - np.asarray(uav_positions[j])))

        n_valid = sum(1 for s in slot_positions if s is not None)

        return {
            'phase_uniformity': phase_uniformity,
            'min_inter_uav_distance': float(min(inter_dists)) if inter_dists else 0.0,
            'mean_encirclement_radius': float(np.mean(dists_to_target)),
            'slot_valid_rate': n_valid / max(len(slot_positions), 1),
        }
