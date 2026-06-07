"""
目标重关联: 马氏距离门控判断新检测是否属于原目标
"""
import numpy as np


class ReassociationModule:

    def __init__(self, gate_threshold=9.21):
        self.gate_threshold = gate_threshold
        self.attempt_count = 0
        self.success_count = 0
        self.reacquisition_times = []
        self._lost_step = None

    def mark_lost(self, step):
        self._lost_step = step

    def check_reacquisition(self, kf, z, step):
        z_pred = kf.H @ kf.x
        S = kf.H @ kf.P @ kf.H.T + np.eye(2) * 0.5
        residual = z - z_pred

        try:
            D2 = float(residual.T @ np.linalg.inv(S) @ residual)
        except np.linalg.LinAlgError:
            D2 = float('inf')

        self.attempt_count += 1

        if D2 < self.gate_threshold:
            self.success_count += 1
            if self._lost_step is not None:
                self.reacquisition_times.append(step - self._lost_step)
                self._lost_step = None
            return True, D2
        return False, D2

    @property
    def success_rate(self):
        return self.success_count / max(self.attempt_count, 1)

    @property
    def mean_reacquisition_time(self):
        if self.reacquisition_times:
            return float(np.mean(self.reacquisition_times))
        return 0.0

    def get_metrics(self):
        return {
            'reassociation_attempt_count': self.attempt_count,
            'reassociation_success_count': self.success_count,
            'reassociation_success_rate': self.success_rate,
            'mean_reacquisition_time': self.mean_reacquisition_time,
            'lost_count': self.attempt_count - self.success_count,
        }
