"""
置信度调制卡尔曼滤波器 (CM-KF)
- 检测置信度、遮挡程度、深度不确定度动态调制测量噪声 R
- 支持 TRACKED / OCCLUDED_TRACK / LOST 状态
- alpha_o=0, alpha_d=0 退化为标准 KF
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DT


class ConfidenceModulatedKalmanFilter:

    def __init__(self, dt=None, process_noise_std=0.3,
                 alpha_o=2.0, alpha_d=1.5,
                 max_lost_frames=15, gamma=0.9,
                 min_track_confidence=0.1):
        self.dt = dt or DT
        self.alpha_o = alpha_o
        self.alpha_d = alpha_d
        self.max_lost_frames = max_lost_frames
        self.gamma = gamma
        self.min_track_confidence = min_track_confidence

        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        q = process_noise_std ** 2
        dt2 = self.dt ** 2
        dt3 = self.dt ** 3
        dt4 = self.dt ** 4
        self.Q = q * np.array([
            [dt4 / 4, 0, dt3 / 2, 0],
            [0, dt4 / 4, 0, dt3 / 2],
            [dt3 / 2, 0, dt2, 0],
            [0, dt3 / 2, 0, dt2],
        ], dtype=float)

        self.x = np.zeros(4)
        self.P = np.eye(4) * 10.0
        self.x_pred = np.zeros(4)
        self.P_pred = np.eye(4) * 10.0

        self.miss_count = 0
        self.track_state = 'INIT'
        self.confidence = 0.0
        self._initialized = False

    def init_state(self, z, P0=None):
        self.x[:2] = z
        self.x[2:] = 0.0
        self.P = P0 if P0 is not None else np.diag([1.0, 1.0, 2.0, 2.0])
        self.miss_count = 0
        self.track_state = 'TRACKED'
        self.confidence = 1.0
        self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    def predict(self):
        self.x_pred = self.F @ self.x
        self.P_pred = self.F @ self.P @ self.F.T + self.Q
        return self.x_pred.copy(), self.P_pred.copy()

    def update(self, z, confidence, occlusion_ratio, depth_uncertainty,
               R_base):
        eps = 1e-6
        lambda_c = 1.0 / (confidence ** 2 + eps)
        lambda_o = 1.0 + self.alpha_o * occlusion_ratio
        lambda_d = 1.0 + self.alpha_d * depth_uncertainty ** 2

        R_det = R_base.copy()
        R_occ = np.eye(2) * (2.0 * occlusion_ratio) ** 2
        R_depth = np.eye(2) * depth_uncertainty ** 2
        R = lambda_c * R_det + lambda_o * R_occ + lambda_d * R_depth

        y = z - self.H @ self.x_pred
        S = self.H @ self.P_pred @ self.H.T + R
        K = self.P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = self.x_pred + K @ y
        I4 = np.eye(4)
        self.P = (I4 - K @ self.H) @ self.P_pred

        self.miss_count = 0
        self.track_state = 'TRACKED'
        self.confidence = 0.7 * confidence + 0.3 * self.confidence
        return self.x.copy(), self.P.copy()

    def predict_only(self):
        self.x = self.x_pred.copy()
        self.P = self.P_pred.copy()
        self.P *= 1.02

        self.miss_count += 1
        self.confidence *= self.gamma

        if self.miss_count >= self.max_lost_frames or \
                self.confidence < self.min_track_confidence:
            self.track_state = 'LOST'
        elif self.miss_count >= 3:
            self.track_state = 'OCCLUDED_TRACK'

    def get_position(self):
        return self.x[:2].copy()

    def get_position_cov(self):
        return self.P[:2, :2].copy()

    def get_state_dict(self):
        return {
            'x_est': self.x[:2].copy(),
            'v_est': self.x[2:].copy(),
            'P': self.P.copy(),
            'P_pos': self.P[:2, :2].copy(),
            'confidence': float(self.confidence),
            'miss_count': self.miss_count,
            'track_state': self.track_state,
        }
