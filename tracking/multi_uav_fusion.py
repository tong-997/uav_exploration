"""
多机联合状态融合: 信息滤波 / 协方差交叉
"""
import numpy as np


class MultiUAVFusion:

    def __init__(self, method='info_filter'):
        self.method = method

    def fuse(self, estimates):
        if not estimates:
            return None, None
        if len(estimates) == 1:
            return estimates[0]['x'].copy(), estimates[0]['P'].copy()

        if self.method == 'single_best':
            return self._single_best(estimates)
        elif self.method == 'naive_average':
            return self._naive_average(estimates)
        elif self.method == 'info_filter':
            return self._info_filter(estimates)
        elif self.method == 'covariance_intersection':
            return self._covariance_intersection(estimates)
        return self._info_filter(estimates)

    def _single_best(self, estimates):
        best = max(estimates, key=lambda e: e.get('confidence', 0))
        return best['x'].copy(), best['P'].copy()

    def _naive_average(self, estimates):
        N = len(estimates)
        x_avg = np.mean([e['x'] for e in estimates], axis=0)
        P_avg = np.mean([e['P'] for e in estimates], axis=0) / N
        return x_avg, P_avg

    def _info_filter(self, estimates):
        reg = np.eye(2) * 1e-6
        Omega = np.zeros((2, 2))
        xi = np.zeros(2)
        for e in estimates:
            P_inv = np.linalg.inv(e['P'] + reg)
            Omega += P_inv
            xi += P_inv @ e['x']
        P_fused = np.linalg.inv(Omega + reg)
        x_fused = P_fused @ xi
        return x_fused, P_fused

    def _covariance_intersection(self, estimates):
        if len(estimates) == 2:
            return self._ci_pair(estimates[0], estimates[1])
        result = estimates[0]
        for i in range(1, len(estimates)):
            x_f, P_f = self._ci_pair(
                {'x': result['x'], 'P': result['P']},
                estimates[i])
            result = {'x': x_f, 'P': P_f}
        return result['x'], result['P']

    def _ci_pair(self, e1, e2):
        reg = np.eye(2) * 1e-6
        P1_inv = np.linalg.inv(e1['P'] + reg)
        P2_inv = np.linalg.inv(e2['P'] + reg)

        best_omega = 0.5
        best_det = float('inf')

        for w_int in range(101):
            w = w_int / 100.0
            P_inv = w * P1_inv + (1 - w) * P2_inv
            try:
                P = np.linalg.inv(P_inv)
                d = np.linalg.det(P)
                if d < best_det:
                    best_det = d
                    best_omega = w
            except np.linalg.LinAlgError:
                continue

        P_inv = best_omega * P1_inv + (1 - best_omega) * P2_inv
        P_fused = np.linalg.inv(P_inv)
        x_fused = P_fused @ (best_omega * P1_inv @ e1['x'] +
                              (1 - best_omega) * P2_inv @ e2['x'])
        return x_fused, P_fused

    def fuse_with_metrics(self, estimates, true_pos):
        if not estimates:
            return {'pre_fusion_rmse': float('nan'),
                    'post_fusion_rmse': float('nan'),
                    'cov_trace': float('nan'),
                    'max_eigenvalue': float('nan')}

        pre_errors = [np.linalg.norm(e['x'] - true_pos) for e in estimates]
        pre_rmse = float(np.sqrt(np.mean(np.array(pre_errors) ** 2)))

        x_fused, P_fused = self.fuse(estimates)
        if x_fused is None:
            return {'pre_fusion_rmse': pre_rmse,
                    'post_fusion_rmse': float('nan'),
                    'cov_trace': float('nan'),
                    'max_eigenvalue': float('nan')}

        post_rmse = float(np.linalg.norm(x_fused - true_pos))
        cov_trace = float(np.trace(P_fused))
        max_eig = float(np.max(np.linalg.eigvalsh(P_fused)))

        return {
            'pre_fusion_rmse': pre_rmse,
            'post_fusion_rmse': post_rmse,
            'cov_trace': cov_trace,
            'max_eigenvalue': max_eig,
        }
