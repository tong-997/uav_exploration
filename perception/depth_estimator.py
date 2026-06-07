"""
深度估计模块
- 仿真模式: 从 SimEnvironment 射线投射获取深度
- 真实模式: 加载 disptool 单相机 ONNX 推理
"""
import numpy as np
import os, sys


class DepthEstimator:
    """前视深度估计器"""

    def __init__(self, onnx_path=None, disptools_root=None,
                 fx=368.92, baseline=0.09, mode='erp'):
        self.fx = fx
        self.baseline = baseline
        self.mode = mode
        self._backend = None

        if onnx_path and disptools_root and os.path.isdir(disptools_root):
            sys.path.insert(0, disptools_root)
            try:
                from core.inference.onnx_inference import OnnxInference
                self._backend = OnnxInference(onnx_path)
                print(f'[DepthEstimator] ONNX 后端已加载: {onnx_path}')
            except Exception as e:
                print(f'[DepthEstimator] ONNX 加载失败: {e}, 使用仿真模式')

    @property
    def is_real(self):
        return self._backend is not None

    def infer_real(self, left_bgr, right_bgr):
        """
        真实推理: 左右目 BGR 图像 → (disp, depth, uncertainty)
        """
        if self._backend is None:
            raise RuntimeError("ONNX 后端未加载")
        H, W = left_bgr.shape[:2]
        img = self._backend.process_img(left_bgr, right_bgr, self.mode, resize=True)
        seg_idx, uncert, flow = self._backend(img)
        disp_map = self._backend.post(flow, (W, H))
        depth_map = self.fx * self.baseline / np.maximum(disp_map, 0.1)
        uncert_map = uncert[0, 0]
        return disp_map, depth_map, uncert_map

    @staticmethod
    def depth_to_local_points(angles, depths):
        """
        仿真深度 → 局部坐标点集
        angles, depths: 来自 sim_env.sense_depth()
        返回: (N, 2) 局部坐标系下的点 (前方为 +x)
        """
        xs = depths * np.cos(angles)
        ys = depths * np.sin(angles)
        return np.column_stack([xs, ys])

    @staticmethod
    def local_to_world(points, pos, heading):
        """局部坐标 → 世界坐标"""
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, -s], [s, c]])
        return (R @ points.T).T + pos
