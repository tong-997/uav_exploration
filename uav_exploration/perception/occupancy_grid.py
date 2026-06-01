"""
占据栅格地图: 从前视深度实时构建, 支持多机融合
"""
import numpy as np
from config import GRID_N, GRID_RES, WORLD_SIZE, UNKNOWN, FREE, OCCUPIED, SENSOR_RANGE


class OccupancyGrid:
    """2D 占据栅格"""

    def __init__(self):
        self.grid = np.full((GRID_N, GRID_N), UNKNOWN, dtype=np.int8)
        self._update_count = np.zeros((GRID_N, GRID_N), dtype=np.int32)

    def pos_to_idx(self, pos):
        ix = int(np.clip(pos[0] / GRID_RES, 0, GRID_N - 1))
        iy = int(np.clip(pos[1] / GRID_RES, 0, GRID_N - 1))
        return iy, ix

    def idx_to_pos(self, iy, ix):
        return np.array([ix * GRID_RES + GRID_RES / 2,
                         iy * GRID_RES + GRID_RES / 2])

    def update_from_rays(self, drone_pos, heading, angles, depths):
        """
        向量化射线更新:
        - 射线路径上的格子 → FREE
        - 射线终点 (depth < SENSOR_RANGE) → OCCUPIED
        """
        ray_angles = heading + angles
        cos_a = np.cos(ray_angles)
        sin_a = np.sin(ray_angles)
        step_size = GRID_RES * 0.8
        max_steps = int(SENSOR_RANGE / step_size) + 1
        ts = np.arange(max_steps) * step_size  # (S,)

        # (R, S) 每条射线各采样点
        all_px = drone_pos[0] + cos_a[:, None] * ts[None, :]
        all_py = drone_pos[1] + sin_a[:, None] * ts[None, :]

        # 截断: 只保留 t < depth 的采样点
        mask_valid = ts[None, :] < depths[:, None]
        mask_bounds = (all_px >= 0) & (all_px < WORLD_SIZE) & (all_py >= 0) & (all_py < WORLD_SIZE)
        mask = mask_valid & mask_bounds

        ix = np.clip((all_px / GRID_RES).astype(int), 0, GRID_N - 1)
        iy = np.clip((all_py / GRID_RES).astype(int), 0, GRID_N - 1)

        free_iy = iy[mask]
        free_ix = ix[mask]
        self.grid[free_iy, free_ix] = FREE

        # 终点标记 OCCUPIED
        hit_mask = depths < SENSOR_RANGE - 0.1
        end_x = drone_pos[0] + cos_a * depths
        end_y = drone_pos[1] + sin_a * depths
        end_bounds = hit_mask & (end_x >= 0) & (end_x < WORLD_SIZE) & (end_y >= 0) & (end_y < WORLD_SIZE)
        end_ix = np.clip((end_x[end_bounds] / GRID_RES).astype(int), 0, GRID_N - 1)
        end_iy = np.clip((end_y[end_bounds] / GRID_RES).astype(int), 0, GRID_N - 1)
        self.grid[end_iy, end_ix] = OCCUPIED

    def merge(self, other: 'OccupancyGrid'):
        """融合另一架无人机的栅格 (取更高置信度)"""
        # OCCUPIED 优先, 其次 FREE, 最后 UNKNOWN
        mask_other_occ = other.grid == OCCUPIED
        mask_other_free = (other.grid == FREE) & (self.grid == UNKNOWN)
        self.grid[mask_other_occ] = OCCUPIED
        self.grid[mask_other_free] = FREE

    def get_frontiers(self, min_cluster_size=3):
        """
        找前沿: FREE 格子且邻居中有 UNKNOWN 的
        返回: list of (center_y, center_x) 聚类中心
        """
        free_mask = self.grid == FREE
        unknown_mask = self.grid == UNKNOWN

        # 4 邻域扩张 unknown
        expanded = np.zeros_like(unknown_mask)
        expanded[1:, :]  |= unknown_mask[:-1, :]
        expanded[:-1, :] |= unknown_mask[1:, :]
        expanded[:, 1:]  |= unknown_mask[:, :-1]
        expanded[:, :-1] |= unknown_mask[:, 1:]

        frontier_mask = free_mask & expanded
        frontier_cells = np.argwhere(frontier_mask)  # (N, 2) [iy, ix]

        if len(frontier_cells) == 0:
            return []

        # 简单聚类: 连通分量 (BFS)
        clusters = []
        visited = set()
        for cell in frontier_cells:
            key = (cell[0], cell[1])
            if key in visited:
                continue
            cluster = []
            queue = [key]
            while queue:
                cy, cx = queue.pop(0)
                if (cy, cx) in visited:
                    continue
                if not frontier_mask[cy, cx]:
                    continue
                visited.add((cy, cx))
                cluster.append((cy, cx))
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = cy+dy, cx+dx
                    if 0 <= ny < GRID_N and 0 <= nx < GRID_N:
                        if (ny, nx) not in visited:
                            queue.append((ny, nx))
            if len(cluster) >= min_cluster_size:
                arr = np.array(cluster)
                center = arr.mean(axis=0).astype(int)
                clusters.append((center[0], center[1], len(cluster)))

        # 按簇大小排序
        clusters.sort(key=lambda c: -c[2])
        return clusters

    @property
    def explored_ratio(self):
        total = GRID_N * GRID_N
        known = np.sum(self.grid != UNKNOWN)
        return known / total

    def copy(self):
        g = OccupancyGrid()
        g.grid = self.grid.copy()
        g._update_count = self._update_count.copy()
        return g
