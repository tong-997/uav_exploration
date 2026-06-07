"""
区域分配: Voronoi 分区 + 前沿过滤
"""
import numpy as np
from config import GRID_N, GRID_RES


class VoronoiAllocator:
    """基于 Voronoi 的区域分配"""

    def __init__(self):
        self._voronoi_map = None

    def compute_voronoi(self, drone_positions):
        """
        生成 Voronoi 分区图: voronoi_map[iy, ix] = drone_id
        """
        n_drones = len(drone_positions)
        xs = np.arange(GRID_N) * GRID_RES + GRID_RES / 2
        XX, YY = np.meshgrid(xs, xs)

        dist_maps = np.zeros((n_drones, GRID_N, GRID_N))
        for i, pos in enumerate(drone_positions):
            dist_maps[i] = np.sqrt((XX - pos[0])**2 + (YY - pos[1])**2)

        self._voronoi_map = np.argmin(dist_maps, axis=0)
        return self._voronoi_map

    def filter_frontiers_for_drone(self, frontiers, drone_id):
        """只保留属于该无人机 Voronoi 区域的前沿"""
        if self._voronoi_map is None:
            return frontiers
        filtered = []
        for iy, ix, size in frontiers:
            if self._voronoi_map[iy, ix] == drone_id:
                filtered.append((iy, ix, size))
        return filtered

    def get_area_ratio(self):
        """返回每架无人机的区域占比"""
        if self._voronoi_map is None:
            return []
        total = GRID_N * GRID_N
        ratios = []
        for i in range(int(self._voronoi_map.max()) + 1):
            ratios.append(np.sum(self._voronoi_map == i) / total)
        return ratios
