"""
A* 路径规划: 在占据栅格上规避障碍物
"""
import numpy as np
import heapq
from config import GRID_N, GRID_RES, OCCUPIED, UNKNOWN


class AStarPlanner:
    """栅格 A* 路径规划器"""

    def __init__(self, safety_margin=2):
        """safety_margin: 障碍物膨胀格数"""
        self.safety_margin = safety_margin

    def plan(self, grid, start_pos, goal_pos, treat_unknown_as='free'):
        """
        参数:
          grid      : (GRID_N, GRID_N) int8 占据栅格
          start_pos : (2,) 世界坐标 m
          goal_pos  : (2,) 世界坐标 m
          treat_unknown_as: 'free' | 'occupied'
        返回:
          path: list of (x, y) 世界坐标, 空列表表示无解
        """
        sy, sx = self._pos_to_idx(start_pos)
        gy, gx = self._pos_to_idx(goal_pos)

        cost_map = self._build_cost_map(grid, treat_unknown_as)

        # 起点/终点被膨胀阻塞时, 搜索最近可通行格子
        sy, sx = self._find_nearest_free(cost_map, sy, sx)
        gy, gx = self._find_nearest_free(cost_map, gy, gx)
        if sy < 0 or gy < 0:
            return []

        path_idx = self._astar(cost_map, (sy, sx), (gy, gx))
        if not path_idx:
            return []

        path_world = []
        for iy, ix in path_idx:
            x = ix * GRID_RES + GRID_RES / 2
            y = iy * GRID_RES + GRID_RES / 2
            path_world.append(np.array([x, y]))

        return self._simplify(path_world, grid)

    def _build_cost_map(self, grid, treat_unknown_as):
        """构建代价地图: 膨胀障碍物"""
        blocked = grid == OCCUPIED
        if treat_unknown_as == 'occupied':
            blocked |= (grid == UNKNOWN)

        cost = np.ones((GRID_N, GRID_N), dtype=np.float64)
        # 膨胀
        from scipy.ndimage import binary_dilation
        struct = np.ones((2 * self.safety_margin + 1, 2 * self.safety_margin + 1))
        inflated = binary_dilation(blocked, structure=struct)
        cost[inflated] = np.inf
        return cost

    def _astar(self, cost_map, start, goal, max_iter=50000):
        sy, sx = start
        gy, gx = goal
        open_set = [(0.0, sy, sx)]
        g_score = np.full((GRID_N, GRID_N), np.inf)
        g_score[sy, sx] = 0.0
        came_from = {}
        closed = set()
        it = 0

        while open_set and it < max_iter:
            it += 1
            f, cy, cx = heapq.heappop(open_set)
            if (cy, cx) in closed:
                continue
            closed.add((cy, cx))

            if cy == gy and cx == gx:
                return self._reconstruct(came_from, (gy, gx))

            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < GRID_N and 0 <= nx < GRID_N):
                    continue
                if cost_map[ny, nx] == np.inf:
                    continue
                move_cost = 1.414 if abs(dy) + abs(dx) == 2 else 1.0
                ng = g_score[cy, cx] + move_cost * cost_map[ny, nx]
                if ng < g_score[ny, nx]:
                    g_score[ny, nx] = ng
                    h = np.hypot(ny - gy, nx - gx)
                    came_from[(ny, nx)] = (cy, cx)
                    heapq.heappush(open_set, (ng + h, ny, nx))

        return []

    @staticmethod
    def _reconstruct(came_from, goal):
        path = [goal]
        while goal in came_from:
            goal = came_from[goal]
            path.append(goal)
        path.reverse()
        return path

    @staticmethod
    def _find_nearest_free(cost_map, cy, cx, radius=10):
        """搜索 (cy,cx) 附近最近的可通行格子"""
        if cost_map[cy, cx] < np.inf:
            return cy, cx
        for r in range(1, radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < GRID_N and 0 <= nx < GRID_N:
                        if cost_map[ny, nx] < np.inf:
                            return ny, nx
        return -1, -1

    @staticmethod
    def _pos_to_idx(pos):
        ix = int(np.clip(pos[0] / GRID_RES, 0, GRID_N - 1))
        iy = int(np.clip(pos[1] / GRID_RES, 0, GRID_N - 1))
        return iy, ix

    @staticmethod
    def _simplify(path, grid, step=3):
        """每隔 step 个点取一个, 减少航点数"""
        if len(path) <= 2:
            return path
        simplified = [path[0]]
        for i in range(step, len(path) - 1, step):
            simplified.append(path[i])
        simplified.append(path[-1])
        return simplified
