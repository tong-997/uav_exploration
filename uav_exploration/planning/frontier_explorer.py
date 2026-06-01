"""
前沿探索策略: 选择最优前沿分配给无人机
"""
import numpy as np
from config import GRID_RES, GRID_N


class FrontierExplorer:
    """基于前沿的探索目标选择"""

    def __init__(self, utility_weight_dist=1.0, utility_weight_size=0.5):
        self.w_dist = utility_weight_dist
        self.w_size = utility_weight_size

    def select_frontier(self, frontiers, drone_pos, other_targets=None):
        """
        为一架无人机选择最优前沿

        参数:
          frontiers     : [(iy, ix, size), ...] 来自 OccupancyGrid.get_frontiers()
          drone_pos     : (2,) 当前位置
          other_targets : list of (2,) 其他无人机已选的目标 (避免重复)

        返回:
          target_pos : (2,) 世界坐标, 或 None (无可用前沿)
        """
        if not frontiers:
            return None

        best_score = -np.inf
        best_pos = None

        for iy, ix, size in frontiers:
            pos = np.array([ix * GRID_RES + GRID_RES / 2,
                            iy * GRID_RES + GRID_RES / 2])
            dist = np.linalg.norm(pos - drone_pos)
            if dist < 1.0:
                continue

            # 距离代价 (越近越好)
            score_dist = -self.w_dist * dist
            # 前沿大小奖励 (越大信息增益越多)
            score_size = self.w_size * np.log1p(size)
            score = score_dist + score_size

            # 惩罚与其他无人机目标太近的前沿
            if other_targets:
                for ot in other_targets:
                    if ot is not None:
                        d_other = np.linalg.norm(pos - ot)
                        if d_other < 10.0:
                            score -= 50.0 * (1.0 - d_other / 10.0)

            if score > best_score:
                best_score = score
                best_pos = pos

        return best_pos

    def assign_frontiers(self, frontiers, drone_positions):
        """
        贪心分配: 依次为每架无人机选最优前沿, 避免重叠

        返回: list of (2,) 目标坐标, None 表示无可用前沿
        """
        n = len(drone_positions)
        targets = [None] * n

        # 按探索进度排序, 探索少的优先选
        order = list(range(n))

        for i in order:
            chosen = self.select_frontier(
                frontiers, drone_positions[i],
                other_targets=[t for j, t in enumerate(targets) if j != i]
            )
            targets[i] = chosen

        return targets
