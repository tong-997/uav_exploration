"""
分布式时空防撞: 每架无人机仅依据本地通信范围内的邻居信息独立决策
====================================================================
- 无中心调度, 各机异步运行
- 优先级基于 drone_id (固定规则, 各机可独立计算, 无需协商)
- 每架无人机仅知道通信范围内邻居的位置和广播轨迹
- 类似 EGO-Swarm 的去中心化异步避障设计
"""
import numpy as np
from config import SAFE_RADIUS, MAX_SPEED, DT, COMM_RANGE


class DistributedDeconfliction:
    """分布式防撞 — 每架无人机独立决策"""

    def __init__(self, safe_radius=None):
        self.safe_radius = safe_radius or SAFE_RADIUS
        self.broadcast_paths = {}

    def broadcast_trajectory(self, drone_id, path):
        """无人机广播自身规划轨迹 (模拟 ROS Topic / 局域通信)"""
        self.broadcast_paths[drone_id] = [np.array(p) for p in path] if path else []

    def get_neighbors(self, drone_id, drone_pos, all_positions):
        """获取通信范围内的邻居 (仅本地信息)"""
        neighbors = []
        for j, pos in enumerate(all_positions):
            if j == drone_id:
                continue
            if np.linalg.norm(np.asarray(drone_pos) - np.asarray(pos)) < COMM_RANGE:
                neighbors.append(j)
        return neighbors

    def local_priority(self, drone_id, other_id):
        """
        分布式优先级规则: ID 小的优先
        ——各机可独立计算, 无需全局协商, 保证一致性
        """
        return drone_id < other_id

    def check_and_resolve(self, drone_id, drone_pos, path, all_positions):
        """
        单机独立防撞决策:
        1. 查询通信范围内邻居
        2. 获取邻居广播的轨迹
        3. 若与更高优先级邻居轨迹冲突 → 本机让行(等待)
        4. 若与更低优先级邻居冲突 → 本机保持(对方会让行)

        返回: 修正后的路径
        """
        if not path or len(path) < 2:
            return path

        neighbors = self.get_neighbors(drone_id, drone_pos, all_positions)
        conflict = False

        for nb in neighbors:
            if self.local_priority(drone_id, nb):
                continue

            nb_path = self.broadcast_paths.get(nb, [])
            if not nb_path:
                continue

            n_check = min(len(path), len(nb_path), 20)
            for t in range(n_check):
                d = np.linalg.norm(path[t] - nb_path[t])
                if d < self.safe_radius * 1.5:
                    conflict = True
                    break
            if conflict:
                break

        if conflict:
            wait_steps = 5
            return [drone_pos.copy()] * wait_steps + path

        return path

    def local_emergency_avoidance(self, drone_id, drone_pos, all_positions):
        """
        紧急避让 (反应式, 纯本地):
        远离通信范围内所有过近邻居的合力方向
        """
        neighbors = self.get_neighbors(drone_id, drone_pos, all_positions)
        repulsion = np.zeros(2)
        triggered = False

        for nb in neighbors:
            nb_pos = all_positions[nb]
            diff = drone_pos - nb_pos
            dist = np.linalg.norm(diff)
            if dist < self.safe_radius * 1.5 and dist > 0.01:
                repulsion += diff / (dist * dist)
                triggered = True

        if not triggered:
            return None

        norm = np.linalg.norm(repulsion)
        if norm < 0.01:
            return np.array([1.0, 0.0])
        return repulsion / norm

    def check_proximity(self, drone_id, drone_pos, all_positions):
        """单机视角: 检测是否有邻居过近"""
        neighbors = self.get_neighbors(drone_id, drone_pos, all_positions)
        conflicts = []
        for nb in neighbors:
            d = np.linalg.norm(drone_pos - all_positions[nb])
            if d < self.safe_radius:
                conflicts.append((drone_id, nb, d))
        return conflicts
