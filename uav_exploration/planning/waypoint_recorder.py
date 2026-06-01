"""
航点记录器: 记录无人机走过的 GPS 航点
"""
import numpy as np
import json
from dataclasses import dataclass, field
from typing import List


@dataclass
class Waypoint:
    x: float
    y: float
    timestamp: float
    heading: float
    drone_id: int


class WaypointRecorder:
    """记录和导出航点"""

    def __init__(self, n_drones=3):
        self.waypoints: List[List[Waypoint]] = [[] for _ in range(n_drones)]

    def record(self, drone_id, pos, heading, timestamp):
        # 距离上一个航点太近则跳过
        wps = self.waypoints[drone_id]
        if wps:
            last = wps[-1]
            if np.hypot(pos[0] - last.x, pos[1] - last.y) < 0.5:
                return
        wps.append(Waypoint(
            x=pos[0], y=pos[1],
            timestamp=timestamp,
            heading=heading,
            drone_id=drone_id,
        ))

    def export_json(self, filepath):
        data = {}
        for i, wps in enumerate(self.waypoints):
            data[f'drone_{i}'] = [
                {'x': w.x, 'y': w.y, 't': w.timestamp, 'hdg': w.heading}
                for w in wps
            ]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def get_total_distance(self, drone_id):
        wps = self.waypoints[drone_id]
        if len(wps) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(wps)):
            total += np.hypot(wps[i].x - wps[i-1].x, wps[i].y - wps[i-1].y)
        return total
