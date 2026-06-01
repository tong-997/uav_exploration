#!/usr/bin/env python
"""
frontier_node.py
================
Frontier detection + goal selection.
Each drone independently detects frontiers and selects exploration targets.

Topics:
  sub: ~global_grid       (nav_msgs/OccupancyGrid)
       ~pose              (geometry_msgs/PoseStamped)
  pub: ~exploration_goal  (geometry_msgs/PoseStamped)
       ~frontiers         (uav_exploration_ros/FrontierArray)
"""
import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header
from uav_exploration_ros.msg import Frontier, FrontierArray

UNKNOWN = -1
FREE = 0
OCCUPIED = 100


class FrontierNode:

    def __init__(self):
        rospy.init_node("frontier_explorer", anonymous=False)

        self.drone_id = rospy.get_param("~drone_id", 0)
        self.resolution = rospy.get_param("~grid_resolution", 0.5)
        self.frontier_min = rospy.get_param("~frontier_min_size", 3)
        self.w_dist = rospy.get_param("~utility_weight_dist", 1.0)
        self.w_size = rospy.get_param("~utility_weight_size", 0.5)
        self.done_ratio = rospy.get_param("~exploration_done_ratio", 0.90)
        rate = rospy.get_param("~frontier_rate", 2.0)

        self.grid = None
        self.grid_n = 0
        self.my_pos = None

        # ---- publishers ----
        self.pub_goal = rospy.Publisher(
            "~exploration_goal", PoseStamped, queue_size=2)
        self.pub_frontiers = rospy.Publisher(
            "~frontiers", FrontierArray, queue_size=2)

        # ---- subscribers ----
        rospy.Subscriber("~global_grid", OccupancyGrid, self.grid_cb, queue_size=2)
        rospy.Subscriber("~pose", PoseStamped, self.pose_cb, queue_size=5)

        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self.replan_cb)
        rospy.loginfo("frontier_node ready  drone_id=%d", self.drone_id)

    # -------------------------------------------------------------- #
    def grid_cb(self, msg):
        self.grid_n = msg.info.width
        self.grid = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)

    def pose_cb(self, msg):
        self.my_pos = np.array([msg.pose.position.x, msg.pose.position.y])

    # -------------------------------------------------------------- #
    def replan_cb(self, event):
        if self.grid is None or self.my_pos is None:
            return

        total = self.grid_n * self.grid_n
        known = np.sum(self.grid != UNKNOWN)
        explored_ratio = known / float(total)

        if explored_ratio >= self.done_ratio:
            rospy.loginfo_throttle(10, "Exploration complete: %.1f%%",
                                   explored_ratio * 100)
            return

        # ---- 1. detect frontiers ----
        frontiers = self._detect_frontiers()

        # ---- 2. select best frontier ----
        target = self._select_frontier(frontiers)

        # ---- publish ----
        frontier_msg = FrontierArray()
        frontier_msg.header = Header(stamp=rospy.Time.now(), frame_id="world")
        frontier_msg.explored_ratio = explored_ratio
        for iy, ix, size in frontiers:
            f = Frontier()
            f.center = Point(
                x=ix * self.resolution + self.resolution / 2,
                y=iy * self.resolution + self.resolution / 2,
                z=0)
            f.size = size
            f.drone_id = self.drone_id
            frontier_msg.frontiers.append(f)
        self.pub_frontiers.publish(frontier_msg)

        if target is not None:
            goal = PoseStamped()
            goal.header = Header(stamp=rospy.Time.now(), frame_id="world")
            goal.pose.position.x = target[0]
            goal.pose.position.y = target[1]
            goal.pose.orientation.w = 1.0
            self.pub_goal.publish(goal)

    # -------------------------------------------------------------- #
    def _detect_frontiers(self):
        g = self.grid
        free_mask = g == FREE
        unknown_mask = g == UNKNOWN

        expanded = np.zeros_like(unknown_mask)
        expanded[1:, :] |= unknown_mask[:-1, :]
        expanded[:-1, :] |= unknown_mask[1:, :]
        expanded[:, 1:] |= unknown_mask[:, :-1]
        expanded[:, :-1] |= unknown_mask[:, 1:]

        frontier_mask = free_mask & expanded
        frontier_cells = np.argwhere(frontier_mask)
        if len(frontier_cells) == 0:
            return []

        clusters = []
        visited = set()
        for cell in frontier_cells:
            key = (int(cell[0]), int(cell[1]))
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
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < self.grid_n and 0 <= nx < self.grid_n:
                        if (ny, nx) not in visited:
                            queue.append((ny, nx))
            if len(cluster) >= self.frontier_min:
                arr = np.array(cluster)
                center = arr.mean(axis=0).astype(int)
                clusters.append((int(center[0]), int(center[1]), len(cluster)))

        clusters.sort(key=lambda c: -c[2])
        return clusters

    def _select_frontier(self, frontiers):
        if not frontiers:
            return None
        best_score = -np.inf
        best_pos = None
        for iy, ix, size in frontiers:
            pos = np.array([
                ix * self.resolution + self.resolution / 2,
                iy * self.resolution + self.resolution / 2])
            dist = np.linalg.norm(pos - self.my_pos)
            if dist < 1.0:
                continue
            score = -self.w_dist * dist + self.w_size * np.log1p(size)
            if score > best_score:
                best_score = score
                best_pos = pos
        return best_pos

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = FrontierNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
