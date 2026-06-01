#!/usr/bin/env python
"""
planner_node.py
===============
A* path planning + PositionCommand output.
Each drone plans independently (no cross-drone deconfliction).

Topics:
  sub: ~exploration_goal  (geometry_msgs/PoseStamped)
       ~global_grid       (nav_msgs/OccupancyGrid)
       ~pose              (geometry_msgs/PoseStamped)
  pub: ~pos_cmd           (uav_exploration_ros/PositionCommand)
       ~trajectory        (nav_msgs/Path) — broadcast
       ~vis_path          (nav_msgs/Path) — visualization
"""
import numpy as np
import heapq
import rospy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point, Vector3
from std_msgs.msg import Header
from uav_exploration_ros.msg import PositionCommand
import tf.transformations as tft

UNKNOWN = -1
FREE = 0
OCCUPIED = 100


class PlannerNode:

    def __init__(self):
        rospy.init_node("planner", anonymous=False)

        self.drone_id = rospy.get_param("~drone_id", 0)
        self.resolution = rospy.get_param("~grid_resolution", 0.5)
        self.max_speed = rospy.get_param("~max_speed", 3.0)
        self.safety_margin = rospy.get_param("~safety_margin", 2)
        self.simplify_step = rospy.get_param("~path_simplify_step", 3)
        rate = rospy.get_param("~planner_rate", 5.0)

        self.grid = None
        self.grid_n = 0
        self.my_pos = None
        self.my_yaw = 0.0
        self.goal = None
        self.path = []
        self.path_idx = 0
        self.traj_id = 0

        # ---- publishers ----
        self.pub_cmd = rospy.Publisher("~pos_cmd", PositionCommand, queue_size=2)
        self.pub_traj = rospy.Publisher("~trajectory", Path, queue_size=2)
        self.pub_vis = rospy.Publisher("~vis_path", Path, queue_size=2)

        # ---- subscribers ----
        rospy.Subscriber("~exploration_goal", PoseStamped, self.goal_cb, queue_size=2)
        rospy.Subscriber("~global_grid", OccupancyGrid, self.grid_cb, queue_size=2)
        rospy.Subscriber("~pose", PoseStamped, self.pose_cb, queue_size=5)

        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self.control_cb)
        rospy.loginfo("planner_node ready  drone_id=%d", self.drone_id)

    # -------------------------------------------------------------- #
    # Callbacks
    # -------------------------------------------------------------- #
    def goal_cb(self, msg):
        new_goal = np.array([msg.pose.position.x, msg.pose.position.y])
        if self.goal is None or np.linalg.norm(new_goal - self.goal) > 2.0:
            self.goal = new_goal
            self._replan()

    def grid_cb(self, msg):
        self.grid_n = msg.info.width
        self.grid = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)

    def pose_cb(self, msg):
        p = msg.pose.position
        self.my_pos = np.array([p.x, p.y])
        q = msg.pose.orientation
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.my_yaw = yaw

    # -------------------------------------------------------------- #
    # Planning
    # -------------------------------------------------------------- #
    def _replan(self):
        if self.grid is None or self.my_pos is None or self.goal is None:
            return

        path = self._astar_plan(self.my_pos, self.goal)
        if not path:
            rospy.logwarn_throttle(5, "A* found no path to goal")
            return

        self.path = path
        self.path_idx = 0
        self.traj_id += 1

        self._broadcast_trajectory()

    def _astar_plan(self, start, goal):
        sy, sx = self._pos_to_idx(start)
        gy, gx = self._pos_to_idx(goal)
        cost_map = self._build_cost_map()

        sy, sx = self._nearest_free(cost_map, sy, sx)
        gy, gx = self._nearest_free(cost_map, gy, gx)
        if sy < 0 or gy < 0:
            return []

        path_idx = self._astar(cost_map, (sy, sx), (gy, gx))
        if not path_idx:
            return []

        path_world = []
        for iy, ix in path_idx:
            x = ix * self.resolution + self.resolution / 2
            y = iy * self.resolution + self.resolution / 2
            path_world.append(np.array([x, y]))

        return self._simplify(path_world)

    def _build_cost_map(self):
        blocked = self.grid == OCCUPIED
        cost = np.ones((self.grid_n, self.grid_n), dtype=np.float64)
        try:
            from scipy.ndimage import binary_dilation
            struct = np.ones((2 * self.safety_margin + 1,
                              2 * self.safety_margin + 1))
            inflated = binary_dilation(blocked, structure=struct)
            cost[inflated] = np.inf
        except ImportError:
            cost[blocked] = np.inf
        return cost

    def _astar(self, cost_map, start, goal, max_iter=50000):
        sy, sx = start
        gy, gx = goal
        n = self.grid_n
        open_set = [(0.0, sy, sx)]
        g_score = np.full((n, n), np.inf)
        g_score[sy, sx] = 0.0
        came_from = {}
        closed = set()

        for _ in range(max_iter):
            if not open_set:
                break
            f, cy, cx = heapq.heappop(open_set)
            if (cy, cx) in closed:
                continue
            closed.add((cy, cx))
            if cy == gy and cx == gx:
                return self._reconstruct(came_from, (gy, gx))
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < n and 0 <= nx < n):
                    continue
                if cost_map[ny, nx] == np.inf:
                    continue
                mc = 1.414 if abs(dy) + abs(dx) == 2 else 1.0
                ng = g_score[cy, cx] + mc * cost_map[ny, nx]
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

    def _nearest_free(self, cost_map, cy, cx, radius=10):
        if 0 <= cy < self.grid_n and 0 <= cx < self.grid_n:
            if cost_map[cy, cx] < np.inf:
                return cy, cx
        for r in range(1, radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < self.grid_n and 0 <= nx < self.grid_n:
                        if cost_map[ny, nx] < np.inf:
                            return ny, nx
        return -1, -1

    def _simplify(self, path):
        if len(path) <= 2:
            return path
        s = self.simplify_step
        simplified = [path[0]]
        for i in range(s, len(path) - 1, s):
            simplified.append(path[i])
        simplified.append(path[-1])
        return simplified

    def _pos_to_idx(self, pos):
        ix = int(np.clip(pos[0] / self.resolution, 0, self.grid_n - 1))
        iy = int(np.clip(pos[1] / self.resolution, 0, self.grid_n - 1))
        return iy, ix

    # -------------------------------------------------------------- #
    # Trajectory broadcast
    # -------------------------------------------------------------- #
    def _broadcast_trajectory(self):
        msg = Path()
        msg.header = Header(stamp=rospy.Time.now(), frame_id="world")
        for pt in self.path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = pt[0]
            ps.pose.position.y = pt[1]
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_traj.publish(msg)
        self.pub_vis.publish(msg)

    # -------------------------------------------------------------- #
    # Control loop
    # -------------------------------------------------------------- #
    def control_cb(self, event):
        if self.my_pos is None:
            return

        if not self.path or self.path_idx >= len(self.path):
            if self.goal is not None:
                self._replan()
            return

        target = self.path[self.path_idx]
        diff = target - self.my_pos
        dist = np.linalg.norm(diff)

        if dist < self.resolution * 2:
            self.path_idx += 1
            if self.path_idx >= len(self.path):
                self._publish_cmd(target, np.zeros(2))
                return

        vel = diff / max(dist, 0.01) * min(self.max_speed, dist * 2)
        self._publish_cmd(target, vel)

    def _publish_cmd(self, pos, vel):
        cmd = PositionCommand()
        cmd.header = Header(stamp=rospy.Time.now(), frame_id="world")
        cmd.position = Point(x=pos[0], y=pos[1], z=0)
        cmd.velocity = Vector3(x=vel[0], y=vel[1], z=0)
        cmd.yaw = np.arctan2(vel[1], vel[0]) if np.linalg.norm(vel) > 0.1 else self.my_yaw
        cmd.trajectory_id = self.traj_id
        cmd.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_EXEC
        self.pub_cmd.publish(cmd)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = PlannerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
