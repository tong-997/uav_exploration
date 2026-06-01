#!/usr/bin/env python
"""
grid_map_node.py
================
Maintain a 2D occupancy grid from depth images.
Each drone builds its own independent map (no cross-drone fusion).

Topics:
  sub: ~depth            (sensor_msgs/Image, 32FC1)
       ~pose             (geometry_msgs/PoseStamped) — drone pose in world
  pub: ~local_grid       (nav_msgs/OccupancyGrid)
       ~global_grid      (nav_msgs/OccupancyGrid) — same as local (no fusion)
"""
import numpy as np
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header
from cv_bridge import CvBridge
import tf.transformations as tft

UNKNOWN = -1
FREE = 0
OCCUPIED = 100


class GridMapNode:

    def __init__(self):
        rospy.init_node("grid_map", anonymous=False)
        self.bridge = CvBridge()

        # ---- parameters ----
        self.world_size = rospy.get_param("~world_size", 100.0)
        self.resolution = rospy.get_param("~grid_resolution", 0.5)
        self.grid_n = int(self.world_size / self.resolution)
        self.sensor_range = rospy.get_param("~sensor_range", 15.0)
        self.fov = np.deg2rad(rospy.get_param("~sensor_fov_deg", 90.0))
        self.depth_max = rospy.get_param("~depth_max", 15.0)
        self.fx = rospy.get_param("~fx", 368.92)
        self.drone_id = rospy.get_param("~drone_id", 0)
        rate = rospy.get_param("~grid_rate", 5.0)

        # ---- state ----
        self.grid = np.full((self.grid_n, self.grid_n), UNKNOWN, dtype=np.int8)
        self.pose = None  # (x, y, yaw)

        # ---- publishers ----
        self.pub_local = rospy.Publisher("~local_grid", OccupancyGrid, queue_size=2)
        self.pub_global = rospy.Publisher("~global_grid", OccupancyGrid, queue_size=2)

        # ---- subscribers ----
        rospy.Subscriber("~depth", Image, self.depth_cb, queue_size=2)
        rospy.Subscriber("~pose", PoseStamped, self.pose_cb, queue_size=5)

        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self.publish_cb)
        rospy.loginfo("grid_map_node ready  drone_id=%d  grid=%dx%d",
                      self.drone_id, self.grid_n, self.grid_n)

    # -------------------------------------------------------------- #
    def pose_cb(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pose = (p.x, p.y, yaw)

    def depth_cb(self, msg):
        if self.pose is None:
            return
        depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        self._update_grid_from_depth(depth)

    # -------------------------------------------------------------- #
    def _update_grid_from_depth(self, depth_img):
        """Project depth image columns into 2D occupancy rays."""
        px, py, yaw = self.pose
        h, w = depth_img.shape

        n_cols = min(w, 60)
        cols = np.linspace(0, w - 1, n_cols, dtype=int)
        cx = w / 2.0

        for c in cols:
            col_depths = depth_img[:, c]
            valid = col_depths[(col_depths > 0.3) & (col_depths < self.depth_max)]
            if len(valid) == 0:
                d = self.sensor_range
                hit = False
            else:
                d = float(np.median(valid))
                hit = True

            angle_offset = np.arctan2(c - cx, self.fx)
            ray_angle = yaw + angle_offset

            self._trace_ray(px, py, ray_angle, d, hit)

    def _trace_ray(self, ox, oy, angle, depth, hit):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        step = self.resolution * 0.8
        n_steps = int(min(depth, self.sensor_range) / step)

        for s in range(n_steps):
            t = s * step
            wx = ox + cos_a * t
            wy = oy + sin_a * t
            ix = int(wx / self.resolution)
            iy = int(wy / self.resolution)
            if 0 <= ix < self.grid_n and 0 <= iy < self.grid_n:
                self.grid[iy, ix] = FREE

        if hit and depth < self.sensor_range - 0.1:
            ex = ox + cos_a * depth
            ey = oy + sin_a * depth
            ix = int(ex / self.resolution)
            iy = int(ey / self.resolution)
            if 0 <= ix < self.grid_n and 0 <= iy < self.grid_n:
                self.grid[iy, ix] = OCCUPIED

    # -------------------------------------------------------------- #
    def _grid_to_msg(self, grid, stamp):
        msg = OccupancyGrid()
        msg.header = Header(stamp=stamp, frame_id="world")
        msg.info = MapMetaData()
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_n
        msg.info.height = self.grid_n
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        msg.data = grid.flatten().tolist()
        return msg

    def publish_cb(self, event):
        stamp = rospy.Time.now()
        grid_msg = self._grid_to_msg(self.grid, stamp)
        self.pub_local.publish(grid_msg)
        self.pub_global.publish(grid_msg)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = GridMapNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
