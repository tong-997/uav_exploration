#!/usr/bin/env python
"""
metrics_node.py
===============
Collect depth accuracy and exploration task metrics for comparison experiments.

Depth accuracy (when GT depth is available):
  - MAE, RMSE, delta_1m (% of pixels with |err| < 1m), delta_05m

Exploration task:
  - explored_ratio over time
  - cumulative flight distance

Saves CSV on shutdown to /tmp/uav_exploration/metrics_uavN_{method}.csv

Topics:
  sub: ~depth_estimated   (sensor_msgs/Image, 32FC1) — current method output
       ~depth_gt          (sensor_msgs/Image, 32FC1) — Gazebo ground-truth
       ~frontiers         (uav_exploration_ros/FrontierArray)
       ~pose              (geometry_msgs/PoseStamped)
"""
import os
import csv
import numpy as np
import rospy
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from uav_exploration_ros.msg import FrontierArray


class MetricsNode:

    def __init__(self):
        rospy.init_node("metrics", anonymous=False)
        self.bridge = CvBridge()

        self.drone_id = rospy.get_param("~drone_id", 0)
        self.method = rospy.get_param("~depth_method", "resunet")
        self.output_dir = rospy.get_param(
            "~metrics_output_dir", "/tmp/uav_exploration")

        # ---- state ----
        self.records = []
        self.explored_ratio = 0.0
        self.last_pos = None
        self.cum_dist = 0.0
        self.t0 = None

        # depth accuracy buffers
        self.latest_depth_est = None
        self.latest_depth_gt = None

        # ---- subscribers ----
        rospy.Subscriber("~depth_estimated", Image,
                         self.depth_est_cb, queue_size=2)
        rospy.Subscriber("~depth_gt", Image,
                         self.depth_gt_cb, queue_size=2)
        rospy.Subscriber("~frontiers", FrontierArray,
                         self.frontier_cb, queue_size=2)
        rospy.Subscriber("~pose", PoseStamped,
                         self.pose_cb, queue_size=5)

        # ---- periodic sampling ----
        self.timer = rospy.Timer(rospy.Duration(1.0), self.sample_cb)

        rospy.on_shutdown(self.save_csv)
        rospy.loginfo("metrics_node ready  drone_id=%d  method=%s",
                      self.drone_id, self.method)

    # -------------------------------------------------------------- #
    def depth_est_cb(self, msg):
        self.latest_depth_est = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def depth_gt_cb(self, msg):
        d = self.bridge.imgmsg_to_cv2(msg, "32FC1").copy()
        d[np.isnan(d)] = 0.0
        d[np.isinf(d)] = 0.0
        self.latest_depth_gt = d

    def frontier_cb(self, msg):
        self.explored_ratio = msg.explored_ratio

    def pose_cb(self, msg):
        p = msg.pose.position
        pos = np.array([p.x, p.y])
        if self.last_pos is not None:
            self.cum_dist += np.linalg.norm(pos - self.last_pos)
        self.last_pos = pos

    # -------------------------------------------------------------- #
    def sample_cb(self, event):
        if self.t0 is None:
            self.t0 = rospy.Time.now()

        elapsed = (rospy.Time.now() - self.t0).to_sec()

        # depth accuracy
        mae, rmse, d1m, d05m = -1, -1, -1, -1
        if (self.latest_depth_est is not None and
                self.latest_depth_gt is not None):
            mae, rmse, d1m, d05m = self._compute_depth_metrics(
                self.latest_depth_est, self.latest_depth_gt)

        record = {
            "time_s": round(elapsed, 2),
            "method": self.method,
            "drone_id": self.drone_id,
            "explored_ratio": round(self.explored_ratio, 4),
            "cum_dist_m": round(self.cum_dist, 2),
            "depth_mae": round(mae, 4),
            "depth_rmse": round(rmse, 4),
            "delta_1m": round(d1m, 4),
            "delta_05m": round(d05m, 4),
        }
        self.records.append(record)

        if len(self.records) % 30 == 0:
            rospy.loginfo(
                "[metrics] t=%.0fs  coverage=%.1f%%  dist=%.1fm  MAE=%.3f",
                elapsed, self.explored_ratio * 100,
                self.cum_dist, mae)

    def _compute_depth_metrics(self, est, gt):
        # resize if shape mismatch
        if est.shape != gt.shape:
            import cv2
            gt = cv2.resize(gt, (est.shape[1], est.shape[0]))

        valid = (est > 0) & (gt > 0)
        n = np.sum(valid)
        if n < 100:
            return -1, -1, -1, -1

        e = est[valid]
        g = gt[valid]
        err = np.abs(e - g)

        mae = float(np.mean(err))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        d1m = float(np.mean(err < 1.0))
        d05m = float(np.mean(err < 0.5))

        return mae, rmse, d1m, d05m

    # -------------------------------------------------------------- #
    def save_csv(self):
        if not self.records:
            rospy.logwarn("metrics_node: no records to save")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        fname = "metrics_uav{}_{}.csv".format(self.drone_id, self.method)
        path = os.path.join(self.output_dir, fname)

        keys = self.records[0].keys()
        with open(path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)

        rospy.loginfo("Saved %d metric records -> %s",
                      len(self.records), path)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = MetricsNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
