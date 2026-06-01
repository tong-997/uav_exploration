#!/usr/bin/env python
"""
depth_estimator_node.py
=======================
Switchable stereo depth estimation node.
Supports three backends via ~depth_method parameter:
  - resunet : ONNX ResUNet inference (proposed method)
  - sgbm    : OpenCV StereoSGBM baseline
  - gt      : Gazebo ground-truth depth camera (upper bound)

Topics:
  sub: ~left/image_raw    (sensor_msgs/Image, BGR8)  [resunet, sgbm]
       ~right/image_raw   (sensor_msgs/Image, BGR8)  [resunet, sgbm]
       ~gt_depth          (sensor_msgs/Image, 32FC1)  [gt mode]
  pub: ~depth             (sensor_msgs/Image, 32FC1, meters)
       ~pointcloud        (sensor_msgs/PointCloud2)
       ~uncertainty        (sensor_msgs/Image, 32FC1)
"""
import os
import numpy as np
import cv2
import rospy
import message_filters
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import rospkg


class DepthEstimatorNode:

    def __init__(self):
        rospy.init_node("depth_estimator", anonymous=False)
        self.bridge = CvBridge()

        # ---- common parameters ----
        self.method = rospy.get_param("~depth_method", "resunet")
        self.baseline = rospy.get_param("~baseline", 0.09)
        self.fx = rospy.get_param("~fx", 368.92)
        self.depth_max = rospy.get_param("~depth_max", 15.0)
        self.depth_min = rospy.get_param("~depth_min", 0.3)
        self.input_h = rospy.get_param("~input_h", 768)
        self.input_w = rospy.get_param("~input_w", 768)
        rate = rospy.get_param("~depth_rate", 10.0)

        # ---- publishers ----
        self.pub_depth = rospy.Publisher("~depth", Image, queue_size=2)
        self.pub_cloud = rospy.Publisher("~pointcloud", PointCloud2, queue_size=2)
        self.pub_uncert = rospy.Publisher("~uncertainty", Image, queue_size=2)

        # ---- method-specific init ----
        if self.method == "resunet":
            self._init_resunet()
        elif self.method == "sgbm":
            self._init_sgbm()
        elif self.method == "gt":
            self._init_gt()
        else:
            rospy.logfatal("Unknown depth_method: %s", self.method)
            return

        self.min_period = 1.0 / rate
        self.last_stamp = rospy.Time(0)
        rospy.loginfo("depth_estimator ready  method=%s  (%.1f Hz)",
                      self.method, rate)

    # ================================================================ #
    #  ResUNet backend
    # ================================================================ #
    def _init_resunet(self):
        import onnxruntime
        onnx_raw = rospy.get_param("~onnx_model_path")
        if not os.path.isabs(onnx_raw):
            pkg_dir = rospkg.RosPack().get_path("uav_exploration_ros")
            onnx_raw = os.path.join(pkg_dir, onnx_raw)

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        device = rospy.get_param("~onnx_device", "cpu")
        providers = (["CUDAExecutionProvider"] if device == "cuda"
                     else ["CPUExecutionProvider"])
        self.sess = onnxruntime.InferenceSession(
            onnx_raw, opts, providers=providers)
        rospy.loginfo("ONNX model loaded: %s  (providers=%s)",
                      onnx_raw, self.sess.get_providers())

        self._subscribe_stereo(self._stereo_resunet_cb)

    def _stereo_resunet_cb(self, left_msg, right_msg):
        now = left_msg.header.stamp
        if (now - self.last_stamp).to_sec() < self.min_period:
            return
        self.last_stamp = now

        l_bgr = self.bridge.imgmsg_to_cv2(left_msg, "bgr8")
        r_bgr = self.bridge.imgmsg_to_cv2(right_msg, "bgr8")
        orig_h, orig_w = l_bgr.shape[:2]

        l_resized = cv2.resize(l_bgr, (self.input_w, self.input_h))
        r_resized = cv2.resize(r_bgr, (self.input_w, self.input_h))
        img = np.concatenate((l_resized, r_resized), axis=2).astype(np.float32)
        img = img / 127.5 - 1.0
        img = np.expand_dims(img.transpose(2, 0, 1), axis=0)

        seg_idx, uncertainty, flow = self.sess.run(
            ["seg_idx", "uncertainty", "flow"], {"img": img})

        disp = flow[-1, 0].copy()
        disp[disp < 0.0] = 0.0
        disp_full = cv2.resize(disp, (orig_w, orig_h),
                               interpolation=cv2.INTER_LINEAR)
        disp_full = disp_full / disp.shape[1] * orig_w

        depth = np.where(disp_full > 0.1,
                         self.fx * self.baseline / disp_full,
                         0.0).astype(np.float32)
        depth[depth > self.depth_max] = 0.0
        depth[depth < self.depth_min] = 0.0

        uncert = uncertainty[0, 0].astype(np.float32)
        uncert_full = cv2.resize(uncert, (orig_w, orig_h))

        self._publish_all(depth, uncert_full, left_msg.header)

    # ================================================================ #
    #  SGBM backend
    # ================================================================ #
    def _init_sgbm(self):
        num_disp = rospy.get_param("~sgbm_num_disparities", 128)
        block_size = rospy.get_param("~sgbm_block_size", 9)
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size * block_size,
            P2=32 * 3 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        rospy.loginfo("SGBM initialized: numDisparities=%d blockSize=%d",
                      num_disp, block_size)
        self._subscribe_stereo(self._stereo_sgbm_cb)

    def _stereo_sgbm_cb(self, left_msg, right_msg):
        now = left_msg.header.stamp
        if (now - self.last_stamp).to_sec() < self.min_period:
            return
        self.last_stamp = now

        l_bgr = self.bridge.imgmsg_to_cv2(left_msg, "bgr8")
        r_bgr = self.bridge.imgmsg_to_cv2(right_msg, "bgr8")
        orig_h, orig_w = l_bgr.shape[:2]

        l_gray = cv2.cvtColor(l_bgr, cv2.COLOR_BGR2GRAY)
        r_gray = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2GRAY)

        disp = self.sgbm.compute(l_gray, r_gray).astype(np.float32) / 16.0
        disp[disp <= 0.0] = 0.0

        depth = np.where(disp > 0.1,
                         self.fx * self.baseline / disp,
                         0.0).astype(np.float32)
        depth[depth > self.depth_max] = 0.0
        depth[depth < self.depth_min] = 0.0

        uncert = np.full_like(depth, 0.5)
        uncert[depth == 0.0] = 1.0

        self._publish_all(depth, uncert, left_msg.header)

    # ================================================================ #
    #  Ground-truth backend
    # ================================================================ #
    def _init_gt(self):
        rospy.Subscriber("~gt_depth", Image, self._gt_depth_cb, queue_size=2)
        rospy.loginfo("GT depth mode: subscribing to ~gt_depth")

    def _gt_depth_cb(self, msg):
        now = msg.header.stamp
        if (now - self.last_stamp).to_sec() < self.min_period:
            return
        self.last_stamp = now

        depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        depth = depth.copy()
        depth[np.isnan(depth)] = 0.0
        depth[np.isinf(depth)] = 0.0
        depth[depth > self.depth_max] = 0.0
        depth[depth < self.depth_min] = 0.0

        uncert = np.zeros_like(depth)

        self._publish_all(depth, uncert, msg.header)

    # ================================================================ #
    #  Common helpers
    # ================================================================ #
    def _subscribe_stereo(self, callback):
        sub_left = message_filters.Subscriber("~left/image_raw", Image)
        sub_right = message_filters.Subscriber("~right/image_raw", Image)
        sync = message_filters.ApproximateTimeSynchronizer(
            [sub_left, sub_right], queue_size=5, slop=0.05)
        sync.registerCallback(callback)
        self._sync = sync

    def _publish_all(self, depth, uncertainty, header):
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header = header
        self.pub_depth.publish(depth_msg)

        uncert_msg = self.bridge.cv2_to_imgmsg(uncertainty, encoding="32FC1")
        uncert_msg.header = header
        self.pub_uncert.publish(uncert_msg)

        if self.pub_cloud.get_num_connections() > 0:
            cloud_msg = self._depth_to_cloud(depth, header)
            self.pub_cloud.publish(cloud_msg)

    def _depth_to_cloud(self, depth, header):
        h, w = depth.shape
        cx, cy = w / 2.0, h / 2.0
        fx = self.fx

        step = max(1, min(h, w) // 200)
        rows = np.arange(0, h, step)
        cols = np.arange(0, w, step)
        cc, rr = np.meshgrid(cols, rows)
        dd = depth[rr, cc]
        mask = dd > 0

        z = dd[mask]
        x = (cc[mask] - cx) * z / fx
        y = (rr[mask] - cy) * z / fx

        points = np.zeros((len(z), 3), dtype=np.float32)
        points[:, 0] = x
        points[:, 1] = y
        points[:, 2] = z

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(z)
        msg.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(z)
        msg.is_dense = True
        msg.data = points.tobytes()
        return msg

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = DepthEstimatorNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
