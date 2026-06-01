#!/usr/bin/env python
"""
fake_drone_node.py
==================
Kinematic drone controller for Gazebo simulation.
Subscribes to PositionCommand, moves the Gazebo model via set_model_state,
and publishes the current pose.

Similar to ego-planner-swarm's so3_control / fake_drone.

Topics:
  sub: ~pos_cmd      (uav_exploration_ros/PositionCommand)
  pub: ~pose         (geometry_msgs/PoseStamped) — current pose
       ~odom         (nav_msgs/Odometry) — for compatibility
"""
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from uav_exploration_ros.msg import PositionCommand
import tf.transformations as tft


class FakeDroneNode:

    def __init__(self):
        rospy.init_node("fake_drone", anonymous=False)

        self.drone_id = rospy.get_param("~drone_id", 0)
        self.model_name = rospy.get_param(
            "~model_name", "uav{}".format(self.drone_id))
        self.max_speed = rospy.get_param("~max_speed", 3.0)
        self.max_yaw_rate = rospy.get_param("~max_yaw_rate", 2.0)  # rad/s
        self.flight_height = rospy.get_param("~flight_height", 2.0)
        rate_hz = rospy.get_param("~control_rate", 50.0)

        # state
        self.pos = np.array([5.0, 5.0 + self.drone_id * 45.0, self.flight_height])
        self.yaw = 0.0
        self.target_pos = self.pos.copy()
        self.target_yaw = 0.0
        self.target_vel = np.zeros(3)
        self.has_cmd = False

        # ---- Gazebo service ----
        rospy.loginfo("Waiting for /gazebo/set_model_state ...")
        rospy.wait_for_service("/gazebo/set_model_state", timeout=30.0)
        self.set_state = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState)

        # try to get initial pose from Gazebo
        try:
            rospy.wait_for_service("/gazebo/get_model_state", timeout=5.0)
            get_state = rospy.ServiceProxy(
                "/gazebo/get_model_state", GetModelState)
            resp = get_state(self.model_name, "world")
            if resp.success:
                p = resp.pose.position
                self.pos = np.array([p.x, p.y, p.z])
                q = resp.pose.orientation
                _, _, self.yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
                self.target_pos = self.pos.copy()
                self.target_yaw = self.yaw
        except Exception:
            pass

        # ---- publishers ----
        self.pub_pose = rospy.Publisher("~pose", PoseStamped, queue_size=10)
        self.pub_odom = rospy.Publisher("~odom", Odometry, queue_size=10)

        # ---- subscribers ----
        rospy.Subscriber("~pos_cmd", PositionCommand, self.cmd_cb, queue_size=5)

        # ---- control timer ----
        self.dt = 1.0 / rate_hz
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.control_loop)

        rospy.loginfo("fake_drone ready  id=%d  model=%s  height=%.1f",
                      self.drone_id, self.model_name, self.flight_height)

    def cmd_cb(self, msg):
        self.target_pos = np.array([
            msg.position.x, msg.position.y, self.flight_height])
        self.target_vel = np.array([
            msg.velocity.x, msg.velocity.y, 0.0])
        self.target_yaw = msg.yaw
        self.has_cmd = True

    def control_loop(self, event):
        if self.has_cmd:
            # position tracking with velocity feedforward
            pos_err = self.target_pos - self.pos
            vel_cmd = 2.0 * pos_err + 0.5 * self.target_vel
            speed = np.linalg.norm(vel_cmd)
            if speed > self.max_speed:
                vel_cmd = vel_cmd / speed * self.max_speed
            self.pos += vel_cmd * self.dt

            # yaw tracking
            yaw_err = self._wrap_angle(self.target_yaw - self.yaw)
            yaw_rate = np.clip(2.0 * yaw_err, -self.max_yaw_rate, self.max_yaw_rate)
            self.yaw += yaw_rate * self.dt
            self.yaw = self._wrap_angle(self.yaw)

        # clamp height
        self.pos[2] = max(self.pos[2], 0.1)

        # ---- set Gazebo model state ----
        q = tft.quaternion_from_euler(0, 0, self.yaw)
        state = ModelState()
        state.model_name = self.model_name
        state.pose = Pose(
            position=Point(x=self.pos[0], y=self.pos[1], z=self.pos[2]),
            orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))
        state.reference_frame = "world"
        try:
            self.set_state(state)
        except rospy.ServiceException:
            pass

        # ---- publish pose ----
        now = rospy.Time.now()
        pose_msg = PoseStamped()
        pose_msg.header = Header(stamp=now, frame_id="world")
        pose_msg.pose = state.pose
        self.pub_pose.publish(pose_msg)

        # ---- publish odom ----
        odom = Odometry()
        odom.header = Header(stamp=now, frame_id="world")
        odom.child_frame_id = "{}/base_link".format(self.model_name)
        odom.pose.pose = state.pose
        if self.has_cmd:
            odom.twist.twist.linear = Vector3(
                x=self.target_vel[0], y=self.target_vel[1], z=0)
        self.pub_odom.publish(odom)

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = FakeDroneNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
