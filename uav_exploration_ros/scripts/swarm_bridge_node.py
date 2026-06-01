#!/usr/bin/env python
"""
swarm_bridge_node.py
====================
Lightweight relay: forward local drone pose to a shared namespace
and record waypoints. Similar to ego-planner-swarm's swarm_bridge.

Topics:
  sub: /mavros/local_position/pose  (geometry_msgs/PoseStamped) — from flight controller
  pub: ~pose                        (geometry_msgs/PoseStamped) — unified output to all local nodes
       ~waypoints                   (nav_msgs/Path) — accumulated waypoints for logging
"""
import numpy as np
import json
import os
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header


class SwarmBridgeNode:

    def __init__(self):
        rospy.init_node("swarm_bridge", anonymous=False)

        self.drone_id = rospy.get_param("~drone_id", 0)
        self.min_wp_dist = rospy.get_param("~min_waypoint_dist", 0.5)
        self.output_dir = rospy.get_param("~waypoint_output_dir", "/tmp/uav_exploration")

        self.waypoints = []
        self.last_pos = None

        # ---- publishers ----
        self.pub_pose = rospy.Publisher("~pose", PoseStamped, queue_size=10)
        self.pub_wp = rospy.Publisher("~waypoints", Path, queue_size=2)

        # ---- subscribers ----
        # MAVROS or Gazebo ground-truth pose
        pose_topic = rospy.get_param(
            "~pose_input_topic", "mavros/local_position/pose")
        rospy.Subscriber(pose_topic, PoseStamped, self.pose_cb, queue_size=10)

        rospy.on_shutdown(self.save_waypoints)
        rospy.loginfo("swarm_bridge ready  drone_id=%d", self.drone_id)

    def pose_cb(self, msg):
        # republish with consistent frame
        out = PoseStamped()
        out.header = Header(stamp=msg.header.stamp, frame_id="world")
        out.pose = msg.pose
        self.pub_pose.publish(out)

        p = msg.pose.position
        pos = np.array([p.x, p.y, p.z])
        if self.last_pos is None or np.linalg.norm(pos[:2] - self.last_pos[:2]) >= self.min_wp_dist:
            self.last_pos = pos
            self.waypoints.append({
                "x": float(p.x), "y": float(p.y), "z": float(p.z),
                "t": msg.header.stamp.to_sec()
            })

    def save_waypoints(self):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir,
                            "waypoints_uav{}.json".format(self.drone_id))
        with open(path, "w") as f:
            json.dump(self.waypoints, f, indent=2)
        rospy.loginfo("Saved %d waypoints → %s", len(self.waypoints), path)

        # also publish as Path
        msg = Path()
        msg.header = Header(stamp=rospy.Time.now(), frame_id="world")
        for wp in self.waypoints:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = wp["x"]
            ps.pose.position.y = wp["y"]
            ps.pose.position.z = wp["z"]
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_wp.publish(msg)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = SwarmBridgeNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
