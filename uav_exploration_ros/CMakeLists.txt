cmake_minimum_required(VERSION 3.0.2)
project(uav_exploration_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  message_generation
)

add_message_files(FILES
  Frontier.msg
  FrontierArray.msg
  PositionCommand.msg
)

generate_messages(DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_python_setup()

catkin_package(CATKIN_DEPENDS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  message_runtime
)

catkin_install_python(PROGRAMS
  scripts/depth_estimator_node.py
  scripts/grid_map_node.py
  scripts/frontier_node.py
  scripts/planner_node.py
  scripts/swarm_bridge_node.py
  scripts/fake_drone_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config worlds models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
