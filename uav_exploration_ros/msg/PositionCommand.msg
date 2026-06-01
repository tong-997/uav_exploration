# Compatible with ego-planner quadrotor_msgs/PositionCommand
std_msgs/Header header
geometry_msgs/Point position
geometry_msgs/Vector3 velocity
geometry_msgs/Vector3 acceleration
float64 yaw
float64 yaw_dot
int32 trajectory_id
uint8 trajectory_flag
uint8 TRAJECTORY_STATUS_EMPTY = 0
uint8 TRAJECTORY_STATUS_READY = 1
uint8 TRAJECTORY_STATUS_EXEC  = 2
