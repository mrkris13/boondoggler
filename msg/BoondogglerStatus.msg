# BoondogglerStatus.msg
# Message describing estimator status and hidden variable values

# Header
std_msgs/Header header

# State vector and covariance
float64[] state
float64[] covariance

int32 disturb_mode
int32 in_flight

geometry_msgs/Vector3 gyro_biases
float64 sp_thrust
float64 drag_coefficient