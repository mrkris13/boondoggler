# EstUavState.msg
# Message describing estimated current state of the UAV

# Header
std_msgs/Header header

# State vector and covariance
float64[] state
float64[] covariance

int32 disturb_mode

# State field definitions
int32 IDX_ROLL 		= 0
int32 IDX_PITCH 	= 1
int32 IDX_YAW 		= 2
int32 IDX_VEL_U 	= 3
int32 IDX_VEL_V 	= 4
int32 IDX_VEL_W 	= 5
int32 IDX_THRUST	= 6
int32 IDX_POS_X		= 7
int32 IDX_POS_Y		= 8
int32 IDX_POS_Z		= 9
int32 IDX_DRAG_COEFF	= 10