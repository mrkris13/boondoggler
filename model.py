# model.py
# Author: Kris Frey [kfrey@mit.edu]


import numpy as np
from math import cos, sin, tan
import math

import rospy

# Physical constants
grav_acc = 9.80665 #[m/s]
mass = 1.0 # [kg]

# Vehicle state variable enumeration
VAR_COUNT     = 14
VAR_ROLL      = 0     # attitude euler angles
VAR_PITCH     = 1
VAR_YAW       = 2
VAR_VEL_U     = 3     # body-frame velocities
VAR_VEL_V     = 4
VAR_VEL_W     = 5
VAR_SP_THRUST = 6     # specific thrust
VAR_POS_X     = 7     # local-frame position
VAR_POS_Y     = 8
VAR_POS_Z     = 9
VAR_DRAG_CO   = 10    # drag coefficient
VAR_GBIAS_P   = 11    # gyro biases
VAR_GBIAS_Q   = 12
VAR_GBIAS_R   = 13

# Initial state estimate
init_mu = np.zeros(VAR_COUNT)
init_mu[VAR_POS_Z]      = -0.05
init_mu[VAR_GBIAS_P]    =  0
init_mu[VAR_GBIAS_R]    =  0
init_mu[VAR_GBIAS_Q]    =  0
init_mu[VAR_THRUST]     = grav_acc
init_mu[VAR_OF_FL]      = 1.0
init_mu[VAR_DRAG_CO]    = 1.0

init_Sigmas = np.zeros(VAR_COUNT)
init_Sigmas[VAR_ROLL:VAR_YAW+1]     = 1e-2*np.ones(3)
init_Sigmas[VAR_VEL_U:VAR_VEL_W+1]  = 1e-1*np.ones(3)
init_Sigmas[VAR_SP_THRUST]          = 0.5
init_Sigmas[VAR_POS_X:VAR_POS_Z+1]  = 1e-3*np.ones(3)  # we are pretty certain because we define initial position as origin
init_Sigmas[VAR_DRAG_CO]            = 2.0
init_Sigmas[VAR_GBIAS_P:VAR_GBIAS_R+1] = 1e-2*np.ones(3) 

## Process Models
def process_model(x, u, dt):
# Inputs:
#       x:    Current vehicle state
#       u:      Control signal (raw gyro rates)
#       dt:     Propogation time [sec]
# Outputs:
#       f:      Predicted state
#       Fx:     Jacobian wrt to state
#       Fu:     Jacobian wrt to control signal
#       M:      Measurement covariance matrix rate
#       R:      Process model covariance matrix rate

    return (f, Fx, Fu, M, R)


########################### Observation Models

def observation_acc(x):
# Inputs:
#   x:  	Vehicle state vector 
# Outputs:
#   h: 		Prediction of measurement given current state
#		Hx: 	Jacobian of measurement wrt robot state
#		Q: 		Measurement covariance matrix
    return (h,Hx,Q)

################### Transform Utility functions

# Defines inertial-to-body transform based on input Euler angles
def i2bRotMatrix(x):
    roll = x[VAR_ROLL]
    pitch = x[VAR_PITCH]
    yaw = x[VAR_YAW]

    R_yaw = np.array([[cos(yaw), sin(yaw), 0], \
        [-sin(yaw), cos(yaw), 0], \
        [0, 0, 1]])
    R_pitch = np.array([[cos(pitch), 0, -sin(pitch)], \
        [0, 1, 0], \
        [sin(pitch), 0, cos(pitch)]])
    R_roll = np.array([[1, 0, 0], \
        [0, cos(roll), sin(roll)], \
        [0, -sin(roll), cos(roll)]])

    R_bi = R_roll.dot(R_pitch.dot(R_yaw))
    return R_bi

# Calculate rotation matrix partial derivitive with respect to roll
def i2bRotMatrixJacobianRoll(x):
    roll = x[VAR_ROLL]
    pitch = x[VAR_PITCH]
    yaw = x[VAR_YAW]

    dRbi = np.zeros((3,3))

    dRbi[1, 0] = sin(roll)*sin(yaw) + cos(roll)*sin(pitch)*cos(yaw)
    dRbi[1, 1] = -sin(roll)*cos(yaw) + cos(roll)*sin(pitch)*sin(yaw)
    dRbi[1, 2] = cos(roll)*cos(pitch)

    dRbi[2, 0] = cos(roll)*sin(yaw) - sin(roll)*sin(pitch)*cos(yaw)
    dRbi[2, 1] = -cos(roll)*cos(yaw) - sin(roll)*sin(pitch)*sin(yaw)
    dRbi[2, 2] = -sin(roll)*cos(pitch)

    return dRbi

# Calculate rotation matrix partial derivitive with respect to pitch
def i2bRotMatrixJacobianPitch(x):
    roll = x[VAR_ROLL]
    pitch = x[VAR_PITCH]
    yaw = x[VAR_YAW]

    dRbi = np.zeros((3,3))

    dRbi[0, 0] = -sin(pitch)*cos(yaw)
    dRbi[0, 1] = -sin(pitch)*sin(yaw)
    dRbi[0, 2] = -cos(pitch)

    dRbi[1, 0] = sin(roll)*cos(pitch)*cos(yaw)
    dRbi[1, 1] = sin(roll)*cos(pitch)*sin(yaw)
    dRbi[1, 2] = -sin(roll)*sin(pitch)

    dRbi[2, 0] = cos(roll)*cos(pitch)*cos(yaw)
    dRbi[2, 1] = cos(roll)*cos(pitch)*sin(yaw)
    dRbi[2, 2] = -cos(roll)*sin(pitch)

    return dRbi

# Calculate rotation matrix partial derivitive with respect to yaw
def i2bRotMatrixJacobianYaw(x):
    roll = x[VAR_ROLL]
    pitch = x[VAR_PITCH]
    yaw = x[VAR_YAW]

    dRbi = np.zeros((3,3))

    dRbi[0, 0] = -cos(pitch)*sin(yaw)
    dRbi[0, 1] = cos(pitch)*cos(yaw)
    dRbi[0, 2] = 0

    dRbi[1, 0] = -cos(roll)*cos(yaw) - sin(roll)*sin(pitch)*sin(yaw)
    dRbi[1, 1] = -cos(roll)*sin(yaw) + sin(roll)*sin(pitch)*cos(yaw)
    dRbi[1, 2] = 0

    dRbi[2, 0] = sin(roll)*cos(yaw) - cos(roll)*sin(pitch)*sin(yaw)
    dRbi[2, 1] = sin(roll)*sin(yaw) + cos(roll)*sin(pitch)*cos(yaw)
    dRbi[2, 2] = 0

    return dRbi

# Calculate transform matrix from gyro rates to Euler rates
def gyroRateTF(x):
    roll = x[VAR_ROLL]
    pitch = x[VAR_PITCH]
    yaw = x[VAR_YAW]

    L = np.array([                                          \
        [1, sin(roll)*tan(pitch), cos(roll)*tan(pitch)],    \
        [0, cos(roll), -sin(roll)],                         \
        [0, sin(roll)/cos(pitch), cos(roll)/cos(pitch)] ])

    return L

# Calculate gyro rate transform matrix derivatives
def gyroRateTFJacobianRoll(x):
    roll = x[VAR_ROLL]
    pitch = x[VAR_PITCH]
    yaw = x[VAR_YAW]

    dL = np.array([                                          \
        [0, cos(roll)*tan(pitch), -sin(roll)*tan(pitch)],    \
        [0, -sin(roll), -cos(roll)],                         \
        [0, cos(roll)/cos(pitch), -sin(roll)/cos(pitch)] ])

    return dL

def gyroRateTFJacobianPitch(x):
    roll = x[VAR_ROLL]
    pitch = x[VAR_PITCH]
    yaw = x[VAR_YAW]

    dL = np.array([                                                 \
        [0, sin(roll)/(cos(pitch)**2), cos(roll)/(cos(pitch)**2)],  \
        [0, 0, 0],                                                  \
        [0, sin(roll)*sin(pitch)/cos(pitch)**2, cos(roll)*sin(pitch)/cos(pitch)**2] ])

    return dL

def gyroRateTFJacobianYaw(x):
    roll = x[VAR_ROLL]
    pitch = x[VAR_PITCH]
    yaw = x[VAR_YAW]

    dL = np.zeros((3,3))
    return dL

def enforce_symmetry(A):
    n = A.shape[0]
    if A.shape[1] != n:
        raise Exception('Input matrix is not square.')
    for i in range(n):
        for j in range(i):
            # i,j is coord of element in lower triangle
            # j,i is coord of symmetric element in upper triangle
            m = (A[i,j]+A[j,i])/2.0
            A[i,j] = m
            A[j,i] = m
    return A

def print_state(x):
    print 'Roll = {} [rad]'.format(x[UAV_ROLL])
    print 'Pitch = {} [rad]'.format(x[UAV_PITCH])
    print 'Yaw = {} [rad]'.format(x[UAV_YAW])
    print 'NED Position = ({}, {}, {}) [m]'.format(x[UAV_POS_X], x[UAV_POS_Y], x[UAV_POS_Z])
    print 'Body velocities = ({}, {}, {}) [m/s]'.format(x[UAV_VEL_U], x[UAV_VEL_V], x[UAV_VEL_W])
    print 'Gyro biases = ({}, {}, {}) [rad/s]'.format(x[UAV_GBIAS_P], x[UAV_GBIAS_R], x[UAV_GBIAS_Q])
    print 'Mag reference vector = ({}, {}, {}) [Gauss].'.format(x[UAV_MREF_X], x[UAV_MREF_Y], x[UAV_MREF_Z])
    print 'Drag coefficient = {}'.format(x[UAV_DRAG_CO])
    print 'Thrust = {} [N]'.format(x[UAV_THRUST])
    print 'OptFlow Scale Factor = {}'.format(x[UAV_OF_FL])

    return
    