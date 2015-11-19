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
VAR_GBIAS_P   = 10    # gyro biases
VAR_GBIAS_Q   = 11
VAR_GBIAS_R   = 12
VAR_DRAG_CO   = 13    # drag coefficient

# Initial state estimate
init_mu = np.zeros(VAR_COUNT)
init_mu[VAR_POS_Z]      = -0.05
init_mu[VAR_GBIAS_P]    =  0
init_mu[VAR_GBIAS_R]    =  0
init_mu[VAR_GBIAS_Q]    =  0
init_mu[VAR_SP_THRUST]     = grav_acc
init_mu[VAR_OF_FL]      = 1.0
init_mu[VAR_DRAG_CO]    = 1.0

init_Sigmas = np.zeros(VAR_COUNT)
init_Sigmas[VAR_ROLL:VAR_YAW+1]     = 1e-2*np.ones(3)
init_Sigmas[VAR_VEL_U:VAR_VEL_W+1]  = 1e-1*np.ones(3)
init_Sigmas[VAR_SP_THRUST]          = 0.5
init_Sigmas[VAR_POS_X:VAR_POS_Z+1]  = 1e-3*np.ones(3)  # we are pretty certain because we define initial position as origin
init_Sigmas[VAR_GBIAS_P:VAR_GBIAS_R+1] = 1e-2*np.ones(3)
init_Sigmas[VAR_DRAG_CO]            = 2.0

# Disturbance modes
DISTURB_NOMINAL = 0   # no disturbance detected
DISTURB_ACTIVE  = 1   # disturbance detected, inflate process

## Process Models
def process_model(x, u, dt, disturb_mode):
# Inputs:
#       x:      Current vehicle state
#       u:      Control signal (raw gyro rates)
#       dt:     Propogation time [sec]
#       disturb_mode:  Flag to use disturbance mode
# Outputs:
#       f:      Predicted state
#       Fx:     Jacobian wrt to state
#       Fu:     Jacobian wrt to control signal
#       M:      Measurement covariance matrix rate
#       R:      Process model covariance matrix rate
  
    # extract useful state variables
    vel_u = x[VAR_VEL_U]
    vel_v = x[VAR_VEL_V]
    vel_w = x[VAR_VEL_W]
    thrust = x[VAR_SP_THRUST]
    drag_coeff  = x[VAR_DRAG_CO]
    gyro_biases = x[VAR_GBIAS_P:UAV_GBIAS_P+3]

    body_vels = np.array([vel_u, vel_v, vel_w])

    # extract gyro rates
    corr_gyro = u - gyro_biases

    # gravity vector
    grav_vect = np.array([0, 0, grav_acc])

    ## Calculate mean f
    # calculate inertial-to-body coordinate frame
    R = i2bRotMatrix(x)
    Rt = R.T

    # define transform from gyro body rates to Euler rates
    L_gyro = gyroRateTF(x)

    # calculate state rate vector
    df = np.zeros(VAR_COUNT)
    df[VAR_ROLL:VAR_ROLL+3]     = L_gyro.dot(corr_gyro)
    df[VAR_VEL_U:VAR_VEL_U+3]   = R.dot(grav_vect) + np.array([-drag_coeff*vel_u, -drag_coeff*vel_v, -thrust])
    df[VAR_POS_X:VAR_POS_X+3]   = Rt.dot(body_vels)
    
    # integrate state rate vector to get new mean
    f  = x + dt*df

    ## Calculate Jacobians
    # robot model Jacobian wrt x_v
    dFx = np.zeros((VAR_COUNT, VAR_COUNT))
    # robot model Jacobian wrt u
    dFu = np.zeros((VAR_COUNT, 3))

    # calc partial derivatives of rotation matrix
    dR_dRoll    = i2bRotMatrixJacobianRoll(x)
    dR_dPitch   = i2bRotMatrixJacobianPitch(x)
    dR_dYaw     = i2bRotMatrixJacobianYaw(x)

    # calc partial derivatives of gyro rate transform matrix
    dL_dRoll    = gyroRateTFJacobianRoll(x)
    dL_dPitch   = gyroRateTFJacobianPitch(x)
    dL_dYaw     = gyroRateTFJacobianYaw(x)

    # calc partial derivatives of inverse transform
    dRt_dRoll   = dR_dRoll.T
    dRt_dPitch  = dR_dRoll.T
    dRt_dYaw    = dR_dYaw.T

    # euler angular derivatives
    dFx[VAR_ROLL:VAR_ROLL+3, VAR_ROLL] = dL_dRoll.dot(corr_gyro)
    dFx[VAR_ROLL:VAR_ROLL+3, VAR_PITCH] = dL_dPitch.dot(corr_gyro)
    dFx[VAR_ROLL:VAR_ROLL+3, VAR_YAW] = dL_dYaw.dot(corr_gyro)
    
    dFu[VAR_ROLL:VAR_ROLL+3, 0] = L_gyro.dot(np.array([1, 0, 0]))
    dFu[VAR_ROLL:VAR_ROLL+3, 1] = L_gyro.dot(np.array([0, 1, 0]))
    dFu[VAR_ROLL:VAR_ROLL+3, 2] = L_gyro.dot(np.array([0, 0, 1]))
    
    dFx[VAR_ROLL:VAR_ROLL+3, VAR_GBIAS_P] = L_gyro.dot(np.array([-1, 0, 0]))
    dFx[VAR_ROLL:VAR_ROLL+3, VAR_GBIAS_Q] = L_gyro.dot(np.array([0, -1, 0]))
    dFx[VAR_ROLL:VAR_ROLL+3, VAR_GBIAS_R] = L_gyro.dot(np.array([0, 0, -1]))

    # body-frame translational velocity derivatives
    dFx[VAR_VEL_U:VAR_VEL_U+3, VAR_ROLL]     = dR_dRoll.dot(grav_vect)
    dFx[VAR_VEL_U:VAR_VEL_U+3, VAR_PITCH]    = dR_dPitch.dot(grav_vect)
    dFx[VAR_VEL_U:VAR_VEL_U+3, VAR_YAW]      = dR_dYaw.dot(grav_vect)

    dFx[VAR_VEL_U, VAR_VEL_U]    = -drag_coeff
    dFx[VAR_VEL_U, VAR_DRAG_CO]  = -vel_u
    dFx[VAR_VEL_V, VAR_VEL_V]    = -drag_coeff
    dFx[VAR_VEL_V, VAR_DRAG_CO]  = -vel_v
    dFx[VAR_VEL_W, VAR_SP_THRUST]   = -1.0

    # inertial-frame position derivatives
    dFx[VAR_POS_X:VAR_POS_X+3, VAR_ROLL]   = dRt_dRoll.dot(body_vels)
    dFx[VAR_POS_X:VAR_POS_X+3, VAR_PITCH]  = dRt_dPitch.dot(body_vels)
    dFx[VAR_POS_X:VAR_POS_X+3, VAR_YAW]    = dRt_dYaw.dot(body_vels)
    dFx[VAR_POS_X:VAR_POS_X+3, VAR_VEL_U]  = Rt.dot(np.array([1, 0, 0]))
    dFx[VAR_POS_X:VAR_POS_X+3, VAR_VEL_V]  = Rt.dot(np.array([0, 1, 0]))
    dFx[VAR_POS_X:VAR_POS_X+3, VAR_VEL_W]  = Rt.dot(np.array([0, 0, 1]))

    # integration
    Fx = np.eye(VAR_VAR_COUNT) + dt*dFx
    Fu = dt*dFu

    ## Measurement and process noise matrices -- TUNABLE
    M = np.diag([0.015**2, 0.015**2, 0.015**2]) * dt**2
    
    dR = np.zeros((VAR_VAR_COUNT, VAR_VAR_COUNT))
    dR[VAR_ROLL, VAR_ROLL]   = 0.25**2
    dR[VAR_PITCH, VAR_PITCH] = 0.25**2
    dR[VAR_YAW, VAR_YAW]     = 0.25**2
    dR[VAR_VEL_U, VAR_VEL_U] = 0.25**2
    dR[VAR_VEL_V, VAR_VEL_V] = 0.25**2
    dR[VAR_VEL_W, VAR_VEL_W] = 0.25**2
    dR[VAR_SP_THRUST, VAR_SP_THRUST] = 1.5**2
    dR[VAR_POS_X, VAR_POS_X] = 0.25**2
    dR[VAR_POS_Y, VAR_POS_Y] = 0.25**2
    dR[VAR_POS_Z, VAR_POS_Z] = 0.25**2
    dR[VAR_DRAG_CO, VAR_DRAG_CO] = 1e-6**2
    dR[VAR_OF_FL, VAR_OF_FL] = 0.000001**2
    dR[VAR_GBIAS_P, VAR_GBIAS_P] = 1e-6**2
    dR[VAR_GBIAS_Q, VAR_GBIAS_Q] = 1e-6**2
    dR[VAR_GBIAS_R, VAR_GBIAS_R] = 1e-6**2

    if disturb_mode == DISTURB_ACTIVE:
      # inflate noise matrices appropriately
      dR[VAR_ROLL:VAR_ROLL+3, VAR_ROLL:VAR_ROLL+3]      += 0.30**2*np.eye(3)
      dR[VAR_VEL_U:VAR_VEL_U+3, VAR_VEL_U:VAR_VEL_U+3]  += 0.20**2*np.eye(3)
      dR[VAR_POS_X:VAR_POS_X+3, VAR_POS_X:VAR_POS_X+3]  += 0.10**2*np.eye(3)

    R = dR * dt**2

    return (f, Fx, Fu, M, R)

########################### Observation Models

def observation_acc(x):
# Inputs:
#   x:    Vehicle state vector 
# Outputs:
#   h:    Prediction of measurement given current state
#   Hx:   Jacobian of measurement wrt robot state
#   Q:    Measurement covariance matrix
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
  