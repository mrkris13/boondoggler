#!/usr/bin/env python

from model import *
import numpy as np 

np.set_printoptions(precision=1)

# define linearlization point
X = np.zeros(VAR_COUNT)
X[VAR_ROLL]     = 0.1
X[VAR_PITCH]    = 0.3
X[VAR_YAW]      = 0.3
X[VAR_VEL_U]    = -0.3
X[VAR_VEL_V]    = 1.0
X[VAR_VEL_W]    = 0.5
X[VAR_SP_THRUST]  = grav_acc
X[VAR_POS_X]    = 0.2
X[VAR_POS_Y]    = 0.3
X[VAR_POS_Z]    = 1
X[VAR_GBIAS_P]  = 0.1
X[VAR_GBIAS_Q]  = .10
X[VAR_GBIAS_R]  = 0.1
X[VAR_DRAG_CO]  = 0.2

# we need sample control input as well
U = np.zeros(3)

# now calculate Jacobian at X: A(X)
A = np.zeros([VAR_COUNT, VAR_COUNT])

A[VAR_ROLL:VAR_ROLL+3, VAR_ROLL]  = gyroRateTFJacobianRoll(X).dot(  U - X[VAR_GBIAS_P:VAR_GBIAS_P+3] )
A[VAR_ROLL:VAR_ROLL+3, VAR_PITCH] = gyroRateTFJacobianPitch(X).dot( U - X[VAR_GBIAS_P:VAR_GBIAS_P+3] )
A[VAR_ROLL:VAR_ROLL+3, VAR_YAW]   = gyroRateTFJacobianYaw(X).dot(   U - X[VAR_GBIAS_P:VAR_GBIAS_P+3] )
A[VAR_ROLL:VAR_ROLL+3, VAR_GBIAS_P:VAR_GBIAS_P+3] = gyroRateTF(X)

A[VAR_VEL_U:VAR_VEL_U+3, VAR_ROLL]  = i2bRotMatrixJacobianRoll(X).dot(  grav_vect )
A[VAR_VEL_U:VAR_VEL_U+3, VAR_PITCH] = i2bRotMatrixJacobianPitch(X).dot( grav_vect )
A[VAR_VEL_U:VAR_VEL_U+3, VAR_YAW]   = i2bRotMatrixJacobianYaw(X).dot(   grav_vect )
A[VAR_VEL_U, VAR_VEL_U]   = -X[VAR_DRAG_CO]
A[VAR_VEL_V, VAR_VEL_V]   = -X[VAR_DRAG_CO]
A[VAR_VEL_U, VAR_DRAG_CO] = -X[VAR_VEL_U]
A[VAR_VEL_V, VAR_DRAG_CO] = -X[VAR_VEL_V]
A[VAR_VEL_W, VAR_SP_THRUST]   = 1

A[VAR_POS_X:VAR_POS_X+3, VAR_ROLL]  = i2bRotMatrixJacobianRoll(X).transpose().dot(   X[VAR_VEL_U:VAR_VEL_U+3] )
A[VAR_POS_X:VAR_POS_X+3, VAR_PITCH] = i2bRotMatrixJacobianPitch(X).transpose().dot(  X[VAR_VEL_U:VAR_VEL_U+3] )
A[VAR_POS_X:VAR_POS_X+3, VAR_YAW]   = i2bRotMatrixJacobianYaw(X).transpose().dot(    X[VAR_VEL_U:VAR_VEL_U+3] )
A[VAR_POS_X:VAR_POS_X+3, VAR_VEL_U:VAR_VEL_U+3] = i2bRotMatrix(X).transpose()

dt = 0.02 # 50Hz
Ak = np.eye(VAR_COUNT) + A*dt

# calculate accel measurement Jacobian at X: C(X)
C = np.zeros([3, VAR_COUNT])

C[0, VAR_VEL_U]      = -X[VAR_DRAG_CO]
C[0, VAR_DRAG_CO]    = -X[VAR_VEL_U]
C[1, VAR_VEL_V]      = -X[VAR_DRAG_CO]
C[1, VAR_DRAG_CO]    = -X[VAR_VEL_V]
C[2, VAR_SP_THRUST]  =  1.0

# # sonar measurement Jacobian
# C = np.zeros(VAR_COUNT)
# C[VAR_POS_Z] = 1

# now calculate observability matrix Mo
Mo = np.zeros([3*VAR_COUNT, VAR_COUNT])
A_i = np.eye(VAR_COUNT)
for i in range(VAR_COUNT):
  j = 3*i
  Mo[j:j+3,:] = C.dot(A_i)
  A_i = A_i.dot(Ak)

# output rank of Mo
print 'Rank of Mo is {}\n'.format(np.linalg.matrix_rank(Mo))

observed_vars = abs(np.sum(Mo,0)) > 0
print observed_vars
