#!/usr/bin/env python

# ekf.py
#
# About: The following code implements an Extended Kalman Filter.
#
# Author:  Kris Frey [kfrey@mit.edu]

# Specify Import Classes. Note OS classes needed for file I/O for debugging
# For File I/O
import os
import fileinput
import sys
import re

# For Math Libraries
import numpy as np
from scipy import linalg
import math

import rospy

import model

# For ROS Messages
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Range

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import Quaternion

from tf.transformations import quaternion_from_euler

from boondoggler.msg import EstUavState

class EKF:

  x     = None
  Sigma = None
  u     = None

  prop_time     = None

  disturb_mode  = None

  def __init__(self): 
    rospy.loginfo('EKF: initializing EKF object...')

    self.x = model.init_mu
    self.Sigma = model.init_Sigma
    self.u = model.init_u

    self.disturb_mode = model.DISTURB_NOMINAL

    self.pub_pose = rospy.Publisher('boondoggler/pose', PoseStamped, queue_size=1)
    self.pub_vel = rospy.Publisher('boondoggler/vel', Vector3Stamped, queue_size=1)
    self.pub_uav_state = rospy.Publisher('boondoggler/uav_state', EstUavState, queue_size=1)
    
    self.seq = 0

    rospy.loginfo('EKF: initialized state.')
    return

  def subscribe(self):
    rospy.loginfo('EKF: starting subscribers...')
    
    self.sub_imu = rospy.Subscriber('/mavros/imu/data_raw', Imu, self.msgHandler, queue_size=1)
    rospy.loginfo('     Imu...')

    return

  # Top-level call made by measurement msg subscribers
  def msgHandler(self, msg):
    # initialize filter if not done yet and we just received IMUSample
    if self.prop_time is None:
        if isinstance(msg, Imu):
            self.prop_time = msg.header.stamp
            rospy.loginfo('EKF: started.')
        else:
            rospy.logwarn('EKF: uninitialized - ignoring messages until first IMUSample arrives.')
            return

    # handle particular measurement type
    if isinstance(msg, Imu):
        self.update_IMU(msg)

    else:
        rospy.logerr('EKF: invalid measurement type.')

    self.publish()

    return

  # Message Handlers
  def update_IMU(self, imu):

    ts = imu.header.stamp

    if ts > self.prop_time:
      # use UNCORRECTED gyro rates as control signal
      u = np.array([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z])
      new_prop_time = ts

      # Now calculate dt and propogate
      dt = (new_prop_time - self.prop_time).to_sec()
      if dt > 0.2:
          rospy.logwarn('Propogating filter by larger than expected dt.')
      (x_p, Sigma_p) = self.propogate(self.x, self.Sigma, u, dt, self.disturb_mode)

      self.prop_time = ts
      (self.x, self.Sigma) = model.enforce_bounds(x_p, Sigma_p)
      # model.print_state(self.x)

      self.u = u
    
    else:
      rospy.logwarn('IMU message recieved from the past.')

    # Now do correction based on accelerometer
    z = np.array([imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z])        

    (h, Hx, Q) = model.observation_acc(self.x, self.disturb_mode)
    (x_c, Sigma_c) = self.update(self.x, self.Sigma, z, h, Hx, Q)
    (self.x, self.Sigma) = model.enforce_bounds(x_c, Sigma_c)

    # rospy.loginfo('Completed IMU update.')
    # model.print_state(self.x)
        
    return

  # Propogation
  def propogate(self, mu, Sigma, u, dt, disturb_mode):
    # Inputs:
    #       mu:     Full state mean vector
    #       Sigma:  State covariance
    #       u:      Control signal
    #       dt:     Time elapsed since last propagation
    #       disturb_mode:  Disturbed status of vehicle
    # Outputs:
    #       mu_p:    Predicted state vector after dt
    #       Sigma_p: Predicted covariance after dt

    n = mu.size

    # propogate vehicle state 
    (f, Fx, Fu, M, R) = model.process_model(mu, u, dt, disturb_mode)

    # calculate predicted covariance
    Sigma_p = Fx.dot(Sigma).dot(Fx.transpose()) + Fu.dot(M).dot(Fu.transpose()) + R

    return (f, Sigma_p)


  # Update
  def update(self, mu, Sigma, z, z_pred, Hx, Q):
    # Inputs:
    #       mu:     Full state mean vector  
    #       Sigma:  State covariance 
    #       z:      Observation vector  
    #       z_pred: Predicted observation                   !! Must be propogated to time of observation
    #       Hx:     Observation model Jacobian wrt state    !! Must be propogated to time of observation
    #       Q:      Measurement noise
    # Outputs:
    #       mu_c:    Corrected state vector
    #       Sigma_c: Corrected covariance

    # verify input dimensions        
    n = mu.size
    m = z.size

    if mu.shape != (n,):
      raise Exception('Argument mu has incorrect dimension.')
    if Sigma.shape != (n, n):
      raise Exception('Argument Sigma has incorrect dimensions.')
    if False in np.isfinite(Sigma):
      raise Exception('Argument Sigma has non-finite elements.')
    if False in np.isfinite(Hx):
      raise Exception('Argument Hx has non-finite elements.')

    if m == 1:
      if z.shape != ():
        raise Exception('Argument z has incorrect dimension.')
      if z_pred.shape != ():
        raise Exception('Argument z_pred has incorrect dimension.')
      if Hx.shape != (n,):
        raise Exception('Argument Hx has incorrect dimension.')
      if Q.shape != ():
        raise Exception('Arugment Q has incorrect dimension.')
      if not np.isfinite(z_pred):
        raise Exception('Argument z_pred has non-finite elements.')
      if not np.isfinite(Q):
        raise Exception('Argument Q has non-finite elements.')

    else:
      if z.shape != (m,):
        raise Exception('Argument z has incorrect dimension.')
      if z_pred.shape != (m,):
        raise Exception('Argument z_pred has incorrect dimension.')
      if Hx.shape != (m, n):
        raise Exception('Argument Hx has incorrect dimension.')
      if Q.shape != (m, m):
        raise Exception('Arugment Q has incorrect dimension.')
      if False in np.isfinite(z_pred):
        raise Exception('Argument z_pred has non-finite elements.')
      if False in np.isfinite(Q):
        raise Exception('Argument Q has non-finite elements.')

    # calculate K
    A = Hx.dot(Sigma).dot(Hx.T) + Q
    if A.shape == ():
      if A == 0.0:
        rospy.logerr('EKFUpdate: Division by 0 detected.')
        return (mu, Sigma)
      try:
        A_inv = 1.0/A;
      except LinAlgError as e:
        print 'EKFUpdate: {}'.format(e.strerror)
        print A
        return (mu, Sigma)
    else:
      A_inv = linalg.inv(A)

    K = Sigma.dot(Hx.T).dot(A_inv) # K is nxm

    # update mean
    mu_c = mu + K.dot(z - z_pred)

    # update covariance
    n = mu.size
    I = np.eye(n)
    Sigma_c = (I - K.dot(Hx)).dot(Sigma)

    return (mu_c, Sigma_c)

  def publish(self):
    if self.prop_time == None:
      # filter not initialized
      # Don't publish
      return

    pose = PoseStamped()
    pose.header.stamp = self.prop_time
    pose.header.frame_id = '/world'
    pose.header.seq = self.seq

    roll = self.x[model.VAR_ROLL]
    pitch = self.x[model.VAR_PITCH]
    yaw = self.x[model.VAR_YAW]

    q = quaternion_from_euler(roll, pitch, yaw)
    pose.pose.orientation = Quaternion(*q)

    pose.pose.position.x =  self.x[model.VAR_POS_X]
    pose.pose.position.y =  self.x[model.VAR_POS_Y]
    pose.pose.position.z =  self.x[model.VAR_POS_Z]

    self.pub_pose.publish(pose)

    vels = Vector3Stamped()
    vels.header.stamp = self.prop_time
    vels.header.frame_id = '/body'
    vels.header.seq = self.seq

    vels.vector.x =  self.x[model.VAR_VEL_U]
    vels.vector.y =  self.x[model.VAR_VEL_V]
    vels.vector.z =  self.x[model.VAR_VEL_W]

    self.pub_vel.publish(vels)



    self.seq += 1
    return


rospy.init_node('boondoggler')
ekf = EKF()
ekf.subscribe()

rospy.spin()
