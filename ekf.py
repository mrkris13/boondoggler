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
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Quaternion

from mavros_msgs.msg import OpticalFlowRad

from tf.transformations import quaternion_from_euler

from boondoggler.msg import BoondogglerStatus

class EKF:

  x     = None
  Sigma = None
  u     = None

  prop_time     = None

  disturb_mode  = None

  flight_state  = None
  takeoff_ts    = None

  def __init__(self): 
    rospy.loginfo('EKF: initializing EKF object...')

    self.x = model.init_mu
    self.Sigma = model.init_Sigma
    self.u = model.init_u

    self.disturb_mode = model.DISTURB_NOMINAL
    self.flight_state = model.FLIGHT_STATE_GROUNDED   # STRONG ASSUMPTION we start on the ground

    self.pub_pose = rospy.Publisher('boondoggler/pose', PoseStamped, queue_size=1)
    self.pub_vel = rospy.Publisher('boondoggler/vel', TwistStamped, queue_size=1)
    self.pub_status = rospy.Publisher('boondoggler/status', BoondogglerStatus, queue_size=1)

    self.seq = 0

    rospy.loginfo('EKF: initialized state.')
    return

  def subscribe(self):
    rospy.loginfo('EKF: starting subscribers...')
    
    self.sub_imu = rospy.Subscriber('/mavros/imu/data_raw', Imu, self.msgHandler, queue_size=1)
    rospy.loginfo('     Imu...')
    self.sub_lidarlite = rospy.Subscriber('/mavros/distance_sensor/lidarlite_pub', Range, self.msgHandler, queue_size=1)
    rospy.loginfo('     Lidarlite...')
    self.sub_optflow = rospy.Subscriber('/mavros/px4flow/raw/optical_flow_rad', OpticalFlowRad, self.msgHandler, queue_size=1)
    rospy.loginfo('     PX4Flow...')
    return

  # Top-level call made by measurement msg subscribers
  def msgHandler(self, msg):
    # handle particular measurement type
    if isinstance(msg, Imu):
      self.update_IMU(msg)
    elif isinstance(msg, Range):
      self.update_Lidarlite(msg)
    elif isinstance(msg, OpticalFlowRad):
      self.update_OptFlowRad(msg)
    else:
      rospy.logerr('EKF: invalid measurement type.')

    if self.prop_time is not None:
      self.publish()

    return

  # Message Handlers
  def update_IMU(self, imu):

    ts = imu.header.stamp
    # use UNCORRECTED rates as control input
    gyro_u = np.array([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z])
    acc = np.array([imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z])

    if self.flight_state == model.FLIGHT_STATE_FLIGHT:
      # check for bump/disturbance
      if model.accel_detect_bump(self.x, acc, 9.0):
        self.disturb_mode = model.DISTURB_ACTIVE
        self.bump_time = ts
        rospy.loginfo('Bump detected')
      else:
        # if we're not nominal, but it's been a while since last bump, return to nominal
        if self.disturb_mode != model.DISTURB_NOMINAL and (ts - self.bump_time).to_sec() > 0.25:
          self.disturb_mode = model.DISTURB_NOMINAL
          rospy.loginfo('Return to nominal')

      # propogate if neccessary
      self.propogate_from_imu(ts, gyro_u) 

      # And do correction based on accelerometer
      (h, Hx, Q) = model.observation_acc_flight(self.x, self.disturb_mode)
      (x_c, Sigma_c) = self.update(self.x, self.Sigma, acc, h, Hx, Q)
      (self.x, self.Sigma) = model.enforce_bounds(x_c, Sigma_c)

    elif self.flight_state == model.FLIGHT_STATE_TAKEOFF:
      # just propogate imu
      self.propogate_from_imu(ts, gyro_u)

      # transition to flight mode after short period
      if (ts - self.takeoff_ts).to_sec() > 1.0:
        self.flight_state = model.FLIGHT_STATE_FLIGHT
        rospy.loginfo('Entering flight mode.')

    else:   # we're grounded
      # check for takeoff
      if model.accel_detect_takeoff(self.x, acc):
        rospy.loginfo('Takeoff detected.')
        self.flight_state = model.FLIGHT_STATE_TAKEOFF
        self.takeoff_ts = ts
        # assume we start in disturbed state
        self.disturb_mode = model.DISTURB_ACTIVE
        self.bump_time = ts

      else: # use grounded data
        # # if there is very little motion, we can zupt
        if not model.accel_detect_bump(self.x, acc, 0.05):
          rospy.loginfo('zupting')
          (h, Hx, Q) = model.observation_zupt(self.x)
          (x_c, Sigma_c) = self.update(self.x, self.Sigma, gyro_u, h, Hx, Q)
          (self.x, self.Sigma) = model.enforce_bounds(x_c, Sigma_c)
          self.prop_time = ts

        # otherwise propogate grounded model
        # and use grounded accelerometer model
        else:
          self.propogate_from_imu(ts, gyro_u)
          
          (h, Hx, Q) = model.observation_acc_ground(self.x, self.disturb_mode)
          (x_c, Sigma_c) = self.update(self.x, self.Sigma, acc, h, Hx, Q)
          (self.x, self.Sigma) = model.enforce_bounds(x_c, Sigma_c)
          self.prop_time = ts
    return

  def update_Lidarlite(self, lidarlite):
    ts = lidarlite.header.stamp
    r = lidarlite.range

    # small measurements may be invalid
    if r < 0.05:
      return

    # handle timing issues
    if self.prop_time is not None:
      dt = (ts - self.prop_time).to_sec()
      if dt < -0.1:
        rospy.logwarn('lidarlite message %f seconds old, skipping', -dt)
        return
      elif dt > 0.1:
        rospy.logwarn('lidarlite message %f seconds in future, skipping', dt)
        return
      elif dt > 0:
        (x, Sigma) = self.process(self.x, self.Sigma, self.u, dt, self.disturb_mode)
      else:
        (x, Sigma) = (self.x, self.Sigma)

    # Now do correction
    z = np.array([r])

    (h,Hx,Q) = model.observation_alt_lidar(x, self.disturb_mode)
    (x_c, Sigma_c) = self.update(x, Sigma, z, h, Hx, Q)
    (self.x, self.Sigma) = model.enforce_bounds(x_c, Sigma_c)

    return

  def update_OptFlowRad(self, optflow):
    ts = optflow.header.stamp
    r = optflow.distance

    print 'opt flow distance is ', r
    # small measurements may be invalid
    if r < 0.32:
      return

    # sonar may give wonky measurements when uav isn't level
    if not model.vehicle_is_level(self.x):
      return

    # handle timing issues
    if self.prop_time is not None:
      dt = (ts - self.prop_time).to_sec()
      if dt < -0.1:
        rospy.logwarn('lidarlite message %f seconds old, skipping', -dt)
        return
      elif dt > 0.1:
        rospy.logwarn('lidarlite message %f seconds in future, skipping', dt)
        return
      elif dt > 0:
        (x, Sigma) = self.process(self.x, self.Sigma, self.u, dt, self.disturb_mode)
      else:
        (x, Sigma) = (self.x, self.Sigma)

    # Now do correction
    z = np.array([r])

    (h,Hx,Q) = model.observation_alt_px4flow(x, self.disturb_mode, z)
    (x_c, Sigma_c) = self.update(x, Sigma, z, h, Hx, Q)
    (self.x, self.Sigma) = model.enforce_bounds(x_c, Sigma_c)

    return

  # Propogation
  def propogate_from_imu(self, ts, u):
    # Inputs:
    #       ts:     Timestamp of imu measurement
    #       u:      Imu rates

    if self.prop_time is None:
      # initialize filter
      self.prop_time = ts
      rospy.loginfo('Ekf started')
      return

    dt = (ts - self.prop_time).to_sec()
    if dt > 0: # we need to propogate
      if dt > 0.2:
        rospy.logwarn('Propogating filter by larger than expected dt: %f sec', dt)
      (x_p, Sigma_p) = self.process(self.x, self.Sigma, u, dt, self.disturb_mode)

      (self.x, self.Sigma) = model.enforce_bounds(x_p, Sigma_p)
      self.u = u
      self.prop_time = ts
    
    else:
      rospy.logwarn('IMU message received %f seconds from the past.', -dt)

    return
  
  def propogate_blind(self, ts):
    # Inputs:
    #       ts:     Timestamp of imu measurement
    if ts > self.prop_time: # we need to propogate
      # Now calculate dt and propogate
      dt = (ts - self.prop_time).to_sec()
      if dt > 0.2:
        rospy.logwarn('Propogating filter by larger than expected dt: %f sec', dt)
      (x_p, Sigma_p) = self.process(self.x, self.Sigma, self.u, dt, self.disturb_mode)

      (self.x, self.Sigma) = model.enforce_bounds(x_p, Sigma_p)
      self.prop_time = ts

    return

  def process(self, mu, Sigma, u, dt, disturb_mode):
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

    # propogate vehicle state according to flight state
    if self.flight_state == model.FLIGHT_STATE_FLIGHT:
    	(f, Fx, Fu, M, R) = model.process_model_flight(mu, u, dt, disturb_mode)
    else:
    	(f, Fx, Fu, M, R) = model.process_model_int_imu(mu, u, dt, disturb_mode)

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

    if z.shape != (m,):
      raise Exception('Argument z has incorrect dimension.')
    if z_pred.shape != (m,):
      raise Exception('Argument z_pred has incorrect dimension.')
    if Hx.shape != (m, n):
      raise Exception('Argument Hx has incorrect dimension.')
    if Q.shape != (m, m):
      raise Exception('Argument Q has incorrect dimension.')
    if False in np.isfinite(z_pred):
      raise Exception('Argument z_pred has non-finite elements.')
    if False in np.isfinite(Q):
      raise Exception('Argument Q has non-finite elements.')

    # calculate K
    A = Hx.dot(Sigma).dot(Hx.T) + Q
    try:
      A_inv = linalg.inv(A);
    except LinAlgError as e:
      print 'EKFUpdate: {}'.format(e.strerror)
      print A
      return (mu, Sigma)

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
    # account for double-cover by forcing w > 0
    if q[3] < 0.0:
      q[0] = -q[0]
      q[1] = -q[1]
      q[2] = -q[2]
      q[3] = -q[3]
    pose.pose.orientation = Quaternion(*q)

    pose.pose.position.x =  self.x[model.VAR_POS_X]
    pose.pose.position.y =  self.x[model.VAR_POS_Y]
    pose.pose.position.z =  self.x[model.VAR_POS_Z]

    self.pub_pose.publish(pose)

    vels = TwistStamped()
    vels.header.stamp = self.prop_time
    vels.header.frame_id = '/body'
    vels.header.seq = self.seq

    vels.twist.linear.x =  self.x[model.VAR_VEL_U]
    vels.twist.linear.y =  self.x[model.VAR_VEL_V]
    vels.twist.linear.z =  self.x[model.VAR_VEL_W]

    self.pub_vel.publish(vels)

    status = BoondogglerStatus()
    status.header.stamp     = self.prop_time
    status.header.frame_id  = '/world'
    status.header.seq       = self.seq

    status.disturb_mode     = self.disturb_mode
    status.flight_state     = self.flight_state

    status.sp_thrust        = self.x[model.VAR_SP_THRUST]
    status.drag_coefficient = self.x[model.VAR_DRAG_CO]
    status.gyro_biases.x    = self.x[model.VAR_GBIAS_P]
    status.gyro_biases.y    = self.x[model.VAR_GBIAS_Q]
    status.gyro_biases.z    = self.x[model.VAR_GBIAS_R]

    self.pub_status.publish(status)

    self.seq += 1
    return


rospy.init_node('boondoggler')
ekf = EKF()
ekf.subscribe()

rospy.spin()
