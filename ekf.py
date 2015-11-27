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

    rospy.loginfo('EKF: initialized state.')
    return

  def subscribe(self):
    rospy.loginfo('EKF: starting subscribers...')
    
    self.sub_imu = rospy.Subscriber('/mavros/imu/data_raw', Imu, self.msgHandler)
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
      # convert from ENU->NED
      u = np.array([imu.angular_velocity.x, -imu.angular_velocity.y, -imu.angular_velocity.z])
      new_prop_time = ts

      # Now calculate dt and propogate
      dt = (new_prop_time - self.prop_time).to_sec()
      if dt > 0.2:
          rospy.logwarn('Propogating filter by larger than expected dt.')
      (x_p, Sigma_p) = self.propogate(self.x, self.Sigma, u, dt, self.disturb_mode)

      self.prop_time = ts
      (self.x, self.Sigma) = model.enforce_bounds(x_p, Sigma_p)
      model.print_state(self.x)

      self.u = u
    
    else:
      rospy.logwarn('IMU message recieved from the past.')

    # Now do correction based on accelerometer
    # convert from ENU->NED
    z = np.array([imu.linear_acceleration.x, -imu.linear_acceleration.y, -imu.linear_acceleration.z])        

    (h, Hx, Q) = model.observation_acc(self.x, self.disturb_mode)
    (x_c, Sigma_c) = self.update(self.x, self.Sigma, z, h, Hx, Q)
    (self.x, self.Sigma) = model.enforce_bounds(x_c, Sigma_c)

    # rospy.loginfo('Completed IMU update.')
    model.print_state(self.x)
        
    return
