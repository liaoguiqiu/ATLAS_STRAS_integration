#!/usr/bin/env python3
# Description:
# - Subscribes to real-time streaming video from your built-in webcam.
#
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
 
# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
import cv2 # OpenCV library
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
from std_msgs.msg import Float64MultiArray
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

import time
import numpy as np

class node():
  def __init__(self):
    # Tells rospy the name of the node.
    # Anonymous = True makes sure the node has a unique name. Random
    # numbers are adpub = rospy.Publisher("/topic_integration",String, queue_size=10)ded to the end of the name. 
    rospy.init_node("STRAS_node")
    rospy.loginfo("Node STRAS started")
    # Node is subscribing to the video_frames topic
    self.sub = rospy.Subscriber('/kinematics_out', numpy_msg(Floats), self.callback)
    self.pub = rospy.Publisher("/kinematics_in",numpy_msg(Floats), queue_size=10)
    self.start_time = time.time()


  def callback(self,data):
    kine_state = data.data
    # send data to STRAS through socket
    rospy.loginfo("STRAS: updated kine state by vs: {}".format(kine_state))

    
    
  def start(self):
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
      now = time.time() 
      self.msg = np.array([0,0,0,0,0,0,0,0,0,0])
      self.pub.publish(self.msg)
      rospy.loginfo("STRAS: current kine state: {}".format(self.msg))
      rate.sleep()

if __name__ == '__main__':
  node_pub_sub = node()
  node_pub_sub.start()
