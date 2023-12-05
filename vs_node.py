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
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import time
import numpy as np

class node():
  def __init__(self):
    # Tells rospy the name of the node.
    # Anonymous = True makes sure the node has a unique name. Random
    # numbers are adpub = rospy.Publisher("/topic_integration",String, queue_size=10)ded to the end of the name. 
    rospy.init_node("vs_node")
    rospy.loginfo("Node vs started")
    # Node is subscribing to the video_frames topic
    self.sub_dist = rospy.Subscriber('/distance', Float64, self.callback_dist) # 24hz
    self.sub_kine = rospy.Subscriber('/kinematics_in', numpy_msg(Floats), self.callback_kine) # 50hz
    self.is_kine_entering = False
    self.pub = rospy.Publisher("/kinematics_out",numpy_msg(Floats), queue_size=10)
    self.start_time = time.time()


  def callback_kine(self,data):
    self.latest_state = data.data
    self.is_kine_entering = True
    rospy.loginfo("VS: got kine in")
    

    

  def callback_dist(self,data):
    if self.is_kine_entering:
      rospy.loginfo("VS: got distance")
      now = time.time()
      self.msg = np.copy(self.latest_state)
      print(type(self.latest_state), self.latest_state[-4])
      self.msg[8] = -100
      self.msg[9] = -100
      self.pub.publish(self.msg)
      rospy.loginfo("VS: publishing kine")
    else:
      rospy.loginfo("Got distance but kine is not coming from STRAS")


    
    
  def start(self):
    rospy.spin()

if __name__ == '__main__':
  node_pub_sub = node()
  node_pub_sub.start()
