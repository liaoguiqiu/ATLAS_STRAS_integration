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
import time

def process_image(frame):
  frame_edges = cv2.Canny(frame , 100, 200)
  dist = 3
  return frame_edges, dist


class node():
  def __init__(self):
    # Tells rospy the name of the node.
    # Anonymous = True makes sure the node has a unique name. Random
    # numbers are adpub = rospy.Publisher("/topic_integration",String, queue_size=10)ded to the end of the name. 
    rospy.init_node("detection_node")
    rospy.loginfo("Node detection started")
    # Node is subscribing to the video_frames topic
    self.sub = rospy.Subscriber('/video_frames', Image, self.callback)
    self.pub = rospy.Publisher("/distance",Float64, queue_size=10)
    self.pub_gui = rospy.Publisher("/video_frames_gui",Image, queue_size=10)
    self.msg = Float64()
    self.start_time = time.time()


  def callback(self,data):
    rate = rospy.Rate(5)
    rospy.loginfo("Got frame")
    br = CvBridge()
    # Output debugging information to the terminal
    # Convert ROS Image message to OpenCV image
    current_frame = br.imgmsg_to_cv2(data)
    frame_processed, dist = process_image(current_frame)
    # Display image
    #cv2.imshow("camera", frame_processed)
    self.pub_gui.publish(br.cv2_to_imgmsg(frame_processed))
    now = time.time()
    self.msg.data = now-self.start_time
    self.pub.publish(self.msg)
    #rate.sleep()
    #cv2.waitKey(1)
    
    
  def start(self):
    rospy.spin()

if __name__ == '__main__':
  node_pub_sub = node()
  node_pub_sub.start()
