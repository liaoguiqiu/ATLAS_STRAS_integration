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
import numpy as np

def process_image(frame):
  frame_edges = cv2.Canny(frame , 100, 200)
  dist = 3
  return frame_edges, dist

def mask_red(im):
    out = (im[...,2] > 180)*(im[...,1] < 70)*(im[...,0] < 70)
    out = (out[...,np.newaxis].astype("float")*255.).astype("uint8")
    return out
def get_cmass(m):
    points = list(np.where(m==255))
    x_c = np.mean(points[0])
    y_c = np.mean(points[1])
    s = 1 if np.sum(m/255.) > 20 else 0
    return s,y_c,x_c
    
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
    self.detector = cv2.SimpleBlobDetector_create()
    self.center = (128,128)



  def callback(self,data):
    rate = rospy.Rate(5)
    rospy.loginfo("Got frame")
    br = CvBridge()
    # Output debugging information to the terminal
    # Convert ROS Image message to OpenCV image
    current_frame = br.imgmsg_to_cv2(data)
    current_frame_thr = mask_red(current_frame)
    s,x,y = get_cmass(current_frame_thr)
    #cf = np.repeat(current_frame_thr,axis=2,repeats=3)
    cf = current_frame
    cf = cv2.circle(current_frame, (int(128),int(128)), radius=5, color=(0, 0, 255), thickness=-1)
    if s:
      cf = cv2.circle(cf, (int(x),int(y)), radius=8, color=(0, 255, 0), thickness=-1)
      cf = cv2.arrowedLine(cf, self.center, (int(x),int(y)),
                                     color=(255,0, 0), thickness=4) 

    self.pub_gui.publish(br.cv2_to_imgmsg(cf))
    self.pub.publish(self.msg)

    
  def start(self):
    rospy.spin()

if __name__ == '__main__':
  node_pub_sub = node()
  node_pub_sub.start()
