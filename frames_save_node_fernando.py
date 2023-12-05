#!/home/lucas/anaconda3/envs/atlas_gpu_tf1/bin/python

from genericpath import isdir
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
import cv2 # OpenCV library
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import time
import numpy as np
from geometry_msgs.msg import PoseStamped
import os
from PIL import Image as Img
import pickle as pkl



class node():
    def __init__(self):
        rospy.init_node("frames_node")
        rospy.loginfo("Node frames detection started")
        # Node is subscribing to the video_frames topic
        self.sub_frames = rospy.Subscriber('/video_frames', Image, self.callback_frame)
        self.i_save = 0
        self.output_path = "/media/icube/DATA1/ATLAS_integration/data/newdataset"
        self.output_path_frame = os.path.join(self.output_path,"iw4")
        if not os.path.isdir(self.output_path_frame): os.makedirs(self.output_path_frame)
        self.i = 0


    def callback_frame(self,data):
        rospy.loginfo("Got frame")
        if self.i %20 == 0:
            rospy.loginfo("Saved frame")
            self.current_frame = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
            name = str(self.i_save)
            while len(name) < 5:
                name = "0" + name
            namef = name + ".jpg"
            Img.fromarray(self.current_frame[...,::-1]).save(os.path.join(self.output_path_frame,namef))
            self.i_save += 1
        self.i += 1

if __name__ == '__main__':
    node_sub = node()
    rospy.spin()
