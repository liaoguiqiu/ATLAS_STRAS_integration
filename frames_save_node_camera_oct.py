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
import time


class node():
    def __init__(self):
        rospy.init_node("frames_node")
        rospy.loginfo("Node frames detection started")
        # Node is subscribing to the video_frames topic
        self.sub_frames = rospy.Subscriber('/video_frames', Image, self.callback_frame)
        self.sub_oct = rospy.Subscriber('/OCT_frames_gui_ori', Image, self.callback_oct)

        self.i_save = 0
        self.j_save = 0
        
        
        #self.output_path = "/media/icube/DATA1/ATLAS_integration/data/Endoscopic Phantom stiffer + trans -90 0 alpha 995 +stab"
        self.output_path = "/media/icube/DATA1/ATLAS_integration/data/Endoscopic Phantom stiffer + trans open -5 +stab 4th"
        self.output_path = "/media/icube/DATA1/ATLAS_integration/data/Endoscopic nomoving stiffer Phantom  + trans open -5 +stab 4th"
        self.output_path = "/media/icube/DATA1/ATLAS_integration/data/New-soft moving phamtom 3cm-2 speed -no trans-close 2"

        #self.output_path = "/media/icube/DATA1/ATLAS_integration/data/Endoscopic Phantom + trans -110 0 alpha 995 +stab 4th"

        self.output_path_frame = os.path.join(self.output_path,"force_only")
        if not os.path.isdir(self.output_path_frame): os.makedirs(self.output_path_frame)
        self.output_path_oct = os.path.join(self.output_path,"oct_only")
        if not os.path.isdir(self.output_path_oct): os.makedirs(self.output_path_oct)
        self.i = 0
        self.j = 0
        self.camera_ready = False
        self.last_time=0
        self.this_time = 0
        self.time_sum =0


    def callback_frame(self,data):

        rospy.loginfo("Got frame")
        if self.i %1 == 0:
            rospy.loginfo("Saved frame")
            self.current_frame = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
            self.camera_ready = True
            name = str(self.i_save)
            while len(name) < 5:
                name = "0" + name
            namef = name + ".jpg"
            #Img.fromarray(self.current_frame[...,::-1]).save(os.path.join(self.output_path_frame,namef))
            self.i_save += 1
        self.i += 1
    def callback_oct(self,data):
        rospy.loginfo("Got oct")
        if self.j %1 == 0:
            rospy.loginfo("Saved oct")
            self.current_oct = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
            name = str(self.j_save)
            while len(name) < 5:
                name = "0" + name
            namef = name + ".jpg"
            if self.camera_ready == True:
                Img.fromarray(self.current_oct[...,::-1]).save(os.path.join(self.output_path_oct,namef))
                Img.fromarray(self.current_frame[...,::-1]).save(os.path.join(self.output_path_frame,namef))
                self.j_save += 1
                print("saved oct id"+str(name))
                self.this_time   = time.time()
                print("this time"+str(self.this_time))
                delta = self.this_time - self.last_time
                if self.j_save > 1:
                    self.time_sum = self.time_sum  + delta 
                print(" time"+str(self.time_sum))
                print(" time_avg"+str(self.time_sum/self.j_save))

                self.last_time = self.this_time
        self.j += 1

if __name__ == '__main__':
    node_sub = node()
    rospy.spin()
