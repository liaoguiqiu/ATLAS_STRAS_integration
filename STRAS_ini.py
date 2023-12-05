# Guiqiu's modification based on the stras_node_autonomous.py
#---------------------------------------------
#!/home/lucas/anaconda3/envs/atlas_gpu_tf1/bin/python
 
# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
import cv2 # OpenCV library
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
from std_msgs.msg import Float64MultiArray
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
# from EASE_socket_python.STRAS_socket import STRASsocket  # CommentedLine - FGH
from EASE_socket_python.STRAS_slave import STRASslave  # AddedLine - FGH
from std_msgs.msg import Float32
from std_msgs.msg import Float64

import time
import numpy as np
import pandas as pd
import os

class node():
  def __init__(self):
    """STRAS Node for polyp alignment and constant translation, stoping by the OCT distance or the image error
    """
    # Tells rospy the name of the node.
    # Anonymous = True makes sure the node has a unique name. Random
    # numbers are adpub = rospy.Publisher("/topic_integration",String, queue_size=10)ded to the end of the name.
    rospy.init_node("STRAS_node")
    rospy.loginfo("Node STRAS started")
    # Node is subscribing to the video_frames topic
    self.pub = rospy.Publisher("/kinematics_in", numpy_msg(Floats), queue_size=10)
    self.sub_oct_align = rospy.Subscriber('/oct_alignment', numpy_msg(Floats), self.callback_oct_align)
    
    self.sub_dist = rospy.Subscriber('/distance_x_y', numpy_msg(Floats), self.callback_dist)

    self.sub_oct_dis = rospy.Subscriber('/OCT_dis', numpy_msg(Floats), self.callback_oct_dis)
    self.current_oct_dis = 0
    self.current_arm_tran = 0 # current arm translation
    self.current_arm_ben = 0 # current arm bending


    self.output_path = "~/OCT_projects/OCT_controller_output/"
    self.state_flag = False

    self.pub_kt = rospy.Publisher("/kt",Float32, queue_size=10)
    self.msg_kt = Float32()
    self.pub_kr = rospy.Publisher("/kr",Float32, queue_size=10)
    self.msg_kr = Float32()
    self.pub_kb1 = rospy.Publisher("/kb1",Float32, queue_size=10)
    self.msg_kb1 = Float32()
    self.pub_kb2 = rospy.Publisher("/kb2",Float32, queue_size=10)
    self.msg_kb2 = Float32()

    self.pub_dx_pix = rospy.Publisher("/dx_pix",Float32, queue_size=10)
    self.msg_dx_pix = Float32()
    self.pub_dx_piy = rospy.Publisher("/dy_pix",Float32, queue_size=10)
    self.msg_dx_piy = Float32()

    self.dxv = []
    self.dyv = []
    self.b1v = []
    self.b2v = []
    self.rv = []
    self.tv = []
    self.times_v = []

    self.dxList = []
    self.dyList = []
    self.octDistList = []

    self.start_time = time.time()
    self.move_time = 2  # Wait until node is active

    self.host = '192.168.12.103'
    self.port_slave = 6666



    self.oct_dis_received = False
    self.flag_received = False
    self.i_loop = 0
    self.i_updates = 0
    self.send_command = True

    self.gain =0.01  # between 500-2000
    self.eps = 5
    self.cutoff = 3
    self.trans_P1 = 13
    self.trans_P2 = 0
    self.disturb_P1 = 0.5
    self.disturb_P2 = -3
    self.trans_stage = 1 #  1: go to the point 1; 2 ; go to the point 2
    self.distrub_stage = 1 #  1: go to the point 1; 2 ; go to the point 2

    self.dy_l = np.zeros(5)
    self.state_filter = np.zeros(5)

    self.dy_integral = 0 
    self.Enable_OCT_FLag = True
    self.STRAS_slave = STRASslave(host=self.host, port=self.port_slave)
    self.STRAS_slave.easyStart()
    self.target_y = -1

  def callback_oct_align(self, data):
    align_data =  data.data
    value =  align_data[0]
    if (value > 0.5):
      self. Enable_OCT_FLag = True
      self.STRAS_slave = STRASslave(host=self.host, port=self.port_slave)
      self.STRAS_slave.easyStart()
    else:
      self.Enable_OCT_FLag = False
    print("OCTEnable"+str(value))
    """[summary]

    Args:
        data ([type]): [description]
    """
    pass
  def callback(self, data):
    """[summary]

    Args:
        data ([type]): [description]
    """
    pass
  def callback_oct_dis(self,data):
    """A callback function when the OCT distance is received

    Args:
        data (Float64): OCT distance
    """
    
    if (self.Enable_OCT_FLag == False):
      
      return
    print("Got oct dis" )
    
         

    self.STRAS_slave.getState(False)
    kine_state_in = self.STRAS_slave.axisCurrVal.astype("float32").copy()
    # print(kine_state_in[1])
    self.current_state = kine_state_in
    self.state_flag = True

          
    time_stamp = time.time() - self.start_time
    if time_stamp > self.move_time:
     
      kine_state = self.current_state
      kine_state_updated = np.copy(self.current_state)
     

      # targetx = kine_state[1] - self.gain*self.dx
      # targety = kine_state[0] - self.gain*self.dy
      #new_gain =self.gain*2* abs(self.dy) /100
       
      
      #targety =     np.mean(self.state_filter) + deltay # PD with differential
      targety =  0
      
      # targety = 0 # disable the bending
      self.current_arm_ben = kine_state[9] # record the current  arm bending

      # targetz = kine_state[3] + 0.07  # 0.1 is a constant trasnlation
      # the translation of the arm
      ##############################################""
      targety  = -1
      ######################################""
      targetOCTT = 1
      
      self.current_arm_tran = kine_state[4] # record the current translation state 
      
      ##############################################""
      #targetz = self.trans_P2
      ######################################""
      # rospy.loginfo("dx{}".format(targetx))
      target_RR = -51 
      # target_RR = target_RR + 180
      ##############################################""
      targetEB = 0
      ######################################""
      target_main_r = -90    
      # target_main_r = 0
      target_tl =70
      target_grasper = 8
      #self.current_arm_tran = kine_state[4] # record the current translation state 
      # rospy.loginfo("dy{}".format(targety))

      # if ((self.dx < self.cutoff and self.dy < self.cutoff) and
      #     (self.dx > -self.cutoff and self.dy > -self.cutoff)):
      #   self.state_flag = False
      #   print('Stoped by Visual Servoing')
      # if (self.oct_dis_received and self.current_oct_dis < 5):
      #   self.state_flag = False
      #   print('Stoped by OCT distance')
      #   # rospy.loginfo("REACH GOAL")
      self.STRAS_slave.getState(False)
      kine_state_in = self.STRAS_slave.axisCurrVal.astype("float32").copy()
      # print(kine_state_in[1])
      self.current_state = kine_state_in
      self.state_flag = True
      if  self.state_flag:
        print("This is old state", kine_state[:2])
        # kine_state_updated[1] =  targetx
        # kine_state_updated[0] =  targety
        # kine_state_updated[1] =  targetx
        # kine_state_updated[9] =  targety
        kine_state_updated[6] =  targety
        # kine_state_updated[9] =  0
        kine_state_updated[10] =  target_grasper
        kine_state_updated[2] =  target_main_r
        # kine_state_updated[3] =  targetz
        kine_state_updated[4] =  targetOCTT
        kine_state_updated[1] =  targetEB
        kine_state_updated[5] =  target_RR
        kine_state_updated[7] =  target_tl
        
        # kine_state_updated[10] =  tqrget_grasp

        self.i_updates += 1
        rospy.loginfo("STRAS: sending updated state to STRAS: {}".format(kine_state_updated[:2]))

        #  slave_idx = [1,2]
        slave_idx = ['EBV', 'EBH']
        outputCommand = []
        # for i,j,k in zip(slave_idx, kine_state_updated[:2], [0.01]*len(slave_idx)):
        #   outputCommand.append([i,j,k])
        # outputCommand.append(['EBV', kine_state_updated[0], 0.05 ])
        # outputCommand.append(['ET', kine_state_updated[3], 0.05 ])
        outputCommand.append(['AL', kine_state_updated[10], 0.05 ])
        outputCommand.append(['EBH', kine_state_updated[1], 0.1 ])
        outputCommand.append(['ER', kine_state_updated[2], 0.1 ])

        outputCommand.append(['TR', kine_state_updated[4], 0.1 ])
        outputCommand.append(['BR', kine_state_updated[6], 0.1])
        outputCommand.append(['RR', kine_state_updated[5], 0.1 ])
        outputCommand.append(['TL', kine_state_updated[7], 0.05 ])



        print(outputCommand)


        #   # print("output command", outputCommand)
        self.updated_state = outputCommand
        self.flag_received = True
      

      
      # self.msg = kine_state_in
      # self.pub.publish(self.msg)

      # self.msg_kt.data = kine_state_in[3]
      # self.msg_kr.data = kine_state_in[2]
      # self.msg_kb1.data = kine_state_in[0]
      # self.msg_kb2.data = kine_state_in[1]
      # self.pub_kt.publish(self.msg_kt)
      # self.pub_kr.publish(self.msg_kr)
      # self.pub_kb1.publish(self.msg_kb1)
      # self.pub_kb2.publish(self.msg_kb2)


      # if self.oct_dis_received:
        # print(self.current_oct_dis)

      #rospy.loginfo("STRAS: current kine state received from STRAS")
      if self.flag_received:
        rospy.loginfo("Sending message")
        print(self.flag_received)
        print("updated state", self.updated_state)
        # self.STRAS_slave.send_GoToPos(self.updated_state,False)
        self.STRAS_slave.send_GoToPos_byLabel(self.updated_state,False)
        pass
        self.flag_received = False
      # Adding X,Y pixels errors
      self.msg_dx_pix = self.dxv.copy()
      self.msg_dx_piy = self.dyv.copy()
      self.pub_dx_pix.publish(self.msg_dx_pix)
      self.pub_dx_pix.publish(self.msg_dx_piy)
    else:
      rospy.loginfo("{}s to start".format(self.move_time-time_stamp))


  def callback_dist(self,data):
    """A callback function when the image errors (from polyp to center of image) is received

    Args:
        data ([type]): [description]
    """
    pass




  def start(self):
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
  
      rate.sleep()

if __name__ == '__main__':
  node_pub_sub = node()
  node_pub_sub.start()

