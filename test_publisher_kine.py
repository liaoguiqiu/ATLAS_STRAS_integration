import rospy
from std_msgs.msg import Float64MultiArray
import time
import numpy as np

rospy.init_node("test_pub")
rospy.loginfo("Node started")

pub = rospy.Publisher("/kinematics_out",Float64MultiArray, queue_size=10)
rate = rospy.Rate(5)
start = time.time()
msg = Float64MultiArray()
while not rospy.is_shutdown():
    msg.data = np.array([1,1,1,0,0,0,0,0,0,0])
    pub.publish(msg)
    rate.sleep()

rospy.loginfo("\n\nExiting")

rospy.loginfo("Exiting")
