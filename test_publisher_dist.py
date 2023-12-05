import rospy
from std_msgs.msg import Float64
import time
import numpy as np

rospy.init_node("test_pub")
rospy.loginfo("Node started")

pub = rospy.Publisher("/distance",Float64, queue_size=10)
rate = rospy.Rate(5)
start = time.time()
msg = Float64()
while not rospy.is_shutdown():
    msg.data = 0.37
    pub.publish(msg)
    rate.sleep()

rospy.loginfo("\n\nExiting")

rospy.loginfo("Exiting")
