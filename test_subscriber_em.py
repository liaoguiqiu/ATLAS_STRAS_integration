import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from std_msgs.msg import UInt8
import time

def callback_fn(msg):
    msg_decoded = msg
    rospy.loginfo("Pose received: {}".format(msg_decoded))

rospy.init_node("em_node_listener")
rospy.loginfo("Node started")

sub = rospy.Subscriber("/pose_tip1",PoseStamped, callback_fn)

rospy.spin()
