
import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Int16

MAX_LINEAR_VEL = 0.5
MAX_ANGULAR_VEL = 0.76

MAX_DEL_VX = 0.05
MAX_DEL_RZ = 0.1

cmdvel_topic = '/sim_p3at/cmd_vel'
# cmdvel_topic = '/RosAria/cmd_vel'
joy_topic = '/joy'
forwardroad_topic = '/forwardroad'
crossing_topic = '/crossing_type'

class joy_to_cmdvel():
    def __init__(self):
        self.joysub = rospy.Subscriber(joy_topic, Joy, self.joyCB)
        self.cmdpub = rospy.Publisher(cmdvel_topic, Twist, queue_size=1)
        self.forwardroadpub = rospy.Publisher(forwardroad_topic, Bool, queue_size=1)
        self.forwardroad_flag = True
        self.crossingpub = rospy.Publisher(crossing_topic, Int16, queue_size=1)
        self.crossing_type = -1
        self.MAX_LINEAR_VEL = MAX_LINEAR_VEL
        self.MAX_ANGULAR_VEL = MAX_ANGULAR_VEL
        self.lastvx = 0.0
        self.lastrz = 0.0
        self.vx = 0.0
        self.rz = 0.0
        self.cntctrl = 10

    def joyCB(self, data):
        if(data.buttons[0] > 0.9):
            # A
            msg = Twist()
            msg.linear.x = 0
            msg.angular.z = 0
            self.lastvx = 0
            self.lastrz = 0
            self.cmdpub.publish(msg)
            return
        elif(data.buttons[4] > 0.9):
            # LB
            self.forwardroad_flag = not self.forwardroad_flag
            msg = Bool()
            msg.data = self.forwardroad_flag
            self.forwardroadpub.publish(msg)
        elif(data.buttons[5] > 0.9):
            # RB
            msg = Int16()
            self.crossing_type += 1
            if(self.crossing_type > 1):
                self.crossing_type = 0
            msg.data = self.crossing_type
            self.crossingpub.publish(msg)
        elif(data.axes[7] > 0.5):
            self.MAX_LINEAR_VEL = 1.0
            print('[joy]set max linear velocity to 1.0m/s.')
        elif(data.axes[7] < -0.5):
            self.MAX_LINEAR_VEL = 0.5
            print('[joy]set max linear velocity to 0.5m/s.')

        forbackward_ctrl = data.axes[1]
        rotate_ctrl = data.axes[3]
        self.vx = forbackward_ctrl * self.MAX_LINEAR_VEL
        self.rz = rotate_ctrl * self.MAX_ANGULAR_VEL
        if(abs(self.vx) > 0.05 or abs(self.rz) > 0.01):
            self.cntctrl = 10
        else:
            self.vx = 0.
            self.rz = 0.

        msg = Twist()
        msg.linear.x = self.vx
        msg.angular.z = self.rz

    def go(self):
        msg = Twist()
        msg.linear.x = self.vx
        msg.angular.z = self.rz
        if(msg.linear.x - self.lastvx > MAX_DEL_VX):
            msg.linear.x = self.lastvx + MAX_DEL_VX
        elif(msg.linear.x - self.lastvx < -MAX_DEL_VX):
            msg.linear.x = self.lastvx - MAX_DEL_VX
        if(msg.angular.z - self.lastrz > MAX_DEL_RZ):
            msg.angular.z = self.lastrz + MAX_DEL_RZ
        elif(msg.angular.z - self.lastrz < -MAX_DEL_RZ):
            msg.angular.z = self.lastrz - MAX_DEL_RZ
        self.lastvx = msg.linear.x
        self.lastrz = msg.angular.z
        self.cmdpub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('topomapping', anonymous=True)
    rate = rospy.Rate(10)
    tm = joy_to_cmdvel()
    print('[JOY]joy to cmdvel start')
    print('[JOY]RB = switch crossing type')
    while not rospy.is_shutdown():
        tm.go()
        rate.sleep()