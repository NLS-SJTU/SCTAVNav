
import time
import struct
import rospy
import PyKDL
import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import Imu
from std_msgs.msg import Int16, Bool
from cv_bridge import CvBridge
from PIL import Image as pilImage

from myzmq.zmq_comm import zmq_comm_cli_c
from myzmq.zmq_cfg import *
from myzmq import jpeg_compress

# params
# mode: 0-sim; 1-bag; 2:real
MODE = 0
MIN_TURN_RAD = np.pi / 6
MIN_TURN_BACK_RAD = np.pi * 0.75
MIN_MOVE_DIST = 0.5

if(MODE == 2):
    # real
    img_topic = '/sensors/stereo_cam/left/image_rect_color'
    imgr_topic = '/sensors/stereo_cam/right/image_rect_color'
    # depth_topic = '/sensors/stereo_cam/depth/depth_registered'

    # img_topic = '/kinect2/qhd/image_color_rect'
    # depth_topic = '/kinect2/qhd/image_depth_rect'

    odom_topic = '/sensors/stereo_cam/odom'
    imu_topic = '/sensors/imu'
elif(MODE == 1):
    # bag
    imgcompress_topic = '/sensors/stereo_cam/left/image_rect_color/compressed'
    imgrcompress_topic = '/sensors/stereo_cam/right/image_rect_color/compressed'
    # depthcompress_topic = '/sensors/stereo_cam/depth/depth_registered/compressedDepth'
    # imgcompress_topic = '/kinect2/qhd/image_color_rect/compressed'
    # depthcompress_topic = '/kinect2/qhd/image_depth_rect/compressed'
    odom_topic = '/sensors/stereo_cam/odom'
    imu_topic = '/sensors/imu'
else:
    # sim
    img_topic = '/camera/left/image_raw'
    imgr_topic = '/camera/right/image_raw'
    # img_topic = '/kinect/rgb/image_raw'
    # depth_topic = '/kinect/depth/image_raw'
    odom_topic = '/sim_p3at/odom'
    imu_topic = '/imu'

action_topic = '/se_order'
crossing_topic = '/crossing_type'
forwardroad_topic = '/forwardroad'
odomtarget_topic = '/targetP_odom'

order_dict = {'H': 0, 'F': 1, 'L': 2, 'R': 3}
crossing_type_str = ['road', 'crossing', 'unknow']


class ros_wrapper():
    def __init__(self):
        self.reset()
        self.bridge = CvBridge()
        self.zmqclient = zmq_comm_cli_c(name_main,ip_main,port_main)
        self.init_ros()
        self.cnt = 0

    def init_ros(self):
        self.odomsub = rospy.Subscriber(odom_topic, Odometry, self.odomCB)
        self.imusub = rospy.Subscriber(imu_topic, Imu, self.imuCB)
        self.forwardroadsub = rospy.Subscriber(forwardroad_topic, Bool, self.forwardroadCB)
        self.actionpub = rospy.Publisher(action_topic, Int16, queue_size=1)
        self.crossingsub = rospy.Subscriber(crossing_topic, Int16, self.crossingCB)
        self.odomtargetpub = rospy.Publisher(odomtarget_topic, Pose, queue_size=1)
        if(MODE == 1):
            sub1 = Subscriber(imgcompress_topic, CompressedImage)
            sub6 = Subscriber(imgrcompress_topic, CompressedImage)
            # sub2 = Subscriber(depthcompress_topic, CompressedImage)
            self.tss1 = ApproximateTimeSynchronizer([sub1, sub6], 10, 0.05)
            self.tss1.registerCallback(self.synstereocompressCB)
            # self.tss1.registerCallback(self.synzedcompressCB)
            # self.tss1.registerCallback(self.synkinectcompressCB)
        else:
            sub1 = Subscriber(img_topic, Image)
            sub6 = Subscriber(imgr_topic, Image)
            # sub2 = Subscriber(depth_topic, Image)
            self.tss1 = ApproximateTimeSynchronizer([sub1, sub6], 10, 0.05)
            # self.tss1 = ApproximateTimeSynchronizer([sub1, sub2], 10, 0.05)
            self.tss1.registerCallback(self.synstereoCB)
            # self.tss1.registerCallback(self.synimgdepthCB)               
            
    def reset(self):
        self.image = None
        self.imager = None
        self.depth = None
        self.odom = None # [x,y,z]
        self.directions = None #[R,P,Y]
        self.crossing_type = -1
        self.lastodom = None
        self.lastdirections = None
        self.forwardroad = True
        self.matchednodeodom = None

    def process(self):
        if(not self.zmqclient.query({'start':None})):
            self.reset()
            return

        # compute transform from last timestep
        params, is_moving = self.prepare_information()

        if(not is_moving):
            return

        print('[BrainNavi]start a job.')
        for param in params:
            # send to BrainNavi
            self.zmqclient.execute(param)

            # get localization and action result
            working = True
            while(working):
                working = self.zmqclient.query({'working':None})
                time.sleep(0.1)
            ret = self.zmqclient.get_result({'phoneapp':None})

        self.lastodom = self.odom
        self.lastdirections = self.directions

        # publish nav result
        if(self.zmqclient.query({'mode':None}) == 1):            
            # local target action
            localtarget = ret['localtarget']
            odomtarget = ret['odomtarget']
            print(ret['nav_res'][0], ret['nav_res'][1], 'local target: ',localtarget, ' odom target: ',odomtarget)
            if(not type(odomtarget) == type(None)):
                msg = Pose()
                msg.position.x = odomtarget[0]
                msg.position.y = odomtarget[1]
                msg.position.z = 0
                self.odomtargetpub.publish(msg)
            else:
                # discrete action
                msg = Int16()
                if(self.crossing_type == 1):
                    msg.data = order_dict[ret['nav_res'][1]]
                else:
                    msg.data = order_dict[ret['nav_res'][0]]
                print('action ', ret['nav_res'][0], ret['nav_res'][1])
                self.actionpub.publish(msg)
        print('[BrainNavi]end a job.')

    def prepare_information(self):
        if(type(self.image) == type(None)):
            # print('[BrainNavi]no image input to topic '+img_topic)
            return None, False
        if(type(self.odom) == type(None)):
            # print('[BrainNavi]no odom input to topic '+odom_topic)
            return None, False
        if(type(self.depth) == type(None) and type(self.imager) == type(None)):
            return None, False

        if(MODE == 2):
            reallnglat = [self.odom[0][0], self.odom[0][1]]
        else:
            reallnglat = [0,0]

        if(type(self.lastodom) == type(None)):
            params = [{'phoneapp':{'image':[jpeg_compress.img_rgb_to_jpeg(self.image)], 'imager':[jpeg_compress.img_rgb_to_jpeg(self.imager)], 'depth':[self.depth], 'realdir':self.directions.GetRPY()[2], 'reallnglat':reallnglat, 'action':'H', 'crossing_type':self.crossing_type, 'distance':[0,1], 'odom':self.odom}}]
            is_moving = True
            return params, is_moving

        del_direction = self.directions.Inverse() * self.lastdirections
        delyaw = -del_direction.GetRPY()[2]
        # print(delyaw)

        deldist = np.linalg.norm(np.array([self.odom[0]]) - np.array([self.lastodom[0]]))

        is_moving = False
        action = None
        if(deldist > MIN_MOVE_DIST):
            print('[BrainNavi]moving forward (deldist:{}, delyaw:{})'.format(deldist, delyaw))
            action = 'F'
            is_moving = True
        elif(abs(delyaw) > MIN_TURN_RAD):
            if(not self.forwardroad):
                print('[BrainNavi]holding (no road ahead)')
                action = 'H'
            elif(delyaw > 0):
                if(self.crossing_type == 0):
                    if(delyaw > MIN_TURN_BACK_RAD):
                        print('[BrainNavi]turning back (deldist:{}, delyaw:{})'.format(deldist, delyaw))
                        action = 'TB'
                else:
                    print('[BrainNavi]turning left (deldist:{}, delyaw:{})'.format(deldist, delyaw))
                    action = 'L'
            else:
                if(self.crossing_type == 0):
                    if(delyaw < -MIN_TURN_BACK_RAD):
                        print('[BrainNavi]turning back (deldist:{}, delyaw:{})'.format(deldist, delyaw))
                        action = 'TB'
                else:
                    print('[BrainNavi]turning right (deldist:{}, delyaw:{})'.format(deldist, delyaw))
                    action = 'R'
            is_moving = True
        elif(self.crossing_type > 0):
            print('[BrainNavi]to crossing')
            is_moving = True
            if(delyaw > MIN_TURN_RAD*0.2):
                print('[BrainNavi]turning left (deldist:{}, delyaw:{})'.format(deldist, delyaw))
                action = 'L'
            elif(delyaw < -MIN_TURN_RAD*0.2):
                print('[BrainNavi]turning right (deldist:{}, delyaw:{})'.format(deldist, delyaw))
                action = 'R'
            elif(deldist > MIN_MOVE_DIST*0.2):
                action = 'F'
                print('[BrainNavi]moving forward (deldist:{}, delyaw:{})'.format(deldist, delyaw))
            else:
                action = 'H'
        
        params = [{'phoneapp':{'image':[jpeg_compress.img_rgb_to_jpeg(self.image)], 'imager':[jpeg_compress.img_rgb_to_jpeg(self.imager)], 'depth':[self.depth], 'realdir':self.directions.GetRPY()[2], 'reallnglat':reallnglat, 'action':action, 'crossing_type':self.crossing_type, 'distance':[deldist, deldist/2], 'odom':self.odom}}]

        return params, is_moving

    def crossingCB(self, data):
        if(not self.crossing_type == data.data):
            print('[BrainNavi]current crossing type is '+crossing_type_str[data.data])
        self.crossing_type = data.data

    def decodeDepth(self, data):
        nparr = np.fromstring(data.data[12:], np.uint8)
        depthraw = cv2.imdecode(nparr, cv2.IMREAD_ANYDEPTH)#IMREAD_UNCHANGED 
        # print(nparr.shape, depthraw.shape, data.format, np.max(depthraw), np.min(depthraw))
        rawheader = data.data[:12]
        [compfmt, dqA, dqB] = struct.unpack('iff', rawheader)
        depthimgscaled = dqA / (depthraw.astype(np.float32) - dqB)
        depthimgscaled[depthraw==0] = 0
        return depthimgscaled

    def forwardroadCB(self, data):
        self.forwardroad = data.data

    def imuCB(self, data):
        self.directions = PyKDL.Rotation.Quaternion(data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w)

    def odomCB(self, data):
        q = PyKDL.Rotation.Quaternion(data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w)
        self.odom = [[data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z], q]

    def synzedcompressCB(self, imgcompressdata, depthcompressdata):  
        nparr = np.fromstring(imgcompressdata.data, np.uint8)
        self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.depth = self.decodeDepth(depthcompressdata)

    def synstereocompressCB(self, imgcompressdata, imgrcompressdata): 
        nparr = np.fromstring(imgcompressdata.data, np.uint8)
        self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        nparr = np.fromstring(imgrcompressdata.data, np.uint8)
        self.imager = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def synstereoCB(self, imgdata, imgrdata): 
        self.image = self.bridge.imgmsg_to_cv2(imgdata, desired_encoding='bgr8')
        self.imager = self.bridge.imgmsg_to_cv2(imgrdata, desired_encoding='bgr8')

    def synimgdepthCB(self, imgdata, depthdata): 
        self.image = self.bridge.imgmsg_to_cv2(imgdata, desired_encoding='bgr8')
        self.depth = self.bridge.imgmsg_to_cv2(depthdata, desired_encoding='32FC1')

    def synkinectcompressCB(self, imgcompressdata, depthcompressdata): 
        print(self.cnt)
        nparr = np.fromstring(imgcompressdata.data, np.uint8)
        self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        nparr = np.fromstring(depthcompressdata.data, np.uint8)
        self.depth = cv2.imdecode(nparr, -1) 
        self.depth = self.depth / 1000.0




####################
# main #
####################
if __name__ == '__main__':
    rospy.init_node('BrainNavi', anonymous=True)
    rate = rospy.Rate(5)
    rw = ros_wrapper()
    print('[BrainNavi]ros start. Press start in UI to run.')
    while not rospy.is_shutdown():
        rw.process()
        rate.sleep()