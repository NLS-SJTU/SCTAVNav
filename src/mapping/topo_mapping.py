#!/usr/bin/env python
# coding=utf-8

import sys, os
sys.path.append('..')
from myzmq.misc import run_thread_c
import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import utils.utils as utils
import networkx as nx
from pygraphviz import AGraph

import cv2
import time
import json
import PyKDL
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image as pilImage
from myzmq.zmq_comm import zmq_comm_cli_c
from myzmq.misc import run_thread_c
from myzmq.zmq_cfg import *
from myzmq import jpeg_compress


timefolder = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
data_save_path = './data/'+timefolder+'/'#'./data/gazebo_ampt_all/'
img_save_path = data_save_path + 'images/'
if(not os.path.exists(data_save_path)):
    os.mkdir(data_save_path)
    os.mkdir(img_save_path)
    print('map data saving to '+data_save_path)

# params
# add new images
# mode: 0-sim; 1-bag; 2:real
MODE = 0

USE_HFNET = True

MIN_TURN_RAD = np.pi / 6
MIN_TURN_BACK_RAD = np.pi * 0.75
MIN_MOVE_DIST = 5.0
MIN_FEATURE_DIST_FOR_NEW_IMAGE = 0.8
# loop close
MAX_FEATURE_DIST_FOR_LOOPCLOSURE = 0.4
MAX_ANGLE_ERR = 0.8
TOP_SIMILAR_FOR_LOOP_DETECTION = 5
IGNORE_LATEST_N_WHEN_LOOP_DETECTION = 5

MAX_NODE_SIZE = MIN_MOVE_DIST / 2
MAX_LINEAR_VEL = 0.5
MAX_ANGULAR_VEL = 0.57


if(MODE == 2):
    # real
    img_topic = '/sensors/stereo_cam/left/image_rect_color'
    odom_topic = '/sensors/stereo_cam/odom'
    imu_topic = '/sensors/imu'
    intrinK = np.array([[699.6599731445312, 0.0, 633.25], \
                    [0.0, 699.6599731445312, 370.7355041503906], \
                    [0.0, 0.0, 1.0]])
elif(MODE == 1):
    # bag
    imgcompress_topic = '/sensors/stereo_cam/left/image_rect_color/compressed'
    odom_topic = '/sensors/stereo_cam/odom'
    imu_topic = '/sensors/imu'
    intrinK = np.array([[699.6599731445312, 0.0, 633.25], \
                    [0.0, 699.6599731445312, 370.7355041503906], \
                    [0.0, 0.0, 1.0]])
else:
    # sim
    img_topic = '/camera/left/image_raw'
    odom_topic = '/sim_p3at/odom'
    imu_topic = '/imu'
    intrinK = np.array([[185.7, 0.0, 320.5], \
                  [0.0, 185.7, 240.5], \
                  [0.0, 0.0, 1]])

joy_topic = '/joy'

# important data format
# data_list: [[img_name, last_image_name, next_image_name, state_id, relative_position_from_last_image, absolute_direction], ...]
# state_dict = {'state_name': {'images':[img_name,...], 'tostate':state_name1, 'fromstate':state_name2}, ...}
# node_dict = {'node_name': {'states':[state_name,...], 'neighbornodes':[node_name1,...]}, ...}

class topo_mapping(run_thread_c):
    def __init__(self):
        run_thread_c.__init__(self, 'mapping')
        self.state_dict = {}
        self.node_dict = {}
        self.data_list = []
        self.feature_list = []
        self.image_cnt = 0
        self.current_loop_image = 0
        self.current_state = -1
        self.new_node_id = 0
        self.image = None
        self.odom = None
        self.directions = None
        self.comp_direction = PyKDL.Rotation.Quaternion(0,0,0,1)
        self.lastodom = None
        self.lastdirections = None
        self.last_state = -1
        self.last_img_id = -1
        self.running = True
        self.bridge = CvBridge()
        self.zmqinit()
        self.rosinit()

    def zmqinit(self):
        self.vprzmqclient = zmq_comm_cli_c(name_single_location, ip_single_location, port_single_location)
        self.vprzmqclient2 = zmq_comm_cli_c(name_single_location, ip_single_location, port_single_location)
        self.vprzmqclient3 = zmq_comm_cli_c(name_single_location, ip_single_location, port_single_location)

    def rosinit(self):
        self.joysub = rospy.Subscriber(joy_topic, Joy, self.joyCB)
        if(MODE == 1):
            self.tss = ApproximateTimeSynchronizer([Subscriber(imgcompress_topic, CompressedImage), Subscriber(imu_topic, Imu), Subscriber(odom_topic, Odometry)], 15, 0.005)
            self.tss.registerCallback(self.allcompressCB)
        else:
            self.tss = ApproximateTimeSynchronizer([Subscriber(img_topic, Image), Subscriber(imu_topic, Imu), Subscriber(odom_topic, Odometry)], 15, 0.005)
            self.tss.registerCallback(self.allCB)

    def myprint(self, strs):
        rospy.loginfo(strs)

    def main_loop(self):
        self.loopclosure()
        time.sleep(0.1)
        return self.running

    def dist_between_nodes(self, node1, node2):
        states_in_node1 = self.node_dict[node1]['states']
        states_in_node2 = self.node_dict[node2]['states']

        is_connected = False

        for state1 in states_in_node1:
            for state2 in states_in_node2:
                if(state1 in self.state_dict[state2]['tostate'] or state1 in self.state_dict[state2]['fromstate']):
                    is_connected = True
                    sc1 = state1
                    sc2 = state2
                    break

        if(is_connected):
            dist = -1
            for img1 in self.state_dict[sc1]['images']:
                for img2 in self.state_dict[sc2]['images']:
                    if(img1 == self.data_list[int(img2)][1]):
                        dist = np.linalg.norm(self.data_list[int(img2)][4])
                        break
                    elif(img1 == self.data_list[int(img2)][2]):
                        dist = np.linalg.norm(self.data_list[int(img1)][4])
                        break
                if(dist >= 0):
                    break
        else:
            dist = -1

        return dist

    def merge_state_to_node(self, last_state, current_state):
        if(last_state == None):
            # the first state, new node
            new_node_name = str(self.new_node_id).zfill(6)
            self.node_dict[new_node_name] = {'states':[current_state], 'neighbornodes':[]}
            self.state_dict[current_state]['node'] = new_node_name
            self.new_node_id += 1
            self.myprint('[MAPPING]generate the first node {}'.format(self.new_node_id-1))
        else:
            # not first state
            last_node_name = self.state_dict[last_state]['node']
            last_state_node = int(last_node_name)
            current_node_name = self.state_dict[current_state]['node']
            current_state_node = int(current_node_name)
            if(not last_state_node == current_state_node):
                dist_between_nodes = np.linalg.norm(np.array(self.data_list[self.current_loop_image][4]))
                if(dist_between_nodes > MAX_NODE_SIZE):
                    # two states are not at the same node
                    if(current_state_node < 0):
                        # no node is assigned to current state, new node
                        new_node_name = str(self.new_node_id).zfill(6)
                        self.node_dict[new_node_name] = {'states':[current_state], 'neighbornodes':[last_node_name]}
                        self.node_dict[last_node_name]['neighbornodes'].append(new_node_name)
                        self.state_dict[current_state]['node'] = new_node_name
                        self.new_node_id += 1
                        self.myprint('[MAPPING]generate a new node {}'.format(self.new_node_id-1))
                    else:
                        # current state already belongs to a node, connect to last node
                        if(not last_node_name in self.node_dict[current_node_name]['neighbornodes']):
                            self.node_dict[current_node_name]['neighbornodes'].append(last_node_name)
                            self.node_dict[last_node_name]['neighbornodes'].append(current_node_name)
                else:
                    # two states are at the same node
                    if(current_state_node >= 0):
                        # there already a node for current state, delete this node and put the states to the merged node
                        states_in_del_node = self.node_dict[current_node_name]['states']
                        states_in_merge_node = self.node_dict[last_node_name]['states']
                        for state in states_in_del_node:
                            if(not state in states_in_merge_node):
                                self.node_dict[last_node_name]['states'].append(state)
                                self.state_dict[state]['node'] = last_node_name
                        neighbornodes_in_del_node = self.node_dict[current_node_name]['neighbornodes']
                        neighbornodes_in_merge_node = self.node_dict[last_node_name]['neighbornodes']
                        self.myprint('[MAPPING]node '+current_node_name+' and '+last_node_name+' are merge for state '+current_state+' and '+last_state)
                        for neighbor in neighbornodes_in_del_node:
                            if(not neighbor in neighbornodes_in_merge_node):
                                # the neighbors of the node to delete are not the neighbors of the node to merge
                                dist = self.dist_between_nodes(neighbor, current_node_name)

                                self.node_dict[last_node_name]['neighbornodes'].append(neighbor)
                                self.node_dict[neighbor]['neighbornodes'].remove(current_node_name)
                                self.node_dict[neighbor]['neighbornodes'].append(last_node_name)
                            else:
                                # the neighbors of the node to delelte are the neighbors of the node to merge
                                self.node_dict[neighbor]['neighbornodes'].remove(current_node_name)
                        self.node_dict.pop(current_node_name)
                    else:
                        # no node is assigned to current state
                        self.state_dict[current_state]['node'] = last_node_name
                        self.node_dict[last_node_name]['states'].append(current_state)
                        self.myprint('[MAPPING]state '+current_state+' is merge to node '+last_node_name+' with state '+last_state)

    def loopclosure(self):
        # no image to deal with
        if(len(self.data_list) <= 0 or self.current_loop_image >= len(self.data_list)):
            return
        self.myprint(' \033[1;33;40m[LOOPCLOSURE]<<<<<start loop closure detection for image {}\033[0m'.format(self.current_loop_image))
        is_loopclose = False
        loop_image_id = -1
        if(self.current_loop_image - IGNORE_LATEST_N_WHEN_LOOP_DETECTION > 0):
            # compute distance between an image and other images
            imgfeature = self.feature_list[self.current_loop_image]
            similar_indexs, similar_dists = utils.get_get_top_similar_images_hffeature(imgfeature, self.feature_list[:(self.current_loop_image - IGNORE_LATEST_N_WHEN_LOOP_DETECTION)], TOP_SIMILAR_FOR_LOOP_DETECTION)

            # double check using fundamantal mat
            # print(similar_indexs, similar_dists)
            # current_loop_dir = self.data_list[self.current_loop_image][5][2]
            R_occ = PyKDL.Rotation.RPY(self.data_list[self.current_loop_image][5][0], self.data_list[self.current_loop_image][5][1], self.data_list[self.current_loop_image][5][2])
            current_loop_dir = (self.comp_direction * R_occ).GetRPY()[2]
            current_image = cv2.imread(img_save_path+self.data_list[self.current_loop_image][0]+'.jpg', 0)
            # cv2.imshow('1', current_image)
            # print(similar_dists, similar_indexs)
            # cv2.waitKey(1)
            similar_dir = []
            loop_img_cv = None
            for i in range(len(similar_dists)):
                if(similar_dists[i] > MAX_FEATURE_DIST_FOR_LOOPCLOSURE):
                    similar_dir.append('None')
                    break
                # potential_loop_dir = self.data_list[similar_indexs[i]][5][2]
                R_ooc = PyKDL.Rotation.RPY(self.data_list[similar_indexs[i]][6][0], self.data_list[similar_indexs[i]][6][1], self.data_list[similar_indexs[i]][6][2])
                R_ocr = PyKDL.Rotation.RPY(self.data_list[similar_indexs[i]][5][0], self.data_list[similar_indexs[i]][5][1], self.data_list[similar_indexs[i]][5][2])
                R_or = R_ooc * R_ocr
                potential_loop_dir = R_or.GetRPY()[2]
                err_dir = abs(potential_loop_dir - current_loop_dir)
                if(err_dir > np.pi):
                    err_dir = 2*np.pi - err_dir
                similar_dir.append(err_dir)
                if(err_dir > MAX_ANGLE_ERR):
                    print('[{}]distcheck: True({}); dircheck: False({}))'.format(similar_indexs[i],round(similar_dists[i], 3), round(err_dir,3), err_dir))
                    continue
                todetect_image = cv2.imread(img_save_path+self.data_list[similar_indexs[i]][0]+'.jpg', 0)
                # print(i, similar_dists[i], similar_indexs[i])
                # cv2.imshow('2', todetect_image)
                # cv2.waitKey(0)
                # fundamantal_check_ok = loop_closure.fundamantal_check(current_image, todetect_image)
                if(not USE_HFNET):
                    fundamantal_check_ok, R_rc, t_rc, numpnp, numkeyp = loop_closure.getRt(current_image, todetect_image, intrinK)
                else:
                    rescur = self.vprzmqclient3.get_result({'allres':jpeg_compress.img_rgb_to_jpeg(current_image)})
                    restod = self.vprzmqclient3.get_result({'allres':jpeg_compress.img_rgb_to_jpeg(todetect_image)})
                    fundamantal_check_ok, R_rc, t_rc = loop_closure.getRt_hf(current_image, todetect_image, rescur['keypoints'], rescur['local_descriptors'], restod['keypoints'], restod['local_descriptors'], intrinK, intrinK)
                print('[{}]distcheck: True({}); dircheck: True({}); fundacheck: {}.'.format(similar_indexs[i], round(similar_dists[i], 3), round(err_dir,3), fundamantal_check_ok))
                if(fundamantal_check_ok):
                    feature_dist = similar_dists[i]
                    loop_image_id = similar_indexs[i]
                    loop_img_cv = todetect_image
                    R_rc = PyKDL.Rotation(R_rc[0,0],R_rc[0,1],R_rc[0,2],R_rc[1,0],R_rc[1,1],R_rc[1,2],R_rc[2,0],R_rc[2,1],R_rc[2,2])
                    # self.comp_direction = R_or * R_rc * R_occ.Inverse()
                    is_loopclose = True
                    break
        else:
            similar_indexs = [-1]
            similar_dists = [-1]
            similar_dir = [-1]

        # update compensated direction
        comp_rpy = self.comp_direction.GetRPY()
        self.data_list[self.current_loop_image][6] = [comp_rpy[0], comp_rpy[1], comp_rpy[2]]
        relapose_after_comp = self.comp_direction * PyKDL.Vector(self.data_list[self.current_loop_image][4][0], self.data_list[self.current_loop_image][4][1], self.data_list[self.current_loop_image][4][2])
        # self.data_list[self.current_loop_image][4] = [relapose_after_comp[0],relapose_after_comp[1],relapose_after_comp[2]]

        if(is_loopclose):
            # merge current image to loop state and connect last state to loop state
            loop_state_key = self.data_list[loop_image_id][3]
            last_image_name = self.data_list[self.current_loop_image][1]
            last_state_key = self.data_list[int(last_image_name)][3]
            last_state_to_loop_states_keys = self.state_dict[loop_state_key]['fromstate']

            if(not last_state_key in last_state_to_loop_states_keys):
                # connect last state to loop state
                self.state_dict[loop_state_key]['fromstate'].append(last_state_key)
                self.state_dict[last_state_key]['tostate'].append(loop_state_key)
                # else: there already exists a connection between last state and loop state, just add image to loop state
            self.state_dict[loop_state_key]['images'].append(self.data_list[self.current_loop_image][0])
            self.data_list[self.current_loop_image][3] = loop_state_key
            self.data_list[self.current_loop_image][7] = str(loop_image_id).zfill(6)
            loop_rpy = R_rc.GetRPY()
            self.data_list[self.current_loop_image][8] = [loop_rpy[0], loop_rpy[1], loop_rpy[2]]
            self.data_list[self.current_loop_image][9] = [t_rc[0], t_rc[1], t_rc[2]]
            comprpy = self.comp_direction.GetRPY()
            self.myprint(' \033[1;32;40m[LOOPCLOSURE]detect a loop closure and merge image {} and {} to state '.format(loop_image_id, self.current_loop_image)+loop_state_key+' with feature distance {}, direction error {}. Current compensated direction is ({},{},{})\033[0m'.format(feature_dist, err_dir, comprpy[0], comprpy[1], comprpy[2]))

            # merge to node graph
            self.erase_node = self.state_dict[loop_state_key]['node']
            self.merge_state_to_node(last_state_key, loop_state_key)
        else:
            # add new state
            new_state_id = len(self.state_dict.keys())
            new_state_key = str(new_state_id).zfill(6)
            last_image_name = self.data_list[self.current_loop_image][1]
            if(int(last_image_name) >= 0):
                last_state_key = self.data_list[int(last_image_name)][3]
                fromstate = [last_state_key]
                self.state_dict[last_state_key]['tostate'].append(new_state_key)
            else:
                fromstate = []
                last_state_key = None

            self.state_dict[new_state_key] = {'fromstate': fromstate, 'tostate': [], 'images': [self.data_list[self.current_loop_image][0]], 'node':'-1'}
            self.data_list[self.current_loop_image][3] = new_state_key

            # merge to node graph
            self.merge_state_to_node(last_state_key, new_state_key)

            self.myprint('[LOOPCLOSURE]no loop closure. The most similar image {} is with feature distance {}, direction error {}'.format(similar_indexs[0], similar_dists[0], similar_dir[0]))


        self.current_loop_image += 1
        if(self.current_loop_image % 10 == 0):
            self.save()
        self.draw_graph()

    def draw_graph(self):
        justupdateimg = self.current_loop_image - 1
        justimgstate = self.data_list[justupdateimg][3]
        justimgnode = self.state_dict[justimgstate]['node']
        if(justupdateimg < 1):
            plt.scatter([0], [0])
            self.nodedraw_dict[justimgnode] = [0,0]
        else:
            lastupdateimg = self.current_loop_image - 2
            lastimgstate = self.data_list[lastupdateimg][3]
            lastimgnode = self.state_dict[lastimgstate]['node']
            if(justimgnode in self.nodedraw_dict.keys()):
                # add line
                justimgcoord = self.nodedraw_dict[justimgnode]
                if(lastimgnode == justimgnode):
                    if(not self.erase_node == justimgnode and not self.erase_node == None):
                        color = 'green'
                        print('[draw]loop erase node '+self.erase_node)
                        lastimgcoord = self.nodedraw_dict[self.erase_node]
                        plt.plot([lastimgcoord[0], justimgcoord[0]],[lastimgcoord[1], justimgcoord[1]], color=color, linewidth=2)
                else:
                    print('[draw]reach existing node and link '+lastimgnode+' '+justimgnode)
                    lastimgcoord = self.nodedraw_dict[lastimgnode]
                    color = 'black'
                    plt.plot([lastimgcoord[0], justimgcoord[0]],[lastimgcoord[1], justimgcoord[1]], color=color, linewidth=2)
                # while(lastimgnode == justimgnode):
                #     color = 'green'
                #     lastupdateimg -= 1
                #     if(lastupdateimg < 0):
                #         break
                #     lastimgstate = self.data_list[lastupdateimg][3]
                #     lastimgnode = self.state_dict[lastimgstate]['node']
            else:
                # new node, add node and line
                print('[draw]add new node '+justimgnode)
                lastimgcoord = self.nodedraw_dict[lastimgnode]
                delxyz = self.data_list[int(justupdateimg)][4]
                justimgcoord = [lastimgcoord[0]+delxyz[0],lastimgcoord[1]+delxyz[1]]
                self.nodedraw_dict[justimgnode] = justimgcoord
                plt.plot([lastimgcoord[0], justimgcoord[0]],[lastimgcoord[1], justimgcoord[1]], color='blue', linewidth=2, marker="o",markersize=5,markerfacecolor="red")
        plt.axis("equal")
        plt.pause(1)


    def record(self, force_record = False):
        if(type(self.odom) == type(None)):
            print('[MAPPING]no odom topic in '+odom_topic)
            return
        if(type(self.directions) == type(None)):
            print('[MAPPING]no imu topic in '+imu_topic)
            return
        if(type(self.image) == type(None)):
            print('[MAPPING]no image topic in '+img_topic)
            return

        # featuren distance
        if(not force_record):
            imgfeature = self.vprzmqclient.get_result({'vector':jpeg_compress.img_rgb_to_jpeg(self.image)})
        else:
            imgfeature = self.vprzmqclient2.get_result({'vector':jpeg_compress.img_rgb_to_jpeg(self.image)})
        if(self.last_img_id >= 0):
            # distance and direction
            phisical_distance = np.linalg.norm(np.array(self.odom[0]) - np.array(self.lastodom[0]))
            direction_difference = (self.lastdirections.Inverse() * self.directions).GetRPY()[2]
            if(force_record or phisical_distance >= MIN_MOVE_DIST or abs(direction_difference) >= MIN_TURN_RAD):
                pass
            else:
                return
            # there is last image
            lastimgfeature = self.feature_list[self.last_img_id]
            feature_distance = utils.compute_distance(imgfeature, lastimgfeature)
            if(feature_distance < MIN_FEATURE_DIST_FOR_NEW_IMAGE):
                return
        else:
            feature_distance = 0
            phisical_distance = 0
            direction_difference = 0

        self.myprint(' \033[1;36;40m[MAPPING]>>>>>start record image {}\033[0m'.format(self.image_cnt))
        self.feature_list.append(imgfeature)
        
        img_name = str(self.image_cnt).zfill(6)
        img_dir = self.directions.GetRPY()
        img_dir = [img_dir[0], img_dir[1], img_dir[2]]

        if(self.image_cnt <= 0):
            imgfrom = -1
            img_pose_from_last_img_list = [0,0,0]
        else:
            imgfrom  = self.last_img_id

            # compute relative pose from last recorded image to current image in IMU frame
            q_io = self.lastdirections * self.lastodom[1].Inverse() 
            t_ic = q_io * (PyKDL.Vector(self.odom[0][0], self.odom[0][1], self.odom[0][2]) - PyKDL.Vector(self.lastodom[0][0], self.lastodom[0][1], self.lastodom[0][2]))
            img_pose_from_last_img_list = [t_ic[0], t_ic[1], t_ic[2]]

        imgfromname = str(self.last_img_id).zfill(6)
        toimgname = '-00001'
        state_name = '-00001'
        loop_image_name = '-00001'
        loop_rpy = [0.,0.,0.]
        loop_t = [0.,0.,0.]
        # record information
        # last one is a compensated direction updated in loop closure
        self.data_list.append([img_name, imgfromname, toimgname, state_name, img_pose_from_last_img_list, img_dir, [0.,0.,0.], loop_image_name, loop_rpy, loop_t])
        if(self.last_img_id >= 0):
            self.data_list[self.last_img_id][2] = img_name
        cv2.imwrite(img_save_path+img_name+'.jpg', self.image)

        self.myprint('[MAPPING]record an image with distances between current image and last recorded image: {}(feature), {}(phisical), {}(direction)'.format(feature_distance, phisical_distance, direction_difference))
        self.myprint('[MAPPING]relative position from last record in IMU frame: ({}, {}, {})'.format(img_pose_from_last_img_list[0], img_pose_from_last_img_list[1], img_pose_from_last_img_list[2]))
        self.myprint('[MAPPING]absolute direction: ({}, {}, {})'.format(img_dir[0], img_dir[1], img_dir[2]))

        self.lastodom = self.odom
        self.lastdirections = self.directions
        self.last_img_id = len(self.data_list)-1
        self.image_cnt += 1

    def save(self):
        if(self.current_loop_image < len(self.state_dict)):
            print('[MAPPING]loopclosure dection not finished, cannot save.')
        else:
            print('[MAPPING]start saving map.')
            # turn into nodemap and save

            # save
            with open(data_save_path+'data_list.json', 'w') as f:
                json.dump(self.data_list, f, indent=2)
            
            with open(data_save_path+'state_dict.json', 'w') as f:
                json.dump(self.state_dict, f, indent=2)

            with open(data_save_path+'node_dict.json', 'w') as f:
                json.dump(self.node_dict, f, indent=2)

            np.save(data_save_path+'globalvecs.npy', np.array(self.feature_list))

            print('[MAPPING]save map done!')

    def joyCB(self, data):
        if(data.buttons[4] > 0.9):
            # LB
            self.record(True)
        elif(data.buttons[5] > 0.9):
            # RB
            self.save()

    def imgCB(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        self.record()

    def imgcompressCB(self, data):
        nparr = np.fromstring(data.data, np.uint8)
        self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.record()

    def allCB(self, imgdata, imudata, odomdata):
        self.image = self.bridge.imgmsg_to_cv2(imgdata, desired_encoding='bgr8')
        self.directions = PyKDL.Rotation.Quaternion(imudata.orientation.x, imudata.orientation.y, imudata.orientation.z, imudata.orientation.w)
        q = PyKDL.Rotation.Quaternion(odomdata.pose.pose.orientation.x, odomdata.pose.pose.orientation.y, odomdata.pose.pose.orientation.z, odomdata.pose.pose.orientation.w)
        self.odom = [[odomdata.pose.pose.position.x, odomdata.pose.pose.position.y, odomdata.pose.pose.position.z], q]

        self.record()

    def allcompressCB(self, imgcompressdata, imudata, odomdata):
        nparr = np.fromstring(imgcompressdata.data, np.uint8)
        self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        self.directions = PyKDL.Rotation.Quaternion(imudata.orientation.x, imudata.orientation.y, imudata.orientation.z, imudata.orientation.w)

        q = PyKDL.Rotation.Quaternion(odomdata.pose.pose.orientation.x, odomdata.pose.pose.orientation.y, odomdata.pose.pose.orientation.z, odomdata.pose.pose.orientation.w)
        self.odom = [[odomdata.pose.pose.position.x, odomdata.pose.pose.position.y, odomdata.pose.pose.position.z], q]

        self.record()

    def imuCB(self, data):
        self.directions = PyKDL.Rotation.Quaternion(data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w)

    def odomCB(self, data):
        q = PyKDL.Rotation.Quaternion(data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w)
        self.odom = [[data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z], q]



