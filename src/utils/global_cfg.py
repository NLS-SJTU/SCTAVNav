#!/usr/bin/python
# coding=utf-8

import math
import numpy as np

#program root dir
ROOT_DIR = '..'
MAP_PATH_ROOT = ROOT_DIR+'/map/'
EXPER_PATH = '/home/uav/BrainNavi/hfnet/hfnet/savedmodels/hfnet_tf_saved_models'

# before running, change the map id
use_map = 0
if(use_map == 0):
    MAP_DIR = ROOT_DIR+'/map/ampt_bag/'
    lng_max = 30.0
    lng_min = -25.0
    lat_max = 20.0
    lat_min = -20.0
    ID_DIGIT = 6
    topN = 0.05
    obimg_wid = 640
    obimg_hgt = 480
    intrinK = np.array([[185.7, 0.0, 320.5], \
                  [0.0, 185.7, 240.5], \
                  [0.0, 0.0, 1]])
elif(use_map == 1):
    MAP_DIR = ROOT_DIR+'/map/hall0/'
    lng_max = 50.0
    lng_min = -25.0
    lat_max = 30.0
    lat_min = -20.0
    ID_DIGIT = 6
    topN = 0.05
    obimg_wid = 1280
    obimg_hgt = 720
    use_cam = 1
    intrinK = np.array([[699.6599731445312, 0.0, 633.25], \
                        [0.0, 699.6599731445312, 370.7355041503906], \
                        [0.0, 0.0, 1.0]])
elif(use_map == 2):
    MAP_DIR = ROOT_DIR+'/map/sb2-hf-d2.5/'
    lng_max = 70.0
    lng_min = -130.0
    lat_max = 80.0
    lat_min = -60.0
    ID_DIGIT = 6
    topN = 0.05
    obimg_wid = 1280
    obimg_hgt = 720
    use_cam = 1
    intrinK = np.array([[699.6599731445312, 0.0, 633.25], \
                        [0.0, 699.6599731445312, 370.7355041503906], \
                        [0.0, 0.0, 1.0]])
else:
    # a sample
    MAP_DIR = ROOT_DIR+'/map/tmp/'
    # map range, lng-width, lat-height
    lng_max = 10.0
    lng_min = -10.0
    lat_max = 10.0
    lat_min = -10.0
    # digits of node id
    ID_DIGIT = 6
    # return top N similar image from global vector comparison
    topN = 0.05
    # reference image width and height
    obimg_wid = 1280
    obimg_hgt = 720
    # intrinsic matrix of reference image
    intrinK = np.array([[699.6599731445312, 0.0, 633.25], \
                        [0.0, 699.6599731445312, 370.7355041503906], \
                        [0.0, 0.0, 1.0]])

MAPIMG_DIR = MAP_DIR+'map.jpg'
# choose the camera on your robot
# 0: sim;, 1: zed; 2: kinect
use_cam = 0
if(use_cam == 0):
    obimg_wid = 640
    obimg_hgt = 480
    BASE_LINE = 0.2
    intrinKquery = np.array([[185.7, 0.0, 320.5], \
                  [0.0, 185.7, 240.5], \
                  [0.0, 0.0, 1]])
    intrinKqueryR = np.array([[185.7, 0.0, 320.5], \
                  [0.0, 185.7, 240.5], \
                  [0.0, 0.0, 1]])
elif(use_cam == 1):
    obimg_wid = 1280
    obimg_hgt = 720
    BASE_LINE = 0.120004
    intrinKquery = np.array([[699.6599731445312, 0.0, 633.25], \
                        [0.0, 699.6599731445312, 370.7355041503906], \
                        [0.0, 0.0, 1.0]])
    intrinKqueryR = np.array([[696.2000122070312, 0.0, 640.6300048828125], \
                        [0.0, 696.2000122070312, 351.8489990234375], \
                        [0.0, 0.0, 1.0]])
    intrinKdepth = np.array([[674.4291381835938, 0.0, 626.818359375], \
                             [0.0, 674.4291381835938, 362.15350341796875], \
                             [0.0, 0.0, 1.0]])
elif(use_cam == 2):
    obimg_wid = 960
    obimg_hgt = 540
    intrinKquery = np.array([[540.68603515625, 0.0, 479.75], \
                        [0.0, 540.68603515625, 269.75], \
                        [0.0, 0.0, 1.0]])


#some flags for origin setting
NEED_UI = True
# map image
IMG_WID=1072
IMG_HGT=797
IMG_SZ=IMG_WID*IMG_HGT

MAP_X_SCALE= (lng_max - lng_min) / IMG_WID       # width scale
MAP_Y_SCALE=-(lat_max - lat_min) / IMG_HGT       # height scale

MAP_WID=1072
MAP_HGT=797

MARK_SZ=1

# how many and which directions are wanted at one time, at least 0 is wanted
WANTED_DIRECTIONS = np.array([ 0.])#math.pi/2, 0., -math.pi/2, math.pi/3, 0., -math.pi/3

SIM_FORWARD_COV_RATE = 0.2

#buttons,---a button list for init maybe better
#l:left
#u:up
#w:width
#h:height
button_h = 50
reset_l = 30
reset_u = 20
reset_w = 140

cancel_l = 200
cancel_u = 20
cancel_w = 170

start_l = 30
start_u = 100
start_w = 330

shutdown_l = 30
shutdown_u = 700
shutdown_w = 330

simmode_l = 25
simmode_u = 210
simmode_w = 170

gtloc_l = 220
gtloc_u = 210
gtloc_w = 170

pureloc_l = 20
pureloc_u = 310
pureloc_w = 330

forward_l = 50
forward_u = 470
forward_w = 40

left_l = forward_l-forward_w
left_u = forward_u+50
left_w = forward_w

right_l = forward_l+forward_w
right_u = forward_u+50
right_w = forward_w

backward_l = forward_l
backward_u = forward_u+50
backward_w = forward_w

order_l = left_l
order_u = left_u + 50
order_w = 200

lastorder_l = left_l
lastorder_u = left_u + 100
lastorder_w = 200

withaction_l = 180
withaction_u = 410
withaction_w = 200

log_l = 180
log_u = 520
log_w = 80

nextaction_l = log_l + 30
nextaction_u = log_u + 100
nextaction_w = 200
