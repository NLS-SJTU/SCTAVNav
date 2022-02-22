# coding=utf-8



import sys, os, time
version_flag = sys.version[0]
import numpy as np
import matplotlib.pyplot as plt
import pymap3d as pm
import cv2
import copy
import math
from PIL import Image
from datetime import datetime

from utils.global_cfg      import *
from mapping.map_handler import map_handler
from myzmq.zmq_comm import *
from myzmq.zmq_cfg import *
from log.log_handler import *


def HSVtoRGB(h,s,v):
    c = v*s
    x = c*(1-math.fabs(h/60%2-1))
    m = v-c
    if(0 <= h < 60):
        rr=c
        gg=x
        bb=0
    elif(60 <= h < 120):
        rr=x
        gg=c
        bb=0
    elif(120 <= h < 180):
        rr=0
        gg=c
        bb=x
    elif(180 <= h < 240):
        rr=0
        gg=x
        bb=c
    elif(240 <= h < 300):
        rr=x
        gg=0
        bb=c
    elif(300 <= h < 360):
        rr=c
        gg=0
        bb=x
    return (int((rr+m)*255), int((gg+m)*255), int((bb+m)*255))

class se_ui(zmq_comm_svr_c):#
    def __init__(self, rootdir=ROOT_DIR, use_zmq=True):
        self.rootdir = rootdir
        #init flags states
        self.loc_with_action = True
        self.shutdown = False
        self.start_F = False
        self.sim = False
        self.pureloc = 0 # 0-loc, 1-nav, 2-supervise nav
        self.gtloc = 0 # 0-gt, 1-orgin photo, 2-sjtu2 phone photo
        self.point_or_path = False
        self.just_move = False
        self.working_real = False
        self.maph = map_handler()
        self.maph.reload_map(MAP_DIR)
        self.node_map = self.maph.detail_node_map #node_map
        self.detail_node_map = self.maph.detail_node_map
        self.img_buffer = []
        self.imgr_buffer = []
        self.depth_buffer = []
        self.crossing_type = -1
        self.WANTED_DIRECTIONS = WANTED_DIRECTIONS
        self.log_message = ''
        self.DATA_SZ=len(self.maph.data_list)

        self.log_flag = 0
        self.log_h = log_handler()

        #start position states
        self.real_pos_id = 1
        self.real_odom = [0,0]  #dellng dellat
        self.real_dir = 0.0     #east is 0
        self.dist_to_node = 0.0
        self.real_lnglat = [121.438503,31.034022]
        self.x = 0
        self.y = 0
        self.pathlength = 0
        self.nav_end_id = 1
        #estimate pos states
        self.pos_id = []
        self.odom = None
        self.odom_fix_ref = None
        self.curodom = None
        self.dir = []
        self.blief = []
        self.distance = [0.0,1.0]
        self.move_drift = [0.0,0.0]

        self.R_b_in_ref = np.array([[1.,0.,0.], [0,1,0], [0,0,1]])
        self.t_b_in_ref = np.array([0.,0.,0.])
        self.refnode_id = -1
        self.refimg_id = -1
        self.localtarget = None
        self.odomtarget = None
        self.comp_direction = np.array([[1.,0.,0.], [0,1,0], [0,0,1]])

        self.updateloc = False
        self.shouldprocess = True

        self.path_id_list = []
        self.path_blief = []
        self.est_path =[]

        #the last node in the list is destination
        self.node_id_list = []

        self.order = 'H'
        self.lastorder = 'H'
        self.nextorder = 'H'
        #zmq
        if(use_zmq):
            self.initzmqs()

        # draw all points in map
        print('[UI]read map image from '+self.rootdir)
        self.map_sjtu = cv2.imread(MAPIMG_DIR)#'../map/map_sjtu.png'
        if(type(self.map_sjtu) == type(None)):
            self.map_sjtu = np.ones((MAP_HGT, MAP_WID, 3), dtype=np.uint8) * 255
        if(NEED_UI):
            cv2.namedWindow('UI', flags=0)
            cv2.resizeWindow('UI', 800, 600)
            cv2.setMouseCallback('UI',self.mouseCallback)

        lng_mean = (lng_max+lng_min)/2
        lat_mean = (lat_max+lat_min)/2
        
        self.map_simp = np.zeros(self.map_sjtu.shape, np.uint8)
        self.map_simp = self.map_sjtu.copy()
        x0,y0=MAP_WID//2,MAP_HGT//2

        # draw nodes
        tmpline = 3

        # draw detail nodes
        drawallnodes = True
        for nk in self.detail_node_map.keys():
            lng,lat = self.get_data_coord(int(nk))
            pos_x, pos_y = self.lnglat_to_imgcoord(lng,lat)
            #self.map_simp[pos_y-MARK_SZ:pos_y+MARK_SZ+1,pos_x-MARK_SZ:pos_x+MARK_SZ+1]=[0,0,0]
            cntnei = 0
            for neighbor in self.detail_node_map[nk].keys():
                if(neighbor[:6] == 'unknow'):
                    continue
                cntnei += 1
                lng1,lat1 = self.get_data_coord(int(neighbor))
                near_x, near_y = self.lnglat_to_imgcoord(lng1,lat1)
                realdist = self.maph.get_distance_between_neibornodes(nk, neighbor)
                fakedist = math.sqrt((lng-lng1)**2 + (lat-lat1)**2)
                if(abs(fakedist / realdist - 1) > 0.3):
                    cv2.line(self.map_simp, (pos_x, pos_y), (near_x, near_y), (0,0,0))
                else:
                    cv2.line(self.map_simp, (pos_x, pos_y), (near_x, near_y), (255,0,147))
            if(drawallnodes or cntnei>2):
                cv2.rectangle(self.map_simp, (pos_x-3, pos_y-3), (pos_x+3, pos_y+3), (190,20,160))                
            if(self.maph.is_crossing_node(nk)):
                cv2.rectangle(self.map_simp, (pos_x-3, pos_y-3), (pos_x+3, pos_y+3), (190,20,160), 3)
        #add control bar
        cbar = np.zeros((IMG_HGT, 400,3), np.uint8)

        cv2.rectangle(cbar, (reset_l,reset_u), (reset_l+reset_w,reset_u+button_h), (255,255,255), thickness=2, lineType=1, shift=0)
        cv2.putText(cbar, "reset", (reset_l+25,reset_u+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        cv2.rectangle(cbar, (cancel_l,cancel_u), (cancel_l+cancel_w,cancel_u+button_h), (255,255,255), thickness=2, lineType=1, shift=0)
        cv2.putText(cbar, "cancel", (cancel_l+20,cancel_u+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        cv2.rectangle(cbar, (shutdown_l,shutdown_u), (shutdown_l+shutdown_w,shutdown_u+button_h), (255,255,255), thickness=2, lineType=1, shift=0)
        cv2.putText(cbar, "shut down", (shutdown_l+60,shutdown_u+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        cv2.rectangle(cbar, (start_l,start_u), (start_l+start_w,start_u+button_h), (255,255,255), thickness=2, lineType=1, shift=0)
        cv2.putText(cbar, "start/pause", (start_l+50,start_u+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        cv2.rectangle(cbar, (simmode_l,simmode_u), (simmode_l+simmode_w,simmode_u+button_h), (255,255,255), thickness=2, lineType=1, shift=0)
        cv2.putText(cbar, "sim/real", (simmode_l+5,simmode_u+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        cv2.rectangle(cbar, (gtloc_l,gtloc_u), (gtloc_l+gtloc_w,gtloc_u+button_h), (255,255,255), thickness=2, lineType=1, shift=0)
        cv2.putText(cbar, "locsrc", (gtloc_l+5,gtloc_u+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        cv2.rectangle(cbar, (pureloc_l,pureloc_u), (pureloc_l+pureloc_w,pureloc_u+button_h), (255,255,255), thickness=2, lineType=1, shift=0)
        cv2.putText(cbar, "loc/nav/sup", (pureloc_l+5,pureloc_u+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        cv2.rectangle(cbar, (forward_l,forward_u), (forward_l+forward_w,forward_u+button_h), (255,255,255), thickness=2, lineType=1, shift=0)
        cv2.putText(cbar, "F", (forward_l+5,forward_u+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        cv2.rectangle(cbar, (left_l,left_u), (left_l+left_w,left_u+button_h), (255,255,255), thickness=2, lineType=1, shift=0)
        cv2.putText(cbar, "L", (left_l+5,left_u+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        cv2.rectangle(cbar, (right_l,right_u), (right_l+right_w,right_u+button_h), (255,255,255), thickness=2, lineType=1, shift=0)
        cv2.putText(cbar, "R", (right_l+5,right_u+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        cv2.rectangle(cbar, (backward_l,backward_u), (backward_l+backward_w,backward_u+button_h), (255,255,255), thickness=2, lineType=1, shift=0)
        cv2.putText(cbar, "B", (backward_l+5,backward_u+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        cv2.rectangle(cbar, (withaction_l,withaction_u), (withaction_l+withaction_w,withaction_u+button_h), (255,255,255), thickness=2, lineType=1, shift=0)
        cv2.putText(cbar, "loc action", (withaction_l+5,withaction_u+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        cv2.rectangle(cbar, (log_l,log_u), (log_l+log_w,log_u+button_h), (255,255,255), thickness=2, lineType=1, shift=0)
        cv2.putText(cbar, "log", (log_l+5,log_u+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        self.map_simp = np.hstack((self.map_simp, cbar))
        self.map_backup = copy.deepcopy(self.map_simp)


    def processInloop(self):
        if(not self.shouldprocess):
            return
        if(self.start_F):
            lastorder = 'H'
            if(self.sim):
                self.working_real = True
                if(self.pureloc == 0):
                    if(self.gtloc == 0):
                        self.get_img_from_simulatorInloop()
                        self.start_F = False
                    elif(len(self.node_id_list) == 1):
                        self.img_buffer = self.get_img_from_simulatorInloop()
                        self.get_loc_resInloop(self.img_buffer, self.lastorder, self.maph.check_crosing_type(str(self.real_pos_id).zfill(ID_DIGIT)), self.depth_buffer, self.imgr_buffer)
                        self.start_F = False
                        self.just_move = True
                    elif(len(self.node_id_list) > 1):
                        print('[UI]you should keep number of red circles <=1.')
                        self.start_F = False
                    else:
                        #path location with human controling
                        if(self.just_move):
                            self.img_buffer = self.get_img_from_simulatorInloop()
                            st = time.time()
                            self.get_loc_resInloop(self.img_buffer, self.lastorder, self.maph.check_crosing_type(str(self.real_pos_id).zfill(ID_DIGIT)), self.depth_buffer, self.imgr_buffer)
                            print('[ui]time for a loop is:{}'.format(time.time()-st))
                            self.lastorder = self.order
                            # self.just_move = False
                elif(self.pureloc == 1):
                    # navigation
                    self.just_move = True
                    if(self.gtloc == 0):
                        st = time.time()
                        #navigation with real pos
                        lastorder = self.order
                        # action = self.get_nav_resInloop([[self.real_pos_id],[self.real_dir],[1]])
                        action, nextaction = self.get_nav_resInloop([[self.real_pos_id,self.real_dir,1]])
                        print('[ui]time for a loop is:{}'.format(time.time()-st))
                        self.nextorder = nextaction
                        self.move_in_map(action, 1)
                    else:
                        #navigation with est pos
                        #fetch the picture of this position
                        #give to loc module
                        lastorder = self.order
                        self.img_buffer = self.get_img_from_simulatorInloop()
                        st = time.time()
                        loc_res = self.get_loc_resInloop(self.img_buffer, self.lastorder, self.maph.check_crosing_type(str(self.real_pos_id).zfill(ID_DIGIT)), self.depth_buffer, self.imgr_buffer)
                        action, nextaction = self.get_nav_resInloop(loc_res)
                        print('[ui]time for a loop is:{}'.format(time.time()-st))
                        self.nextorder = nextaction
                        self.move_in_map(action, 1)
                else:
                    # supervise navigation
                    if(self.just_move):
                        self.lastorder = self.order
                        self.img_buffer = self.get_img_from_simulatorInloop()
                        loc_res = self.get_loc_resInloop(self.img_buffer, self.lastorder, self.maph.check_crosing_type(str(self.real_pos_id).zfill(ID_DIGIT)), self.depth_buffer, self.imgr_buffer)
                        action, nextaction = self.get_nav_resInloop([[self.real_pos_id,self.real_dir,1]])
                        self.nextorder = nextaction
                        self.order = action
                        # self.just_move = False
                self.working_real = False
            else:
                # real world usage
                if(self.gtloc == 0):
                    if(self.pureloc == 1): 
                        # for navigate with real pose
                        if(self.working_real):
                            st = time.time()
                            action, nextaction = self.get_nav_resInloop([[self.real_pos_id,self.real_dir,1]])
                            self.order = action
                            self.nextorder = nextaction
                            self.working_real = False
                            self.just_move = True
                            if(action == 'H'):
                                self.start_F = False
                            print('[ui]time for a loop is:{}'.format(time.time()-st))
                    else:
                        print('[ui]no correct mission mode is set.')
                        self.start_F = False
                elif(self.gtloc == 2):
                    lastorder = self.lastorder
                    # for cell phone app
                    #self.just_move = True
                    if(self.pureloc == 0):
                        # just loc
                        if(self.working_real):
                            st = time.time()
                            self.show_current_imgs()
                            self.get_loc_resInloop(self.img_buffer, self.lastorder, self.crossing_type, self.depth_buffer, self.imgr_buffer)
                            self.working_real = False
                            self.just_move = True
                            print('[ui]time for a loop is:{}'.format(time.time()-st))
                    elif(self.pureloc == 1):
                        # navigation
                        if(self.working_real):
                            st = time.time()
                            self.show_current_imgs()
                            loc_res = self.get_loc_resInloop(self.img_buffer, self.lastorder, self.crossing_type, self.depth_buffer, self.imgr_buffer)
                            action, nextaction = self.get_nav_resInloop(loc_res)
                            self.order = action
                            self.nextorder = nextaction
                            self.working_real = False
                            self.just_move = True
                            print('[ui]time for a loop is:{}'.format(time.time()-st))
                    else:
                        print('[ui]no correct mission mode is set.')
                        self.start_F = False
                else:
                    print('[no correct loc mode is set.')
            # log
            if(self.just_move and self.log_flag > 0):
                if(self.log_flag > 1):
                    imgbuf = self.img_buffer
                else:
                    imgbuf = [None] * len(self.WANTED_DIRECTIONS)
                self.log_h.log(imgbuf, self.WANTED_DIRECTIONS, self.real_dir, self.real_lnglat, self.crossing_type, lastorder, self.distance, [self.pos_id, self.dir, self.blief], self.order, self.log_message)
            if(not self.start_F):
                self.start_task(False, 1)
            self.just_move = False
        self.shouldprocess = False

    def showFigure(self):
        if(not self.shouldprocess):
            self.update_loc_data()
            if(self.sim):
                self.update_sim_data()
            self.shouldprocess = True
        
        if(NEED_UI):
            self.updateFigure()
            cv2.imshow('UI', self.map_simp)
            cv2.waitKey(1)

    # update picture for UI display
    def updateFigure(self):
        self.map_simp = copy.deepcopy(self.map_backup)
        if(self.sim):
            statestr = 'sim'
        else:
            statestr = 'real'
        cv2.putText(self.map_simp, statestr, (IMG_WID+simmode_l+20,simmode_u+80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        if(self.start_F):
            statestr = 'start'
        else:
            statestr = 'pause'
        cv2.putText(self.map_simp, statestr, (IMG_WID+start_l+50,start_u+80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        if(self.pureloc == 0):
            statestr = 'loc'
        elif(self.pureloc == 1):
            statestr = 'nav'
        else:
            statestr = 'sup'
        cv2.putText(self.map_simp, statestr, (IMG_WID+pureloc_l+50,pureloc_u+80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        if(self.gtloc == 0):
            statestr = 'gt'
        elif(self.gtloc == 1):
            statestr = 'estbaidu'
        elif(self.gtloc == 2):
            statestr = 'estphone2'
        cv2.putText(self.map_simp, statestr, (IMG_WID+gtloc_l+10,gtloc_u+80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        if(self.order == 'H'):
            statestr = 'Order: Hold'
        elif(self.order == 'F'):
            statestr = 'Order: Move Forward'
        elif(self.order == 'L'):
            statestr = 'Order: Turn Left'
        elif(self.order == 'R'):
            statestr = 'Order: Turn Right'
        cv2.putText(self.map_simp, statestr, (IMG_WID+order_l,order_u+60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        if(self.lastorder == 'H'):
            statestr = 'LastA: H'
        elif(self.lastorder == 'F'):
            statestr = 'LastA: F'
        elif(self.lastorder == 'L'):
            statestr = 'LastA:  L'
        elif(self.lastorder == 'R'):
            statestr = 'LastA: R'
        cv2.putText(self.map_simp, statestr, (IMG_WID+lastorder_l,lastorder_u+60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        if(self.nextorder == 'H'):
            statestr = 'NextA: H'
        elif(self.nextorder == 'F'):
            statestr = 'NextA: F'
        elif(self.nextorder == 'L'):
            statestr = 'NextA:  L'
        elif(self.nextorder == 'R'):
            statestr = 'NextA: R'
        cv2.putText(self.map_simp, statestr, (IMG_WID+nextaction_l,nextaction_u+60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        if(self.loc_with_action):
            statestr = 'yes'
        else:
            statestr = 'no'
        cv2.putText(self.map_simp, statestr, (IMG_WID+withaction_l+50,withaction_u+80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        if(self.log_flag == 2):
            statestr = 'all'
        elif(self.log_flag == 1):
            statestr = 'noimg'
        else:
            statestr = 'None'
        cv2.putText(self.map_simp, statestr, (IMG_WID+log_l+100,log_u+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

        self.drawAllPos()
        self.drawRealRefPos()
        self.drawClickPoints()

    # mouse callback
    def click_input(self, inx, iny):
        if(inx >= IMG_WID):
            inx = inx - IMG_WID
            if(start_l < inx < (start_l+start_w) and start_u < iny < (start_u+button_h)):
                print('[UI]start/pause')
                self.start_F = not self.start_F
                self.start_task(self.start_F)
                if(self.start_F):
                    print(self.node_id_list)
                    if(not self.pureloc==0):
                        if(len(self.node_id_list)==2):
                            print('[UI]set des pos')
                            self.startid = self.real_pos_id
                            self.nav_end_id = self.node_id_list[1]
                            self.nav_cli.config({'set_des_pos':self.node_id_list[1]})
                        else:
                            print('[UI]navigation mode need two clicked nodes to be start/end node.')
                            self.start_task(False)
            elif(cancel_l < inx < (cancel_l+cancel_w) and cancel_u < iny < (cancel_u+button_h) and not self.start_F):
                print('[UI]cancel')
                if(len(self.node_id_list) > 0):
                    self.node_id_list.pop()
            elif(reset_l < inx < (reset_l+reset_w) and reset_u < iny < (reset_u+button_h)):
                #print('[UI]reset')
                self.reset()
                if(self.gtloc > 0):
                    self.loc_cli.reset()
            elif(forward_l < inx < (forward_l+forward_w) and forward_u < iny < (forward_u+button_h)):
                print('[UI]move forward')
                self.move_in_map('F')
            elif(left_l < inx < (left_l+left_w) and left_u < iny < (left_u+button_h)):
                print('[UI]move left')
                self.move_in_map('L')
            elif(right_l < inx < (right_l+right_w) and right_u < iny < (right_u+button_h)):
                print('[UI]move right')
                self.move_in_map('R')
            elif(backward_l < inx < (backward_l+backward_w) and backward_u < iny < (backward_u+button_h)):
                print('[UI]move backward is no longer used')
            elif(pureloc_l < inx < (pureloc_l+pureloc_w) and pureloc_u < iny < (pureloc_u+button_h) and not self.start_F):
                print('[UI]loc/nav switch')
                self.pureloc += 1
                if(self.pureloc > 2):
                    self.pureloc = 0
            elif(simmode_l < inx < (simmode_l+simmode_w) and simmode_u < iny < (simmode_u+button_h) and not self.start_F):
                print('[UI]simmode switch')
                self.sim = not self.sim
            elif(gtloc_l < inx < (gtloc_l+gtloc_w) and gtloc_u < iny < (gtloc_u+button_h) and not self.start_F):
                print('[UI]gtloc switch')
                self.loc_cli.config({'directions_of_imgs':self.WANTED_DIRECTIONS})
                self.gtloc += 1
                if(self.gtloc > 2):
                    self.gtloc = 0
                if(self.sim):
                    if(self.gtloc == 0):
                        self.simulator_cli.config({'use_phone_image':False})
                    elif(self.gtloc == 1):
                        self.simulator_cli.config({'use_phone_image':False})
                    elif(self.gtloc == 2):
                        self.simulator_cli.config({'use_phone_image':True})
            elif(shutdown_l < inx < (shutdown_l+shutdown_w) and shutdown_u < iny < (shutdown_u+button_h)):
                print('[UI]shutdown')
                self.shutdown = not self.shutdown
            elif(withaction_l < inx < (withaction_l+withaction_w) and withaction_u < iny < (withaction_u+button_h)):
                print('[UI]loc with action switch')
                self.loc_with_action = not self.loc_with_action
                self.loc_cli.config({'update_with_action':self.loc_with_action})
            elif(log_l < inx < (log_l+log_w) and log_u < iny < (log_u+button_h)):
                print('[UI]log switch')
                self.log_flag = 1 + self.log_flag
                if(self.log_flag > 2):
                    self.log_flag = 0
        else:
            if(self.start_F):
                return
            lng,lat = self.imgcoord_to_lnglat(inx, iny)
            idx,_ = self.search_node_by_lnglat(lng, lat)
            print('you are clicking xy(%d,%d)->lnglat(%f,%f) node:%d, lng=%f, lat=%f' % (inx, iny, lng, lat, idx, self.maph.data_list[idx][1],self.maph.data_list[idx][2]))
            #
            self.node_id_list.append(idx)
            if(len(self.node_id_list) == 1):
                #para = {'mode':'pos','lng':simulator.data_list[idx][1], 'lat':simulator.data_list[idx][2],'dir':self.real_dir}
                para = {'initLng':self.maph.data_list[idx][1],'initLat':self.maph.data_list[idx][2], 'dir':self.real_dir}
                if(self.sim):
                    self.simulator_cli.reset(para)
                self.real_pos_id = idx

    def show_current_imgs(self):
        return
        plt.clf()
        i = 0
        for img in self.img_buffer:
            if(not type(img) == type(None)):
                img_rgb = jpg_to_img_rgb(img)
                plt.subplot(1,len(self.WANTED_DIRECTIONS),i+1)
                plt.imshow(img_rgb)
            i = i+1
        plt.pause(0.001)

    # this simulator is no longer used
    def get_img_from_simulatorInloop(self):
        st = time.time()
        retimgs = []
        #plt.clf()
        print('[UI]move distance {}m.'.format(self.distance[0]))
        for i in range(len(self.WANTED_DIRECTIONS)):
            dirs = self.WANTED_DIRECTIONS[i]
            img = self.simulator_cli_inloop.get_result({'heading':dirs})
            if(type(img) == type(None)):
                retimgs.append(None)
                continue
            imgsend = np.frombuffer(img,dtype=np.uint8)
            img_rgb = jpg_to_img_rgb(imgsend)
            #plt.subplot(1,len(self.WANTED_DIRECTIONS),i+1)
            #plt.imshow(img_rgb)
            retimgs.append(imgsend)
        print('[ui]time for an image from simulator is: {}'.format(time.time()-st))
        #plt.pause(0.001)
        return retimgs

    # update robot location data from simulator
    def update_sim_data(self):
        lastlnglat = [self.real_lnglat[0], self.real_lnglat[1]]
        self.real_lnglat[0], self.real_lnglat[1], self.real_dir = self.simulator_cli.query('lla')
        dx,dy,_ = pm.geodetic2enu(lastlnglat[1], lastlnglat[0], 0, self.real_lnglat[1], self.real_lnglat[0],0)
        self.distance[0] = math.sqrt(dx**2 + dy**2)
        self.distance[1] = self.distance[0] * SIM_FORWARD_COV_RATE
        self.real_pos_id, self.dist_to_node = self.search_node_by_lnglat(self.real_lnglat[0], self.real_lnglat[1])
        self.crossing_type = self.maph.check_crosing_type(str(self.real_pos_id).zfill(ID_DIGIT))
        lastx = self.x
        lasty = self.y
        self.x, self.y, _ = self.simulator_cli.query('enu')
        dist = math.sqrt((self.x-lastx)**2+(self.y-lasty)**2)
        if (self.start_F and 1 < dist < 100):
            # add path lenght only when start, fix a bug here
            self.pathlength += dist
            print('[UI]move {} meters till now.'.format(self.pathlength))

    # update estimate robot location data from location module
    def update_loc_data(self):
        # print('1111')
        if(self.gtloc > 0):
            res = self.loc_cli.get_result('est_pos')
            self.pos_id, self.dir, self.blief = res['est_pos']
            if(np.linalg.norm(res['t_b_in_ref'] - self.t_b_in_ref) > 0.001):
                # new relative pose is calculated
                self.R_b_in_ref = res['R_b_in_ref']
                self.t_b_in_ref = res['t_b_in_ref']
                self.refnode_id = res['refnode_id']
                self.refimg_id = res['refimg_id']
                self.comp_direction = res['comp_direction']
                self.odom_fix_ref = self.curodom
            # print('2222')
            return res['est_pos']
        else:
            return [[],[],[]]

    def update_loc_dataInloop(self):
        if(self.gtloc > 0):
            res = self.loc_cli_inloop.get_result('est_pos')
            self.pos_id, self.dir, self.blief = res['est_pos']      
            if(np.linalg.norm(res['t_b_in_ref'] - self.t_b_in_ref) > 0.001):
                # new relative pose is calculated
                self.R_b_in_ref = res['R_b_in_ref']
                self.t_b_in_ref = res['t_b_in_ref']
                self.refnode_id = res['refnode_id']
                self.refimg_id = res['refimg_id']
                self.comp_direction = res['comp_direction']
                self.odom_fix_ref = self.curodom
            return res['est_pos']
        else:
            return [[],[],[]]

    # get localization result
    def get_loc_resInloop(self, imgsend=[], action=None, crossing_type=-1, depthsend=[], imgrsend=[]):
        st = time.time()
        if(self.loc_with_action):
            self.loc_cli_inloop.execute({'image':imgsend,'action':action, 'crossing_type':crossing_type, 'distance':self.distance, 'realdir':self.real_dir, 'depth':depthsend, 'imager':imgrsend})
        else:
            self.loc_cli_inloop.execute({'image':imgsend, 'crossing_type':crossing_type, 'distance':self.distance, 'realdir':self.real_dir, 'depth':depthsend, 'imager':imgrsend})
        locworking = True
        while(locworking):
            locworking = self.loc_cli_inloop.query({'locworking':None})
            time.sleep(0.02)
        loc_res = self.update_loc_dataInloop()
        self.est_pos_list = []
        for i in range(len(loc_res[0])):
            self.est_pos_list.append([loc_res[0][i], loc_res[1][i], loc_res[2][i]])
        print('real pos:',self.real_pos_id,self.real_dir)
        print('est pos:',self.est_pos_list)
        if(len(self.est_pos_list) == 0):
            dist_between_real_and_est = -1
        else:
            estlng = self.maph.data_list[int(self.est_pos_list[0][0]+0.001)][1]
            estlat = self.maph.data_list[int(self.est_pos_list[0][0]+0.001)][2]
            dist_between_real_and_est = self.maph.cal_dist_of_lnglat(self.real_lnglat, [ estlng, estlat ])
        print('[ui]time for a localization result is: {}'.format(time.time()-st))
        print('distance between best est and real pos: ', dist_between_real_and_est)
        return self.est_pos_list

    # get navigation result from planner
    def get_nav_resInloop(self, est_pos_list=[[]]):
        st = time.time()
        # est_pos_list = []
        # for i in range(len(loc_res[0])):
            # est_pos_list.append([loc_res[0][i], loc_res[1][i], loc_res[2][i]])
        self.nav_cli_inloop.execute({'pos_now':est_pos_list,'crossing_type':self.crossing_type})
        navworking = True
        while(navworking):
            navworking = self.nav_cli_inloop.query({'navworking':None})
        ret = self.nav_cli_inloop.get_result({'action':None})
        action = ret['action']
        nextaction = ret['nextaction']
        if(not type(action)==type('f')):
            action = action.decode('utf-8')
        self.shortestlength = self.nav_cli_inloop.query({'totalpathlength':self.node_id_list[0]})
        # get relative target in body frame
        if(self.refnode_id >= 0):
            ret = self.nav_cli_inloop.get_result({'localtarget':{'refodom':self.odom_fix_ref, 'curodom':self.curodom, 'refnode_id':self.refnode_id, 'refimg_id': self.refimg_id, 'R_b_in_ref':self.R_b_in_ref, 't_b_in_ref':self.t_b_in_ref, 'realdir':self.real_dir, 'comp_direction':self.comp_direction}})
            self.localtarget = ret['localtarget']
            self.odomtarget = ret['odomtarget']
            if(action == 'H' and np.linalg.norm(self.localtarget[:2]) < 2.0):
                self.start_F = False
        else:
            self.localtarget = None
            self.odomtarget = None
        print('[ui]time for a decision result is: {}'.format(time.time()-st))
        print('[UI]shortest path length is {}'.format(self.shortestlength))
        return action, nextaction


    # draw all estimated position in red arrow
    def drawAllPos(self):
        selfpos_id = copy.deepcopy(self.pos_id)
        selfdir = copy.deepcopy(self.dir)
        selfblief = copy.deepcopy(self.blief)
        selfestpath = copy.deepcopy(self.est_path)
        if(not len(selfpos_id) == len(selfdir) == len(selfblief)):
            return
        for j in range(len(selfpos_id)):
            i = len(selfpos_id) - 1 - j
            lng, lat = self.get_data_coord(selfpos_id[i])
            x, y = self.lnglat_to_imgcoord(lng, lat)
            x = int(x) # + self.odom[i][0]
            y = int(y) # + self.odom[i][1]
            # image coordinate
            lx = int(x + 9*math.cos(selfdir[i]+0.4))
            ly = int(y - 9*math.sin(selfdir[i]+0.4))
            rx = int(x + 9*math.cos(selfdir[i]-0.4))
            ry = int(y - 9*math.sin(selfdir[i]-0.4))
            mx = int(x + 15*math.cos(selfdir[i]))
            my = int(y - 15*math.sin(selfdir[i]))
            color = HSVtoRGB(240, selfblief[i], 0.8)#(255*self.blief[i], 0, 255*self.blief[i])
            cv2.line(self.map_simp, (mx,my), (lx,ly), color, 2, 1, 0)
            cv2.line(self.map_simp, (rx,ry), (mx,my), color, 2, 1, 0)
            cv2.line(self.map_simp, (x,y), (mx,my), color, 2, 1, 0)

        for j in range(len(self.path_id_list)):
            lng, lat = self.get_data_coord(self.path_id_list[j][0])
            lastx ,lasty = self.lnglat_to_imgcoord(lng, lat)
            color = HSVtoRGB(350, self.path_blief[j], 0.8)
            for i in range(1,len(self.path_id_list[j])):
                lng, lat = self.get_data_coord(self.path_id_list[j][i])
                x, y = self.lnglat_to_imgcoord(lng, lat)
                cv2.line(self.map_simp, (x,y), (lastx,lasty), color, 2, 1, 0)
                lastx = x
                lasty = y

        if(not type(selfestpath) == type(None) and len(selfestpath) > 0):
            if(type(selfestpath[0][0]) == type('string')):
                for j in range(len(selfestpath)):
                    state = selfestpath[j][0]
                    prob = selfestpath[j][1]
                    color = HSVtoRGB(350, 1, 0.8)
                    nodeid = int(state[:ID_DIGIT])
                    lng, lat = self.get_data_coord(nodeid)
                    x, y = self.lnglat_to_imgcoord(lng, lat)
                    cv2.circle(self.map_simp, (x,y), 5, color,2)
            elif(type(selfestpath[0][0]) == type(int(1))):
                for j in range(len(selfestpath)):
                    prob = selfestpath[j][1]
                    color = HSVtoRGB(350, 1, 0.8)
                    nodeid = int(selfestpath[j][0])
                    lng, lat = self.get_data_coord(nodeid)
                    x, y = self.lnglat_to_imgcoord(lng, lat)
                    cv2.circle(self.map_simp, (x,y), 5, color,2)

    # draw pose with triangle
    def drawOnePos(self, node_id, p_odom, p_dir, colorbgr):
        '''
        lng, lat = self.get_data_coord(node_id)
        lng = lng + self.real_odom[0]
        lat = lat + self.real_odom[1]
        '''
        lng = self.real_lnglat[0]
        lat = self.real_lnglat[1]
        x, y = self.lnglat_to_imgcoord(lng, lat)
        #x = int(x + p_odom[0])
        #y = int(y + p_odom[1])
        lx = int(x + 20*math.cos(p_dir+0.4))
        ly = int(y - 20*math.sin(p_dir+0.4))
        rx = int(x + 20*math.cos(p_dir-0.4))
        ry = int(y - 20*math.sin(p_dir-0.4))
        cv2.line(self.map_simp, (x,y), (lx,ly), colorbgr, 2, 1, 0)
        cv2.line(self.map_simp, (rx,ry), (lx,ly), colorbgr, 2, 1, 0)
        cv2.line(self.map_simp, (x,y), (rx,ry), colorbgr, 2, 1, 0)

    def drawRealRefPos(self):
        colorbgr = (255,0,0)
        lng = self.real_lnglat[0]
        lat = self.real_lnglat[1]
        x, y = self.lnglat_to_imgcoord(lng, lat)
        compdir = np.arctan2(self.comp_direction[1,0], self.comp_direction[0,0])
        diraftercomp = self.real_dir + compdir
        lx = int(x + 20*math.cos(diraftercomp+0.4))
        ly = int(y - 20*math.sin(diraftercomp+0.4))
        rx = int(x + 20*math.cos(diraftercomp-0.4))
        ry = int(y - 20*math.sin(diraftercomp-0.4))
        cv2.line(self.map_simp, (x,y), (lx,ly), colorbgr, 2, 1, 0)
        cv2.line(self.map_simp, (rx,ry), (lx,ly), colorbgr, 2, 1, 0)
        cv2.line(self.map_simp, (x,y), (rx,ry), colorbgr, 2, 1, 0)

    # draw the points clicked by mouse
    def drawClickPoints(self):
        tmpnode_id_list = copy.deepcopy(self.node_id_list)
        for i in range(len(tmpnode_id_list)):
            ids = tmpnode_id_list[i]
            x,y = self.lnglat_to_imgcoord(self.maph.data_list[ids][1], self.maph.data_list[ids][2])
            if(i == 0):
                color = (30,20,20)
            else:
                color = (30,20,250)
            cv2.circle(self.map_simp, (x,y), 5, color,2)

    #lng lat to image x y
    def lnglat_to_imgcoord(self, lng, lat):
        lng0,lat0=lng-lng_min,lat-lat_min
        pos_x = int(round(lng0/MAP_X_SCALE))
        pos_y = IMG_HGT + int(round(lat0/MAP_Y_SCALE))
        return pos_x, pos_y

    #image x y to lng lat
    def imgcoord_to_lnglat(self, x, y):
        lng = x*MAP_X_SCALE+lng_min
        lat = (y-IMG_HGT)*MAP_Y_SCALE+lat_min
        return lng, lat

    # get coordinate of node n
    def get_data_coord(self,n): return self.maph.data_list[n][1], self.maph.data_list[n][2]

    # search nearest node id by lnglat
    def search_node_by_lnglat(self,lng,lat):
        dmin=np.inf
        nmin=-1
        for node in self.node_map.keys():
            lng0,lat0=self.get_data_coord(int(node))
            d=(lng-lng0)**2+(lat-lat0)**2
            if d<dmin:
                dmin=d
                nmin=node
        nmin = int(nmin)
        return int(nmin), dmin

    def mouseCallback(self, event, x, y, flags, param):
        if(event == cv2.EVENT_LBUTTONUP):
            self.click_input(x, y)


    # move from one node to the nearest node according to order
    def move_in_map(self,action,flag=0):
        if((self.lastorder == 'L' and action == 'R') or (self.lastorder == 'R' and action == 'L')):
            action = 'F'
        print('[UI]action:'+action)
        # self.lastorder = action
        self.order = action
        self.just_move = True
        randomturn = np.random.randn()*0.05 # sigma=0.5, normal randomness
        max_turnangle = 1.57/2 #90degree
        min_turnangle = 0.001
        targetposid = str(self.real_pos_id).zfill(ID_DIGIT)
        if(action == 'F'):
            nearest_node_id = str(self.real_pos_id).zfill(ID_DIGIT)
            nearest_angle = math.pi*2
            now_node = self.detail_node_map[str(self.real_pos_id).zfill(ID_DIGIT)]
            for node in now_node.keys():
                #if(node == 'unknow'):
                #    continue
                #here errangle can be more accurate by calculating with two points' lnglat
                #err_angle = self.cal_angle_lnglat(data_list[int(node)][1], data_list[int(node)][2], self.real_lnglat[0], self.real_lnglat[1]) - self.real_dir #now_node[node]['dir']
                #print(node)
                err_angle = now_node[node]['dir'] - self.real_dir
                if(err_angle > math.pi):
                    err_angle = err_angle - math.pi*2
                elif(err_angle < -math.pi):
                    err_angle = err_angle + math.pi*2
                print(node, err_angle)
                if(math.fabs(err_angle) < math.fabs(nearest_angle)):
                    nearest_angle = err_angle
                    nearest_node_id = node
            print('nearest angle:',nearest_angle)
            if(nearest_node_id=='unknow'):
                nearest_node_id = str(self.real_pos_id).zfill(ID_DIGIT)
                dx = 0.0
                dy = 0.0
            else:
                dx, dy,_ = pm.geodetic2enu(self.maph.data_list[int(nearest_node_id)][2], self.maph.data_list[int(nearest_node_id)][1], 0, self.real_lnglat[1], self.real_lnglat[0],0)
            targetposid = nearest_node_id
            normalize = math.sqrt(dx**2+dy**2)
            tmpmovedrift = copy.deepcopy(self.move_drift)
            dx = normalize * math.cos(nearest_angle)
            dy = normalize * math.sin(nearest_angle)
            # add some randome in movement
            #self.move_drift[0] = (2.0 * np.random.rand() - 1) * dx * 0.2
            #self.move_drift[1] = (2.0 * np.random.rand() - 1) * dy * 0.2
            #dx = dx - tmpmovedrift[0] + self.move_drift[0]
            #dy = dy - tmpmovedrift[1] + self.move_drift[1]
            print('dist:{},dx({}),dy({}),randx({}),randy({})'.format(normalize,dx,dy,self.move_drift[0],self.move_drift[1]))
            #turn_angle = nearest_angle + randomturn
            nextforwardnode = self.maph.get_neighbor_node(nearest_node_id, nearest_angle + self.real_dir, 'F')
            turn_angle = self.detail_node_map[nearest_node_id][nextforwardnode]['dir'] - self.real_dir
            if(flag == 0):
                self.simulator_cli.execute({'mode':'move','dx':dx,'dy':dy,'dyaw':turn_angle,'outImageSize':(640,480)})
            else:
                self.simulator_cli_inloop.execute({'mode':'move','dx':dx,'dy':dy,'dyaw':turn_angle,'outImageSize':(640,480)})
        elif(action == 'L'):
            nearest_left_node_id = str(self.real_pos_id).zfill(ID_DIGIT)
            nearest_angle = math.pi*2
            now_node = self.node_map[str(self.real_pos_id).zfill(ID_DIGIT)]
            print('at node:'+str(self.real_pos_id).zfill(ID_DIGIT))
            for node in now_node.keys():
                #if(node == 'unknow'):
                #    continue
                #here errangle can be more accurate by calculating with two points' lnglat
                err_angle = now_node[node]['dir'] - self.real_dir - 0.01 #self.cal_angle_lnglat(data_list[int(node)][1], data_list[int(node)][2], self.real_lnglat[0], self.real_lnglat[1])
                if(err_angle <= 0):
                    err_angle = err_angle + math.pi*2
                print(node, err_angle)
                if(err_angle < nearest_angle):
                    nearest_angle = err_angle
                    nearest_left_node_id = node
            print('nearest angle:',nearest_angle)
            #self.simulator_cli.execute({'mode':'move','dx':0,'dy':0,'dyaw':nearest_angle,'outImageSize':(640,480)})
            #if(abs(nearest_angle) > max_turnangle):
            #    turn_angle = max_turnangle
            #elif(abs(nearest_angle) < min_turnangle):
            #    turn_angle = min_turnangle
            #else:
            turn_angle = nearest_angle + min_turnangle
            if(flag == 0):
                self.simulator_cli.execute({'mode':'move','dx':0,'dy':0,'dyaw':turn_angle,'outImageSize':(640,480)})
            else:
                self.simulator_cli_inloop.execute({'mode':'move','dx':0,'dy':0,'dyaw':turn_angle,'outImageSize':(640,480)})
        elif(action == 'R'):
            nearest_right_node_id = str(self.real_pos_id).zfill(ID_DIGIT)
            nearest_angle = -math.pi*2
            now_node = self.node_map[str(self.real_pos_id).zfill(ID_DIGIT)]
            print('at node:'+str(self.real_pos_id).zfill(ID_DIGIT))
            for node in now_node.keys():
                err_angle = now_node[node]['dir'] - self.real_dir + 0.01
                if(err_angle >= 0):
                    err_angle = err_angle - math.pi*2
                print(node, err_angle)
                if(err_angle > nearest_angle):
                    nearest_angle = err_angle
                    nearest_right_node_id = node
            print('nearest angle:',nearest_angle)
            #self.simulator_cli.execute({'mode':'move','dx':0,'dy':0,'dyaw':nearest_angle,'outImageSize':(640,480)})
            #if(abs(nearest_angle) > max_turnangle):
            #    turn_angle = -max_turnangle
            #elif(abs(nearest_angle) < min_turnangle):
            #    turn_angle = -min_turnangle
            #else:
            turn_angle = nearest_angle - min_turnangle
            if(flag == 0):
                self.simulator_cli.execute({'mode':'move','dx':0,'dy':0,'dyaw':turn_angle,'outImageSize':(640,480)})
            else:
                self.simulator_cli_inloop.execute({'mode':'move','dx':0,'dy':0,'dyaw':turn_angle,'outImageSize':(640,480)})
        elif(action == 'H' and self.dist_to_node>25):
            nearest_angle = self.cal_angle_lnglat(self.maph.data_list[self.real_pos_id][1], self.maph.data_list[self.real_pos_id][2], self.real_lnglat[0], self.real_lnglat[1]) - self.real_dir
            dx, dy,_ = pm.geodetic2enu(self.maph.data_list[self.real_pos_id][2], self.maph.data_list[self.real_pos_id][1], 0, self.real_lnglat[1], self.real_lnglat[0],0)
            normalize = math.sqrt(dx**2+dy**2)
            dx = normalize * math.cos(nearest_angle)
            dy = normalize * math.sin(nearest_angle)
            print('dist:%f,dx(%f),dy(%f)'%(normalize,dx,dy))
            if(flag == 0):
                self.simulator_cli.execute({'mode':'move','dx':dx,'dy':dy,'dyaw':nearest_angle,'outImageSize':(640,480)})
            else:
                self.simulator_cli_inloop.execute({'mode':'move','dx':dx,'dy':dy,'dyaw':nearest_angle,'outImageSize':(640,480)})
        if(self.dist_to_node < 25 and action == 'H' and self.pathlength > 50):
            # length = self.nav_cli.query({'totalpathlength':self.startid})self.real_pos_id == self.nav_end_id
            print('[UI]nav end with path length {} meters.'.format(self.pathlength))
            if(self.gtloc > 0):
                if(flag == 0):
                    self.est_path = self.loc_cli.query({'sim_path_loc_result':0})
                else:
                    self.est_path = self.loc_cli_inloop.query({'sim_path_loc_result':0})
                print(self.est_path)
            self.start_F = False
            return
        self.lastorder = action
        self.crossing_type = self.maph.check_crosing_type(targetposid)
        print('[UI]crossing type:', self.crossing_type)
        print('')     
        time.sleep(0.5)

    def cal_angle_lnglat(self, tarlng, tarlat, orilng, orilat):
        res = math.atan2(tarlat-orilat, tarlng-orilng)
        return res

    def start_task(self, setto=True,flag=0):
        self.start_F = setto
        self.just_move = False
        print('[UI]start ', self.start_F)
        if(self.gtloc > 0):# and not self.start_F
            if(flag):
                self.est_path = self.loc_cli_inloop.query({'sim_path_loc_result':0})
            else:
                self.est_path = self.loc_cli.query({'sim_path_loc_result':0})
            if(not type(self.est_path)==type(None) and len(self.est_path) > 0 and type(self.est_path[0][0])==type(u'sb')):
                for i in range(len(self.est_path)):
                    self.est_path[i] = (str(self.est_path[i][0]), self.est_path[i][1])
            print('estimation path:',self.est_path)

        # activate log
        if(self.start_F):
            print(self.node_id_list)
            if(not self.pureloc==0):
                if(len(self.node_id_list)==2):
                    print('[UI]set des pos')
                    self.startid = self.real_pos_id
                    self.nav_end_id = self.node_id_list[1]
                    if(flag):
                        self.nav_cli_inloop.config({'set_des_pos':self.node_id_list[1]})
                    else:
                        self.nav_cli.config({'set_des_pos':self.node_id_list[1]})
                else:
                    print('[UI]navigation mode need two clicked nodes to be start/end node.')
                    self.start_F = False
            strtime = self.log_h.strtimenow()
            if(flag):
                if(self.pureloc > 0):
                    self.nav_cli_inloop.config({'start':strtime})
                self.loc_cli_inloop.config({'start':strtime})
            else:
                if(self.pureloc > 0):
                    self.nav_cli.config({'start':strtime})
                self.loc_cli.config({'start':strtime})
        else:
            if(flag):
                if(self.pureloc > 0):
                    self.nav_cli_inloop.config({'end':None})
                self.loc_cli_inloop.config({'end':None})
            else:
                if(self.pureloc > 0):
                    self.nav_cli.config({'end':None})
                self.loc_cli.config({'end':None})

        if(self.log_flag > 0):
            if(self.start_F):
                self.log_h.start_new_log()
            else:
                self.log_h.end_log()
                self.log_h.log_estpath([self.node_id_list, self.est_path])

    def initzmqs(self):
        zmq_comm_svr_c.__init__(self, name=name_main, ip=ip_main, port=port_main)
        self.simulator_cli = zmq_comm_cli_c(name=name_sim, ip=ip_sim, port=port_sim)
        self.simulator_cli_inloop = zmq_comm_cli_c(name=name_sim, ip=ip_sim, port=port_sim)
        self.loc_cli = zmq_comm_cli_c(name=name_location, ip=ip_location, port=port_location)
        self.loc_cli_inloop = zmq_comm_cli_c(name=name_location, ip=ip_location, port=port_location)
        self.nav_cli = zmq_comm_cli_c(name=name_nav, ip=ip_nav, port=port_nav)
        self.nav_cli_inloop = zmq_comm_cli_c(name=name_nav, ip=ip_nav, port=port_nav)


    # zmq
    def main_loop(self):
        '''
        try:
            self.zmq_sub_msg = self.skt.recv(flags=zmq.NOBLOCK)
        except:
            if(self.shutdown):
                return False
            return True

        return True
        '''
        self.processInloop()
        try:
            rx=self.skt.recv(flags=zmq.NOBLOCK)
        except:
            if(self.shutdown):
                return False
            else:
                return True
        
        if(version_flag == '2'):
            name,api_name,param = pickle.loads(rx)
        else:
            name,api_name,param = pickle.loads(rx,encoding='bytes')
        if name != self.name:
            print('[WRN] name mis-match (%s/%s)'%(name,self.name))
        
        if   api_name=='reset'     : self.skt.send(pickle.dumps(self.reset     (param)))
        elif api_name=='config'    : self.skt.send(pickle.dumps(self.config    (param)))
        elif api_name=='query'     : self.skt.send(pickle.dumps(self.query     (param)))
        elif api_name=='get_result': self.skt.send(pickle.dumps(self.get_result(param)))
        elif api_name=='execute'   : self.skt.send(pickle.dumps(self.execute   (param)))
        elif api_name=='stop':
            self.skt.send(pickle.dumps(None))
            return False
        else:
            print('unknown api name '+api_name)
            self.skt.send(pickle.dumps(None))
        return True

    def reset(self, param=None):
        print('[UI]reset')
        self.node_id_list = []
        self.path_id_list = []
        self.path_blief = []
        self.est_path = []
        #self.pos_id = []
        #self.odom = []
        #self.dir = []
        #self.blief = []
        self.start_F = False
        self.pathlength = 0
        self.x = 0
        self.y = 0
        self.lastorder = 'H'
        self.order = 'H'
        self.nextorder = 'H'
        self.working_real = False
        self.move_drift = [0.0,0.0]
        self.refnode_id = -1

        return None

    def query(self,param=None):
        if param is None: return
        if isinstance(param,str): param=[param]
        
        res={}
        if('sim_robot_location' in param):
            res['sim_robot_location']=[self.real_pos_id, self.dist_to_node, self.real_lnglat, self.real_dir]
            #self.real_odom
        elif('sim_single_loc_result' in param):
            res['real_lnglatdir'] = [self.real_pos_id, self.real_lnglat[0], self.real_lnglat[1], self.real_dir]
            est_pos_list = []
            for i in range(len(self.pos_id)):
                est_pos_list.append([self.pos_id[i], self.dir[i], self.blief[i]])
            res['est_result'] = est_pos_list
        elif('sim_path_loc_result' in param):
            res['sim_path_loc_result'] = self.loc_cli_inloop.query('sim_path_loc_result')
        elif('just_move' in param):
            res['just_move'] = self.just_move
        elif('pathlength' in param):
            res['pathlength'] = self.pathlength
        elif('nav_end' in param):
            res['nav_end'] = self.start_F
        elif('totalpathlength' in param):
            length = self.nav_cli_inloop.query({'totalpathlength':param['totalpathlength']})
            res['totalpathlength'] = length
        elif('order' in param):
            res['order'] = self.order
        elif('working' in param):
            res['working'] = self.working_real
        elif('start' in param):
            res['start'] = self.start_F
        elif('crossing_type' in param):
            res['crossing_type'] = self.crossing_type
        elif('mode' in param):
            res['mode'] = self.pureloc
        
        return list(res.values())[0] if len(res)==1 else res

    def config(self,param=None):
        if param is None: return
        if isinstance(param,str): param=[param]

        res = {}
        if('start' in param):
            if(param['start']):
                self.start_F = True
                if(self.gtloc > 0):
                    self.loc_cli_inloop.reset()    
                self.start_task(self.start_F,1)
                print('[UI]start mission')
                print(self.node_id_list)
            else:
                self.start_F = False
                self.start_task(self.start_F,1)
                print('[UI]end mission')
                res['est_path'] = self.est_path
                print(res)
        elif('mode' in param):
            if(param['mode']=='nav'):
                self.pureloc = 1
                print('[UI]nav mode')
            elif(param['mode']=='loc'):
                self.pureloc = 0
                print('[UI]loc mode')
            elif(param['mode']=='sup'):
                self.pureloc = 2
                print('[UI]sup mode')
        elif('set_init_pos' in param):
            idx,_ = self.search_node_by_lnglat(param['set_init_pos']['initLng'], param['set_init_pos']['initLat'])
            self.real_pos_id = idx
            self.startid = self.real_pos_id
            self.real_lnglat[0] = self.maph.data_list[idx][1]
            self.real_lnglat[1] = self.maph.data_list[idx][2]
            self.real_odom = [0,0]
            self.simulator_cli_inloop.reset({'initLng':self.maph.data_list[idx][1],'initLat':self.maph.data_list[idx][2],'dir':param['set_init_pos']['initDir']})
            print('setinitpos:')
            self.loc_cli_inloop.reset()
            print('[UI]set init pos')
        elif('set_des_pos' in param):
            idx,_ = self.search_node_by_lnglat(param['set_des_pos']['desLng'], param['set_des_pos']['desLat'])
            self.node_id_list = []
            self.node_id_list.append(self.real_pos_id)
            self.node_id_list.append(idx)
            self.nav_cli_inloop.config({'set_des_pos':idx})
            self.nav_end_id = idx
            print('[UI]set des pos', self.node_id_list[1])
            print('[ui]real id', self.real_pos_id)
        elif('gtloc' in param):
            self.gtloc = param['gtloc']
            if(self.sim):
                if(self.gtloc == 1):
                    self.simulator_cli_inloop.config({'use_phone_image':False})
                    print('[UI]set use BAIDU image')
                elif(self.gtloc == 2):
                    self.simulator_cli_inloop.config({'use_phone_image':True})
                    print('[UI]set use PHONE image')
        elif('loc_action' in param):
            self.loc_with_action = param['loc_action']
            self.loc_cli_inloop.config({'update_with_action':self.loc_with_action})
            print('[UI]set localization with action:', self.loc_with_action)
        elif('simulation' in param):
            self.sim = param['simulation']
            print('[UI]set simulation:', self.sim)
        elif('wanted_directions' in param):
            self.WANTED_DIRECTIONS = param['wanted_directions']
            self.loc_cli_inloop.config({'directions_of_imgs':self.WANTED_DIRECTIONS})
            print('[UI]set wanted directions:', self.WANTED_DIRECTIONS)
        elif('log' in param):
            self.log_flag = param['log']
            print('[UI]set log:', self.log_flag)

        return list(res.values())[0] if len(res)==1 else res

    def execute(self,param=None):
        if param is None: return
        if isinstance(param,str): param=[param]

        res={}
        if('move' in param):
            self.move_in_map(param['move'], flag=1)
            self.just_move = True
            #print('move '+param['move'])
        elif('detailmove' in param):
            self.simulator_cli_inloop.execute({'mode':'move','dx':param['detailmove'][0],'dy':param['detailmove'][1],'dyaw':param['detailmove'][2]})
            self.just_move = True
        elif('phoneapp' in param):
            self.img_buffer = param['phoneapp']['image']
            if('depth' in param['phoneapp']):
                self.depth_buffer = param['phoneapp']['depth']
            if('imager' in param['phoneapp']):
                self.imgr_buffer = param['phoneapp']['imager']
            self.curodom = param['phoneapp']['odom']
            if(param['phoneapp']['action'] == None):
                self.lastorder = 'H'
            else:
                self.lastorder = param['phoneapp']['action']
            if('distance' in param['phoneapp']):
                # [dist, cov]
                self.distance = param['phoneapp']['distance']
            self.crossing_type = param['phoneapp']['crossing_type']
            self.real_lnglat = param['phoneapp']['reallnglat']
            lastrealdir = self.real_dir
            self.real_dir = param['phoneapp']['realdir']
            self.real_pos_id,_ = self.search_node_by_lnglat(self.real_lnglat[0],self.real_lnglat[1])
            if('message' in param['phoneapp']):
                self.log_message = param['phoneapp']['message']
            errdir = 0
            if(self.lastorder == 'H'):
                errdir = self.maph.cal_diff_dir(lastrealdir, self.real_dir)
                if(abs(errdir) > 0.175):
                    if(self.crossing_type == 0):
                        # road, L is the same as R
                        self.lastorder = 'L'
                    elif(self.crossing_type == 1):
                        # crossing, analyze turning
                        if(abs(errdir) < 2.79):
                            if(errdir > 0):
                                self.lastorder = 'L'
                            else:
                                self.lastorder = 'R'
                        else:
                            self.lastorder = 'TB'
            self.working_real = True
            print(lastrealdir, self.real_dir, errdir)
            print('[UI]last action:'+self.lastorder)
            print('[UI]phone app execute.')

        return list(res.values())[0] if len(res)==1 else res

    def get_result(self,param=None):
        if param is None: return
        if isinstance(param,str): param=[param]

        res = {}
        if('phoneapp' in param):
            res['loc_res'] = [self.pos_id, self.dir, self.blief]
            res['nav_res'] = [self.order, self.nextorder]
            res['localtarget'] = self.localtarget
            res['odomtarget'] = self.odomtarget
            print('[UI]phone app get result.')

        return list(res.values())[0] if len(res)==1 else res
