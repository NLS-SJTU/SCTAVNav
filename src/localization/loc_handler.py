# coding=utf-8

import time, sys
sys.path.append('..')
import pylab as plt
import random
import math
from localization.path_loc import *
from mapping.map_handler import *
from utils.global_cfg import MAP_DIR, topN, intrinK, intrinKquery, intrinKqueryR, obimg_wid, obimg_hgt, BASE_LINE
import utils.utils as utils
import PyKDL
from PIL import Image as pilImage

from myzmq.zmq_comm import *
from myzmq.zmq_cfg import *
from myzmq.jpeg_compress import jpg_to_img_rgb, img_rgb_to_jpeg
from log.log_handler import log_handler


version_flag = sys.version[0]
getRelativeDir = False#True#

class loc_handler(zmq_comm_svr_c):
    def __init__(self):
        self.shutdown = False
        self.withaction = True
        self.estpos = [[],[],[]]
        self.maph = map_handler()
        self.maph.reload_map(MAP_DIR)
        self.pathloc = path_pos_c(self.maph)
        self.inputimg = []#np.zeros([640,480,3], np.uint8)
        self.inputdepth = []#np.zeros([640,480], np.float32)
        self.inputimgr = []
        self.imgname = ''
        self.realdir = 0
        self.topN = topN #len(detail_node_map)
        self.work = False
        self.crossing_type = 0
        self.action = None
        self.directions_of_imgs = [0.0]
        self.distance = [0.0,0.1]
        self.R_b_in_ref = np.array([[1.,0.,0.], [0,1,0], [0,0,1]])
        self.t_b_in_ref = np.array([0.,0.,0.])
        self.refnode_id = -1
        self.refimg_id = -1
        self.comp_direction = PyKDL.Rotation.RPY(0,0,0)
        self.log_h = log_handler()
        self.initzmqs()

    def initzmqs(self):
        zmq_comm_svr_c.__init__(self, name=name_location, ip=ip_location, port=port_location)
        self.single_loc_cli = zmq_comm_cli_c(name=name_single_location, ip=ip_single_location, port=port_single_location)

    #how to get direction?
    def updatepos(self):
        st = time.time()
        single_loc = self.get_single_loc_witha()
        full_single_loc = self.make_prob_full_witha(single_loc)
        st1 = round(time.time(), 4)
        self.pathloc.update_path(full_single_loc, self.action, self.crossing_type, self.distance, self.realdir)
        self.log_h.log_data([st1, round(time.time(),4)], 'bayest')
        re = self.pathloc.get_last_prob()
        self.estpos = self.get_top_prob_witha(re)
        #print('est pos:',self.estpos)
        self.work = False
        print('[LOCHANDLER]time for a localization update is {} seconds.'.format(time.time()-st))
        self.log_h.end_line()

    # get states with probability higher than 0.0005 and rank from high to low
    def get_top_prob_witha(self, inprob):
        ret = [[],[],[]]
        for state in inprob.keys():
            if(inprob[state][0] > 0.0005):
                nodeid = int(state[:ID_DIGIT])
                prob = inprob[state][0]
                if(state[-1] == 'w'):
                    desstr = 'unknow'
                else:
                    desstr = state[-ID_DIGIT:]
                #print('get top prob with a: '+state)
                direction = self.maph.detail_node_map[state[:ID_DIGIT]][desstr]['dir']
                if(len(ret[0]) == 0):
                    ret[0].append(nodeid) 
                    ret[1].append(direction) 
                    ret[2].append(prob) 
                else:
                    if(prob < ret[2][-1]):
                        ret[0].append(nodeid) 
                        ret[1].append(direction) 
                        ret[2].append(prob) 
                    elif(prob > ret[2][0]):
                        ret[0].insert(0, nodeid)
                        ret[1].insert(0, direction)
                        ret[2].insert(0, prob)
                    else:
                        head = 0
                        end = len(ret[0]) - 1
                        while((end-head) > 1):
                            mid = int((head+end)/2)
                            #print(head,end,ret[2][mid] , inprob[kk][0])
                            if(ret[2][mid] > prob):
                                head = mid
                            else:
                                end = mid
                        #print('headend:',head,end)
                        ret[0].insert(end, nodeid)
                        ret[1].insert(end, direction)
                        ret[2].insert(end, prob)
                # print('add est:',ret)
        return ret

    # let the probability of not observed state to a small prob
    def make_prob_full_witha(self, inprob):
        for state in self.maph.map_states.keys():
            if(not state in inprob.keys()):
                inprob[state] = 0.00001

        return inprob


    # return: 
    # @ matched: whether matched to any ref image
    # @ res: if matched, the res fix the prob of matched image to 1
    def check_with_geo_cons(self, imgref, retref, idimgref, idnoderef, imgquery, ret, imgdepth, imgqueryr, retr, baseline=0.120004):
        matched = False
        # check with depth
        if(not type(imgdepth) == type(None)):
            print('compute Rt with depth')
            matched, R, t = utils.getRtPNP_hf_depth(imgref, imgquery, retref['keypoints'], retref['local_descriptors'], ret['keypoints'], ret['local_descriptors'], imgdepth, intrinK, intrinKquery)
        # check with stereo
        if(not matched and not type(imgqueryr) == type(None)):
            print('compute Rt with stereo')
            # matched0, R_12, t12 = utils.getRt_hf(imgquery, imgqueryr, ret['keypoints'], ret['local_descriptors'],retr['keypoints'], retr['local_descriptors'], intrinKquery, intrinKqueryR)
            st = time.time()
            matched1, R_r1, t_r1 = utils.getRt_hf(imgquery, imgref, ret['keypoints'], ret['local_descriptors'], retref['keypoints'], retref['local_descriptors'], intrinKquery, intrinK)
            if(not matched1):
                return False, None
            matched2, R_r2, t_r2 = utils.getRt_hf(imgqueryr, imgref, retr['keypoints'], retr['local_descriptors'], retref['keypoints'], retref['local_descriptors'], intrinKqueryR, intrinK)
            if(not matched2):
                return False, None
            print('time for  R t computation: {}s'.format(time.time()-st))
            st = time.time()
            # the two t may be wrong?
            res = np.array([[self.maph.data_list[idnoderef][1], self.maph.data_list[idnoderef][2], self.maph.img_data_list[idimgref][4], idnoderef, 1.0]])
            R, t, matched = utils.compute_scale(R_r1, t_r1, R_r2, t_r2, baseline)
            print('time for scale computation: {}s'.format(time.time()-st))

        if(matched):
            res = np.array([[self.maph.data_list[idnoderef][1], self.maph.data_list[idnoderef][2], self.maph.img_data_list[idimgref][4], idnoderef, 1.0]])
            self.R_b_in_ref = R
            self.t_b_in_ref = t.reshape(-1)
            self.refnode_id = idnoderef
            self.refimg_id = idimgref
            R_rc = PyKDL.Rotation(R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1], R[2][2])
            R_ic = PyKDL.Rotation.RPY(0,0,self.realdir)
            imginfo = self.maph.get_information_with_imgid(idimgref)
            R_ir = PyKDL.Rotation.RPY(imginfo[5][0], imginfo[5][1], imginfo[5][2])
            if(len(imginfo) > 6):
                R_ircomp = PyKDL.Rotation.RPY(imginfo[6][0], imginfo[6][1], imginfo[6][2])
            else:
                R_ircomp = PyKDL.Rotation.RPY(0,0,0)
            # self.comp_direction = R_ircomp * R_ir * R_rc * R_ic.Inverse()
            print('fix ref node:', self.refnode_id, ' fix ref img:',self.refimg_id)
            print('compensated dir: ', self.comp_direction.GetRPY())
            # self.comp_direction = np.array([[self.comp_direction[0,0], self.comp_direction[0,1], self.comp_direction[0,2]], [self.comp_direction[1,0], self.comp_direction[1,1], self.comp_direction[1,2]], [self.comp_direction[2,0], self.comp_direction[2,1], self.comp_direction[2,2]]])
            print('\033[1;32;40mmatched\033[0m')
            return matched, res
        else:
            print('\033[1;33;40mnot matched\033[0m')
            return False, res


    # return: [[lng,lat,dir,id,prob], ...]
    def get_raw_single_loc_result(self):
        res = np.array([])
        i = 0
        for imgid in range(len(self.directions_of_imgs)):
            oneimg = self.inputimg[imgid]
            if(type(oneimg) == type(None)):
                continue
            curdir_comp = self.realdir + self.comp_direction.GetRPY()[2]
            st = time.time()
            st1 = round(st, 4)
            ret = self.single_loc_cli.query({'mode':'image','image':oneimg,'retNum':self.topN,'realdir':curdir_comp})
            self.log_h.log_data([st1, round(time.time(),4)], 'leftt')
            maxprob_img_id = ret['maxprobimgid']
            one = ret['res']
            print('[loc]get one result from network in {} s.'.format(time.time()-st))
            # try to find R t, if succeed, this node will be set as one
            # match R t with more ref frames
            if(self.directions_of_imgs[i] == 0 and len(self.inputdepth) > 0):
                imgref = self.maph.get_image_with_imgid(maxprob_img_id)
                print('[loc]matching with image {}'.format(maxprob_img_id))
                if(not type(imgref) == type(None)):
                    imgquery = jpg_to_img_rgb(oneimg)
                    imgdepth = self.inputdepth[imgid]
                    imgqueryr = self.inputimgr[imgid]
                    if(type(imgqueryr) == type(None)):
                        retr = None
                        imgqueryr = None
                    else:
                        st1 = round(time.time(), 4)
                        retr = self.single_loc_cli.get_result({'allres':imgqueryr})
                        self.log_h.log_data([st1, round(time.time(),4)], 'rightt')
                        imgqueryr = jpg_to_img_rgb(imgqueryr)
                    # match by orb feature
                    # utils.getRt(imgref, imgquery, intrinK)
                    # matched, R, t = utils.getRTPNP(imgref, imgquery, imgdepth, intrinK, intrinKquery)
                    # get ref keypoint info and match by hfnet
                    st1 = round(time.time(), 4)
                    retref = self.single_loc_cli.get_result({'allres': img_rgb_to_jpeg(imgref)})
                    self.log_h.log_data([st1, round(time.time(),4)], 'reft')
                    st1 = round(time.time(), 4)
                    matched, tmpres = self.check_with_geo_cons(imgref, retref, maxprob_img_id, ret['res'][0][0][3], imgquery, ret, imgdepth, imgqueryr, retr, BASE_LINE)
                    self.log_h.log_data([st1, round(time.time(),4)], 'geocont')
                    if(matched):
                        res = tmpres
                        self.log_h.log_data([1], 'geook')
                        break
                    else:
                        pos,_,_ = self.pathloc.get_max_prob_pos()
                        if(type(pos) == type('sss')):                            
                            imgref, refid = self.maph.get_image_by_nodedir(pos[:ID_DIGIT], curdir_comp)
                            if(not refid == maxprob_img_id):
                                # the new observation is not ok.
                                # if the last similar image is not the current similar image, match with last image
                                print('[loc]rematch with image {}'.format(refid))
                                retref = self.single_loc_cli.get_result({'allres': img_rgb_to_jpeg(imgref)})
                                matched, tmpres = self.check_with_geo_cons(imgref, retref, refid, int(pos[:ID_DIGIT]), imgquery, ret, imgdepth, imgqueryr, retr, BASE_LINE)
                                if(matched):
                                    self.log_h.log_data([1], 'geook')
                                else:
                                    self.log_h.log_data([0], 'geook')
                                # if matched the last similar image, keep the prob high
                                if(not type(tmpres) == type(None)):
                                    res = tmpres
                                    break
                        else:
                            self.log_h.log_data([0], 'geook')
                else:
                    print('no reference image.')
            else:
                print('no depth input.')
            i += 1
            #[[lng,lat,dir,id,prob], ...]
            one = np.array([[one[i][0][0], one[i][0][1], one[i][0][2], one[i][0][3], one[i][1]] for i in range(len(one))])
            if(getRelativeDir):
                # this is + because it is clockwise(while nwu is counterclockwise)
                # get relative dir of image from vpr
                one[:,2] += self.directions_of_imgs[imgid]
                for nres in range(len(one)):
                    nid = int(one[nres][3]+0.001)
                    one[nres,2] = one[nres,2] 
            else:
                # get absolute dir from vpr
                one[:,2] -= self.directions_of_imgs[imgid]
            if(math.isnan(one[0][4])):
                continue
            if(len(res) == 0):
                res = one
            else:
                res = np.concatenate([res,one],axis=0)
        if(not len(res) == 0):
            res[:,4] = res[:,4] / np.sum(res[:,4])
        return res

    # return: {state1:prob1, state2:prob2, ...}
    def get_single_loc_witha(self):
        st = time.time()
        print('send img to single loc')
        res = self.get_raw_single_loc_result()
        #print('aaaaaaaaaaa',res)
        tt = time.time() - st
        print('get one single loc result use {} second'.format(tt))
        prob_vec = [0.00001]*self.maph.states_num
        ret = {}
        for aa in res:
            #print(aa)
            nodeid = int(aa[3]+0.01) # in case the float fall to aa[3]-1
            # relative dir of image
            # nodedir = data_list[nodeid][-1]/180*math.pi - aa[2] - math.pi/2
            # absolu dir
            nodedir = aa[2]
            # print(str(nodeid).zfill(ID_DIGIT), nodedir)
            #old with no prob of dir
            state = self.maph.check_state(str(nodeid).zfill(ID_DIGIT), nodedir)
            stateid = self.maph.check_id_to_prob(state)
            if(aa[4] > prob_vec[stateid]):
                prob_vec[stateid] = aa[4]    
            ret[state] = prob_vec[stateid]

        print('single loc:', ret)
        return ret

    def reset(self,param=None):
        print('[LOCHANDLER]reset')
        self.pathloc.rest_prob()
        self.estpos = [[],[],[]]
        self.refnode_id = -1
        self.log_h.end_log()
        return None

    def execute(self,param=None):
        if param is None: return
        if isinstance(param,str): param=[param]
        
        res={}
        if('image' in param):
            self.log_h.new_line()
            print('[location]get image')
            self.inputimg = []
            for oneimg in param['image']:
                if(type(oneimg) == type(None)):
                    self.inputimg.append(None)
                    continue
                img = np.frombuffer(oneimg,dtype=np.uint8)
                self.inputimg.append(img)
            self.inputimgr = []
            for oneimg in param['imager']:
                if(type(oneimg) == type(None)):
                    self.inputimgr.append(None)
                    continue
                img = np.frombuffer(oneimg,dtype=np.uint8)
                self.inputimgr.append(img)                
            self.inputdepth = []
            for oneimg in param['depth']:
                if(type(oneimg) == type(None)):
                    self.inputdepth.append(None)
                    continue
                img = np.frombuffer(oneimg,dtype=np.float32).reshape(obimg_hgt,obimg_wid)
                self.inputdepth.append(img)
            self.crossing_type = param['crossing_type']
            self.distance = param['distance']
            self.log_h.log_data(self.distance, 'dist')
            self.realdir = param['realdir']
            print('self.crossing_type',self.crossing_type)
            print('distance', self.distance)
            print('realdir', self.realdir)
            if(self.withaction):
                self.action = param['action']
                if(type(self.action) == type(u'sb')):
                    self.action = str(self.action)
                print('action', self.action)
            #img = np.frombuffer(img,dtype=np.uint8)
            #img = jpg_to_img_rgb(img)
            #self.inputimg = img#cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.work = True

        return list(res.values())[0] if len(res)==1 else res

    def get_result(self,param=None):
        if param is None: return
        if isinstance(param,str): param=[param]
        
        res={}
        if('est_pos' in param): 
            res['est_pos'] = self.estpos
            res['R_b_in_ref'] = self.R_b_in_ref
            res['t_b_in_ref'] = self.t_b_in_ref
            res['refnode_id'] = self.refnode_id
            res['refimg_id'] = self.refimg_id
            res['comp_direction'] = np.array([[self.comp_direction[0,0], self.comp_direction[0,1], self.comp_direction[0,2]], [self.comp_direction[1,0], self.comp_direction[1,1], self.comp_direction[1,2]], [self.comp_direction[2,0], self.comp_direction[2,1], self.comp_direction[2,2]]])

        return list(res.values())[0] if len(res)==1 else res

    def query(self,param=None):
        if param is None: return
        if isinstance(param,str): param=[param]
        
        res={}
        if('locworking' in param):
            res['locworking'] = self.work
        elif('sim_path_loc_result' in param):
            res['sim_path_loc_result'] = self.pathloc.get_pos_backtrace()
            self.pathloc.print_backtrace()

        return list(res.values())[0] if len(res)==1 else res

    def config(self,param=None):
        if param is None: return
        if isinstance(param,str): param=[param]

        res={}
        if('top_n_node' in param):
            self.topN = param['top_n_node']
            print('[LOCHANDLER]set top {} nodes from single loc.'.format(self.topN))
        elif('update_with_action' in param):
            self.withaction = param['update_with_action']
            self.pathloc.withaction = self.withaction
            self.reset()
            print('[LOCHANDLER]set localize with action: {}'.format(self.withaction))
        elif('directions_of_imgs' in param):
            self.directions_of_imgs = param['directions_of_imgs']
            print('[LOCHANDLER]set wanted directions: ', self.directions_of_imgs)
        elif('start' in param):
            print('[LOCHANDLER]start!')
            self.log_h.start_new_log_dict(param['start']+'_loc')
        elif('end' in param):
            print('[LOCHANDLER]end!')
            self.log_h.end_log_dict()
        return list(res.values())[0] if len(res)==1 else res

