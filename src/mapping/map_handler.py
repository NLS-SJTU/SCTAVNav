# coding=utf-8

import os
import time,sys
import pylab as plt
import cv2
import math
from scipy import stats
import numpy as np
import pymap3d as pm
import json
from utils.global_cfg import ID_DIGIT

version_flag = sys.version[0]



class map_handler():
    def __init__(self, Buildmode=True, _detail_node_map=None):
        self.detail_node_map = _detail_node_map
        self.map_states = None
        self.id_to_states = None
        self.data_list = None
        self.mapping_data_list = None
        self.img_data_list = None
        self.node_to_imgs = {}
        self.map_path = '/Noneplace'
        if(not Buildmode):
            self.find_crossing()
            self.states_num = len(_id_to_states)
            self.gen_dir_vecotr_of_all_states()

    def reload_map(self, map_path):
        print('[MAP]reload map from '+map_path)
        self.map_path = map_path
        sys.path.append(map_path)

        from allnodemap_range import detail_node_map
        from id_to_states import id_to_states
        from map_states import map_states
        self.detail_node_map = detail_node_map
        self.map_states = map_states
        self.id_to_states = id_to_states
        with open(map_path+'nodemap_info.json') as f:
            self.nodemap_info = json.load(f)
        with open(map_path+'data_list.json') as f:
            self.data_list = json.load(f)
        with open(map_path+'img_data_list.json') as f:
            self.img_data_list = json.load(f)
        with open(map_path+'mapping_data_list.json') as f:
            self.mapping_data_list = json.load(f)
        self.node_to_imgs = {}
        for i in range(len(self.img_data_list)):
            nodeid = str(self.img_data_list[i][3]).zfill(ID_DIGIT)
            if(nodeid in self.node_to_imgs.keys()):
                self.node_to_imgs[nodeid].append(self.img_data_list[i][0])
            else:
                self.node_to_imgs[nodeid] = [self.img_data_list[i][0]]

        sys.path.pop()

        self.find_crossing()
        self.states_num = len(self.id_to_states)
        self.gen_dir_vecotr_of_all_states()
        print('[MAP]reload map done.')

    # prepare the probability of the crossing nodes
    def find_crossing(self):
        crossing_prob = 0.99
        # for with action
        self.crossing_prob_state = np.array([1.0-crossing_prob]*len(self.map_states))
        # for without action
        self.crossing_prob_node = {}
        for nk in self.detail_node_map.keys():
            if(len(self.detail_node_map[nk].keys()) > 2 or 'unknow' in self.detail_node_map[nk].keys()):
                # crossing
                self.crossing_prob_node[nk] = crossing_prob
                for nei in self.detail_node_map[nk].keys():
                    state = nk+'_'+nei
                    id_to_prob = self.check_id_to_prob(state)
                    self.crossing_prob_state[id_to_prob] = crossing_prob
            else:
                self.crossing_prob_node[nk] = 1 - crossing_prob

    def is_crossing_state(self, state):
        id_to_prob = self.check_id_to_prob(state)
        if(self.crossing_prob_state[id_to_prob] > 0.5):
            return True
        else:
            return False

    def is_crossing_node(self, nodestr):
        if(len(self.node_to_imgs[nodestr]) > 1):
            return True
        else:
            return False

    # get probability of crossing nodes or states
    # withaction(bool), node(string)
    # prob_type(int)
    # 0: normal road; 1: crossing; -1: unknown type
    # return
    # if withaction = true: return vector of all states
    # else: return one element of node
    def get_crossing_prob(self, withaction=False, prob_type=0, node=None):
        if(withaction):
            if(prob_type < 0):
                ret = np.array([1.0]*len(self.map_states))
            elif(prob_type == 0):
                ret = 1.0 - self.crossing_prob_state
            else:
                ret = self.crossing_prob_state
        else:
            if(prob_type < 0):
                ret = 1.0
            elif(prob_type == 0):
                ret = 1.0 - self.crossing_prob_node[node]
            else:
                ret = self.crossing_prob_node[node]

        return ret

    # calculate forward prob predict
    def cal_forward_probs(self, fromstate):
        thisnode = fromstate[:ID_DIGIT]
        nextnode = fromstate[-ID_DIGIT:]
        if(nextnode[-1] == 'w'):
            # dead end
            return {fromstate:1.0}
        elif(len(self.detail_node_map[nextnode].keys())==2):
            dirforward = self.detail_node_map[thisnode][nextnode]['dir']
            # road
            nextneighor = self.get_neighbor_node(nextnode, dirforward, 'F')
            # cdf(90/60)-cdf(-90/60)=0.86
            return {nextnode+'_'+nextneighor:0.86,nextnode+'_'+thisnode:0.14}
        else:
            # crossing
            dirforward = self.detail_node_map[thisnode][nextnode]['dir']
            ret = {}
            # print(fromstate)
            for nextneighor in self.detail_node_map[nextnode].keys():
                ret[nextnode+'_'+nextneighor] = self.cal_prob_with_angle_range(dirforward, self.detail_node_map[nextnode][nextneighor]['range'])
            return ret

    # sum up nearby prob to crossing if it is crossing
    def sum_up_state(self, prob, crossing_type):
        if(crossing_type > 0):
            sumup_distance = 50.0
            crossing_reco_rate = 0.99
            for i in range(len(prob)):
                state = self.id_to_states[i]
                if(not 'nearby_crossing_state' in self.map_states[state].keys()):
                    continue
                nearby_crossing_state = self.map_states[state]['nearby_crossing_state']
                # this state is not crossing state, add prob to nearby crossing state
                if(not state == nearby_crossing_state):
                    # if this state is near crossing, add prob to crossing state
                    # or just remove this prob
                    if(self.map_states[state]['nearby_crossing_dist'] < sumup_distance):
                        id_nearby_crossing = self.map_states[nearby_crossing_state]['id_to_prob']
                        prob[id_nearby_crossing] += prob[i]
                    prob[i] = prob[i] / len(prob)
        
        return prob

    def sum_up_node(self, pos_prob, crossing_type):
        if(crossing_type > 0):
            for pos in pos_prob.keys():
                if(len(self.detail_node_map[pos]) == 2 and not ('unknow' in self.detail_node_map[pos].keys())):
                    if(version_flag == 2):
                        state = pos+'_'+self.detail_node_map[pos].keys()[0]
                    else:
                        state = pos+'_'+list(self.detail_node_map[pos].keys())[0]
                    if(not 'nearby_crossing_state' in self.map_states[state].keys()):                        
                        if(version_flag == 2):
                            state = pos+'_'+self.detail_node_map[pos].keys()[1]
                        else:
                            state = pos+'_'+list(self.detail_node_map[pos].keys())[1]
                        if(not 'nearby_crossing_state' in self.map_states[state].keys()):  
                            continue
                    nearby_crossing_state = self.map_states[state]['nearby_crossing_state']
                    node_crossing = nearby_crossing_state[:ID_DIGIT]
                    if(pos == '000016'):
                        print(pos_prob[pos])
                    #print(state, nearby_crossing_state, pos, node_crossing)
                    pos_prob[node_crossing] = (pos_prob[node_crossing][0] + pos_prob[pos][0], pos_prob[node_crossing][1])
                    pos_prob[pos] = (pos_prob[pos][0] / len(pos_prob), pos_prob[pos][1])
                    if(pos == '000016'):
                        print(pos_prob[pos])
        return pos_prob

    # spread crossing state prob to neaby state
    def spread_sumup_prob_to_state(self, sumupprob):
        spread_prob = [ sumupprob[self.map_states[self.map_states[self.id_to_states[i]]["nearby_crossing_state"]]['id_to_prob']] for i in range(len(sumupprob))]
        return spread_prob

    # node(str)
    def check_crosing_type(self, node):
        if(self.crossing_prob_node[node] > 0.5):
            ret = 1
        else:
            ret = 0
        # print('crossing check node '+node+' is {}'.format(ret))
        return ret

    # pos_now=[[id,dir,prob], [...]]
    def is_pos_confirm(self, pos_now):
        if(len(pos_now) == 0):
            return False
        is_confirm = False
        # see if the top pos is large enough
        topprob = pos_now[0][2]
        secondprob = 0
        idtop = pos_now[0][0]
        idsecond = 1
        for i in range(1,len(pos_now)):
            idnode = pos_now[i][0]
            if(self.is_neibor(idtop, idnode)):
                topprob += pos_now[i][2]
            else:
                idsecond = i
                secondprob = pos_now[i][2]
                break
        if(topprob > 0.66 or topprob/2 > secondprob):
            is_confirm = True
        return is_confirm

    def intnodeid_to_strid(self, intid):
        return str(intid).zfill(ID_DIGIT)

    ###
    # for fast check states with action
    # fromnode: string like '00231'; 
    # direction: float; 
    # state: string like '00123_01321', from 123 node heading to 1321 node;
    def check_state(self, fromnode, direction):
        forwardnode = self.get_neighbor_node(fromnode, direction, 'F')
        return fromnode+'_'+forwardnode

    def check_state_withid(self, idnum):
        return self.id_to_states[idnum]

    def check_id_to_prob(self, state):
        if(not state in self.map_states.keys()):
            return -1
        return self.map_states[state]['id_to_prob']

    def check_next_state(self, state, motion):
        if(motion == 'H'):
            return state
        return self.map_states[state][motion]

    def check_next_crossing(self, nodeid, direction):
        state = self.check_state(str(nodeid).zfill(ID_DIGIT), direction)
        nextcrossingstate = self.map_states[state]['to_crossing_state']
        return int(nextcrossingstate[:ID_DIGIT])

    def is_neibor(self, id1, id2):
        return (str(id1).zfill(ID_DIGIT) in self.detail_node_map[str(id2).zfill(ID_DIGIT)].keys())

    # node1,node2 are string as '00021'
    def get_distance_between_neibornodes(self, node1, node2):
        if(node2[-1] == 'w'):
            return self.detail_node_map[node1]['unknow']['dist']
        else:
            return self.detail_node_map[node1][node2]['dist']

    # for just nodes with no action
    def find_nearby(self, fromnode):
        return self.detail_node_map[fromnode].keys()

    # calculate diff from dir1 to dir2 (dir2-dir1), [-pi, pi], left is plus
    def cal_diff_dir(self, dir1 = 0., dir2 = 0.):
        difdir = dir2 - dir1
        mul_2_pi = 2*math.pi
        while(difdir < -math.pi):
            difdir += mul_2_pi
        while(difdir > math.pi):
            difdir -= mul_2_pi
        return difdir

    ###
    # for generate states from allnodemap_range.py
    def cal_del_dir(self, fromnode, direction, tonode):
        if(len(self.detail_node_map[fromnode].keys()) == 1):
            return [1, -1]
        mul_2_pi = 2*math.pi
        while(direction < -mul_2_pi):
            direction += mul_2_pi
        while(direction > mul_2_pi):
            direction -= mul_2_pi
        dir_range = self.detail_node_map[fromnode][tonode]['range']
        delrdir = dir_range[0] - direction
        delldir = dir_range[1] - direction
        if(delrdir > math.pi):
            delrdir = delrdir - math.pi*2
        elif(delrdir < -math.pi):
            delrdir = delrdir + math.pi*2
        if(delldir > math.pi):
            delldir = delldir - math.pi*2
        elif(delldir < -math.pi):
            delldir = delldir + math.pi*2
        return [delldir, delrdir]

    def cal_dist_of_lnglat(self, lnglat1, lnglat2):
        return math.sqrt((lnglat1[0] - lnglat2[0])**2+(lnglat1[1] - lnglat2[1])**2)

    # input node id in string, direction in float, which neighbor you want('F','L','R')
    # return direction node id in string
    def get_neighbor_node(self, node_id_str, direction, whichneighbor):
        if(node_id_str[-1] == 'w'):
            return None
        if(whichneighbor == 'H'):
            return node_id_str
        retneighbor = ''
        nearL = 2*math.pi
        nearR = -2*math.pi
        TBdir = 0.0
        for neighbor in self.detail_node_map[node_id_str].keys():
            motion,reladir = self.get_relative_motion(node_id_str, direction, neighbor)
            # print(neighbor, motion, reladir)

            if(motion == 'F'):
                if(whichneighbor == 'F'):
                    retneighbor = neighbor
                    break
            elif(whichneighbor == 'L'):
                delrot = reladir[1]
                if(delrot < 0):
                    delrot += 2*math.pi
                if(delrot < nearL):
                    nearL = reladir[1]
                    retneighbor = neighbor
            elif(whichneighbor == 'R'):
                delrot = reladir[0]
                if(delrot > 0):
                    delrot -= 2*math.pi
                if(delrot > nearR):
                    nearR = reladir[0]
                    retneighbor = neighbor
            elif(whichneighbor == 'TB'):
                difdir = self.cal_diff_dir(direction, self.detail_node_map[node_id_str][neighbor]['dir'])
                if(abs(difdir) > abs(TBdir)):
                    TBdir = difdir
                    retneighbor = neighbor

        return retneighbor

    def get_relative_motion(self, fromnode, direction, tonode):
        delldir, delrdir = self.cal_del_dir(fromnode, direction, tonode)
        if(delldir > 0 and delrdir < 0):
            ret = 'F'
        elif(math.fabs(delrdir) < math.fabs(delldir)):
            ret = 'L'
        else:
            ret = 'R'

        return ret, [delldir, delrdir]

    def get_state(self, fromnode, direction):
        neighbor = self.get_neighbor_node(fromnode, direction, 'F')
        return fromnode+'_'+neighbor

    def cal_prob_with_angle_dir(self, centerangle, refangle):
        cosdeldir = np.cos(centerangle - refangle)
        if(cosdeldir > 0.00001):
            cosdeldir = 1
        else:
            cosdeldir = 0.00001
        # cosdeldir = np.clip(cosdeldir, 0.00001, 1)
        return cosdeldir

    # cal prob of gausian in angle
    def cal_prob_with_angle_range(self, centerangle, anglerange):
        centeranglerange = [anglerange[0]-centerangle, anglerange[1]-centerangle]
        if(centeranglerange[0] > math.pi):
            centeranglerange[0] = centeranglerange[0] - 2*math.pi
        elif(centeranglerange[0] < -math.pi):
            centeranglerange[0] = centeranglerange[0] + 2*math.pi
        if(centeranglerange[1] > math.pi):
            centeranglerange[1] = centeranglerange[1] - 2*math.pi
        elif(centeranglerange[1] < -math.pi):
            centeranglerange[1] = centeranglerange[1] + 2*math.pi
        if(centeranglerange[0] < centeranglerange[1]):
            prob = stats.norm.cdf(centeranglerange[1]*3/math.pi) - stats.norm.cdf(centeranglerange[0]*3/math.pi)
        else:
            prob = 1 - stats.norm.cdf(centeranglerange[0]*3/math.pi) + stats.norm.cdf(centeranglerange[1]*3/math.pi)
        # print('l,r,p', centeranglerange, prob)
        return prob

    def get_lnglat_by_id(self, nodeid):
        return [self.data_list[nodeid][1], self.data_list[nodeid][2]]

    # search nearest node id by lnglat
    def search_node_by_lnglat(self,lng,lat):
        dmin=np.inf
        nmin=-1
        for node in self.detail_node_map.keys():
            lng0,lat0=self.get_lnglat_by_id(int(node))
            dx,dy,_=pm.geodetic2enu(lat, lng, 0, lat0, lng0,0)
            d = dx**2 + dy**2
            if d<dmin:
                dmin=d
                nmin=node
        #nmin = int(mapto[nmin])
        lng0,lat0=self.get_lnglat_by_id(int(nmin))
        dx,dy,_ = pm.geodetic2enu(lat, lng, 0, data_list[int(nmin)][2], data_list[int(nmin)][1],0)
        dmin2 = dx**2 + dy**2
        return int(nmin), dmin2

    def gen_dir_vecotr_of_all_states(self):
        self.dir_of_states_vec = []
        for i in range(len(self.id_to_states)):
            state = self.id_to_states[i]
            orinode = state[:ID_DIGIT]
            if(state[-1] == 'w'):
                dirnode = 'unknow'
            else:
                dirnode = state[-ID_DIGIT:]
            statedir = self.detail_node_map[orinode][dirnode]['dir']
            self.dir_of_states_vec.append(statedir)
        self.dir_of_states_vec = np.array(self.dir_of_states_vec)

    def gen_prob_mask_with_realdir(self, realdir):
        if(realdir == None):
            return np.array([1.0]*len(self.id_to_states))
        deldirvec = self.dir_of_states_vec - realdir
        probmask = np.cos(deldirvec)
        probmask = (probmask - 0.707)
        retprobmask = np.clip(probmask, 0.0001, 1)
        return retprobmask

    def get_image_with_imgid(self, maxprob_img_id):
        img_path = self.map_path + 'images/'
        if(os.path.exists(img_path)):
            imgname = img_path + self.img_data_list[maxprob_img_id][0]
            img = cv2.imread(imgname)
            print('[maph]read '+imgname)
            return img
        else:
            print('[maph]no image in '+img_path + self.img_data_list[maxprob_img_id][0])
            return None

    def get_image_by_nodedir(self, nodeidstr, dir_aftercomp):
        img_path = self.map_path + 'images/'
        min_diffdir = 100
        min_diffid = -1
        for i in range(len(self.node_to_imgs[nodeidstr])):
            imgidstr = self.node_to_imgs[nodeidstr][i][:ID_DIGIT]
            imgidint = int(imgidstr)
            imgdir = self.mapping_data_list[imgidint][5][2]
            if(len(self.mapping_data_list[imgidint]) == 7):
                imgdir += self.mapping_data_list[imgidint][6][2]
            diffdir = abs(dir_aftercomp - imgdir)
            if(diffdir > math.pi):
                diffdir = math.pi*2 - diffdir
            if(diffdir < min_diffdir):
                min_diffdir = diffdir
                min_diffid = imgidint
        img = self.get_image_with_imgid(min_diffid)
        return img, min_diffid


    def get_information_with_imgid(self, img_id):
        return self.mapping_data_list[img_id]

# for test
if __name__ == '__main__':
    testmaph = map_handler()
    fromnode = '01530'
    direction = -8
    tonode = '00589'

    print(testmaph.get_neighbor_node(fromnode, direction, 'F'))
    #print(testmaph.get_relative_motion(fromnode, direction, tonode))
