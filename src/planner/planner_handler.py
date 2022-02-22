# coding=utf-8

# h:hold
# f:forward
# l:turn left
# r:turn right

import sys
sys.path.append('..')
import numpy as np
import random
import time
import PyKDL

from mapping.map_handler import map_handler
from planner.dijk import djik_c
from myzmq.zmq_comm import *
from myzmq.zmq_cfg import *
from utils.global_cfg import *
from log.log_handler import log_handler


USE_IMUDIR = False

class planner_handler(zmq_comm_svr_c):
    def __init__(self, MAP_DIR=MAP_DIR):
        self.q_val = []
        self.maph = map_handler()
        self.maph.reload_map(MAP_DIR)
        self.dijkc = djik_c(self.maph)
        self.shutdown = False
        self.replan = False
        self.work = False
        self.des_id = 0
        self.action = 'H'
        self.nextaction = 'H'
        self.pos_now = []
        self.lastact = 'H'
        self.turncnt = 0
        self.crossing_type = 0
        self.log_h = log_handler()
        self.initzmqs()
        print('[plan]init ok!')

    def initzmqs(self):
        zmq_comm_svr_c.__init__(self, name=name_nav, ip=ip_nav, port=port_nav)

    def replanall(self):
        self.dijkc.cal_q(self.des_id)
        self.replan = False
        self.action = 'H'
        self.nextaction = 'H'
        self.lastact = 'H'
        self.turncnt = 0

    def get_action(self):
        print(self.pos_now)
        st = round(time.time(), 4)
        self.get_action_dijk()
        self.log_h.log_data([st, round(time.time(), 4)], 'getat')
        print('[plan]time for a decision is {} seconds.'.format(time.time()-st))
        self.work = False

    def get_action_dijk(self):
        # input - pos_now=[[id,dir,prob], [...]]
        self.action, self.nextaction = self.dijkc.get_action(self.pos_now)
        
        # or self.maph.is_pos_confirm(self.pos_now)
        if(self.action == 'H' or not self.crossing_type == 0):
            # only the first step or at crossing will give a different action
            print('[plan]at crossing')
        else:
            #self.action, self.nextaction = self.dijkc.get_action(self.pos_now)
            if(not self.action == 'H'):
                self.action = 'F'
            print('[plan]not at crossing')
        self.lastact = self.action
        
        print('[plan]action:'+self.action+', nextaction:'+self.nextaction)
        print('')

    def get_local_target_noimu(self, refnode_id, refimg_id, R_b_in_ref, t_b_in_ref, refodom, curodom):
        print('Method without imu')
        Rpykdl = curodom[1]
        T_oc = np.array([[Rpykdl[0,0], Rpykdl[0,1], Rpykdl[0,2], curodom[0][0]], \
                         [Rpykdl[1,0], Rpykdl[1,1], Rpykdl[1,2], curodom[0][1]], \
                         [Rpykdl[2,0], Rpykdl[2,1], Rpykdl[2,2], curodom[0][2]], \
                         [0, 0, 0, 1]])
        Rpykdl = refodom[1]
        T_of = np.array([[Rpykdl[0,0], Rpykdl[0,1], Rpykdl[0,2], refodom[0][0]], \
                         [Rpykdl[1,0], Rpykdl[1,1], Rpykdl[1,2], refodom[0][1]], \
                         [Rpykdl[2,0], Rpykdl[2,1], Rpykdl[2,2], refodom[0][2]], \
                         [0, 0, 0, 1]])
        T_rf = np.array([[R_b_in_ref[0,0], R_b_in_ref[0,1], R_b_in_ref[0,2], t_b_in_ref[0]], \
                         [R_b_in_ref[1,0], R_b_in_ref[1,1], R_b_in_ref[1,2], t_b_in_ref[1]], \
                         [R_b_in_ref[2,0], R_b_in_ref[2,1], R_b_in_ref[2,2], t_b_in_ref[2]], \
                         [0, 0, 0, 1]])
        # get mapping data list
        refimginfo = self.maph.get_information_with_imgid(refimg_id)
        Rpykdl = PyKDL.Rotation.RPY(refimginfo[5][0], refimginfo[5][1], refimginfo[5][2])
        T_mor = np.array([[Rpykdl[0,0], Rpykdl[0,1], Rpykdl[0,2], 0], \
                          [Rpykdl[1,0], Rpykdl[1,1], Rpykdl[1,2], 0], \
                          [Rpykdl[2,0], Rpykdl[2,1], Rpykdl[2,2], 0], \
                          [0, 0, 0, 1]])
        T_omo = np.matmul(np.matmul(T_of, np.linalg.inv(T_rf)), np.linalg.inv(T_mor))
        T_moc = np.matmul(np.linalg.inv(T_omo), T_oc)
        t_moc = T_moc[0:3,3]

        p_ct = np.array([0,0,0])
        if(refnode_id < 0):
            print('ref node does not exist! False function called.')
            return p_ct, np.matmul(T_oc[0:3,0:3], p_c) + T_oc[0:3,3]
        st = round(time.time(),4)
        lastdist = np.linalg.norm(T_moc[0:3,3])
        curnode_id = refnode_id
        curnode_id_str = self.maph.intnodeid_to_strid(curnode_id)
        p_mot = np.array([0., 0., 0.])
        # find a proper target node in map odom frame
        nextnodeok = False
        while(True):
            curdisttodest = self.dijkc.get_q(int(curnode_id_str))
            if(curdisttodest == 0):
                # this node is the final destination, directly return localtarget
                break
            # find a neighbor that is closer to desintation
            mindisttodest = 100000
            mindist_nodeid = -1
            print('search neighbors for '+curnode_id_str)
            for neighbor in self.maph.detail_node_map[curnode_id_str].keys():
                # print('neighbor', neighbor, curnode_id_str)
                nei_id = int(neighbor)
                neidisttodest = self.dijkc.get_q(nei_id) #+ self.maph.detail_node_map[curnode_id_str][neighbor]['dist']  
                print(neighbor, neidisttodest)
                # print(self.dijkc.get_q(nei_id), self.maph.detail_node_map[curnode_id_str][neighbor]['dist'])
                if(neidisttodest < mindisttodest):
                    mindist_nodeid = neighbor
                    mindisttodest = neidisttodest
            theta = self.maph.detail_node_map[curnode_id_str][mindist_nodeid]['dir']
            reladist = self.maph.detail_node_map[curnode_id_str][mindist_nodeid]['dist']
            p_mot[0] = p_mot[0] + np.cos(theta) * reladist
            p_mot[1] = p_mot[1] + np.sin(theta) * reladist
            curdist = np.linalg.norm(t_moc - p_mot)
            print('choose neighbor ', mindist_nodeid, curdist)
            curnode_id_str = mindist_nodeid
            if(curdist > lastdist):
                nextnodeok = True
            if(nextnodeok and curdist > 2):
                break
            lastdist = curdist

        # compute target in odom frame and current body frame
        T_mot = np.array([[1, 0, 0, p_mot[0]], \
                          [0, 1, 0, p_mot[1]], \
                          [0, 0, 1, p_mot[2]], \
                          [0, 0, 0, 1]])
        T_ot = np.matmul(T_omo, T_mot)
        T_ct = np.matmul(np.linalg.inv(T_oc), T_ot)

        self.log_h.log_data([st, round(time.time(),4)], 'tart')
        print('[plan]refnode is {}, local target node is {}.'.format(refnode_id, curnode_id_str))
        print('local target: ', T_ct[0:3,3], 'odom target: ', T_ot[0:3,3])
        return T_ct[0:3,3], T_ot[0:3,3]

    def get_local_target(self, refnode_id, R_b_in_ref, t_b_in_ref, refodom, curodom, realdir, comp_direction):
        print('Method with imu')
        # use numpy for mat mult
        Rpykdl = curodom[1]
        R_oc = np.array([[Rpykdl[0,0], Rpykdl[0,1], Rpykdl[0,2]], \
            [Rpykdl[1,0], Rpykdl[1,1], Rpykdl[1,2]], \
            [Rpykdl[2,0], Rpykdl[2,1], Rpykdl[2,2]]])
        t_oc = np.array(curodom[0])
        Rpykdl = refodom[1]
        R_of = np.array([[Rpykdl[0,0], Rpykdl[0,1], Rpykdl[0,2]], \
            [Rpykdl[1,0], Rpykdl[1,1], Rpykdl[1,2]], \
            [Rpykdl[2,0], Rpykdl[2,1], Rpykdl[2,2]]])
        t_of = np.array(refodom[0])
        R_iccomp = np.array([[comp_direction[0,0], comp_direction[0,1], comp_direction[0,2]], \
                             [comp_direction[1,0], comp_direction[1,1], comp_direction[1,2]], \
                             [comp_direction[2,0], comp_direction[2,1], comp_direction[2,2]]])
        R_ic = np.array([[np.cos(realdir), -np.sin(realdir), 0], \
                         [np.sin(realdir), np.cos(realdir), 0], \
                         [0, 0, 1]])
        R_ic = np.matmul(R_iccomp, R_ic)
        R_rf = R_b_in_ref
        t_rf = t_b_in_ref
        R_ci = R_ic.T #np.matmul(np.matmul(R_oc.T, R_of), R_if.T)
        t_ci = np.matmul(R_oc.T, t_of) - np.matmul(R_oc.T, t_oc) - np.matmul(np.matmul(np.matmul(R_oc.T, R_of), R_rf.T), t_rf)
        # print(t_oc, t_of, t_ci, t_rf)
        # find out the most possible node with odom along the path from refnode to destination
        curnode_id = refnode_id
        p_i = np.array([0.,0.,0.]) # in refnode frame
        p_c = t_ci
        lastdist = np.linalg.norm(p_c)
        curnode_id_str = self.maph.intnodeid_to_strid(curnode_id)
        print('refnode:',refnode_id)
        print('R_ci:', R_ci, 't_ci:', t_ci)
        print('R_of:', R_of, 't_of:', t_of)
        print('R_oc:', R_oc, 't_oc:', t_oc)
        print('R_rf:', R_rf, 't_rf:', t_rf)
        print('R_ic:', R_ic)
        st = round(time.time(),4)
        if(refnode_id < 0):
            return p_c, np.matmul(R_oc, p_c) + t_oc
        nextnodeok = False
        while(True):
            curdisttodest = self.dijkc.get_q(int(curnode_id_str))
            if(curdisttodest == 0):
                # this node is the final destination, directly return localtarget
                break
            mindisttodest = 100000
            mindist_nodeid = -1
            for neighbor in self.maph.detail_node_map[curnode_id_str].keys():
                # print('neighbor', neighbor, curnode_id_str)
                nei_id = int(neighbor)
                neidisttodest = self.dijkc.get_q(nei_id) #+ self.maph.detail_node_map[curnode_id_str][neighbor]['dist']
                # print(neidisttodest)
                if(neidisttodest < mindisttodest):
                    mindist_nodeid = neighbor
                    mindisttodest = neidisttodest
            theta = self.maph.detail_node_map[curnode_id_str][mindist_nodeid]['dir']
            reladist = self.maph.detail_node_map[curnode_id_str][mindist_nodeid]['dist']
            p_i[0] = p_i[0] + np.cos(theta) * reladist
            p_i[1] = p_i[1] + np.sin(theta) * reladist
            # see if the node is getting far from current position
            p_c = np.matmul(R_ci, p_i) + t_ci
            curdist = np.linalg.norm(p_c)
            print(mindist_nodeid, 'p_i', p_i, 'p_c', p_c, curdist)
            curnode_id_str = mindist_nodeid
            if(curdist > lastdist):
                nextnodeok = True
            if(nextnodeok and curdist > 2):
                break
            lastdist = curdist

        p_o = np.matmul(R_oc, p_c) + t_oc
        # the last log
        self.log_h.log_data([st, round(time.time(),4)], 'tart')

        print('[plan]refnode is {}, local target node is {}.'.format(refnode_id, curnode_id_str))

        return p_c, p_o
    
    def config(self,param=None):
        if param is None: return
        if isinstance(param,str): param=[param]
        
        res={}
        if('set_des_pos' in param):
            print('[plan]set des pos to node {}'.format(param['set_des_pos']))
            self.des_id = param['set_des_pos']
            self.replan = True
        elif('start' in param):
            print('[plan]start!')
            self.log_h.start_new_log_dict(param['start']+'_nav')
        elif('end' in param):
            print('[plan]end!')
            self.log_h.end_log_dict()

        return list(res.values())[0] if len(res)==1 else res

    def execute(self,param=None):
        if param is None: return
        if isinstance(param,str): param=[param]
        
        res={}
        if('pos_now' in param and 'crossing_type' in param):
            self.pos_now = param['pos_now']
            self.crossing_type = param['crossing_type']
            self.log_h.new_line()
            self.work = True

        return list(res.values())[0] if len(res)==1 else res

    def query(self,param=None):
        if param is None: return
        if isinstance(param,str): param=[param]
        
        res={}
        if('navworking' in param):
            res['navworking'] = self.work
        elif('totalpathlength' in param):
            res['totalpathlength'] = self.dijkc.get_q(param['totalpathlength'])

        return list(res.values())[0] if len(res)==1 else res

    def reset(self, param=None):
        if param is None: return

        return None

    def get_result(self,param=None):
        if param is None: return
        if isinstance(param,str): param=[param]
        
        res={}
        if('action' in param): 
            res['action'] = self.action
            res['nextaction'] = self.nextaction
        elif('localtarget' in param):
            # print(param)
            if(type(param['localtarget']) == type(None)):
                self.log_h.end_line()
                return None
            refnode_id = param['localtarget']['refnode_id']
            refimg_id = param['localtarget']['refimg_id']
            R_b_in_ref = param['localtarget']['R_b_in_ref']
            t_b_in_ref = param['localtarget']['t_b_in_ref']
            refodom = param['localtarget']['refodom']
            curodom = param['localtarget']['curodom']
            realdir = param['localtarget']['realdir']
            comp_direction = param['localtarget']['comp_direction']
            
            if(USE_IMUDIR):
                res['localtarget'], res['odomtarget'] = self.get_local_target(refnode_id, R_b_in_ref, t_b_in_ref, refodom, curodom, realdir, comp_direction)
            else:
                res['localtarget'], res['odomtarget'] = self.get_local_target_noimu(refnode_id, refimg_id, R_b_in_ref, t_b_in_ref, refodom, curodom)
            self.log_h.end_line()

        return list(res.values())[0] if len(res)==1 else res

    def main_loop(self):
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
            self.shutdown = True
            return False
        else:
            print('unknown api name '+api_name)
            self.skt.send(pickle.dumps(None))
        return True

def test():
    navh = planner_handler(MAP_DIR='../../map/hall1/')
    navh.config({'set_des_pos':85})
    navh.replanall()
    refnode_id = 65
    refimg_id = 146
    R_b_in_ref = np.array([[ 1., -0, 0], \
                           [ 0,  1. , 0 ], \
                           [ 0.0,  0.0,  1]])
    t_b_in_ref = np.array([0., -0., -0.3])
    refodom = [[0., 0., 0.], np.array([[1.,0.,0.], [0,1,0], [0,0,1]])]
    curodom = [[0., 0., 0.], np.array([[1.,0.,0.], [0,1,0], [0,0,1]])]
    realdir = 0.75
    # localtarget = navh.get_local_target(refnode_id, R_b_in_ref, t_b_in_ref, refodom, curodom, realdir)
    localtarget = navh.get_local_target_noimu(refnode_id, refimg_id, R_b_in_ref, t_b_in_ref, refodom, curodom)
    print(localtarget)

if __name__ == '__main__':
    test()