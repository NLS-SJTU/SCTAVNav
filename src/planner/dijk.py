# coding=utf-8

import numpy as np
import copy,time
import math
from utils.global_cfg import ID_DIGIT

# action--hold(h), forward(f), turnleft(l), turnright(r)

action_list = ['H','F','L','R']

class djik_c:
    def __init__(self, map_handler):
        self.map_nodes = {}
        self.des_id = 0
        self.map_handler = map_handler
        topomap = self.map_handler.detail_node_map
        if(type(self.map_handler.nodemap_info) == type(None)):
            for stri in topomap.keys():
                #one_node = {'q':10000000, 'neibour':topomap[i]['neibour'], 'dist':topomap[i]['dist'], 'toWhere':-1, 'dealed':False}
                one_node = {'q':10000000, 'neibour':topomap[stri], 'toWhere':'-1', 'dealed':False}
                self.map_nodes[stri] = copy.deepcopy(one_node)
        else:
            for stri in topomap.keys():
                allneibors = copy.deepcopy(topomap[stri])
                for fromneibor in self.map_handler.nodemap_info[stri]['fromnode']:
                    if(not fromneibor in allneibors.keys()):
                        dist = topomap[fromneibor][stri]['dist']
                        direction = topomap[fromneibor][stri]['dir']
                        if(direction > 0):
                            direction -= math.pi
                        else:
                            direction += math.pi
                        allneibors[fromneibor] = {'range':[direction-0.1, direction+0.1], 'dist':dist, 'dir':direction}
                one_node = {'q':10000000, 'neibour':allneibors, 'toWhere':'-1', 'dealed':False}
                self.map_nodes[stri] = copy.deepcopy(one_node)
        self.reset()
        print('[dijk]init map ok')

    def reset(self):
        self.cnt_hold = 2
        for stri in self.map_nodes.keys():#range(len(self.map_nodes)):
            self.map_nodes[stri]['dealed'] = False
            self.map_nodes[stri]['q'] = 10000000

    # get best action
    # input - pos_est=[[id,dir,prob], [...]]
    #
    def get_action(self, pos_est):
        if(len(pos_est) == 0):
            return action_list[1], action_list[1] # return forward
        score = np.array([0.0, 0.0, 0.0, 0.0])
        nextascore = np.array([0.0, 0.0, 0.0, 0.0])
        if(pos_est[0][0] == self.des_id ):
            if(len(pos_est)==1):
                print('prob of destination node is {}%, there is only one est position, I think I reach destination.'.format(100.0*pos_est[0][2]))
                return 'H', 'H'
            elif(pos_est[0][2]/pos_est[1][2] > 2):
                print('prob of destination node is {}%, prob of second node is {}%, I think I reach destination.'.format(100.0*pos_est[0][2], 100.0*pos_est[1][2]))
                return 'H', 'H'
            else:
                topprobs = pos_est[0][2]
                secondprobs = 0.001
                compid = 1
                for i in range(1, len(pos_est)):
                    if(self.map_handler.is_neibor(pos_est[0][0], pos_est[i][0])):
                        topprobs += pos_est[i][2]
                        compid += 1
                    else:
                        secondprobs = pos_est[i][2]
                        break
                if(compid>=len(pos_est) or topprobs/secondprobs > 2):
                    print('prob of top position is {}%, prob of second position is {}%, I think I reach destination.'.format(100.0*topprobs, 100.0*secondprobs))
                    return 'H', 'H'
        for i in range(len(pos_est)):
            # current action
            req = self.get_action_q(pos_est[i]) # distance to target
            # print('node:', pos_est[i])
            # print('q:',req)
            # hold action will no longer consider
            for j in range(4):
                # prob/(1+dist)
                #req[j] = 1.0/(1+req[j])*pos_est[i][2]
                # prob*dist 
                req[j] = pos_est[i][2] * req[j]
                # req[j] = math.exp(-req[j]) * pos_est[i][2]
            # print('prob/(1+q):',req)
            score += req
            # next action
            nextposest = copy.deepcopy(pos_est[i])
            nextposest[0] = self.map_handler.check_next_crossing(pos_est[i][0], pos_est[i][1])
            req = self.get_action_q(nextposest) # distance to target
            # print('node:', nextposest)
            # print('q:',req)
            # hold action will no longer consider
            for j in range(4):
                # prob/(1+dist)
                #req[j] = 1.0/(1+req[j])*nextposest[2]
                # prob*dist
                req[j] = pos_est[i][2] * req[j]
                # req[j] = math.exp(-req[j]) * pos_est[i][2]
            # print('prob/(1+q):',req)
            nextascore += req
        print('all score:')
        print(score)
        # prob/(1+dist)
        #act = action_list[np.argmax(score)]
        # prob*dist
        act = action_list[np.argmin(score)]
        print('all next a score:')
        print(nextascore)
        # prob/(1+dist)
        #nextact = action_list[np.argmax(nextascore)]
        # prob*dist
        nextact = action_list[np.argmin(nextascore)]
        #if(act == action_list[0] and self.cnt_hold > 0 and pos_est[0][2] < 0.75):
        #    act = action_list[-1]  # keep turning left or right
        #    self.cnt_hold -= 1
        #else:
        #    self.cnt_hold = 2
        
        return act, nextact

    # get q of all action h,f,l,r
    # pos = [id, dir, prob]
    def get_action_q(self, pos):
        q = np.array([100000., 100000., 100000., 100000.])
        # print('node q(%d)=%f,prob=%f'%(pos[0], self.map_nodes[str(pos[0]).zfill(ID_DIGIT)]['q'], pos[2]))
        for stri in self.map_nodes[str(pos[0]).zfill(ID_DIGIT)]['neibour'].keys():
            if(stri == 'unknow'):
                continue
            ai = self.get_relative_dir(pos, stri)
            # print('neibor q({})={}, action({})'.format(stri, self.map_nodes[stri]['q'], action_list[ai]))
            if(self.map_nodes[stri]['q'] < q[ai]):
                #self.map_nodes[stri]['q'] < self.map_nodes[str(pos[0]).zfill(ID_DIGIT)]['q'] and 
                q[ai] = self.map_nodes[stri]['q']
        #if(pos[0] == self.des_id):
        #    q[0] = 0
        #q = q*pos[2]
        if(q[2] > 99999):
            q[2] = q[3]
        if(q[3] > 99999):
            q[3] = q[2]
        if(q[1] > 99999):
            q[1] = self.map_nodes[str(pos[0]).zfill(ID_DIGIT)]['q']
        return q

    # get the move order from pos to nextpos
    # return 0--h, 1--f, 2--l, 3--r
    def get_relative_dir(self, pos, nextpos):
        # method with only dir
        '''
        direction = self.map_nodes[str(pos[0]).zfill(ID_DIGIT)]['neibour'][nextpos]['dir']
        deldir = direction - pos[1]
        if(deldir > math.pi):
            deldir = deldir - math.pi*2
        elif(deldir < -math.pi):
            deldir = deldir + math.pi*2

        if(deldir > math.pi/6):#0.05
            a = 2
        elif(deldir < -math.pi/6):
            a = 3
        else:
            a = 1
        '''
        # method with range
        centerdeldir = self.map_handler.cal_diff_dir(pos[1], self.map_nodes[str(pos[0]).zfill(ID_DIGIT)]['neibour'][nextpos]['dir'])
        rdir_of_next = self.map_nodes[str(pos[0]).zfill(ID_DIGIT)]['neibour'][nextpos]['range'][0] #+ math.pi/6
        ldir_of_next = self.map_nodes[str(pos[0]).zfill(ID_DIGIT)]['neibour'][nextpos]['range'][1] #- math.pi/6
        delrdir = rdir_of_next - pos[1]
        delldir = ldir_of_next - pos[1]
        if(delrdir > math.pi):
            delrdir = delrdir - math.pi*2
        elif(delrdir < -math.pi):
            delrdir = delrdir + math.pi*2
        if(delldir > math.pi):
            delldir = delldir - math.pi*2
        elif(delldir < -math.pi):
            delldir = delldir + math.pi*2

        # 3 states
        # print('find dir to {} from {}:'.format(nextpos, pos[0]))
        # print(ldir_of_next, rdir_of_next, pos[1], delldir, delrdir)
        if(abs(centerdeldir) < 0.35):
            a = 1
        elif(delldir > 0 and delrdir < 0):
            a = 1
        elif(math.fabs(delrdir) < math.fabs(delldir)):
            a = 2
        else:
            a = 3
        # print('action='+action_list[a])
        return a

    # dijkstra
    # input des_id:int
    def cal_q(self,des_id):
        self.reset()
        st = time.time()
        self.des_id = des_id
        q = {}   #np.zeros(len(self.map_nodes))
        waitqueue = []   #int
        waitqueue.append(des_id)
        self.map_nodes[str(des_id).zfill(ID_DIGIT)]['q'] = 0
        q[str(des_id).zfill(ID_DIGIT)] = 0
        self.map_nodes[str(des_id).zfill(ID_DIGIT)]['toWhere'] = des_id
        while(len(waitqueue)>0):
            id_now = str(waitqueue[-1]).zfill(ID_DIGIT)
            waitqueue.pop(-1)
            for stri in self.map_nodes[id_now]['neibour'].keys():
                if(stri == 'unknow'):
                    continue
                # if(not id_now in self.map_handler.detail_node_map[stri].keys()):
                #     continue
                id_next = stri   #self.map_nodes[id_now]['neibour'][i]
                if(id_next == self.map_nodes[id_now]['toWhere']):
                    continue
                if(not self.map_nodes[id_next]['dealed']):
                    self.map_nodes[id_next]['q'] = self.map_nodes[id_now]['q'] + self.map_nodes[id_now]['neibour'][id_next]['dist']
                    q[id_next] = self.map_nodes[id_next]['q']
                    self.map_nodes[id_next]['toWhere'] = id_now
                    self.map_nodes[id_next]['dealed'] = True
                    self.insertSortByDist(waitqueue, id_next)
                else:
                    dtmp = self.map_nodes[id_now]['q'] + self.map_nodes[id_now]['neibour'][id_next]['dist']
                    if(dtmp < self.map_nodes[id_next]['q']):
                        self.map_nodes[id_next]['q'] = dtmp
                        q[id_next] = self.map_nodes[id_next]['q']
                        self.map_nodes[id_next]['toWhere'] = id_now
                        if(id_next in waitqueue):
                            waitqueue.remove(id_next)
                            self.insertSortByDist(waitqueue, id_next)
        for stri in q.keys():
            if(len(self.map_nodes[stri]['neibour'].keys()) > 2):
                print(stri, self.map_nodes[stri]['q'])
        print('[dijk]cal q finish')
        print('time for q calculation is {}s>>>>>>>>>>>>>>>>>>'.format(time.time()-st))
        return q
    
    # queue used in dijkstra
    # large at begin, small at end
    def insertSortByDist(self, queue, insertnode_id):
        intid = int(insertnode_id)
        if(len(queue) == 0):
            queue.append(intid)
        elif(self.map_nodes[insertnode_id]['q'] > self.map_nodes[str(queue[0]).zfill(ID_DIGIT)]['q']):
            queue.insert(0, intid)
        elif(self.map_nodes[insertnode_id]['q'] < self.map_nodes[str(queue[-1]).zfill(ID_DIGIT)]['q']):
            queue.append(intid)
        else:
            start = 0
            end = len(queue)-1
            while((end - start)>1):
                half = int((start+end)/2)
                if(self.map_nodes[insertnode_id]['q'] > self.map_nodes[str(queue[half]).zfill(ID_DIGIT)]['q']):
                    end = half
                else:
                    start = half
            queue.insert(end, intid)

    # get q of node
    def get_q(self, id):
        return self.map_nodes[str(id).zfill(ID_DIGIT)]['q']
