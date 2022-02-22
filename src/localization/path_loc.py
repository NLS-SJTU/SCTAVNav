#!/usr/bin/python3
# coding=utf-8

# the crossing type is not important, just set to 0 or -1

import numpy as np
import time
import math
import copy
import pylab as plt
from scipy import stats
from utils.global_cfg import ID_DIGIT

class path_pos_c:
    def __init__(self, maph=None):
        self.maph = maph
        self.map_states = maph.map_states
        self.path = None 
        self.distance_accumulate = np.array([[0.0,1.0]*len(self.maph.id_to_states)])
        self.distance_accumulate.shape = (len(self.maph.id_to_states),2)
        self.distance_accumulate_from_crossing = np.array([0.0, 1.0])
        self.just_pass_crossing = False
    
    # crossing_type:0-road, 1-crossing, -1-unknow
    def update_path(self, prob_list, action=None, crossing_type=0, distance=[0.0,1.0], realdir=None):
        st = time.time()
        self.update_path_withaction_dist(prob_list, action, crossing_type, distance, realdir)
        print('[pathloc]time for path localization is {} seconds.'.format(time.time()-st))

    # init_state: the state from which distance start
    # state: state that currently calculating
    # prob_vec_pred: prediction of prob from p_{t-1}
    # pos_prob: record largest last state
    # init_prob: prob of init state in p_{t-1}
    # nextprob_factor: factor to deal with case that distance passes crossings
    # lastforwarddist: middle distance between last calculated state and current calculating state
    # forwarddist: distance from init state to current calculating state
    # maxforwarddist: max forward predict distance
    # distance: move distance and covariance
    def forward_prob_pred(self, init_state, state, prob_vec_pred, pos_prob, init_prob, nextprob_factor, lastforwarddist, forwarddist, minforwarddist, maxforwarddist, distance):
        if(forwarddist > maxforwarddist or nextprob_factor < 0.140001):
            # go too far or go too much crossing
            return prob_vec_pred, pos_prob

        initstate_id = self.maph.check_id_to_prob(init_state)
        state_id = self.maph.check_id_to_prob(state)
        if(state_id < 0):
            # this state does not exist
            return prob_vec_pred, pos_prob
        nextforwarddist = forwarddist + self.maph.get_distance_between_neibornodes(state[:ID_DIGIT],state[-ID_DIGIT:])
        # if(nextforwarddist < minforwarddist):
            # go too near
            # return prob_vec_pred, pos_prob
        if(state == init_state):
            # self.distance_accumulate[initstate_id] += distance
            maxforwarddist = self.distance_accumulate[initstate_id][0] + 3*self.distance_accumulate[initstate_id][1]
            if(nextforwarddist > maxforwarddist):
                # the movement is too short to next state,stay
                if(init_prob > prob_vec_pred[state_id]):
                    prob_vec_pred[state_id] = init_prob
                    pos_prob[state] = (pos_prob[state][0], init_state)
                return prob_vec_pred, pos_prob
            dist = copy.deepcopy(self.distance_accumulate[initstate_id])
            # self.distance_accumulate[initstate_id] = [0.0,1.0]
        else:
            dist = distance

        x1 = ((lastforwarddist + forwarddist) / 2 - dist[0]) / dist[1]
        x2 = ((nextforwarddist + forwarddist) / 2 - dist[0]) / dist[1]
        transprob = stats.norm.cdf(x2) - stats.norm.cdf(x1)
        probnext = init_prob * transprob * nextprob_factor
        if(probnext > prob_vec_pred[state_id]):
            prob_vec_pred[state_id] = probnext
            pos_prob[state] = (probnext, init_state)
            if(not state == init_state):
                self.distance_accumulate[state_id] = [0.0,1.0]

        # use average prob for crossing
        num_next_state = len(self.maph.map_states[state]['FwithP'])
        prob_next_states = 1.0 / num_next_state
        for next_posible_state in self.maph.map_states[state]['FwithP']:
            prob_vec_pred, pos_prob = self.forward_prob_pred(init_state, next_posible_state, prob_vec_pred, pos_prob, init_prob, nextprob_factor*prob_next_states, forwarddist, nextforwarddist, minforwarddist, maxforwarddist, dist)
            #self.maph.map_states[state]['FwithP'][next_posible_state]

        return prob_vec_pred, pos_prob

    def forward_prob_pred_for_crossing(self, init_state, state, mask_prob_crossing, nextprob_factor, lastforwarddist, forwarddist, minforwarddist, maxforwarddist):
        if(forwarddist > maxforwarddist or nextprob_factor < 0.140001):
            return mask_prob_crossing
        elif(forwarddist < minforwarddist):
            pass
        else:
            initstate_id = self.maph.check_id_to_prob(init_state)
            state_id = self.maph.check_id_to_prob(state)
            nextforwarddist = forwarddist + self.maph.get_distance_between_neibornodes(state[:ID_DIGIT],state[-ID_DIGIT:])
            dist = self.distance_accumulate_from_crossing
            x1 = ((lastforwarddist + forwarddist) / 2 - dist[0]) / dist[1]
            x2 = ((nextforwarddist + forwarddist) / 2 - dist[0]) / dist[1]
            transprob = stats.norm.cdf(x2) - stats.norm.cdf(x1)
            if(transprob > mask_prob_crossing[state_id]):
                mask_prob_crossing[state_id] = transprob

        # move to cal next state
        minFprobstate = list(self.maph.map_states[state]['FwithP'].keys())[0]
        for next_posible_state in self.maph.map_states[state]['FwithP']:
            if(self.maph.map_states[state]['FwithP'][next_posible_state] < self.maph.map_states[state]['FwithP'][minFprobstate]):
                minFprobstate = next_posible_state
        for next_posible_state in self.maph.map_states[state]['FwithP']:
            if(not next_posible_state == minFprobstate):
                self.forward_prob_pred_for_crossing(init_state, next_posible_state, mask_prob_crossing, 1.0, forwarddist, nextforwarddist, minforwarddist, maxforwarddist)

        return mask_prob_crossing

    def gen_mask_prob_with_crossing_est(self, crossing_type, distance):
        if(crossing_type == 1):
            self.just_pass_crossing = True
            self.distance_accumulate_from_crossing = [0.0, 0.0]

        mask_prob_crossing = np.array([0.01/self.maph.states_num]*self.maph.states_num)
        if(self.just_pass_crossing and self.distance_accumulate_from_crossing[0] > 10):
            maxforwarddist = self.distance_accumulate_from_crossing[0]+3*self.distance_accumulate_from_crossing[1]
            minforwarddist = self.distance_accumulate_from_crossing[0]-3*self.distance_accumulate_from_crossing[1]
            # for all crossings states
            for state in self.maph.map_states.keys():
                if(not self.maph.is_crossing_state(state)):
                    continue
                mask_prob_crossing = self.forward_prob_pred_for_crossing(state, state, mask_prob_crossing, 1.0, -10.0, 0.0, minforwarddist, maxforwarddist)
        
        return mask_prob_crossing

    # to avoid wrong clearing accumulated distance
    def init_pred_prob(self, pos_prob_last, distance):
        prob_vec_pred = np.array([0.0]*self.maph.states_num)
        pos_prob={p:(0,None) for p in self.map_states}
        self.distance_accumulate += distance
        for state in self.maph.map_states.keys():
            id_to_prob = self.maph.check_id_to_prob(state)
            dist_to_next_state = self.maph.get_distance_between_neibornodes(state[:ID_DIGIT],state[-ID_DIGIT:])
            accumulated_dist = self.distance_accumulate[id_to_prob]
            init_prob = pos_prob_last[state][0]
            if(accumulated_dist[0] < dist_to_next_state):
                prob_vec_pred[id_to_prob] = init_prob * (1 - accumulated_dist[0] / dist_to_next_state)
                pos_prob[state] = (prob_vec_pred[id_to_prob], state)
            else:
                prob_vec_pred[id_to_prob] = init_prob / 100
                pos_prob[state] = (prob_vec_pred[id_to_prob], state)
        return prob_vec_pred, pos_prob

    # (too slow)
    # update with distance[dist,sigma]
    def update_path_withaction_dist(self,prob_list,action=None,crossing_type=0,distance=0, realdir=None):
        if self.map_states is None: self.map_states=list(prob_list.keys())
        st = time.time()
        if self.path is None:
            pos_prob={p:(0,None) for p in self.map_states}
            prob_vec_ob = np.array([0.0]*self.maph.states_num)
            for state in self.maph.map_states.keys():
                id_to_prob = self.maph.check_id_to_prob(state)
                prob_vec_ob[id_to_prob] = prob_list[state]
            prob_vec_now = prob_vec_ob
            st2 = time.time()
        else:
            pos_prob_last=self.path[-1]
            # pos_prob={p:(0,None) for p in self.map_states}
            prob_vec_ob = np.array([0.0]*self.maph.states_num)
            # prob_vec_pred = np.array([0.0]*self.maph.states_num)
            prob_vec_pred, pos_prob = self.init_pred_prob(pos_prob_last, distance)
            maxforwarddist = distance[0]+3*distance[1]
            minforwarddist = distance[0]-3*distance[1]
            for state in self.maph.map_states.keys():
                id_to_prob = self.maph.check_id_to_prob(state)
                prob_vec_ob[id_to_prob] = prob_list[state]
            
                if(action=='F'):
                    nextprob_factor = 0.9
                    lastforwarddist = -10
                    forwarddist = 0
                    next_state_withmotion = state
                    next_state_id = self.maph.map_states[next_state_withmotion]['id_to_prob']
                    prob_vec_pred, pos_prob = self.forward_prob_pred(state, state, prob_vec_pred, pos_prob, pos_prob_last[state][0], 1.0, lastforwarddist, forwarddist, minforwarddist, maxforwarddist, distance)
                else:
                    # use action to predict
                    # next_state_withmotion = self.maph.check_next_state(state, action)
                    # use real dir to predict
                    next_state_withmotion = self.maph.get_state(state[:ID_DIGIT], realdir)
                    
                    next_state_id = self.maph.map_states[next_state_withmotion]['id_to_prob']
                    nextprob_factor = 0.75

                    probnext = pos_prob_last[state][0]*nextprob_factor
                    if(probnext > prob_vec_pred[next_state_id]):
                        prob_vec_pred[next_state_id] = probnext
                        pos_prob[next_state_withmotion] = (pos_prob[next_state_withmotion][0], state)

                    probnext = pos_prob_last[state][0]*(1.0-nextprob_factor)
                    if(probnext > prob_vec_pred[id_to_prob]):
                        prob_vec_pred[id_to_prob] = probnext
                        pos_prob[state] = (pos_prob[next_state_withmotion][0], state)
            st2 = time.time()
            # this factor is useless if no intersection information is given
            mask_prob_crossing = self.gen_mask_prob_with_crossing_est(crossing_type, distance)
            prob_vec_now = prob_vec_pred * prob_vec_ob * mask_prob_crossing

        st3 = time.time()
        # mask_prob_dir = self.maph.gen_prob_mask_with_realdir(realdir)
        # st4 = time.time()
        print('time for prediction of all states is {}s'.format(st2-st))
        print('time for from crossing mask is {}s'.format(st3-st2))
        # print('time for dir mask is {}s'.format(st4-st3))
        #print(realdir, mask_prob_dir)
        # prob_vec_now = prob_vec_now * mask_prob_dir
        prob_vec_now = self.maph.sum_up_state(prob_vec_now, crossing_type)
        if(crossing_type == 1):
            self.distance_accumulate = np.array([[0.0,1.0]*len(self.maph.id_to_states)])
            self.distance_accumulate.shape = (len(self.maph.id_to_states),2)
        prob_vec_now /= np.sum(prob_vec_now)

        maxprobid = np.argmax(prob_vec_now)
        print('max prob accum distance is ', self.maph.id_to_states[maxprobid], self.distance_accumulate[maxprobid])

        for state in self.maph.map_states.keys():
            pos_prob[state] = (prob_vec_now[self.maph.check_id_to_prob(state)], pos_prob[state][1])
        
        if self.path is None:
            self.path=[pos_prob]
        else:
            self.path.append(pos_prob)
        return pos_prob.copy()
                      
   
    def find_nearby(self,pos): 
        ret = list(self.pos_list[pos].keys())
        if('unknow' in ret):
            ret.remove('unknow')
        return ret
        
    def get_last_prob(self): 
        return None if self.path is None else self.path[-1]
    
    def get_last_prob_value(self):
        pos_prob=self.get_last_prob()
        return None if pos_prob is None else [pos_prob[p][0] for p in self.pos_list]
        
    def rest_prob(self): 
        if self.path is not None: 
            #self.path.clear()
            self.path=None
        self.distance_accumulate = np.array([[0.0,0.0]*len(self.maph.id_to_states)])
        self.distance_accumulate.shape = (len(self.maph.id_to_states),2)
        self.distance_accumulate_from_crossing =  np.array([0.0, 0.0])
        self.just_pass_crossing = False
        return
    
    def get_max_prob_pos(self):
        if self.path is None: return -np.inf, None, None
        
        prob, pos, pre_pos = -np.inf, None, None
        for p in self.path[-1]:
            if prob<self.path[-1][p][0]:
                pos=p
                prob,pre_pos=self.path[-1][p] 
        return pos, prob, pre_pos


    def get_pos_backtrace(self,pos=None):
        if self.path is None: return None
        
        if pos is None: pos,_,_=self.get_max_prob_pos()

        backtrace=[]
        for n in range(len(self.path)-1,-1,-1):
            pos_prob=self.path[n]
            backtrace.append((pos,pos_prob[pos][0]))
            pos=pos_prob[pos][1]    
            if pos is None: break
        
        return backtrace[::-1]
    
    def print_backtrace(self, backtrace=None, pos=None):
        if backtrace is None: backtrace=self.get_pos_backtrace(pos)
        if backtrace is None: return
        for n in range(len(backtrace)):
            pos,prob=backtrace[n]
            print(' -> ', pos, '(%.2e)'%prob)#, end=''
        print('')
    
    
    def dump_path(self):
        if self.path is None: return
        n=0
        for n in range(len(self.path)):
            print('[%d] '%n)#,end=''
            pos_prob=self.path[n]
            for pos in pos_prob:
                prob,pre_pos=pos_prob[pos]
                print(pos,'(%.2e,'%prob,pre_pos,')  ')#,end=''
            print('')


