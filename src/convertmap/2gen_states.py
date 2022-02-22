
import os, sys
sys.path.append('../')
import json
import math
import numpy as np
import copy

from mapping.map_handler import *

ID_DIGIT = 6

def gen_state(detail_node_map):
    states = {}
    maph = map_handler(True, detail_node_map)

    id_to_prob = -1
    motions = ['F', 'L', 'R']
    # generate action to next state
    for node in detail_node_map.keys():
        for neighbor in detail_node_map[node].keys():
            # id_to_prob is to get the prob in the vactor
            state = node+'_'+neighbor
            states[node+'_'+neighbor] = {}
            id_to_prob += 1
            states[node+'_'+neighbor]['id_to_prob'] = id_to_prob

            direction = detail_node_map[node][neighbor]['dir']
            nextneighor = maph.get_neighbor_node(neighbor, direction, 'F')
            if(nextneighor == None):
                states[node+'_'+neighbor]['F'] = node+'_'+neighbor
            else:
                states[node+'_'+neighbor]['F'] = neighbor+'_'+nextneighor
            # forward with prob
            states[state]['FwithP'] = maph.cal_forward_probs(state)

            nextneighor = maph.get_neighbor_node(node, direction, 'L')
            if(nextneighor == ''):
                states[node+'_'+neighbor]['L'] = node+'_'+neighbor
            else:
                states[node+'_'+neighbor]['L'] = node+'_'+nextneighor

            nextneighor = maph.get_neighbor_node(node, direction, 'R')
            if(nextneighor == ''):
                states[node+'_'+neighbor]['R'] = node+'_'+neighbor
            else:
                states[node+'_'+neighbor]['R'] = node+'_'+nextneighor

            nextneighor = maph.get_neighbor_node(node, direction, 'TB')
            if(nextneighor == ''):
                states[node+'_'+neighbor]['TB'] = node+'_'+neighbor
            else:
                states[node+'_'+neighbor]['TB'] = node+'_'+nextneighor

    # generate to which crossing from current state
    for state in states:
        sta = state
        # keep going forward to next crossing
        cnt = 0
        while(True):
            cnt += 1
            if(cnt > 50):
                print('state '+state+' cannot reach a crossing, maybe the graph is broken, fail to build map.')
                return
            id_str = sta[:ID_DIGIT]
            bigturn = False
            if(len(detail_node_map[id_str].keys()) == 2):
                dirs = []
                for nei in detail_node_map[id_str].keys():
                    dirs.append(detail_node_map[id_str][nei]['dir'])
                deldir = abs(dirs[0] - dirs[1])
                while(deldir > math.pi):
                    deldir = 2*math.pi - deldir
                if(deldir < 2.5):
                    bigturn = True
            if(len(detail_node_map[id_str].keys()) > 2 or sta[-1] == 'w' or bigturn):
                # to crossing or to dead end
                states[state]['to_crossing_state'] = sta
                break
            else:
                sta = states[sta]['F']

    # find nearby crossing state
    for state in states:
        id_str = state[:ID_DIGIT]
        state_list = []
        if(states[state]['to_crossing_state'] == state):
            # print('crossing state '+state)
            # start from a crossing
            states[state]['nearby_crossing_state'] = state
            states[state]['nearby_crossing_dist'] = 0.0
            next_state = state
            shouldbreak = False
            nextidstr = next_state[-ID_DIGIT:]
            if(nextidstr[-1]=='w'):
                nextidstr = 'unknow'
            forward_dist_list = [detail_node_map[next_state[:ID_DIGIT]][nextidstr]['dist']]
            # keep going forward to next crossing
            while(not shouldbreak):
                last_state = next_state
                next_state = states[next_state]['F']
                id_next_str = next_state[:ID_DIGIT]
                # if(len(detail_node_map[id_next_str].keys()) > 2 or next_state[-1] == 'w' or ('unknow' in detail_node_map[id_next_str].keys()) or 'nearby_crossing_state' in states[next_state]):
                if(states[next_state]['to_crossing_state'] == next_state):
                    # to a crossing
                    # half to start state, half to end state
                    for i in range(len(state_list)):
                        sta = state_list[i]
                        if(i <= len(state_list)/2):
                            states[sta]['nearby_crossing_state'] = state
                            states[sta]['nearby_crossing_dist'] = forward_dist_list[i]
                        else:
                            states[sta]['nearby_crossing_state'] = next_state
                            states[sta]['nearby_crossing_dist'] = forward_dist_list[-1] - forward_dist_list[i]
                    shouldbreak = True
                    # print('to a crossing state '+next_state, forward_dist_list)
                    # print(len(detail_node_map[id_next_str].keys()) > 2, next_state[-1] == 'w', ('unknow' in detail_node_map[id_next_str].keys()), 'nearby_crossing_state' in states[sta], sta)
                else:
                    state_list.append(next_state)
                    nextidstr = next_state[-ID_DIGIT:]
                    if(nextidstr[-1]=='w'):
                        nextidstr = 'unknow'
                    forward_dist_list.append(forward_dist_list[-1]+detail_node_map[next_state[:ID_DIGIT]][nextidstr]['dist'])
                    # print(next_state, forward_dist_list[-1])

    id_state_vec = ['']*(id_to_prob+1)
    fmap = open('output/map_states.py', 'w')
    fmap.write('map_states={')
    for node in states:
        node_str = "\""+node+"\":"
        jsObj = json.dumps(states[node])
        fmap.writelines(node_str+jsObj+",\n")
        id_state_vec[states[node]['id_to_prob']] = node
    fmap.writelines("}")
    fmap.close()

    savepath = 'output/id_to_states.py'
    id_to_state = open(savepath, 'w')
    id_to_state.write('id_to_states=[')
    for node in id_state_vec:
        id_to_state.write('\"'+node+'\",\n')
    id_to_state.write(']')
    id_to_state.close()
    # print('generate states from range map done. save to \"'+savepath+'\"')

def gen_range_map(afterallnodemap):
    # print('generate range map')
    for id in afterallnodemap:
        dir_list = []
        connect_id_list = []
        for connect_id in afterallnodemap[id]:
            dir_list.append(afterallnodemap[id][connect_id]['dir'])
            connect_id_list.append(connect_id)
        new_dir_list=copy.deepcopy(dir_list)
        new_dir_list.sort()
        divide_line=[]
        for n in range(len(new_dir_list)):
            if(n==0):
                tmp = (new_dir_list[n-1]+new_dir_list[n])/2+math.pi
                if(tmp>math.pi):
                    tmp = tmp -2*math.pi
                elif(tmp<-math.pi):
                    tmp = tmp + 2*math.pi
                divide_line.append(tmp)
            else:
                divide_line.append((new_dir_list[n-1]+new_dir_list[n])/2)
        divide_line.sort()
        dir_ranges = []
#         print(id)
#         print(dir_list)
#         print(divide_line)
#         print('----------------------')
        for n in range(len(dir_list)):
            idx = -10
            for m in range(len(divide_line)):
                if(dir_list[n]<divide_line[m]):
                    idx = m
                    break
            if(idx==-10):
                idx = 0
            range_tmp = [divide_line[idx-1],divide_line[idx]]
            dir_ranges.append(range_tmp)
#         print(dir_ranges)
        for n in range(len(dir_ranges)):
            afterallnodemap[id][connect_id_list[n]]['range']=dir_ranges[n]
    #     print(detail_node_map[id])
    savepath = 'output/allnodemap_range.py'
    fmap = open(savepath, 'w')
    fmap.write('detail_node_map={')
    for node in afterallnodemap:
        node_str = "\'"+node+"\':"
        jsObj = json.dumps(afterallnodemap[node])
        fmap.writelines(node_str+jsObj+",\n")
    fmap.writelines("}")
    fmap.close()
    # print('save to \"'+savepath+'\"')

    return afterallnodemap

if __name__ == '__main__':
    print('convert states')
    with open('input/allnodemap.json') as f:
        detail_node_map = json.load(f)
    node_map_range = gen_range_map(detail_node_map)
    gen_state(node_map_range)