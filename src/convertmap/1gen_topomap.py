# coding=utf-8

import numpy as np
import json
import math

ID_DIGIT = 6

class Image:
    def __init__(self):
        self.img_id = None
        self.last_img_id = None
        self.next_img_id = None
        self.state_id = None
        self.relative_pos = None
        self.direction = None
        self.direction_comp = [0.,0.,0.]
        self.loop_img_id = '-1'
        self.loop_pos = [0.,0.,0.]

    def load_data(self, img_id, data):
        self.img_id = img_id
        self.last_img_id = data[1]
        self.next_img_id = data[2]
        self.state_id = data[3]
        self.relative_pos = data[4]
        self.direction = data[5]
        if(len(data) >= 7):
            self.direction_comp = data[6]
        if(len(data) == 9):
            self.loop_img_id = data[7]
            self.loop_pos = data[8]

class State:
    def __init__(self):
        self.state_id = None
        self.images = []
        self.node_id = None
        self.to_states = []
        self.from_states = []

    def load_data(self, state_id, data):
        self.state_id = state_id
        self.images = data['images']
        self.to_states = data['tostate']
        self.node_id = data['node']
        self.from_states = data['fromstate']


class Node:
    def __init__(self):
        self.node_id = None
        self.states = []
        self.neighbornodes = []

    def load_data(self, node_id, data):
        self.node_id = node_id
        self.states = data['states']
        self.neighbornodes = data['neighbornodes']


def read_json_file(file_path):
    #, mode='r', encoding='utf-8' , encoding='utf-8'
    with open(file_path) as f:
        load_dict = json.load(f)
    return load_dict


def dict_to_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_data_into_dict(data, classType):
    res = {}
    if isinstance(data, list):
        tmp = {data[i][0]: data[i] for i in range(len(data))}
        data = tmp
    for key, value in data.items():
        tmp = classType()
        tmp.load_data(key, value)
        res[key] = tmp
    return res


class Map:
    def __init__(self):
        state_dict_path = 'input/state_dict.json'
        node_dict_path = 'input/node_dict.json'
        data_list_path = 'input/data_list.json'
        states_data = read_json_file(state_dict_path)
        nodes_data = read_json_file(node_dict_path)
        detail_data = read_json_file(data_list_path)
        self.images = load_data_into_dict(detail_data, Image)
        self.states = load_data_into_dict(states_data, State)
        self.nodes = load_data_into_dict(nodes_data, Node)

    def dis_between_nodes(self, node1, node2):
        if(node2 == 'unknow'):
            return 100000000.0
        node1_states = self.nodes[node1].states
        node2_states = self.nodes[node2].states
        is_connected = False
        for state1 in node1_states:
            for state2 in node2_states:
                if state1 in self.states[state2].to_states or state1 in self.states[state2].from_states:
                    is_connected = True
                    sc1 = state1
                    sc2 = state2
                    break
        dist = -1
        if is_connected:
            for img1 in self.states[sc1].images:
                for img2 in self.states[sc2].images:
                    if img1 == self.images[img2].last_img_id:
                        dist = np.linalg.norm(self.images[img2].relative_pos)
                        break
                    elif img1 == self.images[img2].next_img_id:
                        dist = np.linalg.norm(self.images[img1].relative_pos)
                if dist >= 0:
                    break
        return dist

    def angle_between_nodes(self, node1, node2):
        node1_states = self.nodes[node1].states
        node2_states = self.nodes[node2].states
        is_connected = False
        for state1 in node1_states:
            for state2 in node2_states:
                if state1 in self.states[state2].from_states:
                    is_connected = True
                    sc1 = state1
                    sc2 = state2
                    break
                # elif state1 in self.states[state2].to_states:
                #     is_connected = True
                #     sc1 = state1
                #     sc2 = state2
                #     break
        angle = None
        if is_connected:
            for img1 in self.states[sc1].images:
                for img2 in self.states[sc2].images:
                    if img1 == self.images[img2].last_img_id:
                        # to where
                        angle = math.atan2(self.images[img2].relative_pos[1], self.images[img2].relative_pos[0]) + self.images[img2].direction_comp[2] # + np.pi
                        break
                    # elif(img2 == self.images[img1].last_img_id ):
                    #     # from where
                    #     angle = math.atan2(self.images[img1].relative_pos[1], self.images[img1].relative_pos[0]) + np.pi
                    #     break
                if angle is not None:
                    break
            if(angle == None):
                return angle
            if(angle > math.pi):
                angle = angle - math.pi*2
            elif(angle < -math.pi):
                angle = angle + math.pi*2
        return angle

    def get_all_node_map_range(self):
        res = {}  
        res1 = {} # record other information
        for key, value in self.nodes.items():
            res[key] = {}
            res1[key] = {}
            neigs = value.neighbornodes
            neigs_angle = []
            to_neigs = []
            neigs_angle_from = []
            from_neigs = []
            # False: deadend; True: road
            road_or_deadend = (len(neigs) == 2)
            for item in neigs:
                angle = self.angle_between_nodes(key, item)
                # for non-directed graph
                # if angle == None:
                #     angle = self.angle_between_nodes(item, key) - np.pi
                if(not angle == None):
                    neigs_angle.append(angle)
                    to_neigs.append(item)
                    # neigs_angle1.append(angle)
                    # to_neigs1.append(item)
                else:
                    angle = self.angle_between_nodes(item, key)
                    if(angle > 0):
                        angle -= np.pi
                    else:
                        angle += np.pi
                    neigs_angle_from.append(angle)
                    from_neigs.append(item)
                    if(road_or_deadend):
                        neigs_angle.append(angle)
                        to_neigs.append(item)                        
            res1[key]['fromnode'] = from_neigs
            # road node, avoid unknow to another direction
            # if(len(neigs_angle) == 1):
            # for item in neigs:
            #     angle = self.angle_between_nodes(item, key)
            #     if(not angle == None):
            #         if(angle > 0):
            #             angle -= np.pi
            #         else:
            #             angle += np.pi
            #         neigs_angle1.append(angle)
            #         to_neigs1.append(item)
            # convert range to 0 ~ 2pi.....  % (2 * np.pi)
            sort_neigs_angle = sorted(neigs_angle)

            # tmp_list = []
            # if(int(key) == 31):
            #     print(neigs_angle, to_neigs, sort_neigs_angle)
            if len(sort_neigs_angle) == 1:
                # v1 = sort_neigs_angle[0] - np.pi / 2
                # v2 = sort_neigs_angle[0] + np.pi / 2
                # tmp_list.append(v1)
                # tmp_list.append(v2)
                # tmp_list.append(v1)
                to_neigs.append('unknow')
                if(sort_neigs_angle[0] > 0):
                    unknowdir = sort_neigs_angle[0] - np.pi
                else:
                    unknowdir = sort_neigs_angle[0] + np.pi
                sort_neigs_angle.append(unknowdir)
                neigs_angle.append(unknowdir)
            # else:
            #     angle_tmp = sorted(sort_neigs_angle)
            #     angle_tmp.insert(0, angle_tmp[-1] - 2 * np.pi)
            #     angle_tmp.append(angle_tmp[1] - 2 * np.pi)
            #     for i in range(len(angle_tmp) - 1):
            #         mid_value = (angle_tmp[i] + angle_tmp[i + 1]) / 2
            #         if mid_value > np.pi: mid_value -= 2 * np.pi
            #         tmp_list.append(mid_value)
                # todo
                # how to deal with the condition.

            # print('to_neigs:', to_neigs)
            # print('neigs:', neigs_angle, sort_neigs_angle)
            # print("middle angle:", tmp_list)
            # for item in neigs:
            for i in range(len(to_neigs)):
                item = to_neigs[i]
                dist = self.dis_between_nodes(key, item)
                angle = neigs_angle[i]
                # angle = self.angle_between_nodes(key, item)
                # if angle == None:
                #     angle = self.angle_between_nodes(item, key) - np.pi
                # angle = angle % (2 * np.pi)
                # print('angle:', angle)
                # index = sort_neigs_angle.index(angle)
                # print("index:", index)
                # range_1 = tmp_list[index]
                # range_2 = tmp_list[index + 1]
                # print('range:[{}, {}]'.format(range_1, range_2))
                # print('-' * 100)
                res[key][item] = {
                    "dir": angle,
                    'dist': dist,
                    # 'range': [range_1, range_2]
                }

            # deal with dead end node
            if(len(res[key]) == 0):
                item = 'unknow'
                dist = 100000000.0
                state = value.states[0]
                image = self.states[state].images[0]
                angle = self.images[image].direction[2] + self.images[image].direction_comp[2]
                res[key][item] = {
                    "dir": angle,
                    'dist': dist,
                }

        res = {key: res[key] for key in sorted(res.keys())}
        # print(res)

        # compute node data_list
        res2 = []
        maxnodeid = int(sorted(res.keys())[-1])
        # print(sorted(res.keys()))
        for i in range(maxnodeid+1):
            key = str(i).zfill(ID_DIGIT)
            if(key in res.keys()):
                res2.append([key, None, None, key])
            else:
                res2.append([None, None, None, None]) 

        for i in range(len(self.images.keys())):
            key = str(i).zfill(ID_DIGIT)
            value = self.images[key]
            state_id = value.state_id  
            node_id = self.states[state_id].node_id 
            if(int(value.last_img_id) < 0):
                res2[int(node_id)][1] = 0
                res2[int(node_id)][2] = 0
                continue   
            last_state_id = self.images[value.last_img_id].state_id  
            last_node_id = self.states[last_state_id].node_id 
            if(res2[int(node_id)][1] == None):
                comprelapos_x = math.cos(value.direction_comp[2]) * value.relative_pos[0] - math.sin(value.direction_comp[2]) * value.relative_pos[1]
                comprelapos_y = math.cos(value.direction_comp[2]) * value.relative_pos[1] + math.sin(value.direction_comp[2]) * value.relative_pos[0]
                res2[int(node_id)][1] = res2[int(last_node_id)][1] + comprelapos_x
                res2[int(node_id)][2] = res2[int(last_node_id)][2] + comprelapos_y

        return res, res1, res2

    def get_map_states(self):
        res = {}
        for key, value in self.nodes.items():
            res[key] = {}
            neigs = value.neighbornodes
            for item in neigs:
                key_name = '{}_{}'.format(key, item)
                # print(key_name)

    def get_data_list(self):
        res = []
        allkeys = self.images.keys()
        allkeys = sorted(allkeys)
        # for key, value in self.images.items():
        for key in allkeys:
            value = self.images[key]
            state_id = value.state_id
            node_id = self.states[state_id].node_id
            direction = value.direction[2] + value.direction_comp[2]
            tmp = ['{}.jpg'.format(key), None, None, int(node_id), direction]
            res.append(tmp)
        return res


if __name__ == '__main__':
    print('convert topomap')
    m = Map()
    # generate map range
    map_range_data, res1, res2 = m.get_all_node_map_range()
    dict_to_json_file(map_range_data, 'input/allnodemap.json')
    dict_to_json_file(res1, 'output/nodemap_info.json')
    dict_to_json_file(res2, 'output/data_list.json')
    # generate image list
    img_list = m.get_data_list()
    dict_to_json_file(img_list, 'output/img_data_list.json')

    # gen_state(map_range_data)
    
