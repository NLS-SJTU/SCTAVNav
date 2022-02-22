# coding=utf-8

import os
import time
import cv2
import json
import copy
import numpy as np

from myzmq.jpeg_compress import *

class log_handler():
    def __init__(self, log_path='../log/'):
        self.log_path = log_path
        self.sub_folder = ''
        self.start_loging = False
        self.start_loging_dict = False
        self.log_file = None
        self.log_dict = None
        print('[LOG]init done')
        print('[LOG]log format: \"Y-m-d_h-m-s imgname reallng reallat action est_pos\". ')

    def create_folder(self, file_name = None):
        if(file_name == None):
            self.sub_folder = self.strtimenow()
        else:
            self.sub_folder = file_name
        if(not os.path.isdir(self.log_path + self.sub_folder)):
            os.makedirs(self.log_path + self.sub_folder)

    def start_new_log_dict(self, file_name = None):
        if(self.start_loging_dict):
            print('[LOG]already logging dict, now we end your log and start a new log.')
            self.end_log()
        self.start_loging_dict = True
        # self.create_folder(file_name)
        log_file = self.log_path + file_name + '_logdatadict.py'
        self.log_dict = open(log_file, 'w')
        self.log_dict.write('logdatadict=[\n')
        print('[LOG]start new log to '+log_file)

    def end_log_dict(self):
        if(self.start_loging_dict):
            self.start_loging_dict = False
            self.log_dict.write(']')
            self.log_dict.close()
            self.log_dict = None
            print('[LOG]end log.')

    def start_new_log(self, file_name=None):
        if(self.start_loging):
            print('[LOG]already logging, now we end your log and start a new log.')
            self.end_log()
        self.start_loging = True
        self.create_folder(file_name)
        log_file = self.log_path + self.sub_folder + '/logdata.py'
        self.log_file = open(log_file, 'w')
        self.log_file.write('logdata=[\n')
        self.log_data_json = []
        print('[LOG]start new log to '+log_file)

    def end_log(self):
        if(self.start_loging):
            self.start_loging = False
            self.log_file.write(']')
            self.log_file.close()
            self.log_file = None
            fi = open(self.log_path + self.sub_folder+'/logdata.json','w')
            json.dump(self.log_data_json, fi)
            fi.close()
            print('[LOG]end log.')

    def log_data(self, loglist, name):
        if(self.log_dict == None):
            return
        self.log_dict.write('\''+name+'\':[')
        for i in range(len(loglist)-1):
            self.log_dict.write('{:.3f}, '.format(loglist[i]))        
        self.log_dict.write('{:.3f}],'.format(loglist[-1]))

    def end_line(self):
        if(self.log_dict == None):
            return
        self.log_dict.write('},\n')
        self.log_dict.flush()        

    def new_line(self):
        if(self.log_dict == None):
            return
        self.log_dict.write('{')

    def log_estpath(self, estpath):
        fi = open(self.log_path + self.sub_folder+'/estpath.json','w')
        json.dump(estpath, fi)
        fi.close()

    # imgs: [img1, img2, ...], imgs taken at one place, just use img_buffer or imgsend in UI.py
    # est_pos: [id_list, dir_list, prob_list]
    # action: string
    def log(self, imgs=None, wanted_directions=[], realdir=0.0, reallnglat=[0.0,0.0], crossing_type=-1, lasrorder='H', distance=[0.0,1.0], est_pos=np.array([[],[],[]]), action='H', msg=''):
        if(not self.start_loging):
            return
        if(imgs == None):
            print('[LOG]there is no images to log.')
            return

        onelogdata = []

        self.log_file.write('[')
        # log time and real dir lnglat
        strtime = self.strtimenow()
        strlng = format(reallnglat[0], '.6f')
        strlat = format(reallnglat[1], '.6f')
        strdir = format(realdir, '.2f')
        self.log_file.write('\"'+strtime+'\", ')
        self.log_file.write(strdir+', ')
        self.log_file.write(strlng+', ')
        self.log_file.write(strlat+', ')

        onelogdata.append(strtime)
        onelogdata.append(realdir)
        onelogdata.append(reallnglat[0])
        onelogdata.append(reallnglat[1])

        # save images
        self.log_file.write('[')
        imgnames =[]
        for i in range(len(imgs)):
            img = imgs[i]
            if(type(img) == type(None)):
                imgname = 'None'
                self.log_file.write(imgname+', ')
            elif(type(img) == type('abc')):
                imgname = img
                self.log_file.write('\"'+imgname+'\", ')
            else:
                imgname = strlng+'_'+strlat+'_'+format(realdir+wanted_directions[i], '.2f')+'.jpg'
                self.log_file.write('\"'+imgname+'\", ')
                cv2.imwrite(self.log_path + self.sub_folder + '/' + imgname, cv2.cvtColor(jpg_to_img_rgb(img), cv2.COLOR_BGR2RGB))
            imgnames.append(imgname)
        self.log_file.write('], ')

        onelogdata.append(imgnames)

        self.log_file.write(''+str(crossing_type)+',')

        onelogdata.append(crossing_type)

        self.log_file.write('\"'+lasrorder+'\",')

        onelogdata.append(lasrorder)

        # distance
        strdist = format(distance[0], '.2f')
        strstd = format(distance[1], '.2f')
        self.log_file.write('['+strdist+', '+strstd+'], ')

        onelogdata.append(copy.deepcopy(distance))

        self.log_file.write('\"'+action+'\",')

        onelogdata.append(action)

        # save est pos and action
        est_pos_name = strlng+'_'+strlat+'_'+strdir+'.npy'
        np.save(self.log_path + self.sub_folder + '/' + est_pos_name, est_pos)
        self.log_file.write(' \"'+est_pos_name+'\", ')

        onelogdata.append(est_pos_name)
        
        self.log_file.write('\"'+msg+'\", ')

        onelogdata.append(msg)

        self.log_file.write('],\n')

        self.log_data_json.append(onelogdata)

    def strtimenow(self):
        return time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
