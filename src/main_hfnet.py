import cv2, os
import numpy as np
from pathlib import Path
import time
import json
import heapq

#from hfnet.settings import EXPER_PATH
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
tf.contrib.resampler

from utils.global_cfg import MAP_DIR, EXPER_PATH
from myzmq.zmq_comm import *
from myzmq.zmq_cfg import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

with open(MAP_DIR+'img_data_list.json') as f:
    VPR_data_full = json.load(f)
VPR_data_full = np.array(VPR_data_full)


class HFNet:
    def __init__(self, model_path, outputs):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.image_ph = tf.placeholder(tf.float32, shape=(None, None, 3))

        net_input = tf.image.rgb_to_grayscale(self.image_ph[None])
        tf.saved_model.loader.load(
            self.session, [tag_constants.SERVING], str(model_path),
            clear_devices=True,
            input_map={'image:0': net_input})

        graph = tf.get_default_graph()
        self.outputs = {n: graph.get_tensor_by_name(n+':0')[0] for n in outputs}
        self.nms_radius_op = graph.get_tensor_by_name('pred/simple_nms/radius:0')
        self.num_keypoints_op = graph.get_tensor_by_name('pred/top_k_keypoints/k:0')
        
    def inference(self, image, nms_radius=4, num_keypoints=500):
        inputs = {
            self.image_ph: image[..., ::-1].astype(np.float),
            self.nms_radius_op: nms_radius,
            self.num_keypoints_op: num_keypoints,
        }
        return self.session.run(self.outputs, feed_dict=inputs)

def match_with_ratio_test(desc1, desc2, thresh):
    dist = compute_distance(desc1, desc2)
    nearest = np.argpartition(dist, 2, axis=-1)[:, :2]
    dist_nearest = np.take_along_axis(dist, nearest, axis=-1)
    valid_mask = dist_nearest[:, 0] <= (thresh**2)*dist_nearest[:, 1]
    matches = np.stack([np.where(valid_mask)[0], nearest[valid_mask][:, 0]], 1)
    return matches

def compute_distance(desc1, desc2):
    # For normalized descriptors, computing the distance is a simple matrix multiplication.
    return 2 * (1 - desc1 @ desc2.T)

def compute_prob(desc1, desc2):
    return desc1 @ desc2.T

def test():    
    K = np.array([[699.6599731445312, 0.0, 633.25], [0.0, 699.6599731445312, 370.7355041503906],[0.0, 0.0, 1.0]])
    img1=cv2.imread('0.jpg')
    img2=cv2.imread('1.jpg')
    #K = np.array([[185.7, 0.0, 320.5], [0.0, 185.7, 240.5], [0.0, 0.0, 1.0]])
    outputs = ['global_descriptor', 'keypoints', 'local_descriptors']
    model_path = Path(EXPER_PATH, 'saved_models/hfnet')
    hfnet = HFNet(model_path, outputs)
    res1 = hfnet.inference(img1)
    res2 = hfnet.inference(img2)
    print(type(res1['local_descriptors']), type(res1['keypoints']), type(res1['global_descriptor']))
    print(res1['local_descriptors'].shape, res1['keypoints'].shape, res1['global_descriptor'].shape)
    matches = match_with_ratio_test(res1['local_descriptors'], res2['local_descriptors'], 0.8)
    kp1 = []
    kp2 = []
    pts1 = []
    pts2 = []
    cvmatches = []
    for i in range(len(matches)):
        cvmatches.append(cv2.DMatch(i, i, 1))
        kp1.append(cv2.KeyPoint(res1['keypoints'][matches[i][0]][0], res1['keypoints'][matches[i][0]][1], 1))
        kp2.append(cv2.KeyPoint(res2['keypoints'][matches[i][1]][0], res2['keypoints'][matches[i][1]][1], 1))
        pts1.append((res1['keypoints'][matches[i][0]][0], res1['keypoints'][matches[i][0]][1]))
        pts2.append((res2['keypoints'][matches[i][1]][0], res2['keypoints'][matches[i][1]][1]))

    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    matchimg = cv2.drawMatches(img1, kp1, img2, kp2, cvmatches, None, flags=2)
    cv2.imwrite('matches.jpg', matchimg)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.5, prob=0.95)
    pts, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2)
    inmatches = []
    print('essetial match number: ', len(mask))
    print('R',R, 't', t)
    
    for i in range(len(mask)):
        if(mask[i] > 0.5):
            inmatches.append(cvmatches[i])
    matchimg = cv2.drawMatches(img1, kp1, img2, kp2, inmatches, None, flags=2)
    cv2.imwrite('essmatches.jpg', matchimg)

def produce_all_vecs(): 
    imgpath = MAP_DIR+'images/'
    vecs = []
    keypoints = []
    localvecs = []
    outputs = ['global_descriptor', 'keypoints', 'local_descriptors']
    model_path = Path(EXPER_PATH, 'saved_models/hfnet')
    hfnet = HFNet(model_path, outputs)
    for i in range(302):
        imgname = imgpath + str(i).zfill(6) + '.jpg'
        print(imgname)
        img = cv2.imread(imgname)
        res = hfnet.inference(img)
        vecs.append(res['global_descriptor'])
        keypoints.append(res['keypoints'])
        localvecs.append(res['local_descriptors'])
    vecs = np.array(vecs)
    keypoints = np.array(keypoints)
    localvecs = np.array(localvecs)
    print('global vector shape:', vecs.shape, 'keypoints shape:', keypoints.shape, 'local vector shape:', localvecs.shape)
    np.save(imgpath+'globalvecs.npy', vecs)
    #np.save(imgpath+'keypoints.npy', keypoints)
    #np.save(imgpath+'localvecs.npy', localvecs)

class hfnet_handler(zmq_comm_svr_c):
    def __init__(self):
        outputs = ['global_descriptor', 'keypoints', 'local_descriptors']
        model_path = Path(EXPER_PATH, 'saved_models/hfnet')
        self.hfnet = HFNet(model_path, outputs)
        self.image = None
        self.globalvectors = np.load(MAP_DIR+'globalvecs.npy')
        self.Pos_num = 1
        self.initzmq()
        print('load {} vectors.'.format(len(VPR_data_full)))

    def initzmq(self):
        zmq_comm_svr_c.__init__(self, name=name_single_location, ip=ip_single_location, port=port_single_location)

    def compute_all_dists(self):
        res = self.hfnet.inference(self.image)
        return compute_distance(res['global_descriptor'], self.globalvectors)

    def descriptor(self):
        self.res = self.hfnet.inference(self.image)
        return self.res['global_descriptor'].reshape(-1)

    def get_top_similar_info(self):
        self.descriptor()
        allprobs = compute_prob(self.res['global_descriptor'], self.globalvectors)#.tolist()
        # the dirfactor is to remove images which are in opposite direction
        dirfactor = abs(self.realdir - VPR_data_full[:,-1])
        dirfactor = np.where(dirfactor > 1.57, 0.1, 1.0)
        allprobs = (allprobs * dirfactor).tolist()
        if(self.Pos_num <= 0 or self.Pos_num > len(allprobs)):
            self.Pos_num = len(allprobs)
        print('ask for top {} res.'.format(self.Pos_num))
        prob = heapq.nlargest(self.Pos_num, allprobs)
        self.index = list(map(allprobs.index, prob))
        pos_Lng = []
        pos_Lat = []
        pos_Direction = []
        ret = []
        for i in range(len(self.index)):
            ind = self.index[i]
            ret.append([ [VPR_data_full[ind][1], VPR_data_full[ind][2], VPR_data_full[ind][-1], VPR_data_full[ind][-2]], float(prob[i]) ])

        return ret

    def get_result(self, param = None):
        res = None
        s = time.time()
        if('alldist' in param):
            print('ask for all distance')
            self.image = param['alldist']["image"]
            self.image = np.frombuffer(self.image,dtype=np.uint8)
            self.image = jpg_to_img_rgb(self.image)
            res = self.compute_all_dists()
        elif('vector' in param):
            print('ask for global feature vector')
            self.image = param['vector']
            self.image = np.frombuffer(self.image,dtype=np.uint8)
            self.image = jpg_to_img_rgb(self.image)
            res = self.descriptor()
        elif('allres' in param):
            print('ask for global feature, local feature and keypoint')
            self.image = param['allres']
            self.image = np.frombuffer(self.image,dtype=np.uint8)
            self.image = jpg_to_img_rgb(self.image)
            self.descriptor()
            res = self.res
        print('total time for get result:', time.time() - s)
        return res

    def query(self, param = None):
        s = time.time()
        self.image = param["image"]
        self.image = np.frombuffer(self.image,dtype=np.uint8)
        self.image = jpg_to_img_rgb(self.image)
        self.realdir = param['realdir']
        self.Pos_num = int(param["retNum"] * len(self.globalvectors))
        ret = self.get_top_similar_info()
        res = {'maxprobimgid':self.index[0], 'res':ret, 'keypoints':self.res['keypoints'], 'local_descriptors':self.res['local_descriptors']}
        print('probs', ret, 'maxprobimg', self.index[0])
        print('total time for query:', time.time() - s)
        return res


def zmq_main():
    hfh = hfnet_handler()
    hfh.start()
    print('[hfnet]running!')
    while(True):
        time.sleep(1)

####################
# main
####################
if __name__ == '__main__':
    #test()
    #produce_all_vecs()
    zmq_main()
