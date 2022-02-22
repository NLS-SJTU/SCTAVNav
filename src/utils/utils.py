import time
import heapq
import cv2
import PyKDL
import numpy as np
import matplotlib.pylab as plt

MIN_NUM_KEY_POINT = 8

CV_VERSION = cv2.__version__

# cos distance
# def distance_of_features(fea1, fea2):
#     normmul = np.linalg.norm(fea1) * np.linalg.norm(fea2)
#     return np.dot(fea1, fea2) / normmul

# Euclid distance
def distance_of_features(fea1, fea2):
    return np.linalg.norm(fea1 - fea2)

# return index in the feature_list
def get_top_similar_images(imgfeature, feature_list, topN=5):
    # distance_list = np.array(feature_list) - imgfeature
    # distance_list = np.linalg.norm(distance_list, axis=1)
    # distance_list = distance_list.tolist()
    distance_list = []
    for i in range(len(feature_list)):
        distance_list.append(distance_of_features(np.array(feature_list[i]), imgfeature))

    top_dist = heapq.nsmallest(topN, distance_list)
    indexs = list(map(distance_list.index, top_dist))

    return indexs, top_dist

def get_get_top_similar_images_hffeature(imgfeature, feature_list, topN=5):
    distance_list = []
    for i in range(len(feature_list)):
        distance_list.append(compute_distance(imgfeature, feature_list[i]))

    top_dist = heapq.nsmallest(topN, distance_list)
    indexs = list(map(distance_list.index, top_dist))

    return indexs, top_dist

def fundamantal_check(img1, img2):
    orb = cv2.ORB_create()
    kp1 = orb.detect(img1)
    kp2 = orb.detect(img2)
    kp1, des1 = orb.compute(img1, kp1) 
    kp2, des2 = orb.compute(img2, kp2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    # matches = sorted(matches, key = lambda x:x.distance)
    # matchimg = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:30], None, flags=2)
    # cv2.imshow('a',matchimg)
    # cv2.waitKey(0)

    pts1 = []
    pts2 = []
    for m in matches:
        if(m.distance < 50):
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    # print(kp1)
    print('matched key points {}'.format(len(pts1)))
    if(len(pts1)<=MIN_NUM_KEY_POINT or len(pts2)<=MIN_NUM_KEY_POINT):
        # not enough matched key points
        return False

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    # print(F, mask)

    matchednumber = np.sum(mask)
    accuracy = matchednumber / len(mask)
    print('fundamental matched accuracy {} ({}/{})'.format(accuracy, matchednumber, len(mask)))

    return (accuracy>0.7 or matchednumber > MIN_NUM_KEY_POINT)

# numpy input output
def trans_from_cam_to_body(R, t):
    Rswitch = np.array([[0,-1.,0], [0,0,-1.], [1.,0,0]])
    R = np.matmul(np.matmul(Rswitch.T, R), Rswitch)
    t = np.matmul(Rswitch.T, t.reshape(-1))
    tmp = PyKDL.Rotation(R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1], R[2][2])
    RPY = tmp.GetRPY()
    print(R, t, RPY)
    if(abs(RPY[0]) > 0.5 or abs(RPY[1]) > 0.5 or np.linalg.norm(t) > 10):
        # the robot will not flip up and down or get too far from reference frame
        return False, None, None
    return True, R, t

# img1 reference, img2 query, K intrincic
def getRt(img1, img2, K):
    if(type(K) == type(None) or type(img1) == type(None) or type(img2) == type(None)):
        return False, None, None, 0, 0
    orb = cv2.ORB_create(edgeThreshold=0)
    # mask = np.zeros((720,1280),dtype=np.uint8)
    # mask[360:,:] = 255
    kp1 = orb.detect(img1)
    kp2 = orb.detect(img2)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # corner1 = cv2.goodFeaturesToTrack(img1, 500, 0.01, minDistance=30)
    # corner2 = cv2.goodFeaturesToTrack(img2, 500, 0.01, minDistance=30)
    # kp1 = [cv2.KeyPoint(corner1[i][0][0], corner1[i][0][1], 1) for i in range(corner1.shape[0])]
    # kp2 = [cv2.KeyPoint(corner2[i][0][0], corner2[i][0][1], 1) for i in range(corner2.shape[0])]
    kp1, des1 = orb.compute(img1, kp1) 
    kp2, des2 = orb.compute(img2, kp2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    # matches = sorted(matches, key = lambda x:x.distance)
    # matchimg = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:30], None, flags=2)
    # cv2.imshow('a',matchimg)
    # cv2.waitKey(0)
    # outimg1 = img1
    # outimg1 = cv2.drawKeypoints(img1, kp1, outimg1)
    # cv2.imshow('img1kp',outimg1)
    # cv2.resizeWindow('img1kp', 800, 600)
    # outimg2 = img2
    # outimg2 = cv2.drawKeypoints(img2, kp2, outimg2)
    # cv2.imshow('img2kp',outimg2)
    # cv2.resizeWindow('img2kp', 800, 600)
    # cv2.waitKey(1)

    pts1 = []
    pts2 = []
    # i = 0
    propermatches = []
    for i in range(len(matches)):
        m = matches[i]
        if(m.distance < 50):
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            propermatches.append(m)
            # i += 1
            # if(i >= 16):
            #     break
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    print('matched key points {}'.format(len(pts1)))
    if(len(pts1)<=MIN_NUM_KEY_POINT or len(pts2)<=MIN_NUM_KEY_POINT):
        # not enough matched key points
        return False, None, None, 0, 0

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.5, prob=0.95)
    matchednumber = np.sum(mask)
    accuracy = matchednumber / len(mask)
    print('essential matched accuracy {} ({}/{})'.format(accuracy, matchednumber, len(mask)))
    pts, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2)
    #
    inmatches = []
    for i in range(len(mask)):
        if(mask[i] > 0.5):
            inmatches.append(propermatches[i])
    matchimg = cv2.drawMatches(img1, kp1, img2, kp2, inmatches, None, flags=2)
    # cv2.imshow('getrt',matchimg)
    # cv2.resizeWindow('getrt', 800, 600)
    # cv2.waitKey(10)

    # turn to body frame
    Rswitch = np.array([[0,-1.,0], [0,0,-1.], [1.,0,0]])
    R_21 = np.matmul(np.matmul(Rswitch.T, R), Rswitch)
    t_21 = np.matmul(Rswitch.T, t.reshape(-1))
    tmp = PyKDL.Rotation(R_21[0][0], R_21[0][1], R_21[0][2], R_21[1][0], R_21[1][1], R_21[1][2], R_21[2][0], R_21[2][1], R_21[2][2])
    RPY = tmp.GetRPY()
    print('R, t, rpy', R_21, t_21, RPY)
    if(abs(RPY[0]) > 0.8 or abs(RPY[1]) > 0.8):
        return False, None, None, 0, 0

    return (matchednumber > MIN_NUM_KEY_POINT), R_21, t_21, matchednumber, accuracy


# return R_12, t_12, which is pos 2 in 1 frame or transform point from 2 to 1
def getRTPNP(img1, img2, img2_depth, K1, K2):
    if(type(K1) == type(None) or type(img1) == type(None) or type(img2) == type(None)):
        return False, None, None
    if(type(K2) == type(None)):
        K2 = K1
    orb = cv2.ORB_create(nfeatures=500, edgeThreshold=0)
    # mask = np.zeros((720,1280),dtype=np.uint8)
    # mask[360:,:] = 255
    kp1 = orb.detect(img1) #
    kp2 = orb.detect(img2)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # corner1 = cv2.goodFeaturesToTrack(img1, 500, 0.01, minDistance=30)
    # corner2 = cv2.goodFeaturesToTrack(img2, 500, 0.01, minDistance=30)
    # kp1 = [cv2.KeyPoint(corner1[i][0][0], corner1[i][0][1], 1) for i in range(corner1.shape[0])]
    # kp2 = [cv2.KeyPoint(corner2[i][0][0], corner2[i][0][1], 1) for i in range(corner2.shape[0])]
    # print('kpss',len(kp1))
    # outimg = img1
    # outimg = cv2.drawKeypoints(img1, kp1, outimg)
    # cv2.imshow('aaa',outimg)
    kp1, des1 = orb.compute(img1, kp1) 
    kp2, des2 = orb.compute(img2, kp2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    # matchimg = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:-1], None, flags=2)
    # cv2.imshow('match',matchimg)
    # cv2.imshow('depth', img2_depth/5)
    # cv2.waitKey(100)
    # cv2.imwrite('match.jpg', matchimg)
    # cv2.imwrite('img1.jpg', img1)
    # cv2.imwrite('img2.jpg', img2)
    # depsave = np.array(img2_depth/5*255, dtype=np.uint8)
    # cv2.imwrite('depth.png', img2_depth)
    # np.save('depth.npy', img2_depth)

    pts1 = []
    pts2 = []
    i = 0
    propermatches = []
    for m in matches:
        pt2 = np.round(kp2[m.trainIdx].pt).astype(np.int)
        if(m.distance < 50 and not np.isnan(img2_depth[pt2[1]][pt2[0]])):
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            propermatches.append(m)
            i += 1
            if(i >= 50):
                break
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2) # should use round
    print('matched key points {}'.format(len(pts1)))
    # print(pts2)
    # matchimg = cv2.drawMatches(img1, kp1, img2, kp2, propermatches, None, flags=2)
    # cv2.imshow('propermatch',matchimg)
    # cv2.waitKey(1)
    # cv2.imshow('depth', img2_depth/5)
    if(len(pts1)<=MIN_NUM_KEY_POINT or len(pts2)<=MIN_NUM_KEY_POINT):
        # not enough matched key points
        return False, None, None

    # compute 3d coordinate of key points in image 1
    # print('11111',pts2.dtype, pts2, img2_depth.shape, img2_depth[55][55])
    uv1_2 = np.ones((len(pts2), 3))
    uv1_2[:,0:2] = pts2
    pts2 = np.round(pts2).astype(np.int)
    ds_2 = img2_depth[pts2[:,1] ,pts2[:,0]].reshape(len(pts2), 1)
    xyz_2 = (np.matmul(np.linalg.inv(K2), uv1_2.T).T * ds_2)
    # print(xyz_2, pts1)
    # print('2222',xyz_2.shape, pts1.shape)

    # compute R t by pnp
    distcoeffs = None
    trueorfalse,R,t,inliers = cv2.solvePnPRansac(xyz_2, pts1, K1, distcoeffs)
    if(type(inliers) == type(None)):
        return False, None, None
    R,_ = cv2.Rodrigues(R)
    # print('3333',trueorfalse)
    # print(R)
    # print(t)
    # print('inliers number', len(inliers), inliers.reshape(-1))
    # pts1_in = pts1[inliers.reshape(-1)]
    # pts2_in = pts2[inliers.reshape(-1)]
    # matches_in = []
    # for i in range(len(inliers)):
    #     matches_in.append(propermatches[inliers[i][0]])
    # matchimg = cv2.drawMatches(img1, kp1, img2, kp2, matches_in, None, flags=2)
    # cv2.imwrite('inliermatch.jpg',matchimg)
    # cv2.imshow('inliermatch',matchimg)

    # transform to body frame
    Rswitch = np.array([[0,-1.,0], [0,0,-1.], [1.,0,0]])
    R = np.matmul(np.matmul(Rswitch.T, R), Rswitch)
    t = np.matmul(Rswitch.T, t.reshape(-1))
    tmp = PyKDL.Rotation(R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1], R[2][2])
    RPY = tmp.GetRPY()
    print(R, t, RPY)
    if(abs(RPY[0]) > 0.5 or abs(RPY[1]) > 0.5 or np.linalg.norm(t) > 10):
        # the robot will not flip up and down or get too far from reference frame
        return False, None, None

    return trueorfalse, R, t

def compute_distance(desc1, desc2):
    # For normalized descriptors, computing the distance is a simple matrix multiplication.
    return 2 * (1 - np.matmul(desc1, desc2.T))

def match_with_ratio_test(desc1, desc2, thresh=0.8):
    dist = compute_distance(desc1, desc2)
    # sort the put first two index in nearest
    nearest = np.argpartition(dist, 2, axis=-1)[:, :2]
    # dist_nearest = np.take_along_axis(dist, nearest, axis=-1)  # this function is not valid in py2
    #take the first and second similar to see if the first match is special
    dist_nearest = np.array([[dist[i][nearest[i][0]], dist[i][nearest[i][1]]] for i in range(len(nearest))])
    valid_mask = dist_nearest[:, 0] <= (thresh**2)*dist_nearest[:, 1]
    matches = np.stack([np.where(valid_mask)[0], nearest[valid_mask][:, 0]], 1)
    return matches

def pixel_norm(pts, K):
    normpts = np.zeros(pts.shape)
    normpts[:,0] = (pts[:,0] - K[0][2]) / K[0][0]
    normpts[:,1] = (pts[:,1] - K[1][2]) / K[1][1]
    return normpts

def compute_depth_stereo(kpl, desl, kpr, desr, T, Kl, Kr):
    matches = match_with_ratio_test(des1, des2, 0.8)
    mkpl = []
    mkpr = []
    for i in range(len(matches)):
        mkpl.append([kpl[matches[i][0]][0], kpl[matches[i][0]][1]])
        mkpr.append([kpr[matches[i][1]][0], kpr[matches[i][1]][1]])
    Lnormpts = pixel_norm(mkpl, Kl)
    Rnormpts = pixel_norm(mkpr, Kr)
    T1 = np.eye(4)[:3,:]
    pts4d = cv2.triangulatePoints(T1, T, Lnormpts, Rnormpts)
    return pts4d[:,2] / pts4d[:,3], mpkl, mkpr, matches

# kp1, des1: reference frame
# kp2, des2: query frame, left
# kp2r, des2r: query frame, right
# T: stereo param T
# K: intrinc mat

def getRt_hf(img1, img2, kp1, des1, kp2, des2, K1, K2):
    matches = match_with_ratio_test(des1, des2, 0.8)
    kp1 = np.asarray(kp1, dtype=np.float)
    kp2 = np.asarray(kp2, dtype=np.float)
    # cvkp1 = []
    # cvkp2 = []
    # pts1 = []
    # pts2 = []
    # cvmatches = []
    # for i in range(len(matches)):
    #     cvmatches.append(cv2.DMatch(i, i, 1))
    #     cvkp1.append(cv2.KeyPoint(kp1[matches[i][0]][0], kp1[matches[i][0]][1], 1))
    #     cvkp2.append(cv2.KeyPoint(kp2[matches[i][1]][0], kp2[matches[i][1]][1], 1))
    #     pts1.append([kp1[matches[i][0]][0], kp1[matches[i][0]][1]])
    #     pts2.append([kp2[matches[i][1]][0], kp2[matches[i][1]][1]])
    # matchimg = cv2.drawMatches(img1, cvkp1, img2, cvkp2, cvmatches[:20], None, flags=2)
    # cv2.imshow('match',matchimg)
    # cv2.waitKey(0)
    pts1 = [ [kp1[matches[i][0]][0], kp1[matches[i][0]][1]] for i in range(len(matches)) ]
    pts2 = [ [kp2[matches[i][1]][0], kp2[matches[i][1]][1]] for i in range(len(matches)) ]
    print('matched key points {}'.format(len(matches)))
    if(len(pts1)<=MIN_NUM_KEY_POINT):
        return False, None, None

    pts1 = np.asarray(pts1, dtype=np.float)
    pts2 = np.asarray(pts2, dtype=np.float)
    distcoeffs = None
    if(not CV_VERSION[:3] == '4.5.4'):
        E, mask = cv2.findEssentialMat(pts1, pts2, K1, method=cv2.RANSAC, threshold=1., prob=0.99)
    else:
        E, mask = cv2.findEssentialMat(pts1, pts2, K1, distcoeffs, K2, distcoeffs, method=cv2.RANSAC, threshold=1., prob=0.99)

    matchednumber = np.sum(mask)
    accuracy = matchednumber / len(mask)
    print('essential matched accuracy {} ({})'.format(accuracy, matchednumber))
    if(type(mask) == type(None) or type(E) == type(None) or matchednumber < MIN_NUM_KEY_POINT):
        return False, None, None
    if(not CV_VERSION[:5] == '4.5.4'):
        retval, R_21, t, mask_pose = cv2.recoverPose(E, pts1, pts2, mask=mask)
        # tmpR1, tmpR2, tmpt = cv2.decomposeEssentialMat(E)
        # print(t)
    else:
        retval, E, R_21, t, mask_pose = cv2.recoverPose(pts1, pts2, K1, distcoeffs, K2, distcoeffs, E=E, method=cv2.RANSAC, threshold=1., prob=0.99, mask=mask)

    Rswitch = np.array([[0,-1.,0], [0,0,-1.], [1.,0,0]])
    R_21 = np.matmul(np.matmul(Rswitch.T, R_21), Rswitch)
    t_21 = np.matmul(Rswitch.T, t.reshape(-1))
    # R_12 = R_21.T
    # t_12 = -np.matmul(R_12, t_21).reshape(-1)
    # tmp = PyKDL.Rotation(R_12[0][0], R_12[0][1], R_12[0][2], R_12[1][0], R_12[1][1], R_12[1][2], R_12[2][0], R_12[2][1], R_12[2][2])
    tmp = PyKDL.Rotation(R_21[0][0], R_21[0][1], R_21[0][2], R_21[1][0], R_21[1][1], R_21[1][2], R_21[2][0], R_21[2][1], R_21[2][2])
    RPY = tmp.GetRPY()
    # print('R, t, rpy', R_12, t_12, RPY)
    print('R, t, rpy', R_21, t_21, RPY)
    if(abs(RPY[0]) > 0.8 or abs(RPY[1]) > 0.8):
        return False, None, None

    return (matchednumber > MIN_NUM_KEY_POINT), R_21, t_21 #R_12, t_12

def getRtPNP_hf_depth(img1, img2, kp1, des1, kp2, des2, img2_depth, K1, K2):
    matches = match_with_ratio_test(des1, des2, 0.8)
    cvkp1 = []
    cvkp2 = []
    pts1 = []
    pts2 = []
    cvmatches = []
    for i in range(len(matches)):
        cvmatches.append(cv2.DMatch(i, i, 1))
        cvkp1.append(cv2.KeyPoint(kp1[matches[i][0]][0], kp1[matches[i][0]][1], 1))
        cvkp2.append(cv2.KeyPoint(kp2[matches[i][1]][0], kp2[matches[i][1]][1], 1))
        pts1.append([kp1[matches[i][0]][0], kp1[matches[i][0]][1]])
        pts2.append([kp2[matches[i][1]][0], kp2[matches[i][1]][1]])

    pts1 = np.asarray(pts1, dtype=np.float)
    pts2 = np.asarray(pts2)
    print('[hfnetmatch]matched key points {}'.format(len(pts1)))
    if(len(cvmatches)<=MIN_NUM_KEY_POINT):
        # not enough matched key points
        return False, None, None
    
    uv1_2 = np.ones((len(pts2), 3))
    uv1_2[:,0:2] = pts2
    pts2 = np.round(pts2).astype(np.int)
    ds_2 = img2_depth[pts2[:,1] ,pts2[:,0]].reshape(len(pts2), 1)
    xyz_2 = (np.matmul(np.linalg.inv(K2), uv1_2.T).T * ds_2)
    # print(xyz_2[xyz_2[:,2]>0], pts1[xyz_2[:,2]>0])
    print('num of valid depth: {}'.format(np.sum(xyz_2[:,2]>0)))
    # cv2.namedWindow('propermatch', flags=0)
    # cv2.resizeWindow('propermatch', 800, 400)
    # matchimg = cv2.drawMatches(img1, cvkp1, img2, cvkp2, cvmatches, None, flags=2)
    # cv2.imshow('propermatch',matchimg)
    # cv2.waitKey(1)
    # print('3d and 2d point shape:', xyz_2.shape, pts1.shape)

    # compute R t by pnp
    distcoeffs = None
    trueorfalse,R,t,inliers = cv2.solvePnPRansac(xyz_2, pts1, K1, distcoeffs)
    if(type(inliers) == type(None)):
        return False, None, None
    R_12,_ = cv2.Rodrigues(R)

    # transform to body frame
    Rswitch = np.array([[0,-1.,0], [0,0,-1.], [1.,0,0]])
    R_12 = np.matmul(np.matmul(Rswitch.T, R_12), Rswitch)
    t = np.matmul(Rswitch.T, t.reshape(-1))
    tmp = PyKDL.Rotation(R_12[0][0], R_12[0][1], R_12[0][2], R_12[1][0], R_12[1][1], R_12[1][2], R_12[2][0], R_12[2][1], R_12[2][2])
    RPY = tmp.GetRPY()
    print('num of inliers by pnp: {}'.format(len(inliers)))
    print('R, t, rpy', R_12, t, RPY)
    if(abs(RPY[0]) > 0.5 or abs(RPY[1]) > 0.5 or np.linalg.norm(t) > 10):
        # the robot will not flip up and down or get too far from reference frame
        return False, None, None

    return trueorfalse, R_12, t

# compute the distance of current image to reference image by the triangle
def compute_scale(R_r1, inputt_r1, R_r2, inputt_r2, baseline):
    matched = False
    possible_signoft = [[1.,1.], [1.,-1.], [-1.,1.], [-1.,-1.]]

    t1_12 = np.array([0.,-1,0])
    tr_21 = np.matmul(R_r1, -t1_12)
    for i in range(4):
        t_r1 = possible_signoft[i][0] * inputt_r1
        t_r2 = possible_signoft[i][1] * inputt_r2
        # if(t_r1[1] < t_r2[1]):
        #     continue

        theta_1r2 = np.arccos(np.matmul(t_r1, t_r2.T))
        theta_r12 = np.arccos(np.matmul(t_r1, tr_21)) # no need to divide norm because the vectors are normalized
        theta_r21 = np.pi - theta_r12 - theta_1r2
        lr1 = np.sin(theta_r21) * baseline / np.sin(theta_1r2)
        lr2 = np.sin(theta_r12) * baseline / np.sin(theta_1r2)
        print('theta 1r2 ({}), r12 ({}), r21 ({})'.format(theta_1r2, theta_r12, theta_r21))
        print('stereo compute |t_r1|={}, |t_r2|={}'.format(lr1, lr2))
        t = t_r1 * lr1
        print('scaled tr1:', t)
        str2 = t_r2 * lr2
        print('scaled tr2:', str2)
        R = R_r1
        tr_21_2 = t - str2
        err_tr21 = np.arccos(np.matmul(tr_21_2, tr_21) / np.linalg.norm(tr_21_2))
        if(err_tr21 > np.pi):
            err_tr21 = err_tr21 - 2 * np.pi
        print('error of tr21s:', err_tr21, tr_21*baseline, tr_21_2)
        if(theta_r21 <= 0 or theta_1r2 <= 0 or theta_r12 <= 0 or abs(err_tr21) > 0.52):
            matched = False
        elif(lr1 < 3):
            matched = True
            print('this assumption is correct!')
            break
        print('this assumption is wrong!')
    return R, t, matched

def testfundamental():
    img1 = cv2.imread('000326.jpg')
    img2 = cv2.imread('000009.jpg')
    K = np.array([[699.6599731445312, 0.0, 633.25], \
                        [0.0, 699.6599731445312, 370.7355041503906], \
                        [0.0, 0.0, 1.0]])
    st = time.time()
    ret = fundamantal_check(img1, img2)
    et = time.time()
    print('runtime: {}. match: {}'.format(et-st, ret))
    st = time.time()
    ret,R,t,_,_ = getRt(img1, img2, K)
    et = time.time()
    print('runtime: {}. match: {}'.format(et-st, ret))
    print(R.GetRPY())

def test():
    img1 = cv2.imread('1.jpg')
    img2 = cv2.imread('2.jpg')
    K = np.array([[185.7, 0.0, 320.5], \
                  [0.0, 185.7, 240.5], \
                  [0.0, 0.0, 1]])
    
    st = time.time()
    ret = getRt(img1, img2, K)
    et = time.time()
    print('runtime: {}. match: {}'.format(et-st, ret))

def testPNP():
    imgref = cv2.imread('ref.jpg')
    img1 = cv2.imread('1.jpg')
    img1depth = np.load('depth1.npy') #cv2.imread('depth1.png', -1)
    K1 = np.array([[185.7, 0.0, 320.5], \
                  [0.0, 185.7, 240.5], \
                  [0.0, 0.0, 1]])
    K2 = np.array([[320.25492609007654, 0.0, 320.5], \
                  [0.0, 320.25492609007654, 240.5], \
                  [0.0, 0.0, 1]])
    
    st = time.time()
    ret = getRTPNP(imgref, img1, img1depth, K1, K2)
    et = time.time()
    print('runtime: {}. match: {}'.format(et-st, ret))


####################
# main #
####################
if __name__ == '__main__':
    # testPNP()
    testfundamental()
    # test()