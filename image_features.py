from skimage.transform import resize
from skimage.feature   import hog
from skimage.color     import rgb2luv, rgb2gray
from skimage.feature   import local_binary_pattern
from utils             import unpackbits

import numpy as np
import math
import ipdb

def extract_hog(img, shape=None, flatten=True):
    if shape is not None:
        img = resize(img, output_shape=shape, 
                     anti_aliasing=True, mode='reflect')
    
    hog_feature, _ = HOG_descriptor(img)
    if flatten:
        h, w, ch = hog_feature.shape
        hog_feature = np.reshape(hog_feature, (h*w, ch))
    return hog_feature

def extract_lbp(img, shape=None, flatten=True, sampling=(2, 2)):
    if shape is not None:
        img = resize(img, output_shape=shape, 
                     anti_aliasing=True, mode='reflect')
    
    img = rgb2gray(img)
    img_lbp = np.ndarray.astype(local_binary_pattern(img, P=8*3, R=3), np.uint32)
    img_lbp = img_lbp[::sampling[0], ::sampling[1]]
    lbp_feature = unpackbits(img_lbp, num_bits=24)
    if flatten:
        h, w, ch = lbp_feature.shape
        lbp_feature = np.reshape(lbp_feature, (h*w, ch))
    return lbp_feature

def extract_luv(img, shape=None, flatten=True, sampling=(2, 2)):
    if shape is not None:
        img = resize(img, output_shape=shape, 
                     anti_aliasing=True, mode='reflect')
    
    img_luv = rgb2luv(img)
    img_luv = img_luv[::sampling[0], ::sampling[1]]
    if flatten:
        h, w, ch = img_luv.shape
        img_luv = np.reshape(img_luv, (h*w, ch))
    return img_luv

def HOG_descriptor(img):
    
    if len(img.shape)==3 and img.shape[2]==3:
        multichannel = True
    else:
        multichannel = False
    
    fd, hog_img = hog(img, orientations=6, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, 
                      multichannel=multichannel, block_norm='L2-Hys',
                      feature_vector=False)
    h, w, h_block, w_block, ori_size = fd.shape
    fd = np.reshape(fd, (h, w, h_block*w_block*ori_size))
    return fd, hog_img


def bag_of_features(img_kmeans, bb_list, mode='normal'):
    
    if not isinstance(bb_list, list):
        raise Exception('Bounding boxes should be stored in list.')
    if not isinstance(bb_list[0], list) and not isinstance(bb_list[0], tuple):
        bb_list = [bb_list]
    
    features = []
    
    for bb in bb_list:
        x, y, dx, dy = bb[2:]
        
        if mode=='normal':
            pos = img_kmeans[y:y+dy, x:x+dx, :]
            pos = np.reshape(pos, (pos.shape[0]*pos.shape[1], -1))
        elif mode=='hog':
            y1 = math.floor((y-8)/8)
            if y1<0:
                y1 = 0
            y2 = math.ceil((y+dy-8)/8)
            x1 = math.floor((x-8)/8)
            if x1<0:
                x1 = 0
            x2 = math.ceil((x+dx-8)/8)

            if y1==y2:
                y2 += 1
            if x1==x2:
                x2 += 1

            pos = img_kmeans[y1:y2, x1:x2, :]
            pos = np.reshape(pos, (pos.shape[0]*pos.shape[1], -1))
        
        feature = np.sum(pos, axis=0, keepdims=True)
        feature = feature/np.sum(feature)
        features += [feature]
    
    return np.concatenate(features, axis=0)

def propose_negative(bb_list, img_shape, ratio=1):
    
    bb_propose = []
    coord = [bb[2:] for bb in bb_list]
    
    for bb in bb_list:
        x, y, dx, dy = bb[2:]
        for it in range(ratio):
            #pdb.set_trace()
            break_flag = False
            while True:
                x1 = np.random.randint(img_shape[1])
                y1 = np.random.randint(img_shape[0])
                dx1 = np.random.randint(int(img_shape[1]/2))+1
                dy1 = np.random.randint(int(img_shape[0]/2))+1
                
                if x1+dx1>=img_shape[1]-8 or y1+dy1>=img_shape[0]-8:
                    continue
                if not is_overlap(coord, (x1, y1, dx1, dy1), img_shape[0], img_shape[1]):
                    break
            '''
            if np.random.randint(2):
                while True:
                    x1 = x+np.random.randint(dx)-math.floor(dx/2)
                    y1 = y+np.random.randint(dy)-math.floor(dy/2)
                    dx1 = np.random.randint(dx*2)
                    dy1 = np.random.randint(dy*2)
                    if x1+dx1>=430 or y1+dy1>=640:
                        continue
                    if not is_overlap(coord, (x1, y1, dx1, dy1)):
                        break
            else:
                while True:
                    x1 = np.random.randint(430)
                    y1 = y+np.random.randint(640)
                    dx1 = np.random.randint(150)
                    dy1 = np.random.randint(300)
                    if x1+dx1>=430 or y1+dy1>=640:
                        continue
                    if not is_overlap(coord, (x1, y1, dx1, dy1)):
                        break
            '''
            bb_propose += [[bb[0], bb[1], x1, y1, dx1, dy1]]
    return bb_propose

def compute_overlap(coord1, coord2, h, w):
    x1, y1, dx1, dy1 = coord1
    x2, y2, dx2, dy2 = coord2
    img1 = np.zeros((h, w), dtype=np.bool)
    img1[y1:y1+dy1, x1:x1+dx1] = 1
    img2 = np.zeros((h, w), dtype=np.bool)
    img2[y2:y2+dy2, x2:x2+dx2] = 1

    intersection = img1 * img2
    union = img1 | img2
    return intersection.sum()/union.sum()

def is_overlap(coord1s, coord2, h, w):
    is_overlap = False
    for coord1 in coord1s:
        if compute_overlap(coord1, coord2, h, w)>=0.5:
            is_overlap = True
            break
    return is_overlap

