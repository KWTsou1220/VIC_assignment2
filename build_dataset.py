#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import find_ped_img_idx, read_gt, HOG_descriptor
from utils import one_hot_encoder, built_codebook
from skimage.transform import resize
from skimage.color import rgb2luv, rgb2gray
from skimage.feature import local_binary_pattern

import cv2 as cv
import numpy as np
import tqdm
import time
import pickle
import os
import math


# In[2]:


# Functions to be used

def bag_of_features(img_kmeans, bb_list, mode='normal'):
    
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

def propose_negative(bb_list, ratio=1):
    
    bb_propose = []
    coord = [bb[2:] for bb in bb_list]
    
    for bb in bb_list:
        x, y, dx, dy = bb[2:]
        for it in range(ratio):
            #pdb.set_trace()
            break_flag = False
            while True:
                x1 = np.random.randint(640)
                y1 = np.random.randint(430)
                dx1 = np.random.randint(300)+1
                dy1 = np.random.randint(150)+1
                
                if x1+dx1>=640-8 or y1+dy1>=430-8:
                    continue
                if not is_overlap(coord, (x1, y1, dx1, dy1)):
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

def compute_overlap(coord1, coord2, h=430, w=640):
    x1, y1, dx1, dy1 = coord1
    x2, y2, dx2, dy2 = coord2
    img1 = np.zeros((h, w))
    img1[y1:y1+dy1, x1:x1+dx1] = 1
    img2 = np.zeros((h, w))
    img2[y2:y2+dy2, x2:x2+dx2] = 1

    intersection = img1 * img2
    return intersection.sum()/img1.sum()

def is_overlap(coord1s, coord2, h=430, w=640):
    is_overlap = False
    for coord1 in coord1s:
        if compute_overlap(coord1, coord2)>=0.5:
            is_overlap = True
            break
    return is_overlap

def unpackbits(x, num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1,1])
    to_and = 2**np.arange(num_bits).reshape([1,num_bits])
    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])

def kmeans_for_img(kmeans, img):
    h, w, ch = img.shape
    img = np.reshape(img, (h*w, ch))
    img = kmeans.predict(img)
    img = one_hot_encoder(img, 64)
    img = np.reshape(img, (h, w, 64))
    return img


# # Training K-means HoG

# In[3]:

'''
img_ped_idx = find_ped_img_idx()
hog_features = []
for idx in tqdm.tqdm(img_ped_idx):
    img_name = str(idx)
    if len(img_name)==1:
        img_name = '00'+img_name+'.jpg'
    elif len(img_name)==2:
        img_name = '0'+img_name+'.jpg'
    elif len(img_name)==3:
        img_name = img_name+'.jpg'
    img_path = './img1/' + img_name
    
    img = cv.imread(img_path)
    fd, hog_img = HOG_descriptor(img)
    h, w, ch = fd.shape
    fd = np.reshape(fd, (h*w, ch))
    hog_features += [fd]
hog_features = np.concatenate(hog_features, axis=0)

start = time.time()
kmeans_hog = built_codebook(hog_features)
end = time.time()
print('Time: ', end-start)
print('score: ', -kmeans_hog.score(hog_features))

print('Save hog code book and kmeans')
np.save('./Models/hog_features.npy', hog_features)
np.save('./Models/hog_codebook.npy', kmeans_hog.cluster_centers_)
pickle.dump(kmeans_hog, open('./Models/kmeans_hog.sav', 'wb'))

# Load
# loaded_model = pickle.load(open(filename, 'rb'))
'''

# # Training K-means LBP

# In[ ]:


img_ped_idx = find_ped_img_idx()
lbp_features = []
for idx in tqdm.tqdm(img_ped_idx):
    img_name = str(idx)
    if len(img_name)==1:
        img_name = '00'+img_name+'.jpg'
    elif len(img_name)==2:
        img_name = '0'+img_name+'.jpg'
    elif len(img_name)==3:

        img_name = img_name+'.jpg'
    img_path = './img1/' + img_name
    
    img = cv.imread(img_path)
    img = resize(img, output_shape=(430, 640), 
                 anti_aliasing=True, mode='reflect')
    img = rgb2gray(img)
    img_lbp = np.ndarray.astype(local_binary_pattern(img, P=8*3, R=3), np.uint32)
    img_lbp = img_lbp[::2, ::2]
    lbp_feature = unpackbits(img_lbp, num_bits=24)
    h, w, ch = lbp_feature.shape
    lbp_feature = np.reshape(lbp_feature, (h*w, ch))
    lbp_features += [lbp_feature]
        
lbp_features = np.concatenate(lbp_features, axis=0)

start = time.time()
kmeans_lbp = built_codebook(lbp_features)
end = time.time()
print('Time: ', end-start)
print('score: ', -kmeans_lbp.score(lbp_features))


print('Save hog code book and kmeans')
np.save('./Models/lbp_features.npy', lbp_features)
np.save('./Models/lbp_codebook.npy', kmeans_lbp.cluster_centers_)
pickle.dump(kmeans_lbp, open('./Models/kmeans_lbp.sav', 'wb'))

# Load
# loaded_model = pickle.load(open(filename, 'rb'))


# # Training K-means LUV

# In[ ]:


img_ped_idx = find_ped_img_idx()
luv_features = []
for idx in tqdm.tqdm(img_ped_idx):
    img_name = str(idx)
    if len(img_name)==1:
        img_name = '00'+img_name+'.jpg'
    elif len(img_name)==2:
        img_name = '0'+img_name+'.jpg'
    elif len(img_name)==3:
        img_name = img_name+'.jpg'
    img_path = './img1/' + img_name
    
    img = cv.imread(img_path)
    img = resize(img, output_shape=(430, 640), 
                 anti_aliasing=True, mode='reflect')
    img_luv = rgb2luv(img)
    img_luv = img_luv[::2, ::2]
    
    h, w, ch = img_luv.shape
    img_luv = np.reshape(img_luv, (h*w, ch))
    luv_features += [img_luv]
        
luv_features = np.concatenate(luv_features, axis=0)

start = time.time()
kmeans_luv = built_codebook(luv_features)
end = time.time()
print('Time: ', end-start)
print('score: ', -kmeans_luv.score(luv_features))

np.save('./Models/luv_features.npy', luv_features)
np.save('./Models/luv_codebook.npy', kmeans_luv.cluster_centers_)
pickle.dump(kmeans_luv, open('./Models/kmeans_luv.sav', 'wb'))

# Load
# loaded_model = pickle.load(open(filename, 'rb'))


# # Generate dataset

# In[ ]:


kmeans_hog = pickle.load(open('./Models/kmeans_hog.sav', 'rb'))
kmeans_lbp = pickle.load(open('./Models/kmeans_lbp.sav', 'rb'))
kmeans_luv = pickle.load(open('./Models/kmeans_luv.sav', 'rb'))
gt = read_gt('./gt/gt.txt')

for bb in gt:
    bb[2] = math.floor(bb[2]/2)
    bb[3] = math.floor(bb[3]/2)
    bb[4] = math.ceil(bb[4]/2)
    bb[5] = math.ceil(bb[5]/2)


# In[ ]:


neg_pos_ratio = 4

img_ped_idx = find_ped_img_idx(skip=1)
features_pd = []
features_bg = []
for idx in tqdm.tqdm(img_ped_idx):
    
    img_name = str(idx)
    if len(img_name)==1:
        img_name = '00'+img_name+'.jpg'
    elif len(img_name)==2:
        img_name = '0'+img_name+'.jpg'
    elif len(img_name)==3:
        img_name = img_name+'.jpg'
    img_path = './img1/' + img_name
    
    img = cv.imread(img_path)
    img = resize(img, output_shape=(430, 640), 
                 anti_aliasing=True, mode='reflect')
    
    bb_pos = [tmp for tmp in gt if tmp[0]==idx]
    bb_neg = propose_negative(bb_pos, ratio=neg_pos_ratio)
    
    feature_pd = []
    feature_bg = []
    
    # HoG features
    fd, _ = HOG_descriptor(img)
    fd = kmeans_for_img(kmeans_hog, fd)
    
    feature = bag_of_features(fd, bb_pos, mode='hog')
    feature_pd += [feature]
    feature = bag_of_features(fd, bb_neg, mode='hog')
    feature_bg += [feature]
    
    # LBP features
    img_gray = rgb2gray(img)
    img_lbp = np.ndarray.astype(local_binary_pattern(img_gray, P=8*3, R=3), np.uint32)
    lbp_feature = unpackbits(img_lbp, num_bits=24)
    lbp_feature = kmeans_for_img(kmeans_lbp, lbp_feature)
    
    feature = bag_of_features(lbp_feature, bb_pos)
    feature_pd += [feature]
    feature = bag_of_features(lbp_feature, bb_neg)
    feature_bg += [feature]
    
    # LUV features
    img_luv = rgb2luv(img)
    img_luv = kmeans_for_img(kmeans_luv, img_luv)
    
    feature = bag_of_features(img_luv, bb_pos)
    feature_pd += [feature]
    feature = bag_of_features(img_luv, bb_neg)
    feature_bg += [feature]
    
    # Concatenate the features
    feature_pd = np.concatenate(feature_pd, axis=1)
    feature_bg = np.concatenate(feature_bg, axis=1)
    
    features_pd += [feature_pd]
    features_bg += [feature_bg]
    
features_pd = np.concatenate(features_pd, axis=0)
features_pd = np.repeat(features_pd, neg_pos_ratio, axis=0)
features_bg = np.concatenate(features_bg, axis=0)


# In[ ]:


x_train = np.concatenate([features_pd, features_bg], axis=0)
t_train = np.zeros((features_pd.shape[0]+features_bg.shape[0], ))
t_train[0:features_pd.shape[0], ] = 1

np.save('./Dataset/x_train.npy', x_train)
np.save('./Dataset/t_train.npy', t_train)

