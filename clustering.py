from sklearn.cluster import MiniBatchKMeans
from utils import one_hot_encoder
from utils import find_ped_img_idx
from image_features import extract_hog
from image_features import extract_lbp
from image_features import extract_luv

import numpy as np
import cv2 as cv
import os
import tqdm
import pickle
import time
import ipdb


def built_codebook(features, batch_size=500):
    kmeans = MiniBatchKMeans(n_clusters=64, batch_size=batch_size, n_init=10)
    #kmeans = KMeans(n_clusters=64, random_state=0, n_jobs=-1)
    kmeans.fit(features)
    return kmeans

def kmeans_for_img(kmeans, img):
    h, w, ch = img.shape
    img = np.reshape(img, (h*w, ch))
    img = kmeans.predict(img)
    img = one_hot_encoder(img, 64)
    img = np.reshape(img, (h, w, 64))
    return img

def train_kmeans_hog(save_path='./Models'):
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    # To determine whether to train a K-means
    train_hog = True
    if os.path.isfile(os.path.join(save_path, 'kmeans_hog.sav')):
        train_hog = False
        print('Kmeans for HoG exists')
        
    if not train_hog:
        return None
    
    img_ped_idx = find_ped_img_idx()
    hog_features = []
    for idx in tqdm.tqdm(img_ped_idx[::2]):
        img_name = str(idx)
        if len(img_name)==1:
            img_name = '00'+img_name+'.jpg'
        elif len(img_name)==2:
            img_name = '0'+img_name+'.jpg'
        elif len(img_name)==3:
            img_name = img_name+'.jpg'
        img_path = './img1/' + img_name

        img = cv.imread(img_path)
        if train_hog:
            hog_feature = extract_hog(img)
            hog_features += [hog_feature]
    
    # Training kmeans for HoG features
    if train_hog:
        hog_features = np.concatenate(hog_features, axis=0)
        start = time.time()
        kmeans_hog = built_codebook(hog_features)
        end = time.time()
        print('K-means HoG info:')
        print('\tTraining time: ', end-start)
        print('\tTraining score: ', -kmeans_hog.score(hog_features)/hog_features.shape[0])

        np.save(os.path.join(save_path, 'hog_features.npy'), hog_features)
        np.save(os.path.join(save_path, 'hog_codebook.npy'), kmeans_hog.cluster_centers_)
        pickle.dump(kmeans_hog, open(os.path.join(save_path, 'kmeans_hog.sav'), 'wb'))

def train_kmeans_lbp(save_path='./Models'):
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    # To determine whether to train a K-means
    train_lbp = True
    if os.path.isfile(os.path.join(save_path, 'kmeans_lbp.sav')):
        train_lbp = False
        print('Kmeans for LBP exists')
        
    if not train_lbp:
        return None
    
    img_ped_idx = find_ped_img_idx()
    lbp_features = []
    for idx in tqdm.tqdm(img_ped_idx[::4]):
        img_name = str(idx)
        if len(img_name)==1:
            img_name = '00'+img_name+'.jpg'
        elif len(img_name)==2:
            img_name = '0'+img_name+'.jpg'
        elif len(img_name)==3:
            img_name = img_name+'.jpg'
        img_path = './img1/' + img_name

        img = cv.imread(img_path)
        if train_lbp:
            #lbp_feature = extract_lbp(img, shape=(430, 640))
            lbp_feature = extract_lbp(img, sampling=(3, 3))
            lbp_features += [lbp_feature]
    
    # Training kmeans for LBP features
    if train_lbp:
        lbp_features = np.concatenate(lbp_features, axis=0)
        start = time.time()
        kmeans_lbp = built_codebook(lbp_features)
        end = time.time()
        print('K-means LBP info:')
        print('\tTraining time: ', end-start)
        print('\tTraining score: ', -kmeans_lbp.score(lbp_features)/lbp_features.shape[0])

        np.save(os.path.join(save_path, 'lbp_features.npy'), lbp_features)
        np.save(os.path.join(save_path, 'lbp_codebook.npy'), kmeans_lbp.cluster_centers_)
        pickle.dump(kmeans_lbp, open(os.path.join(save_path, 'kmeans_lbp.sav'), 'wb'))
        

def train_kmeans_luv(save_path='./Models'):
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    # To determine whether to train a K-means
    train_luv = True
    if os.path.isfile(os.path.join(save_path, 'kmeans_luv.sav')):
        train_luv = False
        print('Kmeans for LUV exists')
        
    if not train_luv:
        return None
    
    img_ped_idx = find_ped_img_idx()
    luv_features = []
    for idx in tqdm.tqdm(img_ped_idx[::4]):
        img_name = str(idx)
        if len(img_name)==1:
            img_name = '00'+img_name+'.jpg'
        elif len(img_name)==2:
            img_name = '0'+img_name+'.jpg'
        elif len(img_name)==3:
            img_name = img_name+'.jpg'
        img_path = './img1/' + img_name

        img = cv.imread(img_path)
        if train_luv:
            #luv_feature = extract_luv(img, shape=(430, 640))
            luv_feature = extract_luv(img, sampling=(2, 2))
            luv_features += [luv_feature]
    
    # Training kmeans for LUV features
    if train_luv:
        luv_features = np.concatenate(luv_features, axis=0)
        start = time.time()
        kmeans_luv = built_codebook(luv_features)
        end = time.time()
        print('K-means LUV info:')
        print('\tTraining time: ', end-start)
        print('\tTraining score: ', -kmeans_luv.score(luv_features)/luv_features.shape[0])

        np.save(os.path.join(save_path, 'luv_features.npy'), luv_features)
        np.save(os.path.join(save_path, 'luv_codebook.npy'), kmeans_luv.cluster_centers_)
        pickle.dump(kmeans_luv, open(os.path.join(save_path, 'kmeans_luv.sav'), 'wb'))
        
def train_kmeans(save_path='./Models'):
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    # To determine whether to train a K-means
    train_hog = True
    if os.path.isfile(os.path.join(save_path, 'kmeans_hog.sav')):
        train_hog = False
        print('Kmeans for HoG exists')
    train_lbp = True
    if os.path.isfile(os.path.join(save_path, 'kmeans_lbp.sav')):
        train_lbp = False
        print('Kmeans for LBP exists')
    train_luv = True
    if os.path.isfile(os.path.join(save_path, 'kmeans_luv.sav')):
        train_luv = False
        print('Kmeans for LUV exists')
        
    if not (train_hog or train_lbp or train_luv):
        return None
    
    img_ped_idx = find_ped_img_idx()
    hog_features = []
    lbp_features = []
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
        if train_hog:
            hog_feature = extract_hog(img)
            hog_features += [hog_feature]
        if train_lbp:
            lbp_feature = extract_lbp(img, shape=(430, 640))
            lbp_features += [lbp_feature]
        if train_luv:
            luv_feature = extract_luv(img, shape=(430, 640))
            luv_features += [luv_feature]
    
    # Training kmeans for HoG features
    if train_hog:
        hog_features = np.concatenate(hog_features, axis=0)
        start = time.time()
        kmeans_hog = built_codebook(hog_features)
        end = time.time()
        print('K-means HoG info:')
        print('\tTraining time: ', end-start)
        print('\tTraining score: ', -kmeans_hog.score(hog_features))

        np.save(os.path.join(save_path, 'hog_features.npy'), hog_features)
        np.save(os.path.join(save_path, 'hog_codebook.npy'), kmeans_hog.cluster_centers_)
        pickle.dump(kmeans_hog, open(os.path.join(save_path, 'kmeans_hog.sav'), 'wb'))
    
    # Training kmeans for LBP features
    if train_lbp:
        lbp_features = np.concatenate(lbp_features, axis=0)
        start = time.time()
        kmeans_lbp = built_codebook(lbp_features)
        end = time.time()
        print('K-means LBP info:')
        print('\tTraining time: ', end-start)
        print('\tTraining score: ', -kmeans_lbp.score(lbp_features))

        np.save(os.path.join(save_path, 'lbp_features.npy'), lbp_features)
        np.save(os.path.join(save_path, 'lbp_codebook.npy'), kmeans_lbp.cluster_centers_)
        pickle.dump(kmeans_lbp, open(os.path.join(save_path, 'kmeans_lbp.sav'), 'wb'))

    # Training kmeans for LUV features
    if train_luv:
        luv_features = np.concatenate(luv_features, axis=0)
        start = time.time()
        kmeans_luv = built_codebook(luv_features)
        end = time.time()
        print('K-means LUV info:')
        print('\tTraining time: ', end-start)
        print('\tTraining score: ', -kmeans_luv.score(luv_features))

        np.save(os.path.join(save_path, 'luv_features.npy'), luv_features)
        np.save(os.path.join(save_path, 'luv_codebook.npy'), kmeans_luv.cluster_centers_)
        pickle.dump(kmeans_luv, open(os.path.join(save_path, 'kmeans_luv.sav'), 'wb'))
    
    
  
