from utils import find_ped_img_idx, read_gt
from image_features import extract_hog
from image_features import extract_lbp
from image_features import extract_luv
from image_features import propose_negative
from image_features import bag_of_features
from clustering import kmeans_for_img

import pickle
import numpy as np
import os
import math
import tqdm
import cv2 as cv
import pdb

def build_dataset(save_path='./Dataset/', kmeans_path='./Models/', neg_pos_ratio=10):
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        
     # To determine whether to generate dataset
    if os.path.isfile(os.path.join(save_path, 'x_train.npy')) and os.path.isfile(os.path.join(save_path, 't_train.npy')):
        print('Dataset exists')
        return None
    
    kmeans_hog = pickle.load(open(os.path.join(kmeans_path, 'kmeans_hog.sav'), 'rb'))
    kmeans_lbp = pickle.load(open(os.path.join(kmeans_path, 'kmeans_lbp.sav'), 'rb'))
    kmeans_luv = pickle.load(open(os.path.join(kmeans_path, 'kmeans_luv.sav'), 'rb'))
    gt = read_gt('./gt/gt.txt')

    #for bb in gt:
    #    bb[2] = math.floor(bb[2]/2)
    #    bb[3] = math.floor(bb[3]/2)
    #    bb[4] = math.ceil(bb[4]/2)
    #    bb[5] = math.ceil(bb[5]/2)

    img_ped_idx = find_ped_img_idx()
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
        # bounding boxes for pedestrian
        bb_pos = [tmp for tmp in gt if tmp[0]==idx] 
        # bounding boxes for background
        bb_neg = propose_negative(bb_pos, img.shape, ratio=neg_pos_ratio) 

        feature_pd = []
        feature_bg = []

        # HoG features
        hog_feature = extract_hog(img, flatten=False)
        hog_feature = kmeans_for_img(kmeans_hog, hog_feature)

        feature = bag_of_features(hog_feature, bb_pos, mode='hog')
        feature_pd += [feature]
        feature = bag_of_features(hog_feature, bb_neg, mode='hog')
        feature_bg += [feature]

        # LBP features
        lbp_feature = extract_lbp(img, flatten=False, sampling=(1, 1))
        lbp_feature = kmeans_for_img(kmeans_lbp, lbp_feature)

        feature = bag_of_features(lbp_feature, bb_pos)
        feature_pd += [feature]
        feature = bag_of_features(lbp_feature, bb_neg)
        feature_bg += [feature]

        # LUV features
        luv_feature = extract_luv(img, flatten=False, sampling=(1, 1))
        luv_feature = kmeans_for_img(kmeans_luv, luv_feature)

        feature = bag_of_features(luv_feature, bb_pos)
        feature_pd += [feature]
        feature = bag_of_features(luv_feature, bb_neg)
        feature_bg += [feature]

        # Concatenate the features
        feature_pd = np.concatenate(feature_pd, axis=1)
        feature_bg = np.concatenate(feature_bg, axis=1)

        features_pd += [feature_pd]
        features_bg += [feature_bg]

    features_pd = np.concatenate(features_pd, axis=0)
    features_pd = np.repeat(features_pd, neg_pos_ratio, axis=0)
    features_bg = np.concatenate(features_bg, axis=0)

    x_train = np.concatenate([features_pd, features_bg], axis=0)
    t_train = np.zeros((features_pd.shape[0]+features_bg.shape[0], ))
    t_train[0:features_pd.shape[0], ] = 1

    np.save(os.path.join(save_path, 'x_train.npy'), x_train)
    np.save(os.path.join(save_path, 't_train.npy'), t_train)
