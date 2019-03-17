from image_features import bag_of_features
from image_features import extract_hog
from image_features import extract_lbp
from image_features import extract_luv
from utils import unpackbits, load_image
from clustering import kmeans_for_img

import pickle
import os
import math
import numpy as np
import tqdm
import ipdb
import cv2 as cv

def detect(save_path='./gt', model_path='./Models/', image_path='./img1/'):
    detections = []
    
    if os.path.isfile(os.path.join(save_path, 'solution.txt')):
        os.remove(os.path.join(save_path, 'solution.txt'))
    
    xgb = pickle.load(open(os.path.join(model_path, 'xgb.sav'), 'rb'))
    kmeans_hog = pickle.load(open(os.path.join(model_path, 'kmeans_hog.sav'), 'rb'))
    kmeans_lbp = pickle.load(open(os.path.join(model_path, 'kmeans_lbp.sav'), 'rb'))
    kmeans_luv = pickle.load(open(os.path.join(model_path, 'kmeans_luv.sav'), 'rb'))
    
    img_path = [os.path.join(image_path, img_name) for img_name in os.listdir(image_path)]
    
    
    stepSizes = [10, 20, 25, 50, 75]
    windowSizes = [[20, 50], [40, 110], [50, 150], [100, 300], [150, 300]]
    img = cv.imread(img_path[0])
    windows = sliding_window(img_shape=img.shape[0:2], stepSizes=stepSizes, windowSizes=windowSizes)
    
    
    images = load_image(img_path)
    for img_id, img in tqdm.tqdm(images):
        feature = []
        
        # HoG features
        hog_feature = extract_hog(img, flatten=False)
        hog_feature = kmeans_for_img(kmeans_hog, hog_feature)
        feature += [bag_of_features(hog_feature, windows, mode='hog')]
        
        # LBP features
        lbp_feature = extract_lbp(img, flatten=False, sampling=(1, 1))
        lbp_feature = kmeans_for_img(kmeans_lbp, lbp_feature)
        feature += [bag_of_features(lbp_feature, windows)]
        
        # LUV features
        luv_feature = extract_luv(img, flatten=False, sampling=(1, 1))
        luv_feature = kmeans_for_img(kmeans_luv, luv_feature)
        feature += [bag_of_features(luv_feature, windows)]
        
        # Combine all features
        features = np.concatenate(feature, axis=1)
        
        # Detect Pedestrian
        y_train_prob = xgb.predict_proba(features)
        selected_windows_idx, _ = np.where(y_train_prob[:, 1:]>0.99)
        selected_windows = non_max_suppression_fast(np.array(windows)[selected_windows_idx, 2:], overlapThresh=0.5)
        
        for instance_id, selected_window in enumerate(selected_windows):
            detection = [img_id, instance_id+1]
            detection += [selected_window[0]]
            detection += [selected_window[1]]
            detection += [selected_window[2]]
            detection += [selected_window[3]]
            detection += [1, -1, -1, -1]
            detections += [detection]
            with open(os.path.join(save_path, 'solution.txt'), 'a') as f:
                sol = ''
                for num in detection:
                    sol += str(num)+','
                sol = sol[0:-1]+'\n'
                f.write(sol)
        
    return detections
    
def sliding_window(img_shape, stepSizes, windowSizes):
    # slide a window across the image
    windows = []
    if not isinstance(stepSizes, list):
        stepSizes = [stepSizes]
    if len(windowSizes)==2 and not isinstance(windowSizes[0], list):
        windowSizes = [windowSizes]
    if (not len(stepSizes)==len(windowSizes)) and (not len(stepSizes)==1):
        raise Exception('Lenght of step size and window size should be the same')
    elif (not len(stepSizes)==len(windowSizes)):
        stepSizes = [stepSizes[0] for i in windowSizes]
    
    for stepSize, windowSize in zip(stepSizes, windowSizes):
        for y in range(0, img_shape[0]-windowSize[1], stepSize):
            for x in range(0, img_shape[1]-windowSize[0], stepSize):
                # yield the current window
                windows += [(-1, -1, x, y, windowSize[0], windowSize[1])]
    return windows

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes	
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]+x1
    y2 = boxes[:,3]+y1
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
