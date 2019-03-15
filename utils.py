from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage.feature import hog
from skimage import exposure
from skimage.transform import resize

import os
import numpy as np


def HOG_descriptor(img):
    img = resize(img, output_shape=(430, 640), 
                 anti_aliasing=True, mode='reflect')
    
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

def find_ped_img_idx(skip=1):
    img_ped_idx = set()
    with open('./gt/gt.txt') as f:
        for line in f:
            line = line.split(',')
            img_ped_idx.add(int(line[0]))
    img_ped_idx = list(img_ped_idx)
    
    return img_ped_idx[::skip]

def built_codebook(features):
    kmeans = MiniBatchKMeans(n_clusters=64, batch_size=500, n_init=10)
    #kmeans = KMeans(n_clusters=64, random_state=0, n_jobs=-1)
    kmeans.fit(features)
    return kmeans

def read_gt(filename):
    """Read gt and create list of bb-s"""
    assert os.path.exists(filename)
    with open(filename, 'r') as file:
        lines = file.readlines()
    # truncate data (last columns are not needed)
    return [list(map(lambda x: int(x), line.split(',')[:6])) for line in lines]

def one_hot_encoder(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]
