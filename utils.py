import os
import numpy as np
import cv2 as cv

def find_ped_img_idx(skip=1):
    img_ped_idx = set()
    with open('./gt/gt.txt') as f:
        for line in f:
            line = line.split(',')
            img_ped_idx.add(int(line[0]))
    img_ped_idx = list(img_ped_idx)
    
    return img_ped_idx[::skip]


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

def unpackbits(x, num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1,1])
    to_and = 2**np.arange(num_bits).reshape([1,num_bits])
    return (x & to_and).astype(bool).astype(np.int8).reshape(xshape + [num_bits])

def load_image(img_path):
    for path in img_path:
        img = cv.imread(path)
        
        img_name = path.split('/')[-1]
        img_name = img_name.split('.')[0]
        img_id = int(img_name)
        
        yield img_id, img
