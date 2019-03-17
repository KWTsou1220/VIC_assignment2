from clustering import train_kmeans
from clustering import train_kmeans_hog
from clustering import train_kmeans_lbp
from clustering import train_kmeans_luv
from build_dataset import build_dataset
from train_classifier import train_classifier
from detect_pedestrian import detect

def pedestrians(data_root):
    ''' Return a list of bounding boxes in the format frame, bb_id, x,y,dx,dy '''
    #return [[1,1,617,128,20,50]]
    
    # If the memory space is enough you can use the following function, instead of
    # training hog, lbp, luv separately
    # train_kmeans(save_path='./Models')

    print('Training k-means HoG...')
    train_kmeans_hog(save_path='./Models')
    print('Training k-means LBP...')
    train_kmeans_lbp(save_path='./Models')
    print('Training k-means LUV...')
    train_kmeans_luv(save_path='./Models')

    print('Building dataset...')
    build_dataset(save_path='./Dataset/', kmeans_path='./Models/')

    print('Training classifier...')
    train_classifier(save_path='./Models', data_path='./Dataset')
    
    print('Detecting...')
    sol = detect(save_path='./gt', model_path='./Models/', image_path=data_root)
    
    return sol


if __name__ == '__main__':
    pedestrians('./img1/')
    
