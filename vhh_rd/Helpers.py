
import pickle
import os
import cv2

def do_pickle(data, filepath):
    outfile = open(filepath,'wb')
    pickle.dump(data,outfile)
    outfile.close()

def do_unpickle(filepath):
    with open(filepath, 'rb') as pickleFile:
        return pickle.load(pickleFile)

def load_features(img_name, rd):
    """
    Given an img name, loads the corresponding feature
    """
    features_path = rd.get_feature_path(img_name)
    return do_unpickle(features_path) 

def load_img(img_name, rd):
    img_path = os.path.join(rd.extracted_frames_path, img_name)
    img = cv2.imread(img_path)
    return img