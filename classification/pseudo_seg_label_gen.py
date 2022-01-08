import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import cv2
import argparse

import sys  
sys.path.append(r"/content/drive/MyDrive/WSSS/Project")
import dataset
from dataset import data
from utils import imutils
from imutils import *


def pp_with_connected_components(pred, connectivity=4, THRESHOLD=100):
    _, thresh = cv2.threshold(pred,0,255,cv2.THRESH_BINARY)
    num_labels, labels , stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

    for idx, stat in enumerate(stats):
        x, y, w, h, c = stat 
        if w == pred.shape[1] or h == pred.shape[0]:
            continue
        if c < THRESHOLD:
            pred[labels == idx] = 0
    
    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='voc12/train_aug.txt', type=str)
    parser.add_argument("--cam_dir", default='crf_4.0', type=str)
    parser.add_argument("--num_cls", default=21, type=int)
    parser.add_argument("--threshold", default=0.0002, type=float)
    parser.add_argument("--output_dir", default="pseudo_label", type=str)
    args = parser.parse_args()

    name_list = data.load_img_name_list(args.list)
    num_cls = args.num_cls
    threshold = args.threshold
    cam_dir = args.cam_dir
    output_dir = args.output_dir
    palette = imutils.get_palette()

    for idx in range(len(name_list)):
        name = name_list[idx]
        cam_file = os.path.join(cam_dir,'%s.npy'%name)
        cam_dict = np.load(cam_file, allow_pickle=True).item()
        h, w = list(cam_dict.values())[0].shape
        cam = np.zeros((num_cls,h,w),np.float32)

        for key in cam_dict.keys():
            cam[key] = cam_dict[key]
        cam[0,:,:] = threshold 
        cam = np.argmax(cam, axis=0).astype(np.uint8)
        #cam = pp_with_connected_components(cam, THRESHOLD=100) 
        pred_map = Image.fromarray(cam)
        pred_map.putpalette(palette)
        pred_map.save(os.path.join(output_dir, "%s.png" % name))

        print(idx)

  

 