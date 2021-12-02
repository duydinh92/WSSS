import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
import sys  
sys.path.append(r"/content/drive/MyDrive/WSSS/Project")
import dataset
from dataset import data

def get_palette():
    palette = []
    for i in range(256):
        palette.extend((i,i,i))
    palette[:3*21] = np.array([[0, 0, 0],
                            [128, 0, 0],
                            [0, 128, 0],
                            [128, 128, 0],
                            [0, 0, 128],
                            [128, 0, 128],
                            [0, 128, 128],
                            [128, 128, 128],
                            [64, 0, 0],
                            [192, 0, 0],
                            [64, 128, 0],
                            [192, 128, 0],
                            [64, 0, 128],
                            [192, 0, 128],
                            [64, 128, 128],
                            [192, 128, 128],
                            [0, 64, 0],
                            [128, 64, 0],
                            [0, 192, 0],
                            [128, 192, 0],
                            [0, 64, 128]], dtype='uint8').flatten()
    
    return palette

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='voc12/train1.txt', type=str)
    parser.add_argument("--cam_dir", default='cam', type=str)
    parser.add_argument("--num_cls", default=21, type=int)
    parser.add_argument("--threshold", default=0.2, type=float)
    parser.add_argument("--output_dir", default="pseudo_label", type=str)
    args = parser.parse_args()

    name_list = data.load_img_name_list(args.list)
    num_cls = args.num_cls
    threshold = args.threshold
    cam_dir = args.cam_dir
    output_dir = args.output_dir
    palette = get_palette()

    for idx in range(len(name_list)):
        name = name_list[idx]
        cam_file = os.path.join(cam_dir,'%s.npy'%name)
        cam_dict = np.load(cam_file, allow_pickle=True).item()
        h, w = list(cam_dict.values())[0].shape
        cam = np.zeros((num_cls,h,w),np.float32)

        for key in cam_dict.keys():
            cam[key+1] = cam_dict[key]
        cam[0,:,:] = threshold 
        cam = np.argmax(cam, axis=0).astype(np.uint8)
        pred_map = Image.fromarray(cam)
        pred_map.putpalette(palette)
        pred_map.save(os.path.join(output_dir, "%s.png" % name))

  

 