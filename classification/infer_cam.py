import numpy as np
import torch
import cv2
import os
import sys  
sys.path.append(r"/content/drive/MyDrive/WSSS/Project")
import dataset
from dataset import data
import network
import scipy.misc
import importlib
import torchvision
import argparse
from PIL import Image
import torch.nn.functional as F
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",default="/content/drive/MyDrive/WSSS/SEAM/resnet38_SEAM.pth", type=str)
    parser.add_argument("--network", default="network.resnet38_SEAM", type=str)
    parser.add_argument("--infer_list", default="voc12/train1.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", default='/content/VOC2012_train_val/VOC2012_train_val', type=str)
    parser.add_argument("--out_cam", default='cam', type=str)
    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    model.cuda()   

    infer_data_loader = dataset.data.data_loader_for_cam_inference(args)

    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]
        label = label[0]

        img_path = dataset.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        H, W = orig_img.shape[:2]

        # Multisize testing (0.524)
        # cam = np.zeros((20, H, W), dtype=np.float32)
        # with torch.no_grad():
        #     for s in [256, 420, 512]:
        #         _img = F.interpolate(img_list, size=(s, s), mode='bilinear', align_corners=False)
        #         _, _cam = model(_img.cuda())
        #         _cam = F.interpolate(_cam[:,1:,:,:], size=(H,W), mode='bilinear', align_corners=False)[0]
        #         _cam = _cam.cpu().detach().numpy()
        #         cam = cam + _cam
        # cam = cam/3   
        
        # Testing (Multiscale)(0.53- 23 background)
        # cam = np.zeros((20, H, W), dtype=np.float32)
        # with torch.no_grad():
        #     for s in [0.5, 1.0, 1.5]:
        #         _img = F.interpolate(img_list, size=(int(s*H), int(s*W)), mode='bilinear', align_corners=False)
        #         _, _cam = model(_img.cuda())
        #         _cam = F.interpolate(_cam[:,1:,:,:], size=(H,W), mode='bilinear', align_corners=False)[0]
        #         _cam = _cam.cpu().detach().numpy()
        #         cam = cam + _cam
        # cam = cam/3 

        # Single pass
        with torch.no_grad():
          _, cam = model(img_list.cuda())
          cam = F.interpolate(cam[:,1:,:,:], size=(H,W), mode='bilinear', align_corners=False)[0]
          cam = cam.cpu().detach().numpy()

        cam = cam * label.clone().view(20, 1, 1).numpy()
        cam_max = np.max(cam, (1,2), keepdims=True)
        cam_min = np.min(cam, (1,2), keepdims=True)
        cam[cam < cam_min+1e-5] = 0
        norm_cam = (cam-cam_min-1e-5) / (cam_max - cam_min + 1e-5)
        cam_dict = {}
        for i in range(20):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)
        
        print(iter)
