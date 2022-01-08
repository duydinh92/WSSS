import numpy as np
import torch
import cv2
import os
import torchvision
import argparse
from PIL import Image
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms

import sys  
sys.path.append(r"/content/drive/MyDrive/WSSS")
import dataset
from dataset import data
import network
from network.resnet import ResNet_ER_PCM, ResNet
import visualization_cam
from utils import imutils
from imutils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",default="/content/drive/MyDrive/WSSS/Project/experiments/models/train_res152+er+pcm_crop512_lr0.01_epoch8.pth")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--infer_list", default="voc12/train1.txt", type=str)
    parser.add_argument("--voc12_root", default='/content/VOC2012_train_val/VOC2012_train_val', type=str)
    parser.add_argument("--out_cam", default='cam', type=str)
    parser.add_argument("--out_crf", default='crf', type=str)
    args = parser.parse_args()

    crf_alpha = [4, 24]
    
    if not os.path.exists(args.out_cam):
        os.makedirs(args.out_cam)

    # model = ResNet(20, pretrained=False)
    model = ResNet_ER_PCM(21, pretrained=False)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    model.cuda()   

    infer_data_loader = dataset.data.data_loader_for_cam_inference(args)

    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]
        label = label[0]

        img_path = dataset.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        #print(orig_img)
        H, W = orig_img.shape[:2]   

        # # Single pass
        # with torch.no_grad():
        #   cam  = model.net_forward(img_list.cuda(), mode='val')
        #   cam = F.interpolate(cam[:,1:,:,:], size=(H,W), mode='bilinear', align_corners=False)[0]
        #   cam = cam.cpu().detach().numpy()
        
        # # Test time augmentation
        # with torch.no_grad():
        #   cam = model.net_forward(img_list.cuda(), mode='val')
        #   cam = F.interpolate(cam[:,1:,:,:], size=(H,W), mode='bilinear', align_corners=False)[0]
        #   cam = cam.cpu().detach().numpy()

        #   img_list = transforms.RandomHorizontalFlip(p=1.0)(img_list)
        #   _cam = model.net_forward(img_list.cuda(), mode='val')
        #   _cam = F.interpolate(_cam[:,1:,:,:], size=(H,W), mode='bilinear', align_corners=False)[0]
        #   _cam = transforms.RandomHorizontalFlip(p=1.0)(_cam)
        #   _cam = _cam.cpu().detach().numpy()

        #   cam = (cam + _cam)/2

        # Testing (Multiscale)
        cam = np.zeros((20, H, W), dtype=np.float32)
        with torch.no_grad():
            for s in [0.5, 1.0, 1.5, 2.0]:
                _img = F.interpolate(img_list, size=(int(s*H), int(s*W)), mode='bilinear', align_corners=False)
                _cam = model.net_forward(_img.cuda(), mode='val')
                _cam = F.interpolate(_cam[:,1:,:,:], size=(H,W), mode='bilinear', align_corners=False)[0]
                _cam = _cam.cpu().detach().numpy()
                cam = cam + _cam

        cam = cam/4

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

        # # Cam visualization
        # img = visualization_cam.visualization("/content/VOC2012_train_val/VOC2012_train_val/JPEGImages", "/content/drive/MyDrive/WSSS/Project/test/test", img_name)
        # cv2.imwrite(os.path.join(args.out_cam, "%s.png" % img_name), ((1-img)*255).astype(np.uint8))
        
        # CRF
        for t in crf_alpha:
            crf = imutils._crf_with_alpha(orig_img, cam_dict, t)
            folder = args.out_crf + ('_%.1f'%t)
            if not os.path.exists(folder):
                os.makedirs(folder)
            np.save(os.path.join(folder, img_name + '.npy'), crf)
        
        print(iter)
