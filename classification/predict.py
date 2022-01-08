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
import importlib
from scipy import misc
import os.path
import imageio
from PIL import Image

import sys

sys.path.append(r"/content/drive/MyDrive/WSSS/Project")
import dataset
from dataset import data
import network
from network.resnet import ResNet_ER_PCM, ResNet
import visualization_cam
from utils import imutils


def predict(image):
    # image = Image.open("/content/drive/MyDrive/WSSS/Project/test/orig/2007_000032.jpg").convert("RGB")
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    tsfm_infer = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)])
    img_list = tsfm_infer(image).unsqueeze(0)
    _, _, H, W = img_list.shape
    image = np.array(image)

    # Infer CAM
    model = ResNet_ER_PCM(21, pretrained=False)
    model.load_state_dict(torch.load(
        "/content/drive/MyDrive/WSSS/Project/experiments/models/train_res152+er+pcm_crop512_lr0.01_epoch8.pth"))
    model.cuda()
    model.eval()

    cam = np.zeros((20, H, W), dtype=np.float32)
    label = torch.zeros(1, 20)
    with torch.no_grad():
        for s in [0.5, 1.0, 1.5, 2.0]:
            _img = F.interpolate(img_list, size=(int(s * H), int(s * W)), mode='bilinear', align_corners=False)
            _cam, _score = model.net_forward(_img.cuda(), mode='test')
            _cam = F.interpolate(_cam[:, 1:, :, :], size=(H, W), mode='bilinear', align_corners=False)[0]
            _cam = _cam.cpu().detach().numpy()
            cam = cam + _cam
            label = label + _score[:, 1:].cpu().detach()
        cam = cam / 4
        label = label / 4
        label = torch.sigmoid(label)

    cam = cam * (label > 0.3).clone().view(20, 1, 1).numpy()
    cam_max = np.max(cam, (1, 2), keepdims=True)
    cam_min = np.min(cam, (1, 2), keepdims=True)
    norm_cam = (cam - cam_min - 1e-5) / (cam_max - cam_min + 1e-5)
    norm_cam[norm_cam <= 1e-5] = 0

    cam = visualization_cam.visualize(image, norm_cam)
    cam = (cam * 255).astype(np.uint8)

    # Infer Affinity CAM
    model = getattr(importlib.import_module("network.resnet38_aff"), 'Net')()
    model.load_state_dict(torch.load("/content/drive/MyDrive/WSSS/Project/experiments/models/train_affinity_net8.pth"),
                          strict=False)
    model.cuda()
    model.eval()

    img = img_list
    orig_shape = img.shape
    padded_size = (int(np.ceil(img.shape[2] / 8) * 8), int(np.ceil(img.shape[3] / 8) * 8))

    p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
    img = F.pad(img, p2d)

    dheight = int(np.ceil(img.shape[2] / 8))
    dwidth = int(np.ceil(img.shape[3] / 8))

    cam_full_arr = np.zeros((21, orig_shape[2], orig_shape[3]), np.float32)
    for i in range(20):
        cam_full_arr[i + 1] = norm_cam[i]
    cam_full_arr[0] = (1 - np.max(cam_full_arr[1:], (0), keepdims=False)) ** 6
    cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

    with torch.no_grad():
        aff_mat = torch.pow(model.forward(img.cuda(), True), 8)

        trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
        for _ in range(6):
            trans_mat = torch.matmul(trans_mat, trans_mat)

        cam_full_arr = torch.from_numpy(cam_full_arr)
        cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)
        cam_vec = cam_full_arr.view(21, -1)

        cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
        cam_rw = cam_rw.view(1, 21, dheight, dwidth)

        cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)
        cam_rw = np.array(cam_rw[0][1:, :, :].cpu(), dtype=np.float32)

    cam_rw = visualization_cam.visualize(image, cam_rw)
    cam_rw = (cam_rw * 255).astype(np.uint8)
    return cam, cam_rw


if __name__ == '__main__':
    predict()
