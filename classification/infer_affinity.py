import torch
import torchvision
import argparse
import importlib
import numpy as np
from scipy import misc
import torch.nn.functional as F
import os.path
import imageio
from PIL import Image
import cv2

import sys
sys.path.append(r"/content/drive/MyDrive/WSSS")
import dataset
from dataset import data
import network
from utils import imutils


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="/content/drive/MyDrive/WSSS/Project/experiments/models/train_affinity_net8.pth", type=str)
    parser.add_argument("--network", default="network.resnet38_aff", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--cam_dir", default="/content/drive/MyDrive/WSSS/Project/cam", type=str)
    parser.add_argument("--voc12_root", default='/content/VOC2012_train_val/VOC2012_train_val', type=str)
    parser.add_argument("--alpha", default=6, type=float) 
    parser.add_argument("--out_label", default='/content/drive/MyDrive/WSSS/Project/pseudo_label', type=str)
    parser.add_argument("--beta", default=8, type=int) 
    parser.add_argument("--logt", default=6, type=int)  
    parser.add_argument("--crf", default=0, type=int) 

    args = parser.parse_args()

    if not os.path.exists(args.out_label):
        os.makedirs(args.out_label)

    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights), strict=False)

    model.eval()
    model.cuda()

    infer_data_loader = data.data_loader_for_cam_inference(args)
    palette = imutils.get_palette()

    for iter, (name, img, _) in enumerate(infer_data_loader):
        name = name[0]
        print(iter)

        orig_shape = img.shape
        padded_size = (int(np.ceil(img.shape[2]/8)*8), int(np.ceil(img.shape[3]/8)*8))

        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)

        dheight = int(np.ceil(img.shape[2]/8))
        dwidth = int(np.ceil(img.shape[3]/8))

        cam = np.load(os.path.join(args.cam_dir, name + '.npy'), allow_pickle=True).item()

        cam_full_arr = np.zeros((21, orig_shape[2], orig_shape[3]), np.float32)
        for k, v in cam.items():
            cam_full_arr[k+1] = v
        test = cam_full_arr
        cam_full_arr[0] = (1 - np.max(cam_full_arr[1:], (0), keepdims=False))**args.alpha
        cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

        with torch.no_grad():
            aff_mat = torch.pow(model.forward(img.cuda(), True), args.beta)

            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
            for _ in range(args.logt):
                trans_mat = torch.matmul(trans_mat, trans_mat)

            cam_full_arr = torch.from_numpy(cam_full_arr)
            cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)
            cam_vec = cam_full_arr.view(21, -1)

            cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
            cam_rw = cam_rw.view(1, 21, dheight, dwidth)
       
            cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)

            if args.crf>0:    
                img_8 = img[0].numpy().transpose((1,2,0))
                img_8 = np.ascontiguousarray(img_8)
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                img_8[:,:,0] = (img_8[:,:,0]*std[0] + mean[0])*255
                img_8[:,:,1] = (img_8[:,:,1]*std[1] + mean[1])*255
                img_8[:,:,2] = (img_8[:,:,2]*std[2] + mean[2])*255
                img_8[img_8 > 255] = 255
                img_8[img_8 < 0] = 0
                img_8 = img_8.astype(np.uint8)
                cam_rw = cam_rw[0].cpu().numpy()
                cam_rw = imutils.crf_inference(img_8, cam_rw, args.crf) 
                cam_rw = torch.from_numpy(cam_rw).view(1, 21, img.shape[2], img.shape[3]).cuda()

            # def relu(x):
            #     return x * (x > 0)


            # def norm_cam(x, epsilon=1e-5):
            #     x = relu(x)
            #     c, h, w = x.shape
            #     flat_x = np.reshape(x, (c, (h * w)))
            #     max_value = np.reshape(flat_x.max(axis=-1),(c, 1, 1))
    
            #     return relu(x - epsilon) / (max_value + epsilon)


            # def colormap(cam, shape=None, mode=cv2.COLORMAP_JET):
            #     if shape is not None:
            #         h, w, c = shape
            #         cam = cv2.resize(cam, (w, h))
            #     cam = cv2.applyColorMap(cam, mode)
    
            #     return cam
            
            # cam = norm_cam(np.array(cam_rw[0][1:,:,:].cpu(),dtype=np.float32))
            # cam = cam.max(axis=0)
            # cam = (cam * 255).astype(np.uint8)

            # image_file = os.path.join("/content/VOC2012_train_val/VOC2012_train_val/JPEGImages",name+'.jpg')
            # image = np.array(Image.open(image_file).convert('RGB'))
            # h, w, c = image.shape
            
            # cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
            # cam = colormap(cam)
            
            # image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::-1]
            # image = image.astype(np.float32) / 255.
            # cv2.imwrite(os.path.join("/content/drive/MyDrive/WSSS/Project/test/test/cam", "%s.png" % name), ((1-image)*255).astype(np.uint8))
            
            _, cam_rw_pred = torch.max(cam_rw, 1)
            res = np.uint8(cam_rw_pred.cpu().data[0])[:orig_shape[2], :orig_shape[3]]
            pred_map = Image.fromarray(res)
            pred_map.putpalette(palette)
            pred_map.save(os.path.join(args.out_label, "%s.png" % name))
