import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import cv2
import matplotlib
import matplotlib.pyplot as plt
import kornia as K

sys.path.append(r"C:\Users\Admin\Desktop\2021\20211\DL\WSSS")
from dataset import data
from network.resnet import ResNet_ER_PCM
from utils import evaluate_utils, general_utils,  train_utils
from utils.evaluate_utils import *
from utils.general_utils import *
from utils.train_utils import *


if __name__ == '__main__':
    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_classes', default=21, type=int)
    parser.add_argument('--voc12_root', default='/content/VOC2012_train_val/VOC2012_train_val', type=str)
    parser.add_argument('--train_list', default='/content/drive/MyDrive/WSSS/Project/voc12/train1.txt', type=str)
    parser.add_argument('--val_list', default='/content/drive/MyDrive/WSSS/Project/voc12/train.txt', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epoch', default=8, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--input_size', default=648, type=int)
    parser.add_argument('--crop_size', default=512, type=int)
    parser.add_argument('--print_ratio', default=0.1, type=float)
    parser.add_argument('--tag', default='test', type=str)
    args = parser.parse_args()
    
    # General settings
    log_dir = create_directory(f'./experiments/logs/')
    model_dir = create_directory('./experiments/models/')
    checkpoint_dir = create_directory('./experiments/checkpoints/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/')
    log_path = log_dir + f'{args.tag}.txt'
    model_path = model_dir + f'{args.tag}.pth'
    checkpoint_path = checkpoint_dir + f'{args.tag}.pth'
    meta_dic = read_json('voc12/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])
    set_seed(args.seed)

    log_func = lambda string='': log_print(string, log_path)
    log_func('[i] {}'.format(args.tag))
    log_func()
    
    # Load train and val dataset
    train_loader = data.train_data_loader_for_classification(args)
    val_loader = data.val_data_loader_for_classification(args)
    val_iteration = 1#len(train_loader)
    log_iteration = 1#400
    max_iteration = args.max_epoch * val_iteration

    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))
    
    # Network
    load_checkpoint_fn = lambda: load_model(model, checkpoint_path)
    save_checkpoint_fn = lambda: save_model(model, checkpoint_path)
    save_model_fn = lambda: save_model(model, model_path)

    model = ResNet_ER_PCM(args.num_classes, pretrained=True)
    # load_checkpoint_fn()
    param_groups = model.get_parameter_groups(print_fn=None)
    

    model = model.cuda()
    model.train()

    log_func('[i] Total Params: %.2fM'%(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    log_func()

    
    log_func('[i] gpu : {}'.format(str(torch.cuda.get_device_name(0))))
    
    # Loss, Optimizer
    class_loss_fn = nn.MultiLabelSoftMarginLoss().cuda()
    log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
    log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
    log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
    log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))
    log_func()

    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
    ], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=True)
    
    # Define evaluation function
    def evaluate(loader):
        model.eval()
        meter_dic = {th : Calculator_For_mIoU('/content/drive/MyDrive/WSSS/Project/voc12/VOC_2012.json') for th in thresholds}

        with torch.no_grad():
            length = len(loader)
            for step, (images, labels, gt_masks) in enumerate(loader):
                images = images.cuda()

                cams = model.net_forward(images, mode='val')
                cams = cams[:,1:,:,:]

                for batch_index in range(images.size()[0]):
                    gt_mask = (get_numpy_from_tensor(gt_masks[batch_index].squeeze(0))*255).astype(np.uint8)
                    H, W = gt_mask.shape
                    cam = get_numpy_from_tensor(F.interpolate(cams[batch_index].unsqueeze(0), size=(H,W), mode='bilinear', align_corners=False)[0])
                    cam = cam * labels[batch_index].clone().view(20, 1, 1).numpy()
                    cam_max = np.max(cam, (1,2), keepdims=True)
                    cam_min = np.min(cam, (1,2), keepdims=True)
                    cam[cam < cam_min+1e-5] = 0
                    cam = (cam-cam_min-1e-5) / (cam_max - cam_min + 1e-5)
                    cam = cam * labels[batch_index].clone().view(20, 1, 1).numpy()
                    cam = cam.transpose((1,2,0))

                    for th in thresholds:
                        bg = np.ones_like(cam[:, :, 0]) * th
                        pred_mask = np.argmax(np.concatenate([bg[..., np.newaxis], cam], axis=-1), axis=-1).astype(np.uint8)
                        meter_dic[th].add(pred_mask, gt_mask)

                        del bg, pred_mask
                    
                    del gt_mask, cam, cam_max, cam_min

                del images, cams
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                   
                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
              
        print(' ')
        model.train()
        
        best_th = 0.0
        best_mIoU = 0.0

        for th in thresholds:
            mIoU, mIoU_foreground = meter_dic[th].get(clear=True)
            if best_mIoU < mIoU:
                best_th = th
                best_mIoU = mIoU

        return best_th, best_mIoU

    # Train
    train_meter = Average_Meter(['loss', 'loss_cls', 'loss_er', 'loss_ecr'])
    
    best_train_mIoU = -1
    thresholds = list(np.arange(0.10, 0.50, 0.05))
    
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = data.Iterator(train_loader)   
    scale_factor = 0.6
    current_iteration = 0                                                       

    for iteration in range(current_iteration, max_iteration):
        images, labels = train_iterator.get()
        N,_,_,_ = images.size()
        bg_score = torch.ones((N,1))
        labels = torch.cat((bg_score, labels), dim=1)
        images, labels = images.cuda(), labels.cuda().unsqueeze(2).unsqueeze(3)

        (score, cam, cam_rv), (affine_score, affine_cam, affine_cam_rv) = model.net_forward(images, mode='train')
        matrix = model.get_affine_matrix()
        score = score.unsqueeze(2).unsqueeze(3)
        affine_score = affine_score.unsqueeze(2).unsqueeze(3)
        
        loss_rvmin1 = train_utils.adaptive_min_pooling_loss((cam_rv*labels)[:,1:,:,:])
        cam = F.interpolate(train_utils.max_norm(cam),scale_factor=scale_factor,mode='bilinear',align_corners=False)*labels
        cam_rv1 = F.interpolate(train_utils.max_norm(cam_rv),scale_factor=scale_factor,mode='bilinear',align_corners=False)*labels

        loss_rvmin2 = train_utils.adaptive_min_pooling_loss((affine_cam_rv*labels)[:,1:,:,:])
        affine_cam = train_utils.max_norm(affine_cam)*labels
        affine_cam_rv = train_utils.max_norm(affine_cam_rv)*labels
        
        loss_cls1 = class_loss_fn(score[:,1:,:,:], labels[:,1:,:,:])
        loss_cls2 = class_loss_fn(affine_score[:,1:,:,:], labels[:,1:,:,:])
        loss_cls = (loss_cls1 + loss_cls2)/2 + (loss_rvmin1+loss_rvmin2)/2
        
        ns,cs,hs,ws = affine_cam.size()
        loss_er = torch.mean(torch.abs(K.geometry.transform.warp_perspective(cam[:,1:,:,:], matrix, (hs, ws), align_corners=False)*labels[:,1:,:,:] - affine_cam[:,1:,:,:]))
        
        cam[:,0,:,:] = 1-torch.max(cam[:,1:,:,:],dim=1)[0]
        affine_cam[:,0,:,:] = 1-torch.max(affine_cam[:,1:,:,:],dim=1)[0]
        tensor_ecr1 = torch.abs(train_utils.max_onehot(affine_cam.detach()) - K.geometry.transform.warp_perspective(cam_rv, matrix, (hs, ws), align_corners=False)*labels)
        tensor_ecr2 = torch.abs(train_utils.max_onehot(K.geometry.transform.warp_perspective(cam, matrix, (hs, ws), align_corners=False).detach())*labels - affine_cam_rv)
        loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns,-1), k=(int)(21*hs*ws*0.3), dim=-1)[0])
        loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns,-1), k=(int)(21*hs*ws*0.3), dim=-1)[0])
        loss_ecr = loss_ecr1 + loss_ecr2
        
        loss = loss_cls + loss_er + loss_ecr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss': loss.item(), 'loss_cls': loss_cls.item(), 'loss_er': loss_er.item(), 'loss_ecr': loss_ecr.item()
        })

        del images, labels, score, cam, cam_rv, affine_score, affine_cam, affine_cam_rv, loss_cls1, loss_cls2, loss_cls
        del matrix, loss_er, tensor_ecr1, tensor_ecr2, loss_ecr1, loss_ecr2, loss_ecr, loss
        del loss_rvmin1, loss_rvmin2
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        # For Log
        if (iteration + 1) % log_iteration == 0:
            loss, loss_cls, loss_er, loss_ecr = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            data = {
                'iteration' : iteration + 1,
                'learning_rate' : learning_rate,
                'loss' : loss,
                'loss_cls' : loss_cls,
                'loss_er' : loss_er, 
                'loss_ecr' : loss_ecr
            }

            log_func('[i] iteration={iteration:,}, learning_rate={learning_rate:.4f}, loss={loss:.4f}, loss_cls={loss_cls:.4f}, loss_er={loss_er:.4f}, loss_ecr={loss_ecr:.4f}'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/loss_cls', loss_cls, iteration)
            writer.add_scalar('Train/loss_er', loss_er, iteration)
            writer.add_scalar('Train/loss_ecr', loss_ecr, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
        
        # For evaluation
        if (iteration + 1) % val_iteration == 0:
            save_checkpoint_fn()

            threshold, mIoU = evaluate(val_loader)
            
            if best_train_mIoU == -1 or best_train_mIoU < mIoU:
                best_train_mIoU = mIoU

                save_model_fn()
                log_func('[i] save model')

            data = {
                'iteration' : iteration + 1,
                'threshold' : threshold,
                'train_mIoU' : mIoU,
                'best_train_mIoU' : best_train_mIoU
            }
            
            log_func('[i] iteration={iteration:,}, threshold={threshold:.2f}, train_mIoU={train_mIoU:.2f}%, best_train_mIoU={best_train_mIoU:.2f}%'.format(**data))
            
            writer.add_scalar('Evaluation/threshold', threshold, iteration)
            writer.add_scalar('Evaluation/train_mIoU', mIoU, iteration)
            writer.add_scalar('Evaluation/best_train_mIoU', best_train_mIoU, iteration)
      
    writer.close()

    print("Training done")