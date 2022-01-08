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

sys.path.append(r"/content/drive/MyDrive/WSSS/Project")
from dataset import data
from network.resnet import ResNetUp
import network
from utils import evaluate_utils, general_utils,  train_utils
from utils.evaluate_utils import *
from utils.general_utils import *
from utils.train_utils import *


if __name__ == '__main__':
    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_classes', default=20, type=int)
    parser.add_argument('--voc12_root', default='/content/VOC2012_train_val/VOC2012_train_val', type=str)
    parser.add_argument('--train_list', default='/content/drive/MyDrive/WSSS/Project/voc12/train_aug.txt', type=str)
    parser.add_argument('--val_list', default='/content/drive/MyDrive/WSSS/Project/voc12/train.txt', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=3, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--input_size', default=512, type=int)
    parser.add_argument('--crop_size', default=384, type=int)
    parser.add_argument('--print_ratio', default=0.1, type=float)
    parser.add_argument('--tag', default='train_resnet18+up_crop384', type=str)
    args = parser.parse_args()
    
    # General settings
    log_dir = create_directory(f'./experiments/logs/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/')
    log_path = log_dir + f'{args.tag}.txt'
    model_path = model_dir + f'{args.tag}.pth'
    meta_dic = read_json('/content/drive/MyDrive/WSSS/Project/voc12/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])
    set_seed(args.seed)

    log_func = lambda string='': log_print(string, log_path)
    log_func('[i] {}'.format(args.tag))
    log_func()
    
    # Load train and val dataset
    train_loader = data.train_data_loader_for_classification(args)
    val_loader = data.val_data_loader_for_classification(args)
    val_iteration = len(train_loader)
    # log_iteration = int(val_iteration * args.print_ratio)
    log_iteration = 100
    max_iteration = args.max_epoch * val_iteration

    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))
    
    # Network
    load_model_fn = lambda: load_model(model, model_path)
    save_model_fn = lambda: save_model(model, model_path)

    model = ResNetUp(args.num_classes, pretrained=True)
    param_groups = model.get_parameter_groups(print_fn=None)
    model = model.cuda()
    model.train()

    log_func('[i] Total Params: %.2fM'%(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    log_func()

    
    log_func('[i] gpu : {}'.format(str(torch.cuda.get_device_name(0))))
    
    # Loss, Optimizer
    class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='none').cuda()
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

                _, cams, _ = model(images)

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
    train_meter = Average_Meter(['loss'])
    
    best_train_mIoU = -1
    thresholds = list(np.arange(0.10, 0.50, 0.05))
    
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = data.Iterator(train_loader)                                                          

    for iteration in range(max_iteration):
        images, labels = train_iterator.get()
        images, labels = images.cuda(), labels.cuda()

        _, _, scores = model(images)
        loss = class_loss_fn(scores, labels).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss' : loss.item()
        })

        del images, labels, scores, loss
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        # For Log
        if (iteration + 1) % log_iteration == 0:
            loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            data = {
                'iteration' : iteration + 1,
                'learning_rate' : learning_rate,
                'loss' : loss

            }

            log_func('[i] iteration={iteration:,}, learning_rate={learning_rate:.4f}, loss={loss:.4f}'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
        
        # For evaluation
        if (iteration + 1) % val_iteration == 0: 
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