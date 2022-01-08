import os
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
import importlib

import sys
sys.path.append(r"/content/drive/MyDrive/WSSS/Project")
from dataset import data
import network
from utils import evaluate_utils, general_utils,  train_utils, imutils
from utils.evaluate_utils import *
from utils.general_utils import *
from utils.train_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument("--network", default="network.resnet38_aff", type=str)
    parser.add_argument('--voc12_root', default='/content/VOC2012_train_val/VOC2012_train_val', type=str)
    parser.add_argument('--train_list', default='/content/drive/MyDrive/WSSS/Project/voc12/train_aug.txt', type=str)
    parser.add_argument('--batch_size', default=8, type=int) 
    parser.add_argument('--max_epoches', default=8, type=int) 
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--wt_dec', default=1e-4, type=float)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--la_crf_dir", default="/content/drive/MyDrive/WSSS/Project/crf_4.0", type=str)
    parser.add_argument("--ha_crf_dir", default="/content/drive/MyDrive/WSSS/Project/crf_24.0", type=str)
    parser.add_argument("--weights", default="/content/drive/MyDrive/WSSS/Project/ilsvrc-cls_rna-a1_cls1000_ep-0001.params", type=str)
    parser.add_argument('--tag', default='test', type=str)
    args = parser.parse_args()

    # General settings
    log_dir = create_directory(f'./experiments/logs/')
    model_dir = create_directory('./experiments/models/')
    checkpoint_dir = create_directory('./experiments/checkpoints/')
    model_path = model_dir + f'{args.tag}.pth'
    checkpoint_path = checkpoint_dir + f'{args.tag}.pth'
    log_path = log_dir + f'{args.tag}.txt'
    meta_dic = read_json('/content/drive/MyDrive/WSSS/Project/voc12/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/')
    set_seed(args.seed)

    log_func = lambda string='': log_print(string, log_path)
    log_func('[i] {}'.format(args.tag))
    log_func()

    # Model
    load_checkpoint_fn = lambda: load_model(model, checkpoint_path)
    save_checkpoint_fn = lambda: save_model(model, checkpoint_path)
    save_model_fn = lambda: save_model(model, model_path)

    model = getattr(importlib.import_module(args.network), 'Net')()
    param_groups = model.get_parameter_groups()

    # Load train dataset
    train_dataset = data.VOC_Dataset_For_Aff(args.train_list, label_la_dir=args.la_crf_dir, label_ha_dir=args.ha_crf_dir,
                                               voc12_root=args.voc12_root, cropsize=args.crop_size, radius=5,
                    joint_transform_list=[
                        None,
                        None,
                        imutils.RandomCrop(args.crop_size),
                        imutils.RandomHorizontalFlip()
                    ],
                    img_transform_list=[
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray,
                        model.normalize,
                        imutils.HWC_to_CHW
                    ],
                    label_transform_list=[
                        None,
                        None,
                        None,
                        imutils.AvgPool2d(8)
                    ])

 
    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)

    # Optimizwe
    max_step = len(train_dataset) // args.batch_size * args.max_epoches    
    print("Max step: " +str(max_step))

    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
    ], lr=args.lr, momentum=0.9, weight_decay=args.wt_dec, max_step=max_step, nesterov=True)

    if args.weights[-7:] == '.params':
        import network.resnet38d
        assert args.network == "network.resnet38_aff"
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = model.cuda()
    model.train()

    # Train
    avg_meter = Average_Meter(['loss', 'bg_loss', 'fg_loss', 'neg_loss'])
    writer = SummaryWriter(tensorboard_dir)

    for ep in range(args.max_epoches):
       
        for iter, pack in enumerate(train_data_loader):

            aff = model.forward(pack[0].cuda())
     
            bg_label = pack[1][0].cuda(non_blocking=True)
            fg_label = pack[1][1].cuda(non_blocking=True)
            neg_label = pack[1][2].cuda(non_blocking=True)

            bg_count = torch.sum(bg_label) + 1e-5
            fg_count = torch.sum(fg_label) + 1e-5
            neg_count = torch.sum(neg_label) + 1e-5

            bg_loss = torch.sum(- bg_label * torch.log(aff + 1e-5)) / bg_count
            fg_loss = torch.sum(- fg_label * torch.log(aff + 1e-5)) / fg_count
            neg_loss = torch.sum(- neg_label * torch.log(1. + 1e-5 - aff)) / neg_count

            loss = bg_loss/4 + fg_loss/4 + neg_loss/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({
                'loss': loss.item(),
                'bg_loss': bg_loss.item(), 'fg_loss': fg_loss.item(), 'neg_loss': neg_loss.item()
            })


            if (optimizer.global_step) % 100 == 0:
                
                loss, bg_loss, fg_loss, neg_loss  = avg_meter.get(clear=True)
                iteration = optimizer.global_step
                learning_rate = optimizer.param_groups[0]['lr']

                data = {
                'iteration' : iteration,
                'learning_rate' : learning_rate,
                'loss' : loss,
                'bg_loss' : bg_loss,
                'fg_loss' : fg_loss, 
                'neg_loss' : neg_loss
                }

                writer.add_scalar('Train/loss', loss, iteration)
                writer.add_scalar('Train/bg_loss', bg_loss, iteration)
                writer.add_scalar('Train/fg_loss', fg_loss, iteration)
                writer.add_scalar('Train/neg_loss', neg_loss, iteration)
                writer.add_scalar('Train/learning_rate', learning_rate, iteration)
            
                log_func('[i] iteration={iteration:,}, learning_rate={learning_rate:.4f}, loss={loss:.4f}, bg_loss={bg_loss:.4f}, fg_loss={fg_loss:.4f}, neg_loss={neg_loss:.4f}'.format(**data)
                )
            del pack, bg_label, fg_label, neg_label, bg_count, fg_count, neg_count, bg_loss, fg_loss, neg_loss, loss
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        save_checkpoint_fn()
        if ep % 2 == 1:
          torch.save(model.state_dict(), model_dir+args.tag+str(ep+1)+".pth")
          
    print("Training done")
