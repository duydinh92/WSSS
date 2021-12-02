import numpy as np
import json
import sys
sys.path.append(r"/content/drive/MyDrive/WSSS/Project/classification/utils")
from general_utils import read_json


class Calculator_For_mIoU:
    def __init__(self, json_path):
        data = read_json(json_path)
        self.class_names = ['background'] + data['class_names']
        self.classes = len(self.class_names)

        self.clear()

    def get_data(self, pred_mask, gt_mask):
        obj_mask = gt_mask<255
        correct_mask = (pred_mask==gt_mask) * obj_mask
        
        P_list, T_list, TP_list = [], [], []
        for i in range(self.classes):
            P_list.append(np.sum((pred_mask==i)*obj_mask))
            T_list.append(np.sum((gt_mask==i)*obj_mask))
            TP_list.append(np.sum((gt_mask==i)*correct_mask))

        return (P_list, T_list, TP_list)

    def add_using_data(self, data):
        P_list, T_list, TP_list = data
        for i in range(self.classes):
            self.P[i] += P_list[i]
            self.T[i] += T_list[i]
            self.TP[i] += TP_list[i]

    def add(self, pred_mask, gt_mask):
        obj_mask = gt_mask<255
        correct_mask = (pred_mask==gt_mask) * obj_mask

        for i in range(self.classes):
            self.P[i] += np.sum((pred_mask==i)*obj_mask)
            self.T[i] += np.sum((gt_mask==i)*obj_mask)
            self.TP[i] += np.sum((gt_mask==i)*correct_mask)

    def get(self, detail=False, clear=True):
        IoU_dic = {}
        IoU_list = []

        for i in range(self.classes):
            IoU = self.TP[i]/(self.T[i]+self.P[i]-self.TP[i]+1e-10) * 100
            IoU_dic[self.class_names[i]] = IoU
            IoU_list.append(IoU)
      
        mIoU = np.mean(np.asarray(IoU_list))
        mIoU_foreground = np.mean(np.asarray(IoU_list)[1:])
        
        if clear:
            self.clear()
        
        if detail:
            return mIoU, mIoU_foreground, IoU_dic
        else:
            return mIoU, mIoU_foreground

    def clear(self):
        self.TP = []
        self.P = []
        self.T = []
        
        for _ in range(self.classes):
            self.TP.append(0)
            self.P.append(0)
            self.T.append(0)