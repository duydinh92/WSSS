import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
import cv2

import sys  
sys.path.append(r"/content/drive/MyDrive/WSSS/Project")
import dataset
from dataset import data


categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']


def pp_with_connected_components(pred, connectivity=4, THRESHOLD=1000):
    _, thresh = cv2.threshold(pred,0,255,cv2.THRESH_BINARY)
    num_labels, labels , stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

    for idx, stat in enumerate(stats):
        x, y, w, h, c = stat 
        if w == pred.shape[1] or h == pred.shape[0]:
            continue
        if c < THRESHOLD:
            pred[labels == idx] = 0
    
    return pred


def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, threshold=0.5, printlog=True, png=False):
    TP = []
    P = []
    T = []
    IoU = []
    p_list = []
    loglist = {}

    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))
    
    def calculate(start,step,TP,P,T,threshold):
        for idx in range(start,len(name_list),step):
            name = name_list[idx]
            if png:
                predict_file = os.path.join(predict_folder,'%s.png'%name)
                predict = np.array(Image.open(predict_file)) 
            else:
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                predict = np.zeros((21,h,w),np.float32)
          
                for key in predict_dict.keys():
                  predict[key] = predict_dict[key]
            
                predict[0,:,:] = threshold
                predict = np.argmax(predict, axis=0).astype(np.uint8)
                #predict = pp_with_connected_components(predict, THRESHOLD=100)
            
            gt_file = os.path.join(gt_folder,'%s.png'%name)
            gt = np.array(Image.open(gt_file))
            cal = gt<255
            mask = (predict==gt) * cal
      
            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)
                TP[i].release()

    for i in range(8):
        p = multiprocessing.Process(target=calculate, args=(i,8,TP,P,T,threshold))
        p.start()
        p_list.append(p)
    
    for p in p_list:
        p.join()

    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
  
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i]
               
    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou
    if printlog:
        for i in range(num_cls):
            if i%2 != 1:
                print('%11s:%7.3f'%(categories[i],IoU[i]),end='\t')
            else:
                print('%11s:%7.3f'%(categories[i],IoU[i]))
        print('\n======================================================')
        print('%11s:%7.3f'%('mIoU',miou))

    return loglist


def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  '%(key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)


def writelog(filepath, metric):
    filepath = filepath
    logfile = open(filepath,'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S\n'", time.localtime()))
    writedict(logfile, metric)
    logfile.write('=====================================\n') 
    logfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='voc12/train.txt', type=str)
    parser.add_argument("--predict_dir", default='test_rw', type=str)
    parser.add_argument("--voc12_root", default='/content/VOC2012_train_val/VOC2012_train_val', type=str)
    parser.add_argument('--logfile', default='experiments/evaluation/test.txt',type=str)
    parser.add_argument('--t', default=0.2, type=float)
    parser.add_argument('--png', default=True, type=bool)
    parser.add_argument('--curve', default=True, type=bool)
    args = parser.parse_args()

    name_list = data.load_img_name_list(args.list)
    gt_dir = os.path.join(args.voc12_root,'SegmentationClass')

    if not args.curve:
        loglist = do_python_eval(args.predict_dir, gt_dir, name_list, 21, args.t, png=args.png)
        writelog(args.logfile, loglist)
    else:
        l = []
        for i in range(0,5,5):
            t = i/100.0
            loglist = do_python_eval(args.predict_dir, gt_dir, name_list, 21, t, png=args.png)
            l.append(loglist['mIoU'])
            print('%d/60 background score: %.3f\tmIoU: %.3f'%(i, t, loglist['mIoU']))
        writelog(args.logfile, {'mIoU':l})
