# WSSS
This is the implementation of our deep learning project for weakly supervised semantic segmentation topic. Our work consists of two parts, first is generating class activation maps of images in the training dataset to make pseudo label for training segmentation, and then using these pseudo labels to train a segmentation model. We use code from https://github.com/VainF/DeepLabV3Plus-Pytorch to conduct the second phase of our work. We recommend to install our project in Google Colabotary with GPU session, and for it we have running code guides in file RunningCodeGuides.ipynb. 

## Experiments' results
All our experiments consists of pretrained model weights are included in https://drive.google.com/drive/folders/1-hJoRDkYxERWk2MmCK3zZSJxMW6EwuDZ?usp=sharing, you can use it for inference class activation maps or predict segmentation mask.

## Download PASCAL VOC 2012 dataset 
Download dataset in https://drive.google.com/file/d/1Ysl6LUjaxEnt8yQQZs09QDA3nAV5B2EJ/view?usp=sharing and then unzip it.

## Note
Our code is built in Colab with GPU session and make training and inference with GPU so before running it, please make sure your CUDA device is turn on.

## Install python dependencies
```bash
pip install -r requirements.txt
```

## Train ResNet with classification task to generate CAMs
```bash
python classification/train_res+er+pcm.py --voc12_root data_dir --train_list voc12/train_aug.txt --val_list voc12/train.txt
```
## Use trained ResNet to infer CAMs and CAMs processed with dCRFs to have high confidence object score and high confidence background score
```bash
!python classification/infer_cam.py --voc12_root data_dir --infer_list voc12/train_aug.txt --weights trained_resnet_dir
```
## 
