import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os.path
import scipy.misc
from torchvision import transforms


IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
SEG_FOLDER_NAME = "SegmentationClass"
CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']
CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))


class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)
    
    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)
        
        return data


def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME,img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab


def load_image_label_list_from_xml(img_name_list, voc12_root):
    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]


def load_image_label_list_from_npy(img_name_list):
    cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')


def get_mask_path(img_name, voc12_root):
    return os.path.join(voc12_root, SEG_FOLDER_NAME, img_name + '.png')


def load_img_name_list(dataset_path):
    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]
    return img_name_list


class VOC_Dataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, with_name=True, with_label=False, with_mask=False):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.with_name = with_name
        self.with_label = with_label
        self.with_mask = with_mask

        if with_label:
          self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __len__(self):
        return len(self.img_name_list)

    def get_image(self, name):
      return Image.open(get_img_path(name, self.voc12_root)).convert('RGB')

    def get_label(self, name):
      return load_image_label_from_xml(name, self.voc12_root)
    
    def get_mask(self, name):
        mask_path = get_mask_path(name, self.voc12_root)
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        return mask

    def __getitem__(self, idx):
        data = []
        name = self.img_name_list[idx]

        if self.with_name:
          data.append(name)
        
        data.append(self.get_image(name))
        
        if self.with_label:
          data.append(self.label_list[idx])

        if self.with_mask:
          data.append(self.get_mask(name))

        return data


class VOC_Dataset_For_Classification(VOC_Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, with_name=False, with_label=True)
        self.transform = transform

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class VOC_Dataset_For_Evaluation(VOC_Dataset):

    def __init__(self, img_name_list_path, voc12_root, image_transform=None, mask_transform=None):
        super().__init__(img_name_list_path, voc12_root, with_name=False, with_label=True, with_mask=True)
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx):
        image, label, mask = super().__getitem__(idx)
        
        if self.image_transform is not None and self.mask_transform is not None:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        return image, label, mask


class VOC_Dataset_For_CAM_Inference(VOC_Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, with_label=True)
        self.transform = transform

    def __getitem__(self, idx):
        name, image, label = super().__getitem__(idx)

        if self.transform is not None:
            image = self.transform(image)

        return name, image, label


def train_data_loader_for_classification(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)

    tsfm_train = transforms.Compose([transforms.Resize((input_size, input_size)),  
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.RandomCrop(crop_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                    ])

    train_dataset = VOC_Dataset_For_Classification(img_name_list_path=args.train_list, voc12_root=args.voc12_root, transform=tsfm_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    return train_loader


def val_data_loader_for_classification(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)

    tsfm_val_image = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean_vals, std_vals),
                                        ])
    
    tsfm_val_mask = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                        transforms.ToTensor(),
                                       ])

    val_dataset = VOC_Dataset_For_Evaluation(img_name_list_path=args.val_list, voc12_root=args.voc12_root, image_transform=tsfm_val_image, mask_transform=tsfm_val_mask)
    image, label, mask = val_dataset[0]
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    return val_loader

def data_loader_for_cam_inference(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    tsfm_infer = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                    ])
    infer_dataset = VOC_Dataset_For_CAM_Inference(img_name_list_path=args.infer_list, voc12_root=args.voc12_root, transform=tsfm_infer)
    infer_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    return infer_loader