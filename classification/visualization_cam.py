import cv2
import numpy as np
import os
from PIL import Image


def relu(x):
  return x * (x > 0)


def norm_cam(x, epsilon=1e-5):
    x = relu(x)
    c, h, w = x.shape
    flat_x = np.reshape(x, (c, (h * w)))
    max_value = np.reshape(flat_x.max(axis=-1),(c, 1, 1))
    
    return relu(x - epsilon) / (max_value + epsilon)


def colormap(cam, shape=None, mode=cv2.COLORMAP_JET):
    if shape is not None:
        h, w, c = shape
        cam = cv2.resize(cam, (w, h))
    cam = cv2.applyColorMap(cam, mode)
    
    return cam


def visualization(image_folder, predict_folder, name):
  predict_file = os.path.join(predict_folder,'%s.npy'%name)
  predict_dict = np.load(predict_file, allow_pickle=True).item()
  h, w = list(predict_dict.values())[0].shape
  cam = np.zeros((20,h,w),np.float32)
  
  for key in predict_dict.keys():
    cam[key] = predict_dict[key]
  
  cam = norm_cam(cam)
  cam = cam.max(axis=0)
  cam = (cam * 255).astype(np.uint8)

  image_file = os.path.join(image_folder,name+'.jpg')
  image = np.array(Image.open(image_file).convert('RGB'))
  h, w, c = image.shape
  
  cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
  cam = colormap(cam)
  
  image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::-1]
  image = image.astype(np.float32) / 255.
  
  return image

def visualize(image, cam):
  cam = norm_cam(cam)
  cam = cam.max(axis=0)
  cam = (cam * 255).astype(np.uint8)

  h, w, c = image.shape
  
  cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
  cam = colormap(cam)
  
  image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::-1]
  image = image.astype(np.float32) / 255.
  
  return image