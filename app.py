import streamlit as st
import os
from PIL import Image, ImageOps
import torchvision
import torch

import sys  
sys.path.append(r"/content/drive/MyDrive/WSSS/Project/classification")
from classification import predict


st.sidebar.title("""
    Class activation map generator
""")
content_file = st.sidebar.file_uploader(
    "Please upload content image file", type=["jpg", "jpeg", "png"])
if content_file is None:
  st.sidebar.text("You need to upload content image")
if st.sidebar.button('Submit'):
    content_image = Image.open(content_file).convert('RGB')
    cam, cam_aff = predict.predict(content_image)
    cam = Image.fromarray(cam)
    cam_aff = Image.fromarray(cam_aff)
    img_list = [content_image, cam, cam_aff]
    st.image(img_list, width=400)
