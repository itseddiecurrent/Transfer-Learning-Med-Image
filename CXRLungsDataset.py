# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b8c6QhDcQsTktz4qExDTJK8iFH9hwvDN
"""

'''! pip install kaggle
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download nikhilpandey360/chest-xray-masks-and-labels

! unzip chest-xray-masks-and-labels.zip
! rm chest-xray-masks-and-labels.zip'''

import torch
from torch._C import dtype
from torch.utils.data import Dataset
#from torchvision import datasets
#from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
#import os
#import pandas as pd
#from torchvision.io import read_image
from pathlib import Path
from functools import partial
#import cv2
from PIL import Image
import numpy as np
import os
import glob
#import sklearn.neighbors as knn
import skimage.io
#from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision.transforms as transforms
#import glob

class CXRLungs(torch.utils.data.Dataset):
  def __init__(self, root_dir, roi_type='constant'):
    self.img_data = []
    self.mask_data = []
    for root, dirs, files in os.walk(top=root_dir+"/CXR_png"):
      for file in files:
        self.img_data.append(root_dir+"/"+"CXR_png"+"/"+file)
        start = file.split(".")[0]
        ext = "_mask.png"
        self.mask_data.append(root_dir+"/masks"+"/"+start+ext)
    '''for im in self.img_data:
      splitpath = os.path.split(im)[1]
      dir_path = splitpath[0]
      mask_dir = "masks"
      file = os.path.splitext(splitpath[-1])[0]
      self.mask_data.append(os.path.join(dir_path, mask_dir, file+"_mask.png"))
      #filepath = os.path.join(os.path.join(splitpath[:-2]), os.path.splitext(splitpath[-1])[0]+"_mask.png")'''
    print(self.img_data[0])
    print(self.mask_data[0])
    #self.data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize()])
    self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(256,256))])
    try:
      assert len(self.img_data) == len(self.mask_data)
    except:
      raise ValueError
      print("error, number of training images does not match training targets")
    print("number of images and masks: ", len(self.img_data), len(self.mask_data))
    if roi_type == 'constant':
      self.roi_dim = (50,50)
    elif roi_type == "dynamic":
      self.roi_dim = None
    else:
      raise ValueError("ROI type should be 'constant' or 'dynamic' not {}".format(roi_type))

  def __len__(self):
    return len(self.img_data)

  def __getitem__(self, i):
    data_path = self.img_data[i]
    mask_path = self.mask_data[i]
    data_img = skimage.io.imread(data_path)
    mask_img =  skimage.io.imread(mask_path)
    data = self.transform(data_img)
    target = self.transform(mask_img)
    print("verify min and max:", torch.max(data), torch.min(data))
    print(data.dtype, target.dtype)
    #we don;t need h5py  as we are not working with h5py data, 
    '''with h5py.File(pngs, 'r'), h5py.File(masks, 'r') as png_data, mask_data:
        input = png_data.value            
        target = mask_data.value.astype(np.uint8)
        # Padding is done to compensate for the convolution
        input = np.pad(input,((5,5),(5,5)), mode='constant')
        target = np.pad(target,((5,5),(5,5)),mode='constant')
        ##TODO return a crop version for the ROI region of interest'''
    return (data, target)

'''def create_training_datasets(img_path, mask_path):
    train_data = TrainData(img_path, mask_path)
    return train_data'''

#PNGPATH = "data/Lung Segmentation/CXR_png"
#MASKPATH = "data/Lung Segmentation/masks"
if __name__=="__main__":

  root_dir = "LungSegmentation"
  loaded_data = CXRLungs(root_dir=root_dir)
  data, target = loaded_data.__getitem__(1)
  arr_ = np.array(data)
  tar_ = np.array(target)
  fig, ax = plt.subplots(1,2)
  ax[0].imshow(arr_[0,:,:], cmap='gray')
  ax[1].imshow(tar_[0,:,:], cmap='gray')
  plt.show()
