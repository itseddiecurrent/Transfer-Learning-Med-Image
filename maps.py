import torch
from torch._C import dtype
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial

from PIL import Image
import numpy as np
import os
import glob
import skimage.io
#from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision.transforms as transforms
#import glob

class Maps(torch.utils.data.Dataset):
  def __init__(self, root_dir, roi_type='constant'):
    self.img_data = []
    for root, dirs, files in os.walk(top=root_dir):
      for file in files:          
        self.img_data.append(root_dir+"/"+file)
        

    print(self.img_data[0])
    #self.data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize()])
    self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(256,256))])
    #print("number of images and masks: ", len(self.img_data), len(self.mask_data))
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
    img = skimage.io.imread(data_path)
    h, w, c = img.shape
    print(h,w)
    map_split = int(w/2)
    data = self.transform(img[:,:map_split])
    target = self.transform(img[:,map_split:])
    print("verify min and max:", torch.max(data), torch.min(data))
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

  root_dir = "maps/train"
  loaded_data = Maps(root_dir=root_dir)
  data, target = loaded_data.__getitem__(1)

