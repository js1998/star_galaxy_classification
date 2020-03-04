from astropy.io import fits
import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from astropy.nddata.utils import Cutout2D
import glob, os
from esutil import wcsutil
from astropy.table import Table
import pandas as pd
import fitsio
from esutil import wcsutil
import seaborn as sns
import matplotlib.pyplot as plt
import collections
from random import randint
import math
import sys
from astropy.visualization import hist
from torch.utils.data import DataLoader
from astroML.datasets import fetch_imaging_sample, fetch_sdss_S82standards

class PitcairnDataset(data.Dataset):
    def __init__(self, config, mode):
        # mode is "train", "validation" or "test"
        self.config = config
        print("Loading PitcairnDataset")
        data, label = load_data()
        self.data = data
        self.label = label
        print("done.")
    
    def __len__(self):
        return (self.data.shape[0])
    
    def __getitem__(self, index):
        data_cur = self.data[index]
        # data
        data_cur = torch.from_numpy(data_cur.astype(np.float32))
        # label
        label_cur = self.label[index]
        return data_cur, label_cur

def load_data():
    # read input images
    os.chdir("/home/yufeng/scratch/cfis/input_image")
    data = []
    label = []
    for file in glob.glob("*.fits.fz"):
        file_id = file[:-8]
        data_image = fits.open(file, memmap=True)
        label_image = np.array(torch.load('/home/yufeng/projects/rrg-kyi/yufeng/image_classification/data/label_image/'+file_id+'.pt'))
        x_slice = 264
        y_slice = 258
        x_num = 8
        y_num = 18
        for i in range(x_num):
            for j in range(y_num):
                x = np.array(data_image[0].data.T[i*x_slice:(i+1)*x_slice, j*y_slice:(j+1)*y_slice])
                y = np.array(label_image[i*x_slice:(i+1)*x_slice, j*y_slice:(j+1)*y_slice,:])
                if not np.all(y==0.0):
                    data += [x]
                    label += [y]
        break
    data = np.array(data)
    data = np.transpose(data.reshape(data.shape[0], data.shape[1], data.shape[2], 1), (0, 3, 1, 2))
    label = np.array(label)
    label = np.transpose(label, (0, 3, 1, 2))
    data = data[:, :, :256, :256]
    label = label[:, :, :256, :256]
    print(np.array(data).shape)
    print(np.array(label).shape)
    return data, label