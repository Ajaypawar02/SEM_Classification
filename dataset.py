import os
import pandas as pd
import torch 
import numpy as np
import re
import cv2
import torch.nn.functional as F

from torchvision import models 
from tqdm import tqdm
import torch

import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

class Data_class(Dataset):
    def __init__(self, train_df, input_size):
        self.train_df = train_df
        self.input_size = input_size
        
    def __len__(self):
        return len(self.train_df)
    
    def __getitem__(self, index):
        name = self.train_df["name"][index]
        path = self.train_df["path"][index]
        label = self.train_df["label"][index]
        label_name = self.train_df["label_name"][index]
        
#         print(path)
#         print(name)
#         print(path + '/' + name + '/' + ".jpg")

#         print(name)

        image = cv2.imread(path + '/' + name  + ".jpg")
#         print(image)
        image = cv2.resize(image, (self.input_size, self.input_size))
    
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        return {
            "image" : torch.tensor(image), 
            "label" : torch.tensor(label), 
            "name" : name,
            "label_name" : label_name
        }

# get sample images
# temp_dataset = Data_class(train_csv, 256)
# image = temp_dataset.__getitem__(0)["image"]
# target = temp_dataset.__getitem__(0)["label"]
# image.shape