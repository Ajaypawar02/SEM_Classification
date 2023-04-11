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




model = models.vgg19(pretrained = True)

for p in model.parameters() : 
    p.requires_grad = False 
model.classifier = nn.Sequential(
nn.Linear(in_features=25088, out_features=2048) ,
nn.ReLU(),
nn.Linear(in_features=2048, out_features=512) ,
nn.ReLU(),
nn.Dropout(p=0.6), 
nn.Linear(in_features=512 , out_features=5),
# nn.LogSoftmax(dim=1)  
)