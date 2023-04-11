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

def train_fn(data_loader, model, optimizer, device):
    
    model.train()
    
    final_loss = 0
    
    for data in tqdm(data_loader, total = len(data_loader)):
        inputs = data["image"]
        targets = data["label"]
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = loss_fn(outputs, targets)
#         print(loss)
        
        final_loss += loss.item()
        
        loss.backward()
        
        optimizer.step()
        
        
    return final_loss/len(data_loader)
        
        
        
def eval_fn(data_loader, model, device):
    
    model.eval()
    
    final_targets = []
    
    final_outputs = []
    
    with torch.no_grad():
            
        for data in tqdm(data_loader, total = len(data_loader)):
            inputs = data["image"]
            targets = data["label"]
            inputs = inputs.to(device)
            targets = targets.to(device)

            output = model(inputs)
            
            output = F.softmax(output, dim = -1)
            
            ans = torch.argmax(output, dim = -1)
            
            targets = targets.detach().cpu().numpy().tolist()
            
            ans = ans.detach().cpu().numpy().tolist()
            
            final_targets.extend(targets)
#             print(ans)
            
            final_outputs.extend(ans)
            
    return final_outputs, final_targets


