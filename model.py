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

import torch
import torch.nn as nn
import torch.nn.functional as F

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout):
        super().__init__()

        assert image_size % patch_size == 0, "image size must be divisible by the patch size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout) 
            for _ in range(depth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).permute(0, 2, 1)
        b, n, c = x.size()
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]
        
        for layer in self.transformer:
            x = layer(x)
            
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        
        return x


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