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
from model import model
import warnings
warnings.filterwarnings("ignore")



loss_list = []
loss_fn = nn.CrossEntropyLoss()

if __name__ == "__main__":
    
    
    df = train_csv
    
    train_df, valid_df = train_test_split(df, random_state = 42 )
    
    train_df = train_df.reset_index(drop = True)
    
    valid_df = valid_df.reset_index(drop = True)
        
    train_dataset = Data_class(train_df, 256)
    
    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
    
    valid_dataset = Data_class(valid_df, 256)
    
#     efficient_transformer = Linformer(
#     dim=128,
#     seq_len=49+1,  # 7x7 patches + 1 cls-token
#     depth=12,
#     heads=8,
#     k=64
#     )

#     model = ViT(
#         dim=128,
#         image_size=224,
#         patch_size=32,
#         num_classes=5,
#         transformer=efficient_transformer,
#         channels=3,
#     ).to(device)

#     model = Network()

    
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

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    model.to(device)
    
    valid_loader = DataLoader(valid_dataset, batch_size = 16, shuffle = True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)
    
    epochs = 100
    
    best_val = -1
    for epoch in range(epochs):
        loss = train_fn(train_loader, model, optimizer, device)
        print(loss)
        loss_list.append(loss)
        final_out, final_tar = eval_fn(valid_loader, model, device)
        print("================loss============", loss)
        print(metrics.classification_report(final_out, final_tar))
        
        print("================validation===========")
    
        f1_scores = f1_score(final_tar, final_out, average=None, labels=labels)
    
        print("=============f1_scores======================", f1_scores)
        
        f1_mean = f1_scores.mean()
        
        
        print("===============f1_mean=============", f1_mean)
        
        
        if best_val < f1_mean:
            print("======saving model============")
            best_val = f1_mean
            torch.save(model.state_dict(), args.PATH_MODEL_SAVED)



