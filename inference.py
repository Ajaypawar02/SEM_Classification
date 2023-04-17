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





def classify(image):
    image_path = open(image, "rb")
    image = cv2.imread(image_path.name)
    image = cv2.resize(image, (224, 224))
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    
    image = torch.tensor(image)
    image = image.unsqueeze(0).cuda()
    print("Image has been loaded")
    
    out = model(image)
    
    print("Output has been generated")
    
    ans = F.softmax(out, dim = -1)
    final_out = torch.argmax(ans).item()
    
    if final_out == 0:
        response = "Classified image is Powder"
    elif final_out == 1:
        response = "Classified image is biological"
    elif final_out == 2:
        response = "Classified image is Porous Sponge"
    elif final_out == 3:
        response = "Classified image is Patterned Surfaces"
    elif final_out == 4:
        response = "Classified image is Tips"
    

    return response



if __name__ == "__main__":
    # IMAGEDIR = r"C:\Users\ajayp\OneDrive\Desktop\College_project\images"
    print("The process has been started")
    # model.load_state_dict(torch.load(r"C:\Users\ajayp\OneDrive\Desktop\College_project\model_weights.pth"))
    # model.eval()
    # model.cuda()
    print("Model has been loaded")
    IMAGEDIR = "images/"
    image = r"C:\Users\ajayp\OneDrive\Desktop\College_project\data\Patteerned_surfaces.jpg"
    image_path = os.join(image)
    file = open(image, "rb")

    contents =  file.read()
    name = file.name
    with open(f"{IMAGEDIR}{file.name}", "wb") as f:
        f.write(contents)
    # print(classify(image_path))

    return "loaded"


    