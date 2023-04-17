import os
import pandas as pd
import torch 
import numpy as np
import re
import cv2
import torch.nn.functional as F
import uuid
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
import pathlib
from model import model

import warnings
warnings.filterwarnings("ignore")



from fastapi import FastAPI, File, UploadFile

app = FastAPI()


IMAGEDIR = "images/"
def classify(image):
    # image_path = open(image, "rb")
    image = cv2.imread(image.name)
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

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # file.filename = f"{uuid.uuid4()}.jgp"
    # return {"filename": file.filename}

    # with open(f"{r'C:\Users\ajayp\OneDrive\Desktop\College_project\images\'}{file.filename}", "wb") as f:
    #     f.write(contents)
    file.filename = f"{uuid.uuid4()}.jpg"
    contents =  await file.read()
    
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)

    path = r"C:\Users\ajayp\OneDrive\Desktop\College_project\images"
    image_path = os.path.join(path, file.filename)
    
    model.load_state_dict(torch.load(r"C:\Users\ajayp\OneDrive\Desktop\College_project\model_weights.pth"))
    model.eval()
    model.cuda()
    
    print("Model has been Loaded")

    # image_path = r"C:\Users\ajayp\OneDrive\Desktop\College_project\images"  + file.filename
    # image_path = r"C:\Users\ajayp\OneDrive\Desktop\College_project\images\0e0daea9-b925-43e2-862c-430717f9fdfc.jpg"
    print("Image Loading......")
    image = cv2.imread(image_path)
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

    return {"Classified Image" : response}

if __name__ == "__main__":
    app.run()