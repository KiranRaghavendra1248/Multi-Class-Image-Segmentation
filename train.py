# Imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
import os
import random
from tqdm import tqdm
from PIL import Image
import math
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils import *
from models import *


# Create simple transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the image
    transforms.ToTensor(),  # Convert PIL image to tensor
])

# Create datasets
dataset = CelebAMask("/kaggle/input/celebamaskhq/CelebAMask-HQ/CelebA-HQ-img",
                    "/kaggle/input/collated-masks/collated_masks",
                    transform)

train_dataset, test_dataset = get_train_test_datasets(dataset)

# Params
learning_rate = 0.001
num_epochs = 5
num_workers = 2
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running on: ',device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate model
model = VanillaUNet(in_channels=3,num_classes=11)

# Define Criterion, Loss function, LR Scheduler
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

train_loop(model,train_loader,criterion,optimizer,device, scheduler, num_epochs)

test_loop(model,test_loader,criterion,device)