# Imports
import time

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

from models import *
from utils import *

weights_path = "weights.pth.tar"

# Params
desired_fps = 10
delay = int(1000 / desired_fps)  # Convert desired_fps to milliseconds
image_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running on: ',device)

# Instantiate model
model = VanillaUNet(in_channels=3,num_classes=11)
load_checkpoint(weights_path,model)
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.ToPILImage(),                # Convert numpy array to PIL Image
    transforms.Resize((512, 512)),          # Resize to match the input size of your CNN
    transforms.ToTensor(),                   # Convert PIL Image to tensor
])

# Infinite Face Detection Loop
v_cap = cv2.VideoCapture(0)
v_cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size)
v_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size)
# Check if the webcam is opened successfully
if not v_cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
start = time.time()
prev = 0
while(True):
    time_elapsed = time.time()-prev
    ret,frame = v_cap.read()
    # Preprocess the frame
    input_tensor = transform(frame).unsqueeze(0)
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    segmented_masks = generate_segmentation_masks(output)
    segmented_image = generate_image_from_masks(segmented_masks)
    segmented_image = cv2.resize(segmented_image,(frame.shape[1],frame.shape[0]))

    # Concatenate the original frame and segmented image horizontally
    combined_frame = cv2.hconcat([frame, segmented_image])
    # Display the combined frame in a window
    cv2.imshow('Original vs Segmented',combined_frame)

    if cv2.waitKey(delay) & 0xFF==ord('q'):
        break

v_cap.release()
cv2.destroyAllWindows()




