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
import torch.quantization

from utils import *

class CelebAMask(Dataset):
    def __init__(self, images_dir, masks_dir,transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = os.listdir(images_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir,self.image_files[index])
        mask_paths = retrive_masks(img_path,self.masks_dir)
        # Open image
        image = Image.open(img_path)
        # Open masks - create 1 mask or N masks???? 1 mask for ground truth and N masks from CNN seems like a better design choice
        masks = []
        for mask_path in mask_paths:
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
                masks.append(mask_img)
            else:
                masks.append(np.zeros((512, 512)))
        composite_image = np.zeros((512, 512),dtype=np.uint8)
        # Assign unique labels to the pixels in the combined mask
        for i, mask in enumerate(masks, start=1):
            mask_color = np.zeros((512, 512),dtype=np.uint8)
            mask_color[mask == 255] = i
            # Update composite image only where it hasn't been modified yet
            composite_image[np.where((mask_color != 0) & (composite_image == 0))] = mask_color[np.where((mask_color != 0) & (composite_image == 0))]
        return self.transform(image), torch.tensor(composite_image)


class Lapa(Dataset):
    def __init__(self, images_dir, masks_dir,
                 transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = os.listdir(images_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir,self.image_files[index])
        mask_path = img_path.replace("images","labels").replace("jpg", "png")
        # Open image
        image = Image.open(img_path)
        # Open masks - create 1 mask or N masks???? N mask for ground truth and N masks from CNN seems like a better design choice
        mask_img = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.resize(mask_img,(512, 512))
        # Split the masks
        binary_masks = []

        # Iterate over each class
        for class_value in range(0, 11):
            # Create a binary mask for the current class
            binary_mask = (mask_img == class_value).astype(np.float32)
            binary_masks.append(binary_mask)
        # convert to tensor
        torch_tensors_list = [torch.tensor(mask)for mask in binary_masks]
        return self.transform(image), torch.stack(
            torch_tensors_list, dim=0)