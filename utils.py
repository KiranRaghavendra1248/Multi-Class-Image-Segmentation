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

# Save checkpoint
def save_checkpoint(state,filename='weights.pth.tar'):
    print('Saving weights-->')
    torch.save(state,filename)

# Load checkpoint
def load_checkpoint(filename,model,optim=None):
    print('Loading weights-->')
    checkpoint = torch.load(filename,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    if optim!=None:
        optim.load_state_dict(checkpoint['optimizer'])


# We need to generate 1 single mask from all masks
# First we retrieve all masks associated with current image

default_categories = ["_hair", "_l_brow",
                      "_l_eye", "_l_lip",
                      "_mouth", "_nose",
                      "_r_brow", "_r_eye",
                      "_u_lip", "_skin"]

def retrive_masks(image_path, mask_dir,change_extension=True,categories=default_categories):
    image_filename = image_path.split("/")[-1]
    image_name, extension = image_filename.split(".")
    mask_name = image_name.zfill(5)
    mask_paths = []
    masks = categories
    for mask in masks:
        if (change_extension and extension == "png"):
            new_mask_name = mask_name + mask + "." + "jpg"
            mask_paths.append(os.path.join(mask_dir,new_mask_name))
        elif (change_extension and extension == "jpg"):
            new_mask_name = mask_name + mask + "." + "png"
            mask_paths.append(os.path.join(mask_dir,new_mask_name))
    return mask_paths

colors = [
    [0, 0, 0],
    [0, 153, 255],
    [102, 255, 153],
    [0, 204, 153],
    [255, 255, 102],
    [255, 255, 204],
    [255, 153, 0],
    [255, 102, 255],
    [102, 0, 51],
    [255, 204, 255],
    [255, 0, 102]]

def generate_image_from_masks(masks,original_image=None):
    if original_image:
        height, width = original_image.shape[:2]
    else:
        height, width = 512, 512
    composite_image = np.zeros((height, width, 3),dtype=np.uint8)
    # Overlay each mask on the composite image with its respective color
    for i, mask in enumerate(masks):
        mask_color = np.zeros((height, width, 3),dtype=np.uint8)
        mask_color[mask == 255] = colors[i]
        # Update composite image only where it hasn't been modified yet
        composite_image[np.where((mask_color != 0) & (composite_image == 0))] = mask_color[np.where((mask_color != 0) & (composite_image == 0))]
    return composite_image


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

# Create train and val datasets
def get_train_test_datasets(dataset):
    train_size = 0.8
    test_size = 0.2

    # Calculate the number of samples for each set
    num_samples = len(dataset)
    num_train = int(train_size * num_samples)
    num_test = num_samples - num_train

    # Use random_split to split into train and test sets
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

    return train_dataset, test_dataset

def create_image_from_output(output):
    _,indices = torch.max(output,dim=1)
    image_output = torch.zeros((output.shape[0],3,output.shape[2],output.shape[3]))
    h,w = output.shape[2],output.shape[3]
    result = []
    for a in range(h):
        for b in range(w):
            class_index = indices[0][a][b].int()
            result.append(colors[class_index])
    result = np.reshape(result, (h,w,3)).astype(dtype=np.uint8)
    return result




def train_loop(model, dataloader, loss_fun, optimizer, device, scheduler, num_epochs):
    model.train()
    model.to(device)
    min_loss = None
    for epoch in range(num_epochs):
        losses = []
        loop = tqdm(enumerate(dataloader),total=len(dataloader),leave=True)
        for batch, (x, y) in loop:
            # put on cuda
            x = x.to(device)
            y = y.to(device)

            # forward pass
            y_pred = model(x)

            # calculate loss & accuracy
            loss = loss_fun(y_pred.reshape(-1, 11),y.reshape(-1))
            losses.append(loss.detach().item())

            # zero out prior gradients
            optimizer.zero_grad()

            # backprop
            loss.backward()

            # update weights
            optimizer.step()
            scheduler.step()

            # Update TQDM progress bar
            loop.set_description(f"Epoch [{epoch}/{num_epochs}] ")
            loop.set_postfix(loss=loss.detach().item())

        moving_loss = sum(losses) / len(losses)
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        # Save check point
        if min_loss == None:
            min_loss = moving_loss
            save_checkpoint(checkpoint)
        elif moving_loss < min_loss:
            min_loss = moving_loss
            save_checkpoint(checkpoint)
        print('Epoch {0} : Loss = {1}'.format(epoch,moving_loss))


def test_loop(model, dataloader, loss_fun,device):
    model.eval()
    model.to(device)
    losses = []
    loop = tqdm(enumerate(dataloader),total=len(dataloader), leave=True)
    with torch.no_grad():
        for batch, (x, y) in loop:
            # put on cuda
            x = x.to(device)
            y = y.to(device)

            # forward pass
            y_pred = model(x, y)

            # caclulate test loss
            loss = loss_fun(y_pred.reshape(-1, 11),y.reshape(-1))
            losses.append(loss.detach().item())

            # Update TQDM progress bar
            loop.set_postfix(loss=loss.item())