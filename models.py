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


# Define segmentation architectures
class VanillaUNet(nn.Module):
    def __init__(self, in_channels=3,num_classes=11):
        super(VanillaUNet, self).__init__()
        # Encode architecture
        self.dconv1 = self._double_conv(in_channels, 64)
        self.dconv2 = self._double_conv(64, 128)
        self.dconv3 = self._double_conv(128, 256)
        self.dconv4 = self._double_conv(256, 512)
        self.dconv5 = self._double_conv(512, 1024)
        # Decoder architecture
        self.uconv1 = nn.ConvTranspose2d(1024,512,stride=2,kernel_size=2)
        self.dconv6 = self._double_conv(1024, 512)
        self.uconv2 = nn.ConvTranspose2d(512, 256,stride=2,kernel_size=2)
        self.dconv7 = self._double_conv(512, 256)
        self.uconv3 = nn.ConvTranspose2d(256, 128,stride=2,kernel_size=2)
        self.dconv8 = self._double_conv(256, 128)
        self.uconv4 = nn.ConvTranspose2d(128, 64,stride=2,kernel_size=2)
        self.dconv9 = self._double_conv(128, 64)
        # Final conv
        self.final_conv = nn.Conv2d(64,num_classes,kernel_size=1)
        # Max pool
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _double_conv(self, in_channels,
                     out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv

    def forward(self, X):
        # Forward Prop thru Encoder
        # X Shape : [B X 3 X 512 X 512]
        X1 = self.dconv1(X)  # X1 Shape : [B X 64 X 512 X 512]
        X2 = self.max_pool(X1)  # X2 Shape : [B X 64 X 256 X 256]

        X3 = self.dconv2(X2)  # X3 Shape : [B X 128 X 256 X 256]
        X4 = self.max_pool(X3)  # X4 Shape : [B X 128 X 128 X 128]

        X5 = self.dconv3(X4)  # X5 Shape : [B X 256 X 128 X 128]
        X6 = self.max_pool(X5)  # X6 Shape : [B X 256 X 64 X 64]

        X7 = self.dconv4(X6)  # X7 Shape : [B X 512 X 64 X 64]
        X8 = self.max_pool(X7)  # X8 Shape : [B X 512 X 32 X 32]

        X9 = self.dconv5(X8)  # X9 Shape : [B X 1024 X 32 X 32]

        # Forward Prop thru Decoder
        X10 = self.uconv1(X9)  # X10 Shape : [B X 512 X 64 X 64]
        X11 = torch.cat((X10, X7),dim=1)  # X11 Shape : [B X 1024 X 64 X 64]
        X12 = self.dconv6(X11)  # X12 Shape : [B X 512 X 64 X 64]

        X13 = self.uconv2(X12)  # X13 Shape : [B X 256 X 128 X 128]
        X14 = torch.cat((X13, X5),dim=1)  # X14 Shape : [B X 512 X 128 X 128]
        X15 = self.dconv7(X14)  # X15 Shape : [B X 256 X 128 X 128]

        X16 = self.uconv3(X15)  # X16 Shape : [B X 128 X 256 X 256]
        X17 = torch.cat((X16, X3),dim=1)  # X17 Shape : [B X 256 X 256 X 256]
        X18 = self.dconv8(X17)  # X18 Shape : [B X 128 X 256 X 256]

        X19 = self.uconv4(X18)  # X19 Shape : [B X 64 X 512 X 512]
        X20 = torch.cat((X19, X1),dim=1)  # X20 Shape : [B X 128 X 512 X 512]
        X21 = self.dconv9(X20)  # X21 Shape : [B X 64 X 512 X 512]

        # Final Conv Layer
        X22 = self.final_conv(X21)  # X22 Shape : [B X num_classes X 512 X 512]

        return X22