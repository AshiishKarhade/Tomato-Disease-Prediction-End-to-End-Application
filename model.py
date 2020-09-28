import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch import optim
import matplotlib.pyplot as plt


data_dir = "SELECT_YOUR_DATA_DIRECTORY"
"""
images -> train
            -> 9 types of disease including a healthy type
       -> valid
            -> 9 types of disease including a healthy type
"""

train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(225),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transform)
test_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

