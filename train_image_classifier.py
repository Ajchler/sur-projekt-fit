import torch
import torchvision
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

augmentation = v2.Compose([
    v2.RandomResizedCrop(80),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(30),
    v2.ColorJitter(brightness=0.1, hue=0.1, contrast=0.1, saturation=0.1),
    v2.GaussianBlur(3, sigma=(0.1, 2.0)),
    v2.RandomErasing(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(root="data/train", transform=augmentation)
val_dataset = datasets.ImageFolder(root="data/val", transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



print("foo")

