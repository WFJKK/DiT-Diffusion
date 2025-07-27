"""
Prepare CIFAR-10 training and validation datasets with normalization and deterministic splitting.
"""

import torch
from torchvision import transforms, datasets
from torch.utils.data import random_split

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

datasetCifar = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

generator = torch.Generator().manual_seed(42)

n_total = len(datasetCifar)        
n_train = int(0.8 * n_total)       
n_val = n_total - n_train          

train_dataset, val_dataset = random_split(
    datasetCifar,
    [n_train, n_val],
    generator=generator
)






