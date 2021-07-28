from src.datasets.city import City
from torch.utils.data import DataLoader
import torch
from src.models.unet.unet_model import UNet
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from src.metrics.segmentation import _fast_hist, per_class_pixel_accuracy, jaccard_index
from tqdm import tqdm

data_dir = '/home/dsola/repos/PGA-Net/data/'
batch_size = 1

train_set = City(data_dir, split='train', is_transform=True)
val_set = City(data_dir, split='val', is_transform=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                        drop_last=True)