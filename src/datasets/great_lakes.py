import torch
from torch.utils.data import Dataset
from enum import Enum
import numpy as np
import pickle
import os
from typing import List, Tuple


def unpickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


class Lake(str, Enum):
    """Lake options."""
    erie = 'erie'
    ontario = 'ontario'


class Split(str, Enum):
    """Data split options."""
    train = 'train'
    test = 'test'
    val = 'val'


class LakesRandom(Dataset):
    def __init__(self, data_directory: str, lake: Lake, split: Split, epoch_size: int = 10000):
        self.data_directory = data_directory
        self.lake = lake
        self.split = split
        self.epoch_size = epoch_size
        text_file_path = self._get_text_file_path()
        self.img_paths, self.ice_con_paths = self._read_text_file(text_file_path)

    def _get_text_file_path(self) -> str:
        return os.path.join(self.data_directory, f"imlist_{self.split.value}_{self.lake.value}.txt")

    def _read_text_file(self, text_file_path: str) -> Tuple[List[str], List[str]]:
        img_paths, ice_con_paths = [], []
        with open(text_file_path) as f:
            for line in f:
                line = line.strip('\n')
                pkl_file = line.split('/')[-1]
                date = pkl_file.split('_')[0]

                img_path = f"{date}_3_20_HH_HV_patches_{self.lake.value}.npy"
                ice_con_path = pkl_file

                assert img_path in os.listdir(self.data_directory), f"{img_path} not found in {self.data_directory}"
                assert ice_con_path in os.listdir(
                    self.data_directory), f"{ice_con_path} not found in {self.data_directory}"

                img_paths.append(os.path.join(self.data_directory, img_path))
                ice_con_paths.append(os.path.join(self.data_directory, ice_con_path))

        return img_paths, ice_con_paths

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, item):
        i = np.random.randint(0, len(self.img_paths)+1, 1)[0]
        imgs = np.load(self.img_paths[i])
        ice_cons = unpickle(self.ice_con_paths[i])[0]
        assert imgs.shape[0] == len(ice_cons), f"Number of images, {imgs.shape[0]}, does not match the number of " \
                                               f"labels, {len(ice_cons)}. "
        j = np.random.randint(0, imgs.shape[0]+1, 1)[0]
        img, ice_con = imgs[j], ice_cons[j]

        return torch.from_numpy(img), torch.tensor(ice_con)

from torch.utils.data import DataLoader

data_directory = "/home/dsola/repos/PGA-Net/data/patch20"
lakes = Lake.erie
split = Split.train
train_set = LakesRandom(data_directory, lakes, split)
train_loader = DataLoader(train_set, batch_size=5, shuffle=True)

for batch in train_loader:
    img, ice_con = batch[0], batch[1]
    break