from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from enum import Enum
import numpy as np
import pickle
import os
from typing import List, Tuple, Union, Optional


class Lake(str, Enum):
    erie = 'erie'
    ontario = 'ontario'


class Split(str, Enum):
    train = 'train'
    test = 'test'
    val = 'val'


class Label(str, Enum):
    binary = 'binary'
    triple_class = 'triple_class'
    regression = 'regression'


BINARY_WEIGHTS = {'no_weights': [1], Lake.erie.value: [0.95]}  # [0.4640951738517201],


class RMSELoss(nn.Module):
    name = "RMSE"

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs, labels):
        return torch.sqrt(self.mse(outputs, labels))


class Accuracy(nn.Module):
    name = "Accuracy"

    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        outputs = (outputs > 0.5).float()
        assert outputs.shape == labels.shape, \
            f"Output are of shape {outputs.shape} and Labels are of shape {labels.shape}."
        correct = (outputs == labels).sum()
        return correct / len(outputs)


def unpickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


@dataclass
class BaseConfig:
    data_directory: str
    lake: Lake
    binary_labels: bool

    def __iter__(self):
        return iter((self.data_directory, self.lake, self.binary_labels))


@dataclass
class TrainConfig(BaseConfig):
    batch_size: int
    epochs: int
    epoch_size: int
    class_weighted: bool
    split: Split = Split.train
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def criterion(self):
        if self.binary_labels:
            if self.class_weighted:
                pos_weight = torch.FloatTensor(BINARY_WEIGHTS[self.lake.value]).to(device=self.device)
            else:
                pos_weight = torch.FloatTensor(BINARY_WEIGHTS['no_weights']).to(device=self.device)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return nn.MSELoss()


@dataclass
class TestConfig(BaseConfig):
    batch_size: int
    epoch_size: int
    split: Split = Split.test

    @property
    def metric(self):
        if not self.binary_labels:
            return RMSELoss()
        else:
            return Accuracy()


def make_binary(imgs: np.ndarray, ice_cons: list) -> Tuple[np.ndarray, List[int]]:
    is_binary = ((np.array(ice_cons) == 0) | (np.array(ice_cons) == 1))
    imgs_binary = imgs[is_binary]
    ice_cons_binary = np.array(ice_cons)[is_binary]
    return imgs_binary, ice_cons_binary.astype(int).tolist()


class LakesRandom(Dataset):
    def __init__(self, conf: Union[TrainConfig, TestConfig]):
        self.conf = conf
        text_file_path = self._get_text_file_path()
        self.img_paths, self.ice_con_paths = self._read_text_file(text_file_path)

    def _get_text_file_path(self) -> str:
        return os.path.join(self.conf.data_directory, f"imlist_{self.conf.split.value}_{self.conf.lake.value}.txt")

    def _read_text_file(self, text_file_path: str) -> Tuple[List[str], List[str]]:
        img_paths, ice_con_paths = [], []
        with open(text_file_path) as f:
            for line in f:
                line = line.strip('\n')
                pkl_file = line.split('/')[-1]
                date = pkl_file.split('_')[0]

                img_path = f"{date}_3_20_HH_HV_patches_{self.conf.lake.value}.npy"
                ice_con_path = pkl_file

                assert img_path in os.listdir(self.conf.data_directory), \
                    f"{img_path} not found in {self.conf.data_directory}"
                assert ice_con_path in os.listdir(self.conf.data_directory), \
                    f"{ice_con_path} not found in {self.conf.data_directory}"

                img_paths.append(os.path.join(self.conf.data_directory, img_path))
                ice_con_paths.append(os.path.join(self.conf.data_directory, ice_con_path))

        return img_paths, ice_con_paths

    def __len__(self) -> int:
        return self.conf.epoch_size

    def __getitem__(self, item: int) -> Tuple[torch.DoubleTensor, torch.DoubleTensor]:
        i = np.random.randint(0, len(self.img_paths), 1)[0]
        imgs = np.load(self.img_paths[i])
        ice_cons = unpickle(self.ice_con_paths[i])[0]
        assert imgs.shape[0] == len(ice_cons), \
            f"Number of images, {imgs.shape[0]}, does not match the number of labels, {len(ice_cons)}. "
        if self.conf.binary_labels:
            imgs, ice_cons = make_binary(imgs, ice_cons)
        j = np.random.randint(0, imgs.shape[0] + 1, 1)[0]
        img, ice_con = imgs[j - 1], ice_cons[j - 1]
        img = img.transpose(2, 0, 1)

        return torch.from_numpy(img), torch.tensor(ice_con)
