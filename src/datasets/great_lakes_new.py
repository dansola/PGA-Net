from dataclasses import dataclass
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset
from enum import Enum
import numpy as np
import pickle
import os
from typing import List, Tuple, Union, Optional


def wandb_logging(x_vals: list, y_vals: list, classes: int, log_preds: bool = True, log_labels: bool = True) -> None:
    wandb.log({"accuracy_score": accuracy_score(x_vals, y_vals)})
    wandb.log({"precision_score": precision_score(x_vals, y_vals)})  # , average='micro')})
    wandb.log({"recall_score": recall_score(x_vals, y_vals)})  # , average='micro')})
    wandb.log({"f1_score": f1_score(x_vals, y_vals)})  # , average='micro')})
    if log_preds:
        for i in range(2, classes):
            wandb.log({f"n_{i}_preds": (np.array(x_vals) == i).sum()})
        wandb.log({"n_0_preds": (np.array(x_vals) == 0).sum()})
        wandb.log({"n_1_preds": (np.array(x_vals) == 1).sum()})
    if log_labels:
        for i in range(2, classes):
            wandb.log({f"n_{i}_labels": (np.array(y_vals) == i).sum()})
        wandb.log({"n_0_labels": (np.array(y_vals) == 0).sum()})
        wandb.log({"n_1_labels": (np.array(y_vals) == 1).sum()})


class Lake(str, Enum):
    ERIE = 'erie'
    ONTARIO = 'ontario'


class Split(str, Enum):
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'


class Label(str, Enum):
    BINARY = 'binary'
    TRIPLE_CLASS = 'triple_class'
    REGRESSION = 'regression'
    DOUBLE_CLASS = 'double_class'


def make_one_hot(in_array: np.ndarray, classes: int) -> np.ndarray:
    in_array = in_array.astype(int)
    return np.eye(classes)[in_array]


def make_one_hot_torch(in_array: torch.Tensor, classes: int) -> torch.Tensor:
    in_array = in_array.to(dtype=int)
    return torch.eye(classes)[in_array]


class RMSELoss(nn.Module):
    name = "RMSE"

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs, labels):
        return torch.sqrt(self.mse(outputs, labels))


class Accuracy(nn.Module):
    name = "Accuracy"

    def __init__(self, classes):
        super().__init__()
        self.classes = classes

    def forward(self, outputs, labels):
        outputs = torch.argmax(outputs, dim=1)
        outputs = make_one_hot_torch(outputs, classes=self.classes)
        assert outputs.shape == labels.shape, \
            f"Output are of shape {outputs.shape} and Labels are of shape {labels.shape}."
        correct = torch.all(torch.eq(outputs.cpu(), labels.cpu()), dim=1).sum()
        return correct / len(outputs)


def unpickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def make_regression(imgs: np.ndarray, ice_cons: list) -> Tuple[np.ndarray, List[int]]:
    return imgs, ice_cons


def make_binary(imgs: np.ndarray, ice_cons: list) -> Tuple[np.ndarray, List[int]]:
    is_binary = ((np.array(ice_cons) == 0) | (np.array(ice_cons) == 1))
    imgs_binary = imgs[is_binary]
    ice_cons_binary = np.array(ice_cons)[is_binary]
    ice_con_one_hot = make_one_hot(ice_cons_binary, 2)
    return imgs_binary, ice_con_one_hot.astype(int).tolist()


def make_double_class(imgs: np.ndarray, ice_cons: list, lower: float = 0.31,
                      upper: float = 1.0) -> Tuple[np.ndarray, List[List[int]]]:
    ice_cons = np.array(ice_cons)
    is_low = (ice_cons <= lower)
    is_hi = (ice_cons >= upper)
    ice_cons = np.where(is_low, 0, ice_cons)
    ice_cons = np.where(is_hi, 1, ice_cons)
    ice_con_one_hot = make_one_hot(ice_cons, 2)
    return imgs, ice_con_one_hot.astype(int).tolist()


def make_triple_class(imgs: np.ndarray, ice_cons: list, lower: float = 0.0,
                      upper: float = 1.0) -> Tuple[np.ndarray, List[List[int]]]:
    ice_cons = np.array(ice_cons)
    is_zero = (ice_cons <= lower)
    is_one = (ice_cons >= upper)
    is_other = ((ice_cons < upper) & (ice_cons > lower))
    ice_cons = np.where(is_zero, 0, ice_cons)
    ice_cons = np.where(is_other, 1, ice_cons)
    ice_cons = np.where(is_one, 2, ice_cons)
    ice_con_one_hot = make_one_hot(ice_cons, 3)
    return imgs, ice_con_one_hot.astype(int).tolist()


# BINARY_WEIGHTS = {'no_weights': [1], Lake.erie.value: [0.95]}  # [0.4640951738517201], }
LABEL_CONVERTER = {
    Label.BINARY.value: {'function': make_binary, 'classes': 2},
    Label.DOUBLE_CLASS.value: {'function': make_double_class, 'classes': 2},
    Label.TRIPLE_CLASS.value: {'function': make_triple_class, 'classes': 3},
    Label.REGRESSION.value: {'function': make_regression, 'classes': 1},
}


@dataclass
class BaseConfig:
    data_directory: str
    lakes: List[Lake]
    label: Label

    def __iter__(self):
        return iter((self.data_directory, self.lakes, self.label))


@dataclass
class TrainConfig(BaseConfig):
    batch_size: int
    epochs: int
    epoch_size: int
    weight: Optional[list] = None
    split: Split = Split.TRAIN
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def criterion(self):
        if self.label is Label.REGRESSION:
            return nn.MSELoss()
        else:
            return nn.CrossEntropyLoss(weight=self.weight)


@dataclass
class TestConfig(BaseConfig):
    batch_size: int
    epoch_size: int
    split: Split = Split.TEST

    @property
    def metric(self):
        if self.label is Label.REGRESSION:
            return RMSELoss()
        else:
            return Accuracy(classes=LABEL_CONVERTER[self.label.value]['classes'])


class LakesRandom(Dataset):
    def __init__(self, conf: Union[TrainConfig, TestConfig], imgs_paths: Optional[List[str]] = None,
                 ice_con_paths: Optional[List[str]] = None):
        self.conf = conf
        if imgs_paths and ice_con_paths:
            self.img_paths = imgs_paths
            self.ice_con_paths = ice_con_paths
        else:
            self.img_paths, self.ice_con_paths = [], []
            for lake in self.conf.lakes:
                text_file_path = self._get_text_file_path(lake)
                img_paths_lake, ice_con_paths_lake = self._read_text_file(text_file_path, lake)
                self.img_paths += img_paths_lake
                self.ice_con_paths += ice_con_paths_lake

    def _get_text_file_path(self, lake: Lake) -> str:
        return os.path.join(self.conf.data_directory, f"imlist_{self.conf.split.value}_{lake}.txt")

    def _read_text_file(self, text_file_path: str, lake: Lake) -> Tuple[List[str], List[str]]:
        img_paths, ice_con_paths = [], []
        with open(text_file_path) as f:
            for line in f:
                line = line.strip('\n')
                pkl_file = line.split('/')[-1]
                date = pkl_file.split('_')[0]

                img_path = f"{date}_3_20_HH_HV_patches_{lake}.npy"
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
        item_found = False
        while not item_found:
            i = np.random.randint(0, len(self.img_paths), 1)[0]
            imgs = np.load(self.img_paths[i])
            ice_cons = unpickle(self.ice_con_paths[i])[0]
            assert imgs.shape[0] == len(ice_cons), \
                f"Number of images, {imgs.shape[0]}, does not match the number of labels, {len(ice_cons)}. "
            imgs, ice_cons = LABEL_CONVERTER[self.conf.label.value]['function'](imgs, ice_cons)
            if imgs.shape[0] > 0:
                j = np.random.randint(0, imgs.shape[0], 1)[0]
                img, ice_con = imgs[j], ice_cons[j]
                img = img.transpose(2, 0, 1)
                item_found = True

        return torch.from_numpy(img), torch.tensor(ice_con)
