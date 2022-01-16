from dataclasses import dataclass
import torch
import torch.nn as nn
from enum import Enum
from typing import List, Optional
from utils import make_binary, make_double_class, make_triple_class, make_regression, RMSELoss, Accuracy


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