import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pickle
from typing import List, Tuple


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
