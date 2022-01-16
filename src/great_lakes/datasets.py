import torch
from torch.utils.data import Dataset
import numpy as np
import os
from typing import List, Tuple, Union, Optional
from configs import TrainConfig, TestConfig, Lake, LABEL_CONVERTER
from utils import unpickle


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