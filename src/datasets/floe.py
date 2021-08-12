"""
Expected formats:
Image: 1-channel, 8-bit (0->255), TIF
Mask: 4-channel, data channel -> 0th, PNG
Ice_Conc: 1-channel, 0->100 % ice concentration, numpy.ndarray, NPY

Output:
sample{'image': channel 0 (image patch, 0->1), channel 1 (ice_conc_patch, 0->1)
       'mask' : mask_patch [0,1]
"""

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

import os
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2


class DatasetFloe_Ice_Mask(Dataset):
    def __init__(self, patch_size, mode):
        self.path_images = '/home/dsola/repos/PGA-Net/data/floes/images/' + mode
        self.path_masks = '/home/dsola/repos/PGA-Net/data/floes/annotation_masks/'
        self.path_ice_conc = '/home/dsola/repos/PGA-Net/data/floes/ice_conc/'
        self.patchsize = patch_size
        self.img_names = os.listdir(self.path_images)

    def transform(self, img, mask, ice):

        image = TF.to_pil_image(img)
        mask = TF.to_pil_image(mask)
        ice = TF.to_pil_image(ice.astype(np.float32) / 100)

        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            ice = TF.hflip(ice)

        # Random Vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            ice = TF.vflip(ice)

        # To Tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        ice = TF.to_tensor(ice)

        # Stack image and ice so that network accepts them as two channels.
        image_stacked = torch.stack((image.squeeze(), ice.squeeze()), dim=0)

        return image_stacked, mask

    def get_patch(self, file_name):
        img = plt.imread(os.path.join(self.path_images, file_name + '.tif'))
        mask = cv2.imread(os.path.join(self.path_masks, file_name + '.png'))
        ice_conc = np.load(os.path.join(self.path_ice_conc, file_name + '.npy'))

        mask = (mask[:, :, 2] / 128).astype(np.int32)

        img_x, img_y = img.shape

        patch_x = int((img_x - self.patchsize) * random.random())
        patch_y = int((img_y - self.patchsize) * random.random())

        img_patch = img[patch_x: (patch_x + self.patchsize), patch_y:(patch_y + self.patchsize)]
        mask_patch = mask[patch_x: (patch_x + self.patchsize), patch_y:(patch_y + self.patchsize)]
        ice_conc_patch = ice_conc[patch_x: (patch_x + self.patchsize), patch_y:(patch_y + self.patchsize)]

        return img_patch, mask_patch, ice_conc_patch

    def __len__(self):
        return 1000  # We are generating images on the go.

    def __getitem__(self, index):
        file_name = random.choice(self.img_names)[:-4]

        img_patch, mask_patch, ice_conc_patch = self.get_patch(file_name)
        while np.count_nonzero(img_patch == 0) > int((self.patchsize * self.patchsize) * 0.25):
            img_patch, mask_patch, ice_conc_patch = self.get_patch(file_name)

        img_patch, mask_patch = self.transform(img_patch, mask_patch, ice_conc_patch)

        sample = {'image': img_patch, 'mask': mask_patch}

        return sample


class DatasetValidateFloe(Dataset):
    """
    Expecting: image_patches be 4-channel TIF, 8-bit
             : mask_patches be 4-channel TIF, 8-bit
             : ice_con_patches be 1-channel TIF, 0->120, 120 means land, 0->100 is the ice concentration
    """

    def __init__(self):
        self.path_images = '/home/dsola/repos/PGA-Net/data/floes/valid_premade_patches_multi/image_patches/'
        self.path_masks = '/home/dsola/repos/PGA-Net/data/floes/valid_premade_patches_multi/mask_patches/'
        self.path_ice_conc = '/home/dsola/repos/PGA-Net/data/floes/valid_premade_patches_multi/con_patches/'
        self.file_names = os.listdir(self.path_images)

    def transform(self, img, mask, ice):
        # To Tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        ice = TF.to_tensor((np.array(ice) != 120).astype(int) * ice)
        # Ice concentration patches: land_mass = 120
        #                          : ice_conc = 0 -> 100

        img = img.float() / 255  # 0->1
        mask = torch.round(mask)  # [0,1]
        ice = ice.float() / 100  # 0->1

        img = torch.stack((img.squeeze(), ice.squeeze()), dim=0)  # all img, mask, ice as_type(float32)

        return img, mask

    def __len__(self):
        return (len(self.file_names))

    def __getitem__(self, index):
        img = Image.open(self.path_images + self.file_names[index])
        mask = Image.open(self.path_masks + self.file_names[index])
        ice = Image.open(self.path_ice_conc + self.file_names[index])

        img, mask = self.transform(img, mask, ice)
        sample = {'image': img, 'mask': mask}
        return sample
