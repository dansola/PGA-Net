import os
import logging
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

MEANS = [121.4836, 122.35021, 122.517166]
STDS = [58.89167, 58.966404, 59.09349]


class Ice(Dataset):
    def __init__(self, imgs_dir, masks_dir, txt_dir, split, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.txt_dir = txt_dir
        self.split = split
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        if split == "train":
            file_name = os.path.join(self.txt_dir, 'ice_train.txt')

        elif split == "val":
            file_name = os.path.join(self.txt_dir, 'ice_val.txt')

        elif split == "test":
            file_name = os.path.join(self.txt_dir, 'ice_test.txt')

        self.img_ids = [i_id.strip() for i_id in open(file_name)]
        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(imgs_dir, name)
            mask_file = os.path.join(masks_dir, name)
            self.files.append({
                "img": img_file,
                "mask": mask_file
            })
        logging.info(f'Creating dataset with {len(self.files)} examples')

    def __len__(self):
        return len(self.files)

    def process(self, img, mask):
        img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

        img = transforms.CenterCrop(300)(Image.fromarray(img))
        mask = transforms.CenterCrop(300)(Image.fromarray(mask))

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=MEANS, std=STDS)(img).unsqueeze(0)

        mask = transforms.ToTensor()(mask).unsqueeze(0)

        return img, mask

    def __getitem__(self, i):
        datafiles = self.files[i]
        img = cv2.imread(datafiles["img"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = Image.open(datafiles["mask"])
        mask = np.array(mask)[:, :, 0]
        masks = [(mask == v) for v in [0, 128, 255]]
        mask = np.stack(masks, axis=-1).astype('int8')

        assert img.size == mask.size, \
            f'Image and mask {i} should be the same size, but are {img.size} and {mask.size}'

        img, mask = self.process(img, mask)

        return {
            'image': img,
            'mask': mask
        }
