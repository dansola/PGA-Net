import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser(description='Convert multiclass masks into binary masks.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--save-directory', metavar='S', type=str, default='../data/binary_masks',
                        help='Directory where binary masks will be saved.', dest='save_dir')
    parser.add_argument('-l', '--load-directory', metavar='L', type=str, default='../data/masks',
                        help='Directory where multiclass masks will be loaded from.', dest='load_dir')
    return parser.parse_args()


def build_masks(save_dir, load_dir):
    for NUM in range(1, 51):
        mask = plt.imread(os.path.join(load_dir, f'img_{NUM}.tif'))
        mask_1d = np.zeros_like(mask[:, :, 0])
        mask_1d[(mask[:, :, 0] == 1)] = 1
        mask_1d[(mask[:, :, 0] == 2)] = 1
        im = Image.fromarray(mask_1d)
        im.save(os.path.join(save_dir, f'img_{NUM}.tif'))


if __name__ == '__main__':
    args = get_args()
    build_masks(args.save_dir, args.load_dir)