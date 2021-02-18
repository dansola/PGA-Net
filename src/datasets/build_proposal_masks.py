import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser(description='Convert DeepMask proposals into numpy masks.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--save-directory', metavar='S', type=str, default='../data/proposals/binary',
                        help='Directory where proposals will be saved.', dest='save_dir')
    parser.add_argument('-n', '--n-proposals', metavar='P', type=int, default=250,
                        help='Number of proposals', dest='proposals')
    parser.add_argument('-r', '--max-ratio', metavar='B', type=int, default=16,
                        help='Ratio of largest proposal to the size of the image.', dest='max_ratio')
    parser.add_argument('-p', '--plot', metavar='PLT', type=bool, default=False,
                        help='Plot images with proposals.', dest='plot')
    return parser.parse_args()


def build_masks(dir, n_proposals, max_ratio=16, plot=False):
    for NUM in range(1, 51):
        proposal = np.load(f"../data/proposals/masks/masks_{NUM}.npy")
        proposals = np.zeros_like(proposal[:, :, 0])
        for ind in range(n_proposals):
            if proposal[:, :, ind].sum() < (proposal.shape[0] * proposal.shape[1]) / max_ratio:
                proposals[(proposal[:, :, ind] == 1)] = 1
        dst = os.path.join(dir, f'img_{NUM}.npy')
        np.save(dst, proposals)

        if plot:
            img = Image.open(f"../data/imgs/img_{NUM}.tif")
            h, w = img.size
            newW, newH = np.round_(0.5 * w), np.round_(0.5 * h)
            assert newW > 0 and newH > 0, 'Scale is too small'
            img_nd = np.array(img)
            img_nd = skimage.transform.resize(img_nd,
                                              (newW, newH),
                                              mode='edge',
                                              anti_aliasing=False,
                                              anti_aliasing_sigma=None,
                                              preserve_range=True,
                                              order=0)
            img_nd = img_nd.astype(int)
            fig, ax = plt.subplots()
            ax.imshow(img_nd)
            ax.imshow(proposals, alpha=0.2)


if __name__ == '__main__':
    args = get_args()
    build_masks(args.save_dir, args.proposals, max_ratio=args.max_ratio, plot=args.plot)