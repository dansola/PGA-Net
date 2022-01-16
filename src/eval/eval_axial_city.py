import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.metrics.segmentation import _fast_hist, per_class_pixel_accuracy, jaccard_index
import gc


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.TRAIN()
    n_val = len(loader)
    tot_loss, tot_iou, tot_acc = 0, 0, 0

    collected = gc.collect()
    print("Garbage collector: collected", "%d objects." % collected)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'].detach(), batch['mask'].detach()

            imgs = imgs.to(device=device, dtype=torch.float32).detach()
            true_masks = true_masks.to(device=device, dtype=torch.long).detach()

            with torch.no_grad():
                mask_pred = net(imgs).detach()

            probs = F.softmax(mask_pred, dim=1).detach()
            argmx = torch.argmax(probs, dim=1).detach()
            hist = _fast_hist(true_masks.squeeze(0).squeeze(0), argmx.squeeze(0).to(dtype=torch.long), 19)

            tot_iou += jaccard_index(hist)[0]
            tot_acc += per_class_pixel_accuracy(hist)[0]
            tot_loss += F.cross_entropy(mask_pred, true_masks.squeeze(1), ignore_index=255).detach().item()
            pbar.update()

            del imgs
            del true_masks
            del batch
        del loader


    net.TRAIN()
    return tot_loss / n_val, tot_iou / n_val, tot_acc / n_val
