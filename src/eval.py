import torch
import torch.nn.functional as F
from tqdm import tqdm


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot_loss = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            with torch.no_grad():
                mask_pred = net(imgs)

            target = torch.argmax(true_masks.to(dtype=torch.long), dim=1)
            tot_loss += F.cross_entropy(mask_pred, target).item()
            pbar.update()

    net.train()
    return tot_loss / n_val
