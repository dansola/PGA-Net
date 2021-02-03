import torch
from torch import nn
from models.basic_pga.utils import get_image_dicts, build_pos_tensors
import random


class PropAttention(nn.Module):
    def __init__(self, dim, heads, img_crop, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.rand_inds = []
        for _ in range((heads * img_crop) // 2):
            self.rand_inds.append(random.sample(range(img_crop ** 2), img_crop))

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def make_stack(self, x, obj_dict, bg_dict):
        ts = []
        l, w, e = *x.shape,
        x = x.reshape(l * w, e)
        t1, t2 = build_pos_tensors(x, obj_dict, bg_dict, self.rand_inds[0])
        ts.append(t1)
        ts.append(t2)

        for i in range(1, len(self.rand_inds)):
            t1, t2 = build_pos_tensors(x, obj_dict, bg_dict, self.rand_inds[i])
            ts.append(t1)
            ts.append(t2)

        t = torch.stack(ts, dim=0)

        return t

    def forward(self, x, prop, kv=None):
        prop_flat = prop.flatten()
        obj_dict, bg_dict = get_image_dicts(prop_flat)

        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        q = self.make_stack(q, obj_dict, bg_dict)
        k = self.make_stack(k, obj_dict, bg_dict)
        v = self.make_stack(v, obj_dict, bg_dict)

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)

        return out
