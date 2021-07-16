import torch
from torch import nn
from src.models.basic_pga.utils import get_image_dicts, build_pos_tensors, build_rand_inds


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BlockPGA(nn.Module):
    def __init__(self, channels, embedding_dims, img_shape=(300, 300)):
        super(BlockPGA, self).__init__()
        self.channels = channels
        self.embedding_dims = embedding_dims
        self.embedding_dims_double = embedding_dims * 2
        self.img_shape = img_shape

        self.conv1 = conv1x1(self.channels, self.embedding_dims)
        self.bn1 = nn.BatchNorm2d(self.embedding_dims)
        self.relu = nn.ReLU(inplace=True)

        self.attn = PropAttention(dim=self.embedding_dims, heads=2, img_crop=img_shape[0])

        self.conv2 = conv1x1(self.embedding_dims_double, self.embedding_dims)
        self.bn2 = nn.BatchNorm2d(self.embedding_dims)

    def forward(self, x, obj_dict, bg_dict):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x_attn = self.attn(x, obj_dict, bg_dict)
        x_attn = self.relu(x_attn)

        x = torch.cat((x_attn, x), dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class PropAttention(nn.Module):
    def __init__(self, dim, heads, img_crop, dim_heads=None):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads

        inds = build_rand_inds(heads, img_crop)
        inds_tensor = torch.LongTensor(inds)
        self.rand_inds = nn.Parameter(inds_tensor, requires_grad=False)

        self.heads = heads
        self.img_crop = img_crop
        self.to_q = nn.Linear(self.dim_heads, self.dim_heads, bias=False)
        self.to_kv = nn.Linear(self.dim_heads, 2 * self.dim_heads, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def construct(self, t, obj_dict, bg_dict, inds):
        img_flat = t.view(-1, self.img_crop ** 2)

        obj_vecs = []
        bg_vecs = []
        num_obj = self.img_crop // 2
        num_bg = self.img_crop - num_obj
        for i in range(num_obj):
            obj_vecs.append(build_pos_tensors(img_flat, obj_dict, inds[i]))
        for i in range(num_bg):
            bg_vecs.append(build_pos_tensors(img_flat, bg_dict, inds[i + num_obj]))

        obj_tensor = torch.stack(obj_vecs, dim=0)
        bg_tensor = torch.stack(bg_vecs, dim=0)

        all_tensor = torch.cat((obj_tensor, bg_tensor), dim=0)

        return all_tensor

    def destruct(self, t, img, obj_dict, bg_dict, inds):
        ts = torch.chunk(t, 2, dim=0)

        obj_tensor_tuple = torch.chunk(ts[0], ts[0].shape[0], dim=0)
        bg_tensor_tuple = torch.chunk(ts[1], ts[1].shape[0], dim=0)

        for i, vec in enumerate(obj_tensor_tuple):
            img = self.add_vec_to_tensor(img, inds[i], vec.squeeze(0).transpose(1, 0), obj_dict)
        for i, vec in enumerate(bg_tensor_tuple):
            img = self.add_vec_to_tensor(img, inds[i + len(obj_tensor_tuple)], vec.squeeze(0).transpose(1, 0), bg_dict)

        return img

    def add_vec_to_tensor(self, t, inds, vec, obj_dict):
        for i, v in zip(inds.detach().cpu().numpy(), vec):
            obj_ind = obj_dict[i]
            x = obj_ind // self.img_crop
            y = obj_ind % self.img_crop
            t[:, :, x, y] = v
        return t

    def forward(self, x, obj_dict, bg_dict, kv=None):
        x = x.clone().detach()

        img_heads = torch.chunk(x.cpu(), self.heads, dim=1)
        head_list = []
        for inds, h in zip(self.rand_inds, img_heads):
            head_list.append(self.construct(h, obj_dict, bg_dict, inds))

        out = torch.cat(head_list, dim=0).view(self.heads * self.img_crop, self.img_crop, -1).cuda()

        kv = out if kv is None else kv
        q, k, v = (self.to_q(out), *self.to_kv(kv).chunk(2, dim=-1))
        dots = torch.einsum('bie,bje->bij', q, k) * (self.dim_heads ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.view(self.heads * self.img_crop, -1, self.img_crop).cpu()
        ts = torch.chunk(out, self.heads, dim=0)

        new_img_list = []
        for t, img_head, inds in zip(ts, img_heads, self.rand_inds):
            new_img_list.append(self.destruct(t, img_head, obj_dict, bg_dict, inds))
        out_final = torch.cat(new_img_list, dim=1).permute(0, 2, 3, 1).contiguous().cuda()
        out_final = self.to_out(out_final)

        return out_final.permute(0, 3, 1, 2).contiguous()
