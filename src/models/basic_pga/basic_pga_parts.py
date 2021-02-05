import torch
from torch import nn
from models.basic_pga.utils import get_image_dicts, build_pos_tensors, build_rand_inds


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

        # self.pos = PositionalEncodingPermute2D(self.embedding_dims)
        # self.pos = AxialPositionalEmbedding(self.embedding_dims, self.img_shape)
        self.attn = PropAttention(dim=self.embedding_dims, heads=2, img_crop=img_shape[0])

        self.conv2 = conv1x1(self.embedding_dims_double, self.embedding_dims)
        # self.conv2 = conv1x1(self.embedding_dims, self.embedding_dims)
        self.bn2 = nn.BatchNorm2d(self.embedding_dims)

        # self.conv3 = conv1x1(self.embedding_dim

    def forward(self, x, prop):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # pos = self.pos(x)
        # x = torch.cat((pos, x), dim=1)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)

        x_attn = self.attn(x, prop)
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

        self.rand_inds = build_rand_inds(heads, img_crop)

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
        for i, v in zip(inds, vec):
            obj_ind = obj_dict[i]
            x = obj_ind // self.img_crop
            y = obj_ind % self.img_crop
            t[:, :, x, y] = v
        return t

    def forward(self, x, prop, kv=None):
        x = x.clone().detach()
        prop_flat = prop.flatten()
        obj_dict, bg_dict = get_image_dicts(prop_flat)

        img_heads = torch.chunk(x, self.heads, dim=1)
        head_list = []
        for inds, h in zip(self.rand_inds, img_heads):
            head_list.append(self.construct(h, obj_dict, bg_dict, inds))

        out = torch.cat(head_list, dim=0).view(self.heads * self.img_crop, self.img_crop, -1)

        kv = out if kv is None else kv
        q, k, v = (self.to_q(out), *self.to_kv(kv).chunk(2, dim=-1))
        dots = torch.einsum('bie,bje->bij', q, k) * (self.dim_heads ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.view(self.heads * self.img_crop, -1, self.img_crop)
        ts = torch.chunk(out, self.heads, dim=0)

        new_img_list = []
        for t, img_head, inds in zip(ts, img_heads, self.rand_inds):
            new_img_list.append(self.destruct(t, img_head, obj_dict, bg_dict, inds))
        out_final = torch.cat(new_img_list, dim=1).permute(0, 2, 3, 1).contiguous()
        out_final = self.to_out(out_final)
        return out_final.permute(0, 3, 1, 2).contiguous()

# class _PropAttention(nn.Module):
#     def __init__(self, dim, heads, img_crop, dim_heads=None, dim_index=1):
#         assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
#         super().__init__()
#         self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
#         dim_hidden = self.dim_heads * heads
#
#         self.rand_inds = []
#         for _ in range(heads):
#             inds_head = []
#             for _ in range(img_crop // 2):
#                 inds_head.append(random.sample(range(img_crop ** 2), img_crop))
#             self.rand_inds.append(inds_head)
#
#         self.heads = heads
#         self.img_crop = img_crop
#         print(dim, dim_hidden)
#         self.to_q = nn.Linear(dim, dim_hidden, bias=False)
#         self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
#         self.encode = nn.Linear(dim, self.dim_heads)
#         self.to_out = nn.Linear(dim_hidden, dim)
#
#     def make_stack(self, x, obj_dict, bg_dict, rand_ind):
#         ts = []
#         l, w, e = *x.shape,
#
#         # plt.imshow(x.detach().cpu().numpy()[:,:,0])
#         inds = self.rand_inds[rand_ind]
#
#         x = x.reshape(l * w, e)
#
#         t1, t2 = build_pos_tensors(x, obj_dict, bg_dict, inds[0])
#         ts.append(t1)
#         ts.append(t2)
#
#         for i in range(1, len(inds)):
#             t1, t2 = build_pos_tensors(x, obj_dict, bg_dict, inds[i])
#             ts.append(t1)
#             ts.append(t2)
#
#         t = torch.stack(ts, dim=0)
#         return t
#
#     def split_heads(self, x, obj_dict, bg_dict):
#         ts = torch.chunk(x, self.heads, dim=-1)
#         t_heads = []
#         for i, t in enumerate(ts):
#             t_heads.append(self.make_stack(t, obj_dict, bg_dict, i))
#         return torch.cat(t_heads, dim=0)
#
#     def rebuild(self, x, out, obj_dict, bg_dict):
#         print(x.shape, out.shape, len(obj_dict), len(bg_dict))
#         embeds = torch.chunk(x, self.heads, dim=-1)
#         attns = torch.chunk(out, self.heads, dim=-1)
#         for embed, attn, inds in zip(embeds, attns, self.rand_inds):
#             print(embed.shape, attn.shape, len(inds))
#             embed_flat = embed.reshape(self.img_crop**2, -1)
#             attn_flat = attn.reshape(self.img_crop ** 2, -1)
#         pass
#
#     def forward(self, x, prop, kv=None):
#         x = x.squeeze(0).permute(1, 2, 0).contiguous()
#         # x_encode = self.encode(x)
#         # print(x.shape, x_encode.shape)
#         # print(x.shape)
#         prop_flat = prop.flatten()
#         obj_dict, bg_dict = get_image_dicts(prop_flat)
#
#         kv = x if kv is None else kv
#         q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))
#
#         b, t, d, h, e = *q.shape, self.heads, self.dim_heads
#         print(q.shape, k.shape, v.shape)
#         q = self.split_heads(q, obj_dict, bg_dict)
#         k = self.split_heads(k, obj_dict, bg_dict)
#         v = self.split_heads(v, obj_dict, bg_dict)
#
#         print(q.shape, k.shape, v.shape)
#         dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
#         dots = dots.softmax(dim=-1)
#         out = torch.einsum('bij,bje->bie', dots, v)
#         print(out.shape)
#
#         out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
#         self.rebuild(x, out, obj_dict, bg_dict)
#         out = self.to_out(out)
#
#         return out.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
