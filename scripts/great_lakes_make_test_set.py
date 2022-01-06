import os
from src.datasets.great_lakes import Lake, Split, unpickle, make_binary
import numpy as np

data_directory = "/home/dsola/repos/PGA-Net/data/patch20"
lake = Lake.ontario
split = Split.test
binary_labels = True

text_file_path = os.path.join(data_directory, f"imlist_{split.value}_{lake.value}.txt")

if binary_labels:
    binary_tag = "binary"
else:
    binary_tag = "regression"

img_save_dir = f"/home/dsola/repos/PGA-Net/data/patch20/{lake.value}_test_{binary_tag}_imgs"

try:
    os.mkdir(img_save_dir)
except OSError:
    pass
label_save_name = f"/home/dsola/repos/PGA-Net/data/patch20/{lake.value}_test_{binary_tag}_labels.npy"

img_paths, ice_con_paths = [], []
with open(text_file_path) as f:
    for line in f:
        line = line.strip('\n')
        pkl_file = line.split('/')[-1]
        date = pkl_file.split('_')[0]

        img_path = f"{date}_3_20_HH_HV_patches_{lake.value}.npy"
        ice_con_path = pkl_file

        assert img_path in os.listdir(data_directory), \
            f"{img_path} not found in {data_directory}"
        assert ice_con_path in os.listdir(data_directory), \
            f"{ice_con_path} not found in {data_directory}"

        img_paths.append(os.path.join(data_directory, img_path))
        ice_con_paths.append(os.path.join(data_directory, ice_con_path))

label_list = []
done_dict = {}
count = 0
while count < 1000:
    i = np.random.randint(0, len(img_paths), 1)[0]
    imgs = np.load(img_paths[i])
    print(imgs.shape)
    ice_cons = unpickle(ice_con_paths[i])[0]
    assert imgs.shape[0] == len(ice_cons), \
        f"Number of images, {imgs.shape[0]}, does not match the number of labels, {len(ice_cons)}. "
    if binary_labels:
        imgs, ice_cons = make_binary(imgs, ice_cons)
    j = np.random.randint(0, imgs.shape[0] + 1, 1)[0]
    if i not in done_dict.keys():
        done_dict[i] = []
    if j not in done_dict[i] and imgs.shape[0] > 0:
        img, ice_con = imgs[j], ice_cons[j]
        name = f"example{count}"
        np.save(os.path.join(img_save_dir, name)+".npy", img)
        label_list.append([name, ice_con])
        done_dict[i].append(j)
        count += 1
        print(count)

label_array = np.array(label_list)
np.save(label_save_name, label_array)