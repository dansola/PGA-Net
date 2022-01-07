import os
from src.datasets.great_lakes import Lake, Split, unpickle
import numpy as np

data_directory = "/home/dsola/repos/PGA-Net/data/patch20"
lake = Lake.erie
split = Split.test

text_file_path = os.path.join(data_directory, f"imlist_{split.value}_{lake.value}.txt")

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

tot_0, tot_1 = 0, 0
for i in range(len(img_paths)):
    print(f"Counting {i} of {len(img_paths)} files...")
    ice_cons = unpickle(ice_con_paths[i])[0]
    tot_0 = (np.array(ice_cons) == 0).sum()
    tot_1 = (np.array(ice_cons) == 1).sum()
    del ice_cons

tot = tot_0 + tot_1
print(f"{tot_0} zeros, {tot_1} ones, out of {tot} binary labels.")
