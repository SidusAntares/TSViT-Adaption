from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import pickle
import os
from collections import defaultdict

# rootdir = "/data/user/ViT/T31TFM_1618"
# paths_train = "/data/user/ViT/T31TFM_1618/paths/train_paths.csv"(48, 48)
rootdir = "/data/user/ViT/PASTIS/"
paths_train = "/data/user/ViT/PASTIS/fold-paths/folds_1_123_paths.csv" # (24, 24)
df = pd.read_csv(paths_train,header=None)
for i in range(1,df.shape[0]):
    with open(os.path.join(rootdir, df.iloc[i, 0]), 'rb') as f:
        sample = pickle.load(f, encoding='latin1')
        # print(type(sample['labels'])) # <class 'numpy.ndarray'>
        # h,w = sample['labels'].shape
        # print(sample['labels'].shape)(24, 24)
        # print(sample['doy'].shape)(46,)
        print(sample['doy'])
        # print(sample['img'].shape)(46, 10, 24, 24)
        # print(sample.keys())



# dict_keys(['img', 'labels', 'doy'])
# [  3  28  43  48  53  58  78  83  88 103 108 123 133 143 153 158 163 168
#  178 183 188 193 198 203 213 223 233 238 243 248 253 258 263 263 273 273
#  278 283 288 293 298 308 313 328 343 353]