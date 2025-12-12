import os
import numpy as np
import pickle
import pandas as pd
import torch
from collections import defaultdict

eval_csv = pd.read_csv('/data/user/ViT/PASTIS/fold-paths/folds_1_123_paths.csv',header=None)
rootdir = '/data/user/ViT/PASTIS/'
metric = defaultdict(int)
total = 0
for i in range(eval_csv.shape[0]):
    file = os.path.join(rootdir, eval_csv.iloc[i,0])
    total += 24*24
    with open(file, 'rb') as f:
        sample = pickle.load(f,encoding='latin1')
        labels = sample['labels']
        # print(type(labels))
    unique_label,counts = np.unique(labels,return_counts = True)
    for x,y in zip(unique_label,counts):
        metric[x] += y
    print(i)
metric = {k:v/total for k,v in metric.items()}
metric = dict(sorted(metric.items(),key = lambda x:x[1]))
data = pd.DataFrame([metric])
data.to_csv( "/data/user/ViT/D3/test/pastis_train_classes.csv", index=False)
