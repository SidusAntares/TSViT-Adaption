import sys
import os
sys.path.insert(0, os.getcwd())
import torch
from utils.lr_scheduler import build_scheduler
import numpy as np
import os
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input
import pandas as pd
from matplotlib import pyplot as plt



config = read_yaml('configs/transfer/Germany.yaml')
dataloaders = get_dataloaders(config)
name_dict = {0: 'beet', 1: 'meadow', 2: 'potatoes', 3: 'winter wheat', 4: 'winter barley', 5: 'corn'}
metric = torch.zeros(6,366)
total = torch.zeros(6,366)
for step,sample in enumerate(dataloaders['train']):
    # print(sample.keys()) # dict_keys(['inputs', 'labels', 'seq_lengths', 'unk_masks'])
    # print(sample['seq_lengths'].shape) # torch.Size([16])
    # print(sample['seq_lengths']) # tensor([46, 38, 60, 60, 43, 38, 46, 46, 60, 43, 46, 46, 60, 60, 46, 46])
    # print(sample['inputs'].shape) torch.Size([16, 60, 24, 24, 11])
    # print(sample['inputs'][:,:,:,:,-1].shape) torch.Size([16, 60, 24, 24])
    # print(sample['inputs'][0,:,0,0,-1].shape) torch.Size([60])
    # print((sample['inputs'][0,:,0,0,-1]*365.0001).to(torch.int64).shape) torch.Size([60])

    x = sample['inputs']
    x = x.permute(0,1,4,2,3)
    B, T, C, H, W = x.shape
    xt = x[:,:,-1,0,0]
    # # print(x[:,:,-1,0,0].shape)
    # # print(x[:,:,-1,0,1].shape)
    # print((x[0, :, -1, 0,:] * 365.0001).to(torch.int64))
    print((x[0, :, -1, 0,0] * 365.0001).to(torch.int64))
    print((x[0, :, -2, 0,0] * 365.0001).to(torch.int64))
    print((x[0, :, -3, 0,0] * 365.0001).to(torch.int64))
    break
    # # x = x[:,:,:-1]
    # # print(xt.shape) torch.Size([16, 60])
    # xt = (xt * 365.0001).to(torch.int64)
#     nir = x[:,:, 7, :,:]
#     red = x[:,:, 3, :,:]
#     ndvi = (nir - red) / (nir + red+1e-10)
#
#
#     for b in range(B):
#         for t in range(T):
#             for h in range(H):
#                 for w in range(W):
#                     doy = (x[b, t, -1,h, w]*365.0001).item()
#                     doy = int(doy)
#                     if doy == 0:
#                         continue
#                     label = sample['labels'][b,h,w].item()
#                     if label ==6:
#                         continue
#                     doy = int(doy)
#                     label = int(label)
#                     total[label,doy-1]+=1
#                     metric[label,doy-1]+=ndvi[b, t, h, w]
#
#     print(step)
#
# metric = metric/total
# df = pd.DataFrame(metric, index=name_dict.keys(), columns=range(1,366+1))
# df.to_csv('/data/user/ViT/D3/test/pastis_ndvi-doy.csv')

    # print(xt[0])
    # tensor([17, 22, 47, 51, 57, 72, 87, 102, 112, 117, 132, 147, 152, 157,
    #         172, 177, 182, 187, 192, 197, 212, 222, 227, 232, 236, 242, 247, 257,
    #         262, 267, 267, 272, 277, 282, 282, 292, 292, 297, 302, 312, 317, 322,
    #         327, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #         0, 0, 0, 0])
    # xt = F.one_hot(xt, num_classes=365).to(torch.float32)
    # # print(xt.shape) torch.Size([16, 60, 365])
    # xt = xt.reshape(-1, 365)
    # # print(xt.shape) torch.Size([960, 365])

    # for i in range(x.shape[0]):
    #     for j in range(x.shape[1]):
    #         nir = x[i,j,7]
    #         red = x[i,j,3]
    #         doy = x[]
    #         ndvi = (nir-red)/(nir+red)



