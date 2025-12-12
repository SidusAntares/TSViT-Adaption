import sys
import os

sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from models import get_model
from utils.config_files_utils import read_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input
import pandas as pd
import torch.nn.functional as F


# 类别映射
name_dict = {0: 'beet', 1: 'meadow', 2: 'potatoes', 3: 'winter wheat', 4: 'winter barley', 5: 'corn'}
num_classes = len(name_dict)  # 6


def process(save_path, dataloaders, config):
    """
    要求batchsize为1
    尝试input时间维度形状转换为366，便可直接相加，batchsize不再局限为1
    """
    local_device_ids = config['local_device_ids']
    # if len(local_device_ids) > 1:
    #     net = nn.DataParallel(net, device_ids=local_device_ids)
    # net.to(device)
    # net.eval()  # 注意这里是 eval 模式！

    total = torch.zeros(6,366)
    total += 1e-5
    count = torch.zeros(6,10,366)

    for step, sample in enumerate(dataloaders['train']):
            # break
            # print(sample.keys())dict_keys(['inputs', 'labels', 'seq_lengths', 'unk_masks'])
            # inputs = sample['inputs'].to(device)
            # print(sample['seq_lengths']) batchsize
            # print(sample['inputs'].shape) torch.Size([16, 60, 24, 24, 11])
            x = sample['inputs'].permute(0, 1, 4, 2, 3)
            B, T, C, H, W = x.shape
            doy = (x[0, :, -1, 0, 0]* 365.0001).to(torch.int64)

            inputs = sample['inputs'].permute(0,2,3,4,1).reshape(-1,C,T)[:,:-1,:]
            labels = sample['labels'].reshape(-1)
            # time = (sample['inputs'].permute(0,2,3,4,1).reshape(-1,C,T)[0,-1,:]* 365.0001).to(torch.int64)
            # print(doy)(batchsize,time)
            for i in range(6):
                mask = labels == i
                # print("mask.shape",mask.shape)
                # print("inputs.shape",inputs.shape)
                temp = inputs[mask]
                total[i,doy] += 1
                # print("count[i,:,doy]", count[i, :, doy].shape)
                # print(temp.sum(dim=0).shape)
                count[i, :, doy] += temp.sum(dim=0)
            print(step)
    total = total.unsqueeze(1).expand(6,10,366) # torch.Size([6, 10, 366])
    count /= total
    torch.save(count,save_path)




            # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # parser.add_argument('--config', required=True, help='configuration (.yaml) file to use')
    # parser.add_argument('--device', default='0', type=str, help='gpu ids to use')


    # args = parser.parse_args()

    device_ids = [int(d) for d in "0,1,2,3".split(',')]
    config_file = '/data/user/ViT/D3/configs/transfer/PASTIS2PASTIS.yaml'

    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids
    save_path = '/data/user/Raincoat/data/PASTIS/pastis_function_cubic.pt'
    dataloaders = get_dataloaders(config)

    process(save_path, dataloaders, config)