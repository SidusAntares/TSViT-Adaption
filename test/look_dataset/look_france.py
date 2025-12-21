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
from data import main_get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input
from scipy.interpolate import PchipInterpolator
from collections import Counter
import pandas as pd

# 类别映射
name_dict = {0: 'beet', 1: 'meadow', 2: 'potatoes', 3: 'winter wheat', 4: 'winter barley', 5: 'corn'}
# num_classes = len(name_dict)  # 6
# time = np.arange(0, 366,10)
# functions = []
def process(mode,save_path, dataloaders, config):
    os.makedirs(save_path, exist_ok=True)
    label_counter = Counter()
    total_pixels = 0
    for step, sample in enumerate(dataloaders[mode]):
        # print(sample.keys())dict_keys(['labels', 'ids', 'inputs', 'seq_lengths', 'unk_masks'])
            # inputs = sample['inputs'].to(device)
            # print(sample['seq_lengths']) batchsize
        # print(sample['inputs'].shape) #torch.Size([32, 60, 48, 48, 14])
        # print(sample['labels'].shape) #torch.Size([32, 60, 48, 48, 14])
        inputs = sample['inputs']
        inputs = torch.cat([inputs[..., 0:10], inputs[..., 13].unsqueeze(-1)], dim=-1)[
            ..., [2, 1, 0, 4, 5, 6, 3, 7, 8, 9, 10]]
        labels = sample['labels']  # shape: [B, H, W, 1]
        labels = labels.squeeze(-1)  # shape: [B, H, W]
        labels_flat = labels.flatten()  # shape: [B * H * W]

        # 转换为 CPU 并转为 numpy 或直接使用 torch.unique
        # 使用 torch 更高效（避免 numpy 转换开销）
        unique_labels, counts = torch.unique(labels_flat, return_counts=True)
        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            label_counter[label] += count
        total_pixels += labels_flat.numel()
        print(step)
        # x = sample['inputs'].permute(0, 1, 4, 2, 3)
        # B, T, C, H, W = x.shape
        # doy = (x[:, :, -1, 0, 0]* 365.0001).to(torch.int64).reshape(-1).numpy()
        # mask = doy != 0
        # doy = doy[mask]
        # inputs = sample['inputs'].permute(0,2,3,4,1).reshape(-1,C,T)[:,:-1,:].reshape(-1,T) # batch_size = 1 torch.Size([5760, 60])
        #     # print(inputs.shape)
        #     # inputs = inputs[mask.unsqueeze(0).expand_as(inputs)]torch.Size([345600])
        #     # inputs = inputs[mask.unsqueeze(0).expand(inputs.shape[0],inputs.shape[1])]
        # labels = sample['labels'].reshape(-1) # if reshpe(1,-1) then stack will be torch.Size([b, 1, 576])
        # unique_doy ,indices = np.unique(doy, return_inverse=True)

    abel_ratios = {label: count / total_pixels for label, count in label_counter.items()}
    abel_ratios = dict(sorted(abel_ratios.items(),key=lambda item: item[1], reverse=True))
    df = pd.DataFrame([abel_ratios])
    df.to_csv(os.path.join(save_path, 'france_crop_sort.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # parser.add_argument('--config', required=True, help='configuration (.yaml) file to use')
    # parser.add_argument('--device', default='0', type=str, help='gpu ids to use')


    # args = parser.parse_args()

    device_ids = [int(d) for d in "0,1,2,3".split(',')]
    config_file = '/data/user/ViT/D3/configs/transfer/France2France.yaml'

    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids
    # collect_path = '/data/user/Raincoat/data/PASTIS/pastis_function_cubic.pt'
    # save_path = '/data/user/Raincoat/data/PASTIS_and_Germany'
    save_path = '/data/user/ViT/D3/test/look_dataset'
    dataloaders = main_get_dataloaders(config)

    # process("train",save_path, dataloaders, config)
    process("eval",save_path, dataloaders, config)