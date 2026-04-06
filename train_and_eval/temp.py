import sys
import os
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
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
from metrics.loss_functions import get_loss_da
from data import get_loss_data_input
import pandas as pd
import shutil
from datetime import datetime
import glob
from collections import defaultdict


def train_and_evaluate(net, dataloaders, config, device):
    loss_input_fn = get_loss_data_input(config)
    classes = defaultdict(int)
    for step, sample in enumerate(dataloaders['train']):
        ground_truth = loss_input_fn(sample, device)
        target, mask = ground_truth
        target = target.cpu().numpy()
        unique, counts = np.unique(target, return_counts=True)
        print(step)
        for cls , num in zip(unique, counts):
            classes[cls] += num
    for step, sample in enumerate(dataloaders['eval']):
        ground_truth = loss_input_fn(sample, device)
        target, mask = ground_truth
        target = target.cpu().numpy()
        unique, counts = np.unique(target, return_counts=True)
        print(step)
        for cls , num in zip(unique, counts):
            classes[cls] += num
    sum = 0
    for cls in classes:
        sum += classes[cls]

    classes = {cls : classes[cls]/sum for cls in classes}
    classes = {k:f'{v:.01%}' for k,v in classes.items()}
    df = pd.DataFrame([classes])
    save = "/mnt/d/All_Documents/note/Time_Series/Time_Series/数据集整理/遥感时序/Germany"
    df.to_csv(save+'/classes数据集全统计百分比格式.csv', index=False)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0,1', type=str,
                        help='gpu ids to use')


    args = parser.parse_args()
    config_file = args.config
    print(args.device)
    device_ids = [int(d) for d in args.device.split(',')]


    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids

    src_dataloaders = get_dataloaders(config,'src')
    trg_dataloaders = get_dataloaders(config,'trg')

    net = get_model(config, device)

    train_and_evaluate(net, src_dataloaders, config, device)
