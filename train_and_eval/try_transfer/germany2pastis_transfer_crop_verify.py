import sys
import os

sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
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
from collections import defaultdict
import torch.nn.functional as F

name_dict = {0:'beet',1:'meadow',2:'potatoes',3:'winter wheat',4:'winter barley',5:'corn'}

def train_and_evaluate(net, dataloaders, config, device, class_num,class_name):
    # ------------------------------------------------------------------------------------------------------------------#
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    local_device_ids = config['local_device_ids']

    metric_predict = {0:0,1:0,2:0,3:0,4:0,5:0}
    total = 0

    if checkpoint:
        try:
            load_from_checkpoint(net, checkpoint, partial_restore=False)
        except FileNotFoundError as e:
            print(e)
            print("未找到checkpoint： ", checkpoint)
            sys.exit(1)
        else:
            print("已加载模型参数")

    else:
        print("未加载模型参数")

    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)
    loss_input_fn = get_loss_data_input(config)
    net.eval()
    for step,sample in enumerate(dataloaders['eval']):

        # pad = (0, 4, 0, 0, 0, 0, 0, 0, 0, 0)
        # sample['inputs'] = F.pad(sample['inputs'], pad, mode="constant", value=0)
        tail = sample['inputs'][:,:,:,:,10]
        tail = tail.unsqueeze(-1).expand(-1,-1,-1,-1,4)
        sample['inputs'] = torch.cat((sample['inputs'], tail), dim=-1)
        outputs = net(sample['inputs'].to(device))
        outputs = outputs.permute(0, 2, 3, 1)
        _, predicted = torch.max(outputs.data, -1)
        # print(predicted)
        ground_truth = loss_input_fn(sample, device)
        labels, unk_masks = ground_truth
        if unk_masks is not None:
            labels = labels.view(-1)[unk_masks.view(-1)]
            predicted = predicted.view(-1)[unk_masks.view(-1)]
        labels =labels.flatten()
        predicted = predicted.flatten()
        for i,j in zip(labels, predicted):
            if i.item() == class_num : # 我要知道目标类型被识别为什么类型
                metric_predict[j.item()]+=1
                total+=1
                print(f"{class_name}: ",metric_predict)
        print(step)
    metric_predict = dict(sorted(metric_predict.items(), key=lambda item: item[0]))
    metric_predict = {name_dict[k]: v / total for k, v in metric_predict.items()}
    df = pd.DataFrame([metric_predict])
    df.insert(0,"class",f'{class_name}')
    folder_path = os.path.splitext(os.path.basename(args.config))[0]

    rootdir = f"/data/user/ViT/D3/results/transfer/{folder_path}_predict.csv"
    if not os.path.exists(rootdir):
        df.to_csv(rootdir, index=False,mode='w')
    else:
        df.to_csv(rootdir,index=False,mode='a',header =False)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0,1', type=str,
                        help='gpu ids to use')
    parser.add_argument('--lin', action='store_true',
                        help='train linear classifier only')

    args = parser.parse_args()
    config_file = args.config
    print(args.device)
    device_ids = [int(d) for d in args.device.split(',')]
    lin_cls = args.lin

    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids

    dataloaders = get_dataloaders(config)

    net = get_model(config, device)

    train_and_evaluate(net, dataloaders, config, device,0,'beet')
    train_and_evaluate(net, dataloaders, config, device,1,'meadow')
    train_and_evaluate(net, dataloaders, config, device,2,'potatoes')
    train_and_evaluate(net, dataloaders, config, device,3,'winter wheat')
    train_and_evaluate(net, dataloaders, config, device,4,'winter barley')
    train_and_evaluate(net, dataloaders, config, device,5,'corn')
    # name_dict = {0:'beet',1:'meadow',2:'potatoes',3:'winter wheat',4:'winter barley',5:'corn'}

