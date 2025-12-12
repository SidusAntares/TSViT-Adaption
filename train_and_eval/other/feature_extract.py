import sys
import os

sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
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


def train_and_evaluate(net, dataloaders, config, device, lin_cls=False):
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    local_device_ids = config['local_device_ids']
    loss_input_fn = get_loss_data_input(config)
    
    try:
        load_from_checkpoint(net, checkpoint, partial_restore=False)
    except FileNotFoundError as e:
        print(e)
        print("未找到checkpoint： ", checkpoint)
        sys.exit(1)
    else:
        print("已加载模型参数")


    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)

    net.train()
    feature_mean_map = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
    feature_norm_map = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0} # l2范数
    total_map = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

    for step, sample in enumerate(dataloaders['train']):
        # 测试光谱通道数为 1 情况，考虑还有个时间通道进入模型后会剥离
        # pastis
        # mean = sample['inputs'][:,:,:,:,:10].mean(dim = -1,keepdim = True)
        # sample['inputs'] = torch.cat(( mean,sample['inputs'][:,:,:,:,10].unsqueeze(-1) ), dim = -1)
        # germany
        # mean = sample['inputs'][:, :, :, :, :13].mean(dim=-1, keepdim=True)
        # sample['inputs'] = torch.cat((mean, sample['inputs'][:, :, :, :, 14].unsqueeze(-1)), dim=-1)
        feature , _ = net(sample['inputs'].to(device))
        ground_truth = loss_input_fn(sample, device)
        target, mask = ground_truth
        target = target.view(-1)[mask.view(-1)]
        mask = mask.view(-1)

        for i in range(6):
            label_mask = target == i
            dim = config['MODEL']['dim']
            feature_ = feature[:,:,:,i,:].view(-1,dim)[mask][label_mask]
            feature_mean_map[i] += feature_.mean(dim=-1).sum().item()
            feature_norm_map[i] += feature_.norm(p=2).sum().item()
            total_map[i] += feature_.shape[0]
        print(step)
            # print(label_mask.shape)
            # print(target.shape)
            # print(mask.shape)
            # # print(mask.expand(mask.shape[0],dim))
            # print(feature[:,:,:,i,:].view(-1,dim).shape)
            # print(feature[:,:,:,i,:].view(-1,dim)[mask][label_mask].shape)

    feature_mean_map = [key / total_map[key] for key in feature_mean_map]
    feature_norm_map = [key / total_map[key] for key in feature_norm_map]
    # feature_mean_map[
    #     0.0, 2.520082537743276e-07, 3.143171460003143e-05, 1.764649828770145e-06, 9.132732856148042e-06, 2.5435920810856457e-06]
    # feature_norm_map[
    #     0.0, 2.520082537743276e-07, 3.143171460003143e-05, 1.764649828770145e-06, 9.132732856148042e-06, 2.5435920810856457e-06]

    print("feature_mean_map", feature_mean_map)
    print("feature_norm_map", feature_norm_map)




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

    train_and_evaluate(net, dataloaders, config, device)
