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


def compute_confusion_matrix(net, dataloaders, config, device):
    """
    计算整个验证集上的混淆矩阵（真实 vs 预测）
    返回一个 [num_classes, num_classes] 的 numpy 数组
    """
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    local_device_ids = config['local_device_ids']

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
    net.eval()  # 注意这里是 eval 模式！

    loss_input_fn = get_loss_data_input(config)

    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for step, sample in enumerate(dataloaders['eval']):

            # tail = sample['inputs'][:, :, :, :, 10] # germany2pastis
            # tail = tail.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
            # sample['inputs'] = torch.cat((sample['inputs'], tail), dim=-1)

            # input1 = sample['inputs'][:,:,:,:,:10] # pastis2germany
            # input2 = sample['inputs'][:,:,:,:,10:15].mean(dim=-1,keepdim=True)
            # sample['inputs'] = torch.cat((input1, input2), dim=-1)

            if config['DATASETS']['eval']['dataset'] == 'MTLCC':
                mean = sample['inputs'][:, :, :, :, :14].mean(dim=-1, keepdim=True)
                sample['inputs'] = torch.cat((mean, sample['inputs'][:, :, :, :, 14].unsqueeze(-1)), dim=-1)
            else:
                mean = sample['inputs'][:,:,:,:,:10].mean(dim = -1,keepdim = True)
                sample['inputs'] = torch.cat(( mean,sample['inputs'][:,:,:,:,10].unsqueeze(-1) ), dim = -1)

            inputs = sample['inputs'].to(device)
            outputs = net(inputs)

            # 假设你的模型输出是 [B, C, H, W]，需要先 permute 成 [B, H, W, C]，然后取 argmax(-1)
            outputs = outputs.permute(0, 2, 3, 1).contiguous()
            preds = torch.argmax(outputs, dim=-1)  # [B, H, W]

            # 获取 ground truth 标签
            ground_truth = loss_input_fn(sample, device)
            labels, unk_masks = ground_truth

            if unk_masks is not None:
                labels = labels.view(-1)[unk_masks.view(-1)]
                # print(preds.shape)
                # print(unk_masks.shape)
                # torch.Size([24, 24, 24])
                # torch.Size([24, 24, 24, 1])
                preds = preds.view(-1)[unk_masks.view(-1)]

            # 更新混淆矩阵
            for true, pred in zip(labels, preds):
                # print(type(true.item()), type(pred.item()))
                # print(true.item(), pred.item())
                #<class 'float'> <class 'int'> 1.0 0
                t = int(true.item())  # 转为 Python int
                p = pred.item()  # 转为 Python int
                confusion_matrix[t, p] += 1

            if step % 10 == 0:
                print(f"Step {step}, confusion matrix updated.")

    # 转成 numpy
    confusion_matrix = confusion_matrix.cpu().numpy()
    return confusion_matrix


def save_confusion_matrix_to_csv(confusion_matrix, class_names, output_path):
    # 归一化为比例（按行，即每个真实类别的预测分布）
    confusion_matrix_percent = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1, keepdims=True) * 100
    confusion_matrix_percent = np.nan_to_num(confusion_matrix_percent, nan=0.0)  # 避免除以0

    # 保存比例矩阵
    df_percent = pd.DataFrame(confusion_matrix_percent, index=class_names, columns=class_names)
    df_percent.to_csv(output_path)
    print(f"混淆矩阵（比例%）已保存至：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', required=True, help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0', type=str, help='gpu ids to use')
    parser.add_argument('--lin', action='store_true', help='train linear classifier only')

    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(',')]
    config_file = args.config
    lin_cls = args.lin

    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids

    dataloaders = get_dataloaders(config)

    net = get_model(config, device)

    # 混淆矩阵计算
    confusion_matrix = compute_confusion_matrix(net, dataloaders, config, device)

    # 保存为 CSV
    folder_path = os.path.splitext(os.path.basename(args.config))[0]
    csv_output_path = f"/data/user/ViT/D3/results/transfer_2_channels/{folder_path}_confusion_matrix.csv"

    save_confusion_matrix_to_csv(confusion_matrix, list(name_dict.values()), csv_output_path)