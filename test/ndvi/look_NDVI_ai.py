import sys
import os

sys.path.insert(0, os.getcwd())
import torch
import numpy as np
import pandas as pd
from utils.config_files_utils import read_yaml
from data import get_dataloaders

dir = 'configs/transfer/PASTIS.yaml'
# dir = 'configs/transfer/Germany.yaml'
config = read_yaml(dir)
folder = os.path.splitext(os.path.basename(dir))[0]
dataloaders = get_dataloaders(config)
name_dict = {0: 'beet', 1: 'meadow', 2: 'potatoes',
             3: 'winter wheat', 4: 'winter barley', 5: 'corn'}

# 初始化统计张量
metric = torch.zeros(6, 366)
total = torch.zeros(6, 366)

for step, sample in enumerate(dataloaders['train']):
    x = sample['inputs']
    x = x.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
    B, T, C, H, W = x.shape

    # print(doy_all[0,:,0,0])
    # 获取标签并确保整数类型
    labels = sample['labels'].to(torch.int64)  # 确保标签是整数类型
    # 创建有效掩码（过滤doy=0和label=6）
    if folder == 'Germany'  :
        # print(labels.shape)torch.Size([32, 24, 24, 1])
        labels = labels.squeeze(-1)
        nir = x[:, :, 3, :, :]  # 近红外波段
        red = x[:, :, 0, :, :]  # 红光波段
        doy_all = (x[:, :, -2, :, :] * 365.0001).to(torch.int64)
        # print(labels.shape)
    else:
        nir = x[:, :, 6, :, :]  # 近红外波段
        red = x[:, :, 2, :, :]  # 红光波段
        doy_all = (x[:, :, -1, :, :] * 365.0001).to(torch.int64)
    # 向量化计算NDVI
    ndvi = (nir - red) / (nir + red + 1e-10)

    valid_mask = (doy_all > 0) & (doy_all <= 366) & (labels.unsqueeze(1) != 6)

    for b in range(B):
        batch_mask = valid_mask[b]
        if not batch_mask.any():
            continue

        # 提取有效数据并确保整数类型
        batch_doy = (doy_all[b][batch_mask] - 1).to(torch.int64)  # 确保long类型
        batch_labels = labels[b].unsqueeze(0).expand(T, H, W)[batch_mask].to(torch.int64)
        batch_ndvi = ndvi[b][batch_mask]

        # 为每个标签和doy组合创建唯一索引
        combined_idx = batch_labels * 366 + batch_doy

        # 使用bincount进行快速聚合
        unique_idx, counts = torch.unique(combined_idx, return_counts=True)

        # 计算每个唯一索引的NDVI总和
        sums = torch.zeros_like(unique_idx, dtype=torch.float32)
        for i, idx in enumerate(unique_idx):
            mask = combined_idx == idx
            sums[i] = batch_ndvi[mask].sum()

        # 更新全局统计 - 关键修正：确保索引是整数
        for i in range(len(unique_idx)):
            idx = unique_idx[i].item()  # 转换为Python整数
            label = idx // 366
            doy = idx % 366

            # 确保索引在有效范围内
            if 0 <= label < 6 and 0 <= doy < 366:
                count = counts[i].item()
                sum_val = sums[i].item()

                total[label, doy] += count
                metric[label, doy] += sum_val

    print(f'处理进度: {step}批次, 有效像素: {valid_mask.sum().item()}')

# 计算平均NDVI
valid_mask = total > 0
metric[valid_mask] /= total[valid_mask]
metric[~valid_mask] = -1  # 标记无数据的天数

# 创建DataFrame并保存
df = pd.DataFrame(metric.numpy(),
                  index=[name_dict[i] for i in range(6)],
                  columns=range(1, 367))
df.to_csv(f'/data/user/ViT/D3/test/{folder}_ndvi_doy.csv')
print("数据处理完成并已保存")