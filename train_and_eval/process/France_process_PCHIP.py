import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
import sys
import os
from scipy.interpolate import PchipInterpolator
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
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
import sys
import os

sys.path.insert(0, os.getcwd())
import argparse
import torch

from scipy.interpolate import interp1d
from data import get_dataloaders
from utils.config_files_utils import read_yaml



TARGET_TIME = np.arange(0, 366, 10).astype(np.float32)  # (37,)


def process_with_class_filtering_and_averaging(mode, save_path, dataloaders, config, min_samples_per_class=5):
    """
    按类别聚合处理：过滤低样本数类别，只保留满足样本数下限的类别
    """
    os.makedirs(save_path, exist_ok=True)
    all_class_samples = []  # 存储每个批次处理后的类别样本
    all_class_labels = []  # 存储每个批次处理后的类别标签

    for step, sample in enumerate(dataloaders[mode]):
        inputs_raw = sample['inputs']  # [B=32, T, H=24, W=24, C+1]
        labels = sample['labels']  # [32, 24, 24]
        inputs_raw = torch.cat([inputs_raw[..., 0:10], inputs_raw[..., 13].unsqueeze(-1)], dim=-1)[
            ..., [2, 1, 0, 4, 5, 6, 3, 7, 8, 9, 10]]
        B, T, H, W, C_plus_1 = inputs_raw.shape
        C = C_plus_1 - 1  # e.g., 10
        num_classes = 5  # 你提到有6个类别

        # === Step 1: Extract shared DOY sequence from ONE pixel ===
        doy_norm = inputs_raw[0, :, 0, 0, -1]  # [T]
        doy = (doy_norm * 365.0001).cpu().numpy().astype(np.float32)  # [T]
        valid_mask = doy != 0
        doy_valid = doy[valid_mask]  # [T_v]

        if len(doy_valid) < 2:
            print(f"Skip batch {step}: insufficient valid time points")
            continue

        # Labels
        labels_flat = labels.reshape(-1)  # [18432]
        labels_mask = labels_flat != 5
        labels_flat = labels_flat[labels_mask]

        # Handle duplicate DOYs by averaging (rare but safe)
        unique_doy, inverse = np.unique(doy_valid, return_inverse=True)
        T_u = len(unique_doy)

        # === Step 2: Reshape data to [N_pixels, T, C] ===
        data = inputs_raw[..., :-1]  # [32, T, 24, 24, C]
        data_flat = data.permute(0, 2, 3, 1, 4).reshape(-1, T, C)[labels_mask]  # [18432, T, C]
        N_pixels = data_flat.shape[0]
        data_valid = data_flat[:, valid_mask, :]  # [18432, T_v, C]

        # Aggregate duplicates in time (if any)
        if T_u == len(doy_valid):
            agg_data = data_valid  # [N, T_u, C]
        else:
            # Average over duplicate time indices
            agg_data = torch.zeros(N_pixels, T_u, C, device=data.device)
            for j in range(T_u):
                mask_j = inverse == j  # [T_v]
                agg_data[:, j, :] = data_valid[:, mask_j, :].mean(dim=1)
            agg_data = agg_data.cpu().numpy()
        agg_data = agg_data if isinstance(agg_data, np.ndarray) else agg_data.cpu().numpy()

        # === Step 3: 按类别统计样本数量 ===
        unique_labels= torch.unique(labels_flat)
        # print(unique_labels)tensor([0., 1., 3., 4.])
        # unique_labels, count_labels= torch.unique(labels_flat,return_counts=True)
        # valid_classes = [unique_labels[i] for i in range(6) if count_labels[i] >= min_samples_per_class]

        class_counts = {}
        for class_id in unique_labels:
            class_counts[class_id.item()] = (labels_flat == class_id).sum().item()

        # 过滤样本数少于下限的类别
        valid_classes = [int(cls) for cls, count in class_counts.items() if count >= min_samples_per_class]

        if not valid_classes:
            print(f"Skip batch {step}: no classes meet minimum sample requirement of {min_samples_per_class}")
            continue

        # === Step 4: 为有效类别计算均值 ===
        class_means = np.full((num_classes, T_u, C), np.nan, dtype=np.float32)
        class_exists = [False] * num_classes # 创建一个长度为 num_classes 的布尔列表，初始值全部设为 False。

        for class_id in valid_classes:

            class_mask = (labels_flat == class_id).cpu().numpy()
            class_data = agg_data[class_mask]  # [N_class, T_u, C]

            # 计算该类别的均值
            class_mean = np.mean(class_data, axis=0)  # [T_u, C]
            # print(class_means.shape)
            class_means[class_id] = class_mean
            class_exists[class_id] = True

        # === Step 5: 为有效类别进行 PCHIP 插值 ===
        interpolated_classes = np.full((num_classes, C, len(TARGET_TIME)), np.nan, dtype=np.float32)

        for cls in range(num_classes):
            if not class_exists[cls]:
                continue

            # 确保 unique_doy 是严格递增的（np.unique 已保证，但 double-check）
            x = unique_doy
            if len(x) < 2:
                # PCHIP 至少需要两个点
                continue

            for c in range(C):
                y = class_means[cls, :, c]  # shape: [T_u]

                # 构建 PCHIP 插值器
                try:
                    pchip = PchipInterpolator(x, y, extrapolate=False)
                except ValueError as e:
                    print(f"Warning: PCHIP failed for class {cls}, channel {c}: {e}")
                    continue

                # 手动处理外推：只在 [x.min(), x.max()] 内插值，外部设为 NaN
                t_min, t_max = x[0], x[-1]
                valid_target_mask = (TARGET_TIME >= t_min) & (TARGET_TIME <= t_max)
                valid_targets = TARGET_TIME[valid_target_mask]

                if len(valid_targets) > 0:
                    interpolated_vals = pchip(valid_targets)
                    interpolated_classes[cls, c, valid_target_mask] = interpolated_vals

                # 外部区域保持 NaN（已由 full 初始化保证）

        # === Step 6: 只保留当前batch中存在的有效类别 ===
        valid_class_samples = []
        valid_class_labels = []

        for cls in valid_classes:
            if class_exists[cls]:
                class_sample = interpolated_classes[cls:cls + 1, :, :]  # [1, C, 37]
                valid_class_samples.append(torch.from_numpy(class_sample).float())
                valid_class_labels.append(torch.tensor([cls], dtype=torch.long))

        if valid_class_samples:
            batch_class_samples = torch.cat(valid_class_samples, dim=0)  # [num_valid_classes_in_batch, C, 37]
            batch_class_labels = torch.cat(valid_class_labels, dim=0)  # [num_valid_classes_in_batch]

            all_class_samples.append(batch_class_samples)
            all_class_labels.append(batch_class_labels)

        print(
            f"{mode} Step {step}, processed {N_pixels} pixels, found {len(valid_classes)} valid classes (min {min_samples_per_class})")

    # === Step 7: 合并所有批次 ===
    if all_class_samples:
        final_samples_tensor = torch.cat(all_class_samples, dim=0)  # [total_valid_samples, C, 37]
        final_labels_tensor = torch.cat(all_class_labels, dim=0)  # [total_valid_samples]
        # final_samples_tensor = final_samples_tensor[..., 2:-14]
        torch.save({
            'samples': final_samples_tensor,
            'labels': final_labels_tensor
        }, os.path.join(save_path, f'{mode}_france.pt'))
        # }, os.path.join(save_path, f'{mode}_3.pt'))

        print(f"Saved {mode}: samples {final_samples_tensor.shape}, labels {final_labels_tensor.shape}")
        print(f"Unique labels in final dataset: {torch.unique(final_labels_tensor).tolist()}")
    else:
        print(f"No valid samples found for {mode}")


if __name__ == "__main__":
    device_ids = [int(d) for d in "0,1,2,3".split(',')]
    config_file = '/data/user/ViT/D3/configs/init_transfer/France2France.yaml'

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids
    save_path = '/data/user/Raincoat/data/PASTIS_and_Germany_and_France'
    dataloaders = get_dataloaders(config)
    min_samples_per_class = 5  # 设置样本数下限

    # 使用第一种方法：每个batch输出有效类别的样本
    process_with_class_filtering_and_averaging("train", save_path, dataloaders, config, min_samples_per_class)
    process_with_class_filtering_and_averaging("eval", save_path, dataloaders, config, min_samples_per_class)



