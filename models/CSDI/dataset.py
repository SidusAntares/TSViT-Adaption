import pickle
import torch
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset



def align(values_list):
    # (B, T, C+1, H, W) -> (B, T, H, W, C+1)
    B, T, H, W, C_plus_1 = values_list.shape
    C = C_plus_1 - 1

    data = values_list[..., :-1]      # (B, T, H, W, C)
    raw_time = values_list[..., -1]   # (B, T, H, W), assumed in [0, 1)
    # 🔥 关键修复：先 clamp 到 [0, 1)，再转 DOY
    raw_time = torch.clamp(raw_time, 0.0, 1.0 - 1e-6)  # 防止 1.0 出现
    doy = torch.floor(raw_time * 365).long() + 1
    doy= doy.permute(0,2,3,1)
    data = data.permute(0,2,3,1,4)
    N = B * H * W
    data_flat = data.reshape(N, T, C)
    doy_flat = doy.reshape(N, T)

    aligned_vals = torch.zeros(N, 365, C, dtype=data.dtype, device=data.device)
    aligned_masks = torch.zeros(N, 365, dtype=torch.float32, device=data.device)
    batch_idx = torch.arange(N, device=data.device).unsqueeze(1).expand(N, T)
    time_idx = doy_flat - 1
    aligned_vals[batch_idx, time_idx] = data_flat
    aligned_masks[batch_idx, time_idx] = 1.0
    # aligned_vals[doy_flat] = data_flat
    return aligned_vals, aligned_masks



class daDataset(Dataset):
    def __init__(self, original_dataloader, missing_ratio=0.1, seed=0, use_index_list=None):
        """
        original_dataloader: 原始 dataloader，返回 {'inputs': (B, T, C+1, H, W)}
        missing_ratio: 隐藏比例
        use_index_list: 可选，只保留这些样本索引（用于 train/valid/test split）
        """
        self.samples = []
        self.missing_ratio = missing_ratio
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 预加载所有样本
        all_samples = []
        for batch in original_dataloader:
            inputs = batch['inputs']
            aligned_vals, observed_masks = align(inputs)
            N, L, C = aligned_vals.shape
            aligned_vals = aligned_vals.cpu().numpy()
            observed_masks = observed_masks.cpu().numpy()
            for i in range(N):
                obs_mask = observed_masks[i]
                gt_mask = self._create_gt_mask(obs_mask, missing_ratio)
                all_samples.append({
                    'observed_data': aligned_vals[i],
                    'observed_mask': observed_masks[i],
                    'gt_mask': gt_mask,
                    'timepoints': np.arange(365)
                })

        # 如果指定了 use_index_list，则只保留这些
        if use_index_list is not None:
            self.samples = [all_samples[i] for i in use_index_list]
        else:
            self.samples = all_samples

    def _create_gt_mask(self, observed_mask, missing_ratio):
        obs_indices = np.where(observed_mask)[0]
        n_hide = int(len(obs_indices) * missing_ratio)
        if n_hide == 0:
            return observed_mask.copy()
        hide_indices = np.random.choice(obs_indices, n_hide, replace=False)
        gt_mask = observed_mask.copy()
        gt_mask[hide_indices] = 0.0
        return gt_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        K = item['observed_data'].shape[1]
        obs_mask = np.repeat(item['observed_mask'][:, None], K, axis=1)
        gt_mask = np.repeat(item['gt_mask'][:, None], K, axis=1)
        return {
            "observed_data": torch.tensor(item['observed_data'], dtype=torch.float32),
            "observed_mask": torch.tensor(obs_mask, dtype=torch.float32),
            "gt_mask": torch.tensor(gt_mask, dtype=torch.float32),
            "timepoints": torch.tensor(item['timepoints'], dtype=torch.long),
        }

    def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1, original_dataloader=None):
        """
        构建 train/valid/test DataLoader，支持 5-fold 划分。

        Args:
            seed: 随机种子
            nfold: int in [0,4]，指定测试 fold；若为 None，则不分 fold（全部用于训练）
            batch_size: 批大小
            missing_ratio: 掩码比例
            original_dataloader: 原始 dataloader（必须提供！）

        Returns:
            train_loader, valid_loader, test_loader
        """
        assert original_dataloader is not None, "original_dataloader must be provided!"

        # 第一次加载全部数据以获取总长度和索引
        full_dataset = daDataset(
            original_dataloader=original_dataloader,
            missing_ratio=missing_ratio,
            seed=seed,
            use_index_list=None  # 加载全部
        )
        total_len = len(full_dataset)
        indlist = np.arange(total_len)

        np.random.seed(seed)
        np.random.shuffle(indlist)

        if nfold is not None and 0 <= nfold <= 4:
            # 5-fold: 20% test
            start = int(nfold * 0.2 * total_len)
            end = int((nfold + 1) * 0.2 * total_len)
            test_index = indlist[start:end]
            remain_index = np.delete(indlist, np.arange(start, end))
        else:
            # 不做 fold 划分：全部作为训练（可选）
            test_index = []
            remain_index = indlist

        # 在 remaining 中划分 train / valid (70% / 30% of remaining ≈ 56% / 24% of total)
        np.random.seed(seed)
        np.random.shuffle(remain_index)
        num_train = int(len(remain_index) * 0.7)
        train_index = remain_index[:num_train]
        valid_index = remain_index[num_train:]

        # 创建三个子数据集
        train_dataset = daDataset(
            original_dataloader=original_dataloader,
            missing_ratio=missing_ratio,
            seed=seed,
            use_index_list=train_index.tolist()
        )
        valid_dataset = daDataset(
            original_dataloader=original_dataloader,
            missing_ratio=missing_ratio,
            seed=seed,
            use_index_list=valid_index.tolist()
        )
        test_dataset = daDataset(
            original_dataloader=original_dataloader,
            missing_ratio=missing_ratio,
            seed=seed,
            use_index_list=test_index.tolist()
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader