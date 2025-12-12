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
from scipy.interpolate import PchipInterpolator


# 类别映射
name_dict = {0: 'beet', 1: 'meadow', 2: 'potatoes', 3: 'winter wheat', 4: 'winter barley', 5: 'corn'}
num_classes = len(name_dict)  # 6
time = np.arange(0, 366,10)
functions = []
def process(mode,save_path, dataloaders, config):
    # local_device_ids = config['local_device_ids']
    # collect = torch.load(collect_path, map_location='cpu') # torch.Size([6, 10, 366])
    # collect =collect[collect!=0]
    # for i in range(6):
    #     for j in range(10):
    #         data = collect[i,j]
    #         interpolator = PchipInterpolator(time, data, extrapolate=False)
    #         functions.append(interpolator)
    os.makedirs(save_path, exist_ok=True)
    save_sample = []
    save_labels = []
    for step, sample in enumerate(dataloaders[mode]):

            # print(sample.keys())dict_keys(['inputs', 'labels', 'seq_lengths', 'unk_masks'])
            # inputs = sample['inputs'].to(device)
            # print(sample['seq_lengths']) batchsize
            # print(sample['inputs'].shape) torch.Size([16, 60, 24, 24, 11])
        x = sample['inputs'].permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        doy = (x[:, :, -1, 0, 0]* 365.0001).to(torch.int64).reshape(-1).numpy()
        mask = doy != 0
        doy = doy[mask]
        inputs = sample['inputs'].permute(0,2,3,4,1).reshape(-1,C,T)[:,:-1,:].reshape(-1,T) # batch_size = 1 torch.Size([5760, 60])
            # print(inputs.shape)
            # inputs = inputs[mask.unsqueeze(0).expand_as(inputs)]torch.Size([345600])
            # inputs = inputs[mask.unsqueeze(0).expand(inputs.shape[0],inputs.shape[1])]
        labels = sample['labels'].reshape(-1) # if reshpe(1,-1) then stack will be torch.Size([b, 1, 576])
        # print(labels.shape)
            # print(mask.unsqueeze(0).expand(inputs.shape[0],inputs.shape[1]).shape)torch.Size([345600])
            # print(mask)
        batch_tensor = []
        unique_doy ,indices = np.unique(doy, return_inverse=True)
            # print(indices)
            # break
        for i in range(inputs.shape[0]):
            temp = inputs[i][mask].numpy()
                # print(temp.shape)torch.Size([38])
            unique_temp = np.zeros_like(unique_doy)
            for j, _ in enumerate(unique_doy):
                unique_temp[j] = np.mean(temp[indices == j])
                # print(unique_temp.shape)
                # print(unique_doy.shape)
            interpolator = PchipInterpolator(unique_doy, unique_temp, extrapolate=False)
            y = torch.from_numpy(interpolator(time))
            batch_tensor.append(y)
        batch_tensor = torch.stack(batch_tensor,dim=0).reshape(-1,10,time.shape[0])
        # print("batch_tensor.shape",batch_tensor.shape)
        save_sample.append(batch_tensor)
        save_labels.append(labels)
        print(step)
        # if step == 1:
        #     break
    save_sample = torch.stack(save_sample,dim=0).reshape(-1,10,time.shape[0])
    save_labels = torch.stack(save_labels,dim=0).reshape(-1)
    # print("save_sample.shape",save_sample.shape)torch.Size([1152, 10, 37])
    # print("save_labels.shape",save_labels.shape)torch.Size([1152])
    torch.save({'samples':save_sample,'labels':save_labels}, os.path.join(save_path, f'{mode}_1.pt'))



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
    # collect_path = '/data/user/Raincoat/data/PASTIS/pastis_function_cubic.pt'
    save_path = '/data/user/Raincoat/data/PASTIS_and_Germany'
    dataloaders = get_dataloaders(config)

    process("train",save_path, dataloaders, config)
    process("eval",save_path, dataloaders, config)