from dataset import align

import argparse
import torch
import datetime
import json
import yaml
import os
import sys
import os
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

from datetime import datetime
from models.CSDI.main_model import CSDI_Physio
from models.CSDI.utils import train, evaluate


config_file = '/data/user/ViT/TSViT-Adaption/configs/tsvit_transfer/Germany2PASTIS.yaml'
save_path = '/data/user/ViT/TSViT-Adaption/test/look_dataset/germany'
os.makedirs(save_path, exist_ok=True)  # 确保保存路径存在

main_config = read_yaml(config_file)
src_dataloaders = get_dataloaders(main_config, 'src')
trg_dataloaders = get_dataloaders(main_config, 'trg')
for sample in src_dataloaders['train']:
    aligned_values,_ = align(sample)
    print(aligned_values)
    break