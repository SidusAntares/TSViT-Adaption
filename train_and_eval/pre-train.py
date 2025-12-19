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
from models.CSDI.main_model import CSDI_da
from models.CSDI.utils import train, evaluate


parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

config_file = '/data/user/ViT/TSViT-Adaption/configs/tsvit_transfer/Germany2PASTIS.yaml'
save_path = '/data/user/ViT/TSViT-Adaption/test/look_dataset/germany'
os.makedirs(save_path, exist_ok=True)  # 确保保存路径存在

main_config = read_yaml(config_file)
src_dataloaders = get_dataloaders(main_config, 'src')
trg_dataloaders = get_dataloaders(main_config, 'trg')


with open('base.yaml', "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save" + str(args.nfold) + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

model = CSDI_Physio(config, args.device).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
