import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# from train_and_eval.utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
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
from utils.summaries import write_mean_summaries, write_class_summaries
import pandas as pd
import shutil
from datetime import datetime
import glob

def mIou(y_true, y_pred, n_classes):
    """
    Mean Intersect over Union metric.
    Computes the one versus all IoU for each class and returns the average.
    Classes that do not appear in the provided set are not counted in the average.
    Args:
        y_true (1D-array): True labels
        y_pred (1D-array): Predicted labels
        n_classes (int): Total number of classes
    Returns:
        mean Iou (float)
    """
    iou = 0
    n_observed = n_classes
    for i in range(n_classes):
        y_t = (np.array(y_true) == i).astype(int)
        y_p = (np.array(y_pred) == i).astype(int)

        inter = np.sum(y_t * y_p)
        union = np.sum((y_t + y_p > 0).astype(int))

        if union == 0:
            n_observed -= 1
        else:
            iou += inter / union

    return iou / n_observed


def train_and_evaluate(net, dataloaders, config, device, lin_cls=False):

    def train_step(net, sample, loss_fn, optimizer, device):
        optimizer.zero_grad()
        outputs = net(sample['inputs'].to(torch.float32).to(device))
        ground_truth = sample['labels'][:, center, 0].to(torch.int64).to(device)
        unk_masks = sample['unk_masks'][:, center, 0].to(device)
        loss = loss_fn['mean'](outputs[unk_masks], ground_truth[unk_masks])
        loss.backward()
        optimizer.step()
        return outputs, ground_truth, loss

    def evaluate(net, evalloader, loss_fn, config):
        num_classes = config['MODEL']['num_classes']
        Neval = len(evalloader)

        predicted_all = []
        labels_all = []

        net.eval()
        with torch.no_grad():
            for step, sample in enumerate(evalloader):

                if step % 500 == 0:
                    print("Eval step %d of %d" % (step, Neval))

                logits = net(sample['inputs'].to(torch.float32).to(device))

                _, predicted = torch.max(logits.data, -1)

                predicted_all.append(predicted.view(-1).cpu().numpy())
                labels_all.append(sample['labels'][:, center, 0].view(-1).to(torch.int64).cpu().numpy())

                # if step > 5:
                #    break

        print("finished iterating over dataset after step %d" % step)
        print("calculating metrics...")
        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        unk_masks = (target_classes != 19)
        eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                                  n_classes=num_classes, unk_masks=unk_masks)

        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

        un_labels = np.unique(target_classes)

        print("-------------------------------------------------------------------------------------------------------")
        print("Mean (micro) Evaluation metrics (micro/macro), iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
              "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s" %
              (micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
               micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes)))
        print("-------------------------------------------------------------------------------------------------------")
        return [un_labels,
                {"macro": {"Accuracy": macro_acc, "Precision": macro_precision,
                           "Recall": macro_recall, "F1": macro_F1, "IOU": macro_IOU},
                 "micro": {"Accuracy": micro_acc, "Precision": micro_precision,
                           "Recall": micro_recall, "F1": micro_F1, "IOU": micro_IOU},
                 "class": {"Accuracy": class_acc, "Precision": class_precision,
                           "Recall": class_recall,
                           "F1": class_F1, "IOU": class_IOU}}]


    num_classes = config['MODEL']['num_classes']
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    save_epoch_interval = config['CHECKPOINT']['save_epoch_interval']
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)

    folder_name = os.path.basename(os.path.dirname(args.config))
    base_dir = "/data/user/ViT/D2/metric_save/"
    folder_dir = os.path.join(base_dir, folder_name)

    start_global = 1
    start_epoch = 1

    print("current learn rate: ", lr)

    loss_fn = {'all': get_loss(config, device, reduction=None),
               'mean': get_loss(config, device, reduction="mean")}

    # if lin_cls:
    #     print('Train linear classifier only')
    #     trainable_params = get_net_trainable_params(net)[-2:]
    # else:
    #     print('Train network end-to-end')
    #     trainable_params = get_net_trainable_params(net)
    trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad()
    scheduler = build_scheduler(config, optimizer, num_steps_train)

    if checkpoint:
        try:
            load_from_checkpoint(net, checkpoint, partial_restore=False)
        except FileNotFoundError as e:
            print(e)
            print("未找到checkpoint： ",checkpoint)
            sys.exit(1)
        else:
            save_folder_name = os.path.basename(os.path.dirname(checkpoint))
            metric_save_path = os.path.join(folder_dir, save_folder_name)
            optimizer_save_path = glob.glob(os.path.join(metric_save_path, "*_optimizer.pth"))
            scheduler_save_path = glob.glob(os.path.join(metric_save_path, "*_scheduler.pth"))

            start_epoch = int(os.path.splitext(os.path.basename(checkpoint))[0]) + 1
            metrics_csv_path = f"{metric_save_path}/epoch_metrics.csv"
            df = pd.read_csv(metrics_csv_path)
            filtered_df = df.drop(index=range(start_epoch, len(df)))
            filtered_df.to_csv(metrics_csv_path, index=False)  # 保存回原文件（覆盖）

            print("已加载模型参数")

            if optimizer_save_path:
                optimizer_state_dict = torch.load(optimizer_save_path[0], map_location=device)
                optimizer.load_state_dict(optimizer_state_dict)
                print("优化器参数已加载")
            else:
                print("未加载优化器参数")

            if scheduler_save_path:
                scheduler_state_dict = torch.load(scheduler_save_path[0], map_location=device)
                scheduler.load_state_dict(scheduler_state_dict)
                print("scheduler参数已加载")
            else:
                print("未加载scheduler参数")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = os.path.splitext(os.path.basename(args.config))[0]
        metric_save_path = os.path.join(folder_dir, f"{task_name}_{timestamp}")
        # os.makedirs(metric_save_path, exist_ok=True)
        # if os.path.exists(metric_save_path):
        #     print("保存目录已创建")
        # else:
        #     print("未成功创建保存目录")
        #     sys.exit(1)
        # shutil.copy2(args.config, os.path.join(metric_save_path, "config_used.yaml"))
        # metrics_csv_path = f"{metric_save_path}/epoch_metrics.csv"
        print("未加载模型参数\n未加载优化器参数\n未加载scheduler参数")



    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)

    net.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):  # loop over the dataset multiple times
        epoch_train_loss = 0.0
        num_train_batches = 0
        for step, sample in enumerate(dataloaders['train']):

            abs_step = start_global + (epoch - start_epoch) * num_steps_train + step

            logits, labels, loss = train_step(net, sample, loss_fn, optimizer, device)

            epoch_train_loss += loss.item()
            num_train_batches += 1

            # if abs_step % train_metrics_steps == 0:
            if abs_step % 2 == 0:
                batch_metrics = get_mean_metrics(
                    logits=logits, labels=labels, unk_masks=None, n_classes=num_classes, loss=loss, epoch=epoch,
                    step=step)
                print("abs_step: %d, epoch: %d, step: %5d, loss: %.7f, batch_iou: %.4f, batch accuracy: %.4f, batch precision: %.4f, "
                      "batch recall: %.4f, batch F1: %.4f, lr: %.6f" %
                      (abs_step, epoch, step + 1, loss, batch_metrics['IOU'], batch_metrics['Accuracy'], batch_metrics['Precision'],
                       batch_metrics['Recall'], batch_metrics['F1'], optimizer.param_groups[0]["lr"]))
                print('predicted:\t', torch.max(logits.data, -1)[1].cpu().tolist(), end='\n')
                print('ground truths:', labels.cpu().tolist())
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr}")
        scheduler.step_update(abs_step)

        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        print(f"\nRunning Evaluation at End of Epoch {epoch}")
        eval_metrics = evaluate(net, dataloaders['eval'], loss_fn, config)
        net.train()
        macro_iou = eval_metrics[1]['macro']['IOU']
        micro_iou = eval_metrics[1]['micro']['IOU']
        accuracy = eval_metrics[1]['macro']['Accuracy']
        precision = eval_metrics[1]['macro']['Precision']
        recall = eval_metrics[1]['macro']['Recall']
        f1 = eval_metrics[1]['macro']['F1']

        # 构造该 epoch 的指标字典
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "mAcc": accuracy,
            "OA": eval_metrics[1]['micro']['Accuracy'],
            "lr": optimizer.param_groups[0]["lr"],
            "val_macro_IOU": macro_iou,
            "val_micro_IOU": micro_iou,
            "val_precision": precision,
            "val_recall": recall,
            "val_F1": f1,
        }

        # 每个 epoch 训练完立即写入 CSV（防止中断丢失）
        epoch_metrics_df = pd.DataFrame([epoch_metrics])  # 注意是单行数据，所以用 []

        # 判断文件是否存在，决定写入模式
        if not os.path.exists(metrics_csv_path):
            epoch_metrics_df.to_csv(metrics_csv_path, index=False, mode='w')
        else:
            epoch_metrics_df.to_csv(metrics_csv_path, index=False, mode='a', header=False)

        print(f"Epoch {epoch} metrics saved to: {metrics_csv_path}")
        if avg_train_loss < 0.01:
            break

        if epoch % save_epoch_interval == 0:
            checkpoint_filename = os.path.join(metric_save_path, f'{epoch:04d}.pth')
            checkpoint_optimizer_path = os.path.join(metric_save_path, f'{epoch:04d}_optimizer.pth')
            checkpoint_scheduler_path = os.path.join(metric_save_path, f'{epoch:04d}_scheduler.pth')
            torch.save(optimizer.state_dict(), checkpoint_optimizer_path)
            torch.save(scheduler.state_dict(), checkpoint_scheduler_path)
            if len(local_device_ids) > 1:
                torch.save(net.module.state_dict(), checkpoint_filename)
            else:
                torch.save(net.state_dict(), checkpoint_filename)
            print(f"[INFO] 已保存 checkpoint 到：{checkpoint_filename}")
            print(f"[INFO] 已保存 optimizer 到：{checkpoint_optimizer_path}")
            print(f"[INFO] 已保存 scheduler 到：{checkpoint_scheduler_path}")
            # --- 删除旧的 checkpoint，只保留最新的一个 ---
            checkpoint_files = glob.glob(os.path.join(metric_save_path, '*.pth'))
            if len(checkpoint_files) > 1:
                for f in checkpoint_files:
                    if f != checkpoint_filename and f != checkpoint_optimizer_path and f != checkpoint_scheduler_path:
                        os.remove(f)
                        print(f"[INFO] 删除旧 checkpoint：{f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', nargs='+', default=[0, 1], type=int,
                        help='gpu ids to use')
    parser.add_argument('--lin', action='store_true',
                         help='train linear classifier only')

    args = parser.parse_args()
    config_file = args.config
    device_ids = args.device
    lin_cls = args.lin

    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids

    center = config['MODEL']['img_res'] // 2
    dataloaders = get_dataloaders(config)

    net = get_model(config, device)

    train_and_evaluate(net, dataloaders, config, device)
