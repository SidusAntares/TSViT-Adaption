# train_da.py
import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from models import get_model # 确保你的 get_model 能正确加载上面修改后的 TSViT
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders # 确保它能分别返回 src 和 trg loader
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss # 假设这是分割损失
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input
import pandas as pd
import shutil
from datetime import datetime
import glob
# --- 聚类相关 ---
from sklearn.cluster import KMeans
import torch.nn.functional as F # 用于可能的距离计算

# --- 假设你有一个计算聚类损失的函数 (根据论文具体公式实现) ---
def compute_clustering_loss(feat_s, feat_t, cluster_centers, device):
    """
    计算聚类对齐损失 L_cluster (示例性伪代码，需根据论文细节调整).
    Args:
        feat_s (Tensor): 源域特征 [B_s, N_patch, dim]
        feat_t (Tensor): 目标域特征 [B_t, N_patch, dim]
        cluster_centers (np.ndarray): KMeans 聚类中心 [num_clusters, dim]
        device (torch.device): 设备
    Returns:
        Tensor: 聚类损失标量
    """
    # 1. 将特征展平以便计算距离
    flat_feat_s = feat_s.view(-1, feat_s.shape[-1]) # [B_s*N_patch, dim]
    flat_feat_t = feat_t.view(-1, feat_t.shape[-1]) # [B_t*N_patch, dim]
    all_feats = torch.cat([flat_feat_s, flat_feat_t], dim=0) # [Total_N, dim]

    # 2. 将聚类中心转为 Tensor
    centers_tensor = torch.from_numpy(cluster_centers).float().to(device) # [num_clusters, dim]

    # 3. 计算每个特征点到所有聚类中心的距离
    # 使用负欧氏距离的平方作为相似度（越小越相似）
    # sim_matrix shape: [Total_N, num_clusters]
    sim_matrix = -torch.cdist(all_feats, centers_tensor, p=2) ** 2

    # 4. 计算 softmax 概率 (模拟分配概率)
    probs = F.softmax(sim_matrix, dim=1) # [Total_N, num_clusters]

    # 5. 计算熵 (鼓励硬分配，即每个点强烈属于某一个簇)
    # 熵越小越好，这里取负号，使损失最小化时熵也最小
    entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))

    # 6. (可选) 计算源域和目标域特征分布的一致性 (例如 MMD 或者 KL 散度 between domain-wise cluster assignments)
    # 这里简化，可以进一步扩展

    # 示例：假设我们希望最大化源域和目标域分配给同一簇的概率 (启发式)
    # 或者最小化它们分配概率分布的差异 (KL散度)
    # 这部分是论文核心，需要仔细实现

    # 为了演示，我们只返回熵损失的一部分
    # 实际应用中，需要根据论文公式组合不同的项
    return entropy_loss # * beta_cluster 需要在外部乘以超参数

# --- 假设你有一个定期更新聚类中心的函数 ---
def update_cluster_centers(model, src_loader, device, num_clusters, config):
    """
    在源域数据上运行推理，收集特征，并更新聚类中心。
    Args:
        model (nn.Module): 训练中的模型
        src_loader (DataLoader): 源域数据加载器
        device (torch.device): 设备
        num_clusters (int): 聚类数量 (可能等于类别数，也可能不同，取决于论文)
        config (dict): 配置
    Returns:
        np.ndarray: 更新后的聚类中心 [num_clusters, dim]
    """
    collected_features = []
    model.eval() # 设置为评估模式，避免 BatchNorm/Dropout 影响
    with torch.no_grad():
        for step, sample in enumerate(src_loader):
            # 数据预处理 (与训练时一致)
            inputs = sample['inputs'].to(device) # [B, T, C+1, H, W]
            # --- 特殊处理通道数 ---
            # pastis: mean = inputs[:, :, :, :, :10].mean(dim=-1, keepdim=True); inputs = torch.cat((mean, inputs[:, :, :, :, 10].unsqueeze(-1)), dim=-1)
            # germany:
            mean = inputs[:, :, :, :, :13].mean(dim=-1, keepdim=True)
            inputs = torch.cat((mean, inputs[:, :, :, :, 14].unsqueeze(-1)), dim=-1)

            _, da_features = model(inputs, return_da_features=True) # [B, N_patch, dim]
            collected_features.append(da_features.cpu().numpy()) # 收集到 CPU

            # 可以限制收集的数据量以防内存溢出
            # if step > some_limit:
            #     break

    model.train() # 恢复训练模式
    if not collected_features:
        raise ValueError("No features were collected from source domain for clustering.")

    all_features_np = np.concatenate(collected_features, axis=0) # [Total_B*N_patch, dim]
    all_features_flat = all_features_np.reshape(-1, all_features_np.shape[-1]) # [Total_B*N_patch, dim]

    print(f"[Clustering] Running KMeans on {all_features_flat.shape[0]} points...")
    # 使用 sklearn KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(all_features_flat)
    print("[Clustering] KMeans completed.")
    return kmeans.cluster_centers_ # 返回 numpy array [num_clusters, dim]


def train_and_evaluate(net, src_dataloader, trg_dataloader, config, device):
    """域适应训练主循环"""

    def train_step(net, src_sample, trg_sample, loss_fn_seg, optimizer, device, loss_input_fn, cluster_centers, beta_cluster, lambda_entropy):
        optimizer.zero_grad()

        # --- 源域处理 ---
        src_inputs = src_sample['inputs'].to(device)
        # --- 通道数特殊处理 ---
        # pastis: mean = src_inputs[:, :, :, :, :10].mean(dim=-1, keepdim=True); src_inputs = torch.cat((mean, src_inputs[:, :, :, :, 10].unsqueeze(-1)), dim=-1)
        # germany:
        mean_src = src_inputs[:, :, :, :, :13].mean(dim=-1, keepdim=True)
        src_inputs = torch.cat((mean_src, src_inputs[:, :, :, :, 14].unsqueeze(-1)), dim=-1)

        src_logits, src_da_feat = net(src_inputs, return_da_features=True) # [B_s, K, H, W], [B_s, N_patch, dim]
        src_logits = src_logits.permute(0, 2, 3, 1) # [B_s, H, W, K]
        src_ground_truth = loss_input_fn(src_sample, device) # (labels, mask)
        loss_seg = loss_fn_seg['mean'](src_logits, src_ground_truth) # 标量

        # --- 目标域处理 (仅用于 DA) ---
        trg_inputs = trg_sample['inputs'].to(device)
        # --- 通道数特殊处理 ---
        # pastis: mean = trg_inputs[:, :, :, :, :10].mean(dim=-1, keepdim=True); trg_inputs = torch.cat((mean, trg_inputs[:, :, :, :, 10].unsqueeze(-1)), dim=-1)
        # germany:
        mean_trg = trg_inputs[:, :, :, :, :13].mean(dim=-1, keepdim=True)
        trg_inputs = torch.cat((mean_trg, trg_inputs[:, :, :, :, 14].unsqueeze(-1)), dim=-1)

        _, trg_da_feat = net(trg_inputs, return_da_features=True) # [B_t, N_patch, dim]

        # --- 计算域适应损失 (L_cluster) ---
        loss_cluster = torch.tensor(0.0, device=device)
        if cluster_centers is not None:
            loss_cluster = compute_clustering_loss(src_da_feat, trg_da_feat, cluster_centers, device)

        # --- (可选) 计算其他 DA 损失，例如简单的熵最小化 (鼓励目标域预测更确定) ---
        loss_ent = torch.tensor(0.0, device=device)
        if lambda_entropy > 0:
            with torch.no_grad(): # 不对 logits 反向传播梯度到网络
                trg_logits_raw, _ = net(trg_inputs, return_da_features=False) # [B_t, K, H, W]
            trg_probs = F.softmax(trg_logits_raw, dim=1) # [B_t, K, H, W]
            # 计算熵: -sum(p * log(p))
            loss_ent = -torch.mean(torch.sum(trg_probs * torch.log(trg_probs + 1e-8), dim=1)) # 平均像素熵

        # --- 总损失 ---
        total_loss = loss_seg + beta_cluster * loss_cluster + lambda_entropy * loss_ent

        total_loss.backward()
        optimizer.step()

        return src_logits, src_ground_truth, loss_seg, loss_cluster, loss_ent, total_loss

    # --- 评估函数 (保持不变，注意处理 logits.permute) ---
    def evaluate(net, evalloader, loss_fn, config):
        num_classes = config['MODEL']['num_classes']
        predicted_all = []
        labels_all = []
        losses_all = []
        net.eval()
        with torch.no_grad():
            for step, sample in enumerate(evalloader):
                logits = net(sample['inputs'].to(device)) # [B, K, H, W]
                logits = logits.permute(0, 2, 3, 1) # [B, H, W, K]
                _, predicted = torch.max(logits.data, -1) # [B, H, W]
                ground_truth = loss_input_fn(sample, device) # (labels, mask)
                loss = loss_fn['all'](logits, ground_truth) # [B, H, W] or [B, H, W] masked
                target, mask = ground_truth
                if mask is not None:
                    predicted_all.append(predicted.view(-1)[mask.view(-1)].cpu().numpy())
                    labels_all.append(target.view(-1)[mask.view(-1)].cpu().numpy())
                else:
                    predicted_all.append(predicted.view(-1).cpu().numpy())
                    labels_all.append(target.view(-1).cpu().numpy())
                losses_all.append(loss.view(-1).cpu().detach().numpy())

        print("finished iterating over evaluation dataset after step %d" % step)
        print("calculating evaluation metrics...")
        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        losses = np.concatenate(losses_all)

        eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                                  n_classes=num_classes, unk_masks=None)

        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

        un_labels, class_loss = get_per_class_loss(losses, target_classes, unk_masks=None)

        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
              "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s" %
              (losses.mean(), micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
               micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes)))
        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")

        return (un_labels,
                {"macro": {"Loss": losses.mean(), "Accuracy": macro_acc, "Precision": macro_precision,
                           "Recall": macro_recall, "F1": macro_F1, "IOU": macro_IOU},
                 "micro": {"Loss": losses.mean(), "Accuracy": micro_acc, "Precision": micro_precision,
                           "Recall": micro_recall, "F1": micro_F1, "IOU": micro_IOU},
                 "class": {"Loss": class_loss, "Accuracy": class_acc, "Precision": class_precision,
                           "Recall": class_recall,
                           "F1": class_F1, "IOU": class_IOU}}
                )

    # --- 主训练配置 ---
    num_classes = config['MODEL']['num_classes']
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    save_steps = config['CHECKPOINT']["save_steps"]
    save_path = config['CHECKPOINT']["save_path"]
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    save_epoch_interval = config['CHECKPOINT']["save_epoch_interval"]
    num_steps_train = min(len(src_dataloader), len(trg_dataloader)) # 使用较短的 loader 长度
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)
    folder_name = os.path.basename(os.path.dirname(args.config))
    base_dir = "/data/user/ViT/D3/metric_save/" # 确保路径存在且有权限
    folder_dir = os.path.join(base_dir, folder_name)

    # --- DA 特定超参数 ---
    beta_cluster = get_params_values(config.get('DA_SOLVER', {}), 'beta_cluster', 0.1) # L_cluster 的权重
    lambda_entropy = get_params_values(config.get('DA_SOLVER', {}), 'lambda_entropy', 0.0) # 熵最小化的权重
    cluster_update_freq = get_params_values(config.get('DA_SOLVER', {}), 'cluster_update_freq', 1) # 每多少个 epoch 更新一次聚类
    num_clusters_for_kmeans = get_params_values(config.get('DA_SOLVER', {}), 'num_clusters', num_classes) # 聚类数

    start_global = 1
    start_epoch = 1
    cluster_centers = None # 初始化聚类中心为空

    print("Current learning rate: ", lr)
    print("DA Hyperparameters: beta_cluster={}, lambda_entropy={}, cluster_update_freq={}, num_clusters={}".format(
        beta_cluster, lambda_entropy, cluster_update_freq, num_clusters_for_kmeans))

    trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(config, optimizer, num_steps_train)

    # --- 检查点恢复 ---
    if checkpoint:
        try:
            load_from_checkpoint(net, checkpoint, partial_restore=False)
            save_folder_name = os.path.basename(os.path.dirname(checkpoint))
            metric_save_path = os.path.join(folder_dir, save_folder_name)
            optimizer_save_path = glob.glob(os.path.join(metric_save_path, "*_optimizer.pth"))
            scheduler_save_path = glob.glob(os.path.join(metric_save_path, "*_scheduler.pth"))

            start_epoch = int(os.path.splitext(os.path.basename(checkpoint))[0]) + 1
            metrics_csv_path = f"{metric_save_path}/epoch_metrics.csv"
            if os.path.exists(metrics_csv_path):
                df = pd.read_csv(metrics_csv_path)
                if len(df) >= start_epoch:
                    filtered_df = df.drop(index=range(start_epoch-1, len(df))) # drop 从0开始索引
                    filtered_df.to_csv(metrics_csv_path, index=False)
                print("已加载模型参数，并截断 metrics CSV 文件.")

            if optimizer_save_path:
                optimizer_state_dict = torch.load(optimizer_save_path[0], map_location=device)
                optimizer.load_state_dict(optimizer_state_dict)
                print("优化器参数已加载")
            else: print("未加载优化器参数")

            if scheduler_save_path:
                scheduler_state_dict = torch.load(scheduler_save_path[0], map_location=device)
                scheduler.load_state_dict(scheduler_state_dict)
                print("scheduler参数已加载")
            else: print("未加载scheduler参数")

        except FileNotFoundError as e:
            print(e)
            print("未找到checkpoint： ", checkpoint)
            sys.exit(1)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = os.path.splitext(os.path.basename(args.config))[0]
        metric_save_path = os.path.join(folder_dir, f"{task_name}_{timestamp}")
        os.makedirs(metric_save_path, exist_ok=True)
        if os.path.exists(metric_save_path):
            print("保存目录已创建:", metric_save_path)
        else:
            print("未成功创建保存目录")
            sys.exit(1)
        shutil.copy2(args.config, os.path.join(metric_save_path, "config_used.yaml"))
        metrics_csv_path = f"{metric_save_path}/epoch_metrics.csv"
        print("未加载模型/优化器/scheduler参数")

    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)

    loss_input_fn = get_loss_data_input(config)
    loss_fn_seg = {'all': get_loss(config, device, reduction=None), # 用于评估
                   'mean': get_loss(config, device, reduction="mean")} # 用于训练

    # --- 训练主循环 ---
    net.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_train_loss = 0.0
        epoch_cluster_loss = 0.0
        epoch_entropy_loss = 0.0
        num_train_batches = 0

        # --- 定期更新聚类中心 ---
        if epoch % cluster_update_freq == 0 or cluster_centers is None:
            print(f"[Epoch {epoch}] Updating cluster centers...")
            cluster_centers = update_cluster_centers(net, src_dataloader, device, num_clusters_for_kmeans, config)
            print(f"[Epoch {epoch}] Cluster centers updated.")

        # --- 创建联合迭代器 ---
        joint_loaders = zip(src_dataloader, trg_dataloader) # 简单 zip，会在较短的 loader 结束时停止

        for step, (src_sample, trg_sample) in enumerate(joint_loaders):
            abs_step = start_global + (epoch - start_epoch) * num_steps_train + step

            # --- 执行训练步骤 ---
            src_logits, src_gt, loss_seg, loss_cluster, loss_ent, total_loss = train_step(
                net, src_sample, trg_sample, loss_fn_seg, optimizer, device,
                loss_input_fn, cluster_centers, beta_cluster, lambda_entropy
            )

            epoch_train_loss += loss_seg.item()
            epoch_cluster_loss += loss_cluster.item() if isinstance(loss_cluster, torch.Tensor) else loss_cluster
            epoch_entropy_loss += loss_ent.item() if isinstance(loss_ent, torch.Tensor) else loss_ent
            num_train_batches += 1

            # --- 打印训练批次信息 ---
            if abs_step % train_metrics_steps == 0:
                # 计算源域 batch 的 metrics
                if len(src_gt) == 2:
                    labels, unk_masks = src_gt
                else:
                    labels = src_gt
                    unk_masks = None
                batch_metrics = get_mean_metrics(
                    logits=src_logits.permute(0, 3, 1, 2), # 转换回 [B, K, H, W] 供 metrics 函数使用
                    labels=labels, unk_masks=unk_masks, n_classes=num_classes,
                    loss=loss_seg, epoch=epoch, step=step)

                print(
                    "abs_step: %d, epoch: %d, step: %5d, total_loss: %.7f, seg_loss: %.7f, clus_loss: %.7f, ent_loss: %.7f, "
                    "batch_iou: %.4f, batch accuracy: %.4f" %
                    (abs_step, epoch, step + 1, total_loss.item(), loss_seg.item(),
                     loss_cluster.item() if isinstance(loss_cluster, torch.Tensor) else loss_cluster,
                     loss_ent.item() if isinstance(loss_ent, torch.Tensor) else loss_ent,
                     batch_metrics['IOU'], batch_metrics['Accuracy']))

        scheduler.step_update(abs_step) # 注意：通常放在 epoch 循环内，iteration 循环外

        # --- 计算 epoch 平均损失 ---
        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        avg_cluster_loss = epoch_cluster_loss / num_train_batches if num_train_batches > 0 else 0.0
        avg_entropy_loss = epoch_entropy_loss / num_train_batches if num_train_batches > 0 else 0.0

        print(f"\nRunning Evaluation at End of Epoch {epoch}")
        # --- 运行验证 (通常只在源域验证集上) ---
        eval_metrics = evaluate(net, src_dataloader, loss_fn_seg, config) # 假设你有源域验证 loader
        # eval_metrics = evaluate(net, dataloaders['eval'], loss_fn_seg, config) # 如果 dataloaders['eval'] 是验证集
        macro_iou = eval_metrics[1]['macro']['IOU']
        accuracy = eval_metrics[1]['macro']['Accuracy']

        # --- 保存 epoch 指标 ---
        epoch_metrics = {
            "epoch": epoch,
            "avg_train_loss": avg_train_loss,
            "avg_cluster_loss": avg_cluster_loss,
            "avg_entropy_loss": avg_entropy_loss,
            "total_train_loss": avg_train_loss + beta_cluster * avg_cluster_loss + lambda_entropy * avg_entropy_loss,
            "mAcc": accuracy,
            "val_macro_IOU": macro_iou,
            "lr": optimizer.param_groups[0]["lr"],
        }

        epoch_metrics_df = pd.DataFrame([epoch_metrics])
        if not os.path.exists(metrics_csv_path):
            epoch_metrics_df.to_csv(metrics_csv_path, index=False, mode='w')
        else:
            epoch_metrics_df.to_csv(metrics_csv_path, index=False, mode='a', header=False)
        print(f"Epoch {epoch} metrics saved to: {metrics_csv_path}")

        # --- Early Stopping Check (如果需要) ---
        if avg_train_loss < 0.001: # 示例条件
            print("Early stopping triggered by low training loss.")
            break

        net.train() # 确保下一轮是训练模式

        # --- 定期保存模型 ---
        if epoch % save_epoch_interval == 0:
            checkpoint_filename = os.path.join(metric_save_path, f'{epoch:04d}.pth')
            checkpoint_optimizer_path = os.path.join(metric_save_path, f'{epoch:04d}_optimizer.pth')
            checkpoint_scheduler_path = os.path.join(metric_save_path, f'{epoch:04d}_scheduler.pth')

            torch.save(optimizer.state_dict(), checkpoint_optimizer_path)
            torch.save(scheduler.state_dict(), checkpoint_scheduler_path)

            model_state_dict = net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()
            torch.save(model_state_dict, checkpoint_filename)

            print(f"[INFO] 已保存 checkpoint 到：{checkpoint_filename}")
            print(f"[INFO] 已保存 optimizer 到：{checkpoint_optimizer_path}")
            print(f"[INFO] 已保存 scheduler 到：{checkpoint_scheduler_path}")

            # --- 清理旧 checkpoint (保留最新的 N 个，这里简化为只保留最新的一个) ---
            checkpoint_files = glob.glob(os.path.join(metric_save_path, '*.pth'))
            if len(checkpoint_files) > 3: # .pth, _optimizer.pth, _scheduler.pth 算一组
                # 排序并删除最旧的组
                epochs_in_files = [int(os.path.basename(f)[:4]) for f in checkpoint_files if f.endswith('.pth') and not '_optimizer' in f and not '_scheduler' in f]
                if epochs_in_files:
                    oldest_epoch = min(epochs_in_files)
                    files_to_delete = [
                        os.path.join(metric_save_path, f'{oldest_epoch:04d}.pth'),
                        os.path.join(metric_save_path, f'{oldest_epoch:04d}_optimizer.pth'),
                        os.path.join(metric_save_path, f'{oldest_epoch:04d}_scheduler.pth')
                    ]
                    for f in files_to_delete:
                        if os.path.exists(f):
                            os.remove(f)
                            print(f"[INFO] 删除旧 checkpoint：{f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation Training')
    parser.add_argument('--config', required=True, help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0', type=str, help='gpu ids to use, e.g., "0,1"')
    # lin_cls 参数在此任务中可能不需要

    args = parser.parse_args()
    config_file = args.config
    print("Using devices:", args.device)
    device_ids = [int(d) for d in args.device.split(',') if d.isdigit()]
    # lin_cls = args.lin # 不使用

    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids

    # --- 获取源域和目标域的 DataLoader ---
    src_dataloader = get_dataloaders(config, 'src') # 实现需确保返回 src loader
    trg_dataloader = get_dataloaders(config, 'trg') # 实现需确保返回 trg loader

    # --- 获取模型 ---
    net = get_model(config, device) # 确保 get_model 能加载你修改后的 TSViT

    # --- 开始训练 ---
    train_and_evaluate(net, src_dataloader, trg_dataloader, config, device)