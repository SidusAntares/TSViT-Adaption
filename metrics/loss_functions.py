import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from utils.config_files_utils import get_params_values
from copy import deepcopy

def get_loss_da(config, device,method):
    model_config = config['MODEL']
    loss_config = config['SOLVER']
    if method in ['mk-mmd']:
        if method == 'mk-mmd':
            return DANAdapter(model_config['spatial_depth'],
                      kernel_mul=loss_config['mmd_kernel_mul'],
                      kernel_num=loss_config['mmd_kernel_num'],
                      fix_sigma=loss_config['mmd_fix_sigma'],
                      alpha_sum=loss_config['alpha_sum'])

    print('wrong da method')
    sys.exit(1)


def get_loss(config, device, reduction='mean'):
    model_config = config['MODEL']
    loss_config = config['SOLVER']

    print(loss_config['loss_function'])

    if type(loss_config['loss_function']) in [list, tuple]:
        loss_fun = []
        loss_types = deepcopy(loss_config['loss_function'])
        config_ = deepcopy(config)
        for loss_fun_type in loss_types:
            config_['SOLVER']['loss_function'] = loss_fun_type
            loss_fun.append(get_loss(config_, device, reduction=reduction))
        return loss_fun

    else:
    # Cross-Entropy Loss ------------------------------------------------------------------
        if loss_config['loss_function'] == 'cross_entropy':
            num_classes = get_params_values(model_config, 'num_classes', None)
            weight = torch.Tensor(num_classes * [1.0]).to(device)
            if loss_config['class_weights'] not in [None, {}]:
                for key in loss_config['class_weights']:
                    weight[key] = loss_config['class_weights'][key]
            return torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    
    # Masked Cross-Entropy Loss -----------------------------------------------------------
        elif loss_config['loss_function'] == 'masked_cross_entropy':
            mean = reduction == 'mean'
            return MaskedCrossEntropyLoss(mean=mean)

    return None


def per_class_loss(criterion, logits, labels, unk_masks, n_classes):
    class_loss = []
    class_counts = []
    for class_ in range(n_classes):
        idx = labels == class_
        class_loss.append(
            criterion(logits[idx.repeat(1, 1, 1, n_classes)].reshape(-1, n_classes),  # ???
                      labels[idx].reshape(-1, 1),
                      unk_masks[idx].reshape(-1, 1)).detach().cpu().numpy()
        )
        class_counts.append(unk_masks[idx].sum().cpu().numpy())
    class_loss = np.array(class_loss)
    class_counts = np.array(class_counts)
    return np.nan_to_num(class_loss, nan=0.0), class_counts


class MaskedContrastiveLoss(torch.nn.Module):
    def __init__(self, pos_weight=1, reduction="mean"):
        """
        mean: return mean loss vs per element loss
        """
        super(MaskedContrastiveLoss, self).__init__()

        self.pos_weight = pos_weight
        self.reduction = reduction
        self.h = 1e-7

    def forward(self, logits, ground_truth):
        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        else:
            target = ground_truth[0]
            mask = ground_truth[1].to(torch.float32)

        loss = - self.pos_weight * target * logits + (1 - target) * logits
        if mask is not None:
            loss = mask * loss

        if self.reduction == "mean":
            return loss.mean()  # loss.sum() / (mask.sum() - 1)
        return loss


class MaskedBinaryCrossEntropy(torch.nn.Module):
    def __init__(self, reduction="mean", pos_weight=None):
        """
        mean: return mean loss vs per element loss
        """
        super(MaskedBinaryCrossEntropy, self).__init__()
        self.reduction = reduction
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)

    def forward(self, logits, ground_truth):
        if type(ground_truth) == torch.Tensor:
            target = ground_truth
        elif len(ground_truth) == 1:
            target = ground_truth[0]
        else:
            target = ground_truth[0][ground_truth[1]]
            logits = logits[ground_truth[1]]
        return self.loss_fn(logits, target)


class MaskedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, mean=True):
        """
        mean: return mean loss vs per element loss
        """
        super(MaskedCrossEntropyLoss, self).__init__()
        self.mean = mean
    
    def forward(self, logits, ground_truth):
        """
            Args:
                logits: (N,T,H,W,...,NumClasses)A Variable containing a FloatTensor of size
                    (batch, max_len, num_classes) which contains the
                    unnormalized probability for each class.
                target: A Variable containing a LongTensor of size
                    (batch, max_len) which contains the index of the true
                    class for each corresponding step.
                length: A Variable containing a LongTensor of size (batch,)
                    which contains the length of each data in a batch.
            Returns:
                loss: An average loss value masked by the length.
            """
        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        elif len(ground_truth) == 2:
            target, mask = ground_truth
        else:
            raise ValueError("ground_truth parameter for MaskedCrossEntropyLoss is either (target, mask) or (target)")
        
        if mask is not None:
            mask_flat = mask.reshape(-1, 1)  # (N*H*W x 1)
            nclasses = logits.shape[-1]
            logits_flat = logits.reshape(-1, logits.size(-1))  # (N*H*W x Nclasses)
            masked_logits_flat = logits_flat[mask_flat.repeat(1, nclasses)].view(-1, nclasses)
            target_flat = target.reshape(-1, 1)  # (N*H*W x 1)
            masked_target_flat = target_flat[mask_flat].unsqueeze(dim=-1).to(torch.int64)
        else:
            masked_logits_flat = logits.reshape(-1, logits.size(-1))  # (N*H*W x Nclasses)
            masked_target_flat = target.reshape(-1, 1).to(torch.int64)  # (N*H*W x 1)
        masked_log_probs_flat = torch.nn.functional.log_softmax(masked_logits_flat)  # (N*H*W x Nclasses)
        masked_losses_flat = -torch.gather(masked_log_probs_flat, dim=1, index=masked_target_flat)  # (N*H*W x 1)
        if self.mean:
            return masked_losses_flat.mean()
        return masked_losses_flat


class MaskedFocalLoss(nn.Module):
    """
    Credits to  github.com/clcarwin/focal_loss_pytorch
    """

    def __init__(self, gamma=0, alpha=None, reduction=None):
        super(MaskedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, logits, ground_truth):

        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        elif len(ground_truth) == 2:
            target, mask = ground_truth
        else:
            raise ValueError("ground_truth parameter for MaskedCrossEntropyLoss is either (target, mask) or (target)")

        target = target.reshape(-1, 1).to(torch.int64)
        logits = logits.reshape(-1, logits.shape[-1])

        if mask is not None:
            mask = mask.reshape(-1, 1)
            target = target[mask]
            logits = logits[mask.repeat(1, logits.shape[-1])].reshape(-1, logits.shape[-1])

        logpt = F.log_softmax(logits, dim=-1)
        logpt = logpt.gather(-1, target.unsqueeze(-1))
        logpt = logpt.reshape(-1)
        pt = logpt.exp()  # Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != logits.data.type():
                self.alpha = self.alpha.type_as(logits.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.reduction is None:
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                "FocalLoss: reduction parameter not in list of acceptable values [\"mean\", \"sum\", None]")


class MaskedDiceLoss(nn.Module):
    """
    Credits to  github.com/clcarwin/focal_loss_pytorch
    """

    def __init__(self, reduction=None):
        super(MaskedDiceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, ground_truth):

        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        elif len(ground_truth) == 2:
            target, mask = ground_truth
        else:
            raise ValueError("ground_truth parameter for MaskedCrossEntropyLoss is either (target, mask) or (target)")

        target = target.reshape(-1, 1).to(torch.int64)
        logits = logits.reshape(-1, logits.shape[-1])

        if mask is not None:
            mask = mask.reshape(-1, 1)
            target = target[mask]
            logits = logits[mask.repeat(1, logits.shape[-1])].reshape(-1, logits.shape[-1])

        target_onehot = torch.eye(logits.shape[-1])[target].to(torch.float32).cuda()  # .permute(0,3,1,2).float().cuda()
        predicted_prob = F.softmax(logits, dim=-1)

        inter = (predicted_prob * target_onehot).sum(dim=-1)
        union = predicted_prob.pow(2).sum(dim=-1) + target_onehot.sum(dim=-1)

        loss = 1 - 2 * inter / union

        if self.reduction is None:
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                "FocalLoss: reduction parameter not in list of acceptable values [\"mean\", \"sum\", None]")


class FocalLoss(nn.Module):
    """
    Credits to  github.com/clcarwin/focal_loss_pytorch
    """
    def __init__(self, gamma=0, alpha=None, reduction=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction
        
    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction is None:
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                "FocalLoss: reduction parameter not in list of acceptable values [\"mean\", \"sum\", None]")

class MK_MMD_Loss(nn.Module):
    """
    计算 Multi-Kernel Maximum Mean Discrepancy (MK-MMD) 损失。
    基于论文 "Learning Transferable Features with Deep Adaptation Networks" (ICML 2015)。

    支持自动带宽选择和多核组合。
    """

    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        初始化 MK-MMD 损失计算器。

        Args:
            kernel_mul (float): RBF 核的带宽乘子，用于生成多个核。
                                默认值 2.0 是论文中常用的设置。
            kernel_num (int): 使用的 RBF 核的数量。默认为 5。
                              论文中使用 5 个不同带宽的高斯核。
            fix_sigma (float, optional): 固定的 RBF 核带宽 σ。
                                         如果为 None（默认），则根据输入数据自动估算。
        """
        super(MK_MMD_Loss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        计算多核高斯 (RBF) 核矩阵。

        Args:
            source (Tensor): 源域特征 (Ns, D)
            target (Tensor): 目标域特征 (Nt, D)
            kernel_mul (float): 带宽 multiplier.
            kernel_num (int): 核数量.
            fix_sigma (float, optional): 固定带宽.

        Returns:
            list[Tensor]: 每个核的 Gram 矩阵列表 [(Ns+Nt, Ns+Nt), ...] (共 kernel_num 个)
        """
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)  # (Ns+Nt, D)

        # 计算 L2 距离平方矩阵 ||x_i - x_j||^2
        L2_distance = torch.cdist(total, total, p=2).pow(2)  # (Ns+Nt, Ns+Nt)

        if fix_sigma is not None:
            bandwidth = fix_sigma
        else:
            # 根据中位数距离估计带宽
            distances = L2_distance[L2_distance.ne(0)]
            if distances.numel() > 0:
                bandwidth = torch.median(distances)
                bandwidth = bandwidth + 1e-6  # 防止为零
            else:
                bandwidth = 1.0  # fallback

        # 生成多个尺度的核
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # 计算 exp(-||x_i - x_j||^2 / sigma) for each bandwidth
        kernel_val_list = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return kernel_val_list  # List of (Ns+Nt, Ns+Nt) tensors

    def forward(self, source, target):
        """
        计算 MK-MMD 损失。

        Args:
            source (Tensor): 源域特征，形状 (N_source, D)。
            target (Tensor): 目标域特征，形状 (N_target, D)。

        Returns:
            Tensor: 标量 MK-MMD 损失值。
        """
        batch_size_source = int(source.size()[0])

        # 获取所有核的 Gram 矩阵
        kernels = self.gaussian_kernel(
            source, target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma
        )

        # 对所有核求平均，然后分割
        joint_kernels = sum(kernels) / len(kernels)  # Average over all kernels

        # 分割联合核矩阵
        XX = joint_kernels[:batch_size_source, :batch_size_source]  # Source-Source
        YY = joint_kernels[batch_size_source:, batch_size_source:]  # Target-Target
        XY = joint_kernels[:batch_size_source, batch_size_source:]  # Source-Target

        # 计算无偏 MMD^2 估计 (更稳定)
        # 注意：原论文 Long et al. (2015) 使用的是无偏估计
        loss = torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)

        # 确保损失非负
        return torch.clamp(loss, min=0.0)

class DANAdapter(nn.Module):
    def __init__(self, num_layers, kernel_mul=2.0, kernel_num=5, fix_sigma=None, init_alpha_mode='ones',alpha_sum = 1.0):
        super(DANAdapter, self).__init__()
        self.num_layers = num_layers-1
        self.alpha_sum = alpha_sum
        self.mmd_loss_fn = MK_MMD_Loss(kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if init_alpha_mode == 'zeros':
            self.log_alphas = nn.Parameter(torch.zeros(self.num_layers))
        elif init_alpha_mode == 'ones':
            self.log_alphas = nn.Parameter(torch.ones(self.num_layers))
        elif init_alpha_mode == 'random':
            self.log_alphas = nn.Parameter(torch.randn(self.num_layers) * 0.1)
        else:
            raise ValueError("init_alpha_mode must be 'zeros', 'ones', or 'random'")

    def forward(self, source_features_list, target_features_list):
        assert len(source_features_list) == len(target_features_list) == self.num_layers+1, \
            (f"Mismatch in number of layers. Expected {self.num_layers}, "
             f"got {len(source_features_list)} (source) and {len(target_features_list)} (target).")
        alphas = F.softmax(torch.cat([self.log_alphas,torch.zeros(1,device=self.log_alphas.device)],dim=0),dim=0)*self.alpha_sum
        total_weighted_loss = 0.0
        individual_losses = []
        weights = []
        for i, (src_feat, tgt_feat) in enumerate(zip(source_features_list, target_features_list)):
            if src_feat.dim() > 2:
                src_feat = src_feat.view(src_feat.size(0), -1)
            if tgt_feat.dim() > 2:
                tgt_feat = tgt_feat.view(tgt_feat.size(0), -1)
            current_loss = self.mmd_loss_fn(src_feat, tgt_feat)
            current_alpha = alphas[i]
            weighted_loss = current_alpha * current_loss
            total_weighted_loss += weighted_loss
            individual_losses.append(current_loss)
            weights.append(current_alpha)
        return total_weighted_loss, individual_losses, weights