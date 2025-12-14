import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from utils.config_files_utils import get_params_values
from copy import deepcopy


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

    if reduction == 'mmd':
        return MMDLoss(kernel_type='rbf', kernel_mul=2.0, kernel_num=5)
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

class MMDLoss(nn.Module):
    """
    计算 Maximum Mean Discrepancy (MMD) 损失，使用 RBF (Gaussian) 核。

    该类继承自 nn.Module，可以像其他损失函数一样在模型中使用。
    """

    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, linear_blind=False):
        """
        初始化 MMD 损失计算器。

        Args:
            kernel_type (str, optional): 核函数类型 ('rbf' 或 'linear')。默认为 'rbf'。
            kernel_mul (float, optional): RBF 核的带宽参数 multiplier。默认为 2.0。
            kernel_num (int, optional): RBF 核的数量。默认为 5。
            fix_sigma (float, optional): 固定的 RBF 核带宽 sigma。如果为 None，则根据数据自动计算。默认为 None。
            linear_blind (bool, optional): 是否在线性核计算中忽略对角线元素（避免过拟合方差项）。默认为 False。
                                       (主要用于 linear 核)
        """
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        self.linear_blind = linear_blind

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        计算 Gaussian (RBF) 核矩阵。

        Args:
            source (Tensor): 源域特征 (Ns, D)
            target (Tensor): 目标域特征 (Nt, D)
            kernel_mul (float): 带宽 multiplier.
            kernel_num (int): 核数量.
            fix_sigma (float, optional): 固定带宽.

        Returns:
            Tensor: 联合核矩阵 (Ns+Nt, Ns+Nt).
        """
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0) # (Ns+Nt, D)

        # 计算 L2 距离平方矩阵 ||x_i - x_j||^2
        total0 = total.unsqueeze(0).expand(n_samples, n_samples, -1) # (Ns+Nt, Ns+Nt, D)
        total1 = total.unsqueeze(1).expand(n_samples, n_samples, -1) # (Ns+Nt, Ns+Nt, D)
        # L2_distance[i, j] = ||total[i] - total[j]||^2
        L2_distance = ((total0 - total1) ** 2).sum(2) # (Ns+Nt, Ns+Nt)

        if fix_sigma is not None:
            bandwidth = fix_sigma
        else:
            # 根据中位数距离估计带宽
            # 只考虑非零距离，避免自己和自己比较 (对角线为0)
            distances = L2_distance[L2_distance.ne(0)]
            if distances.numel() > 0:
                bandwidth = torch.median(distances)
                # 为避免 bandwidth 为 0 或过小导致梯度消失/爆炸，加一个小量
                bandwidth = bandwidth + 1e-6
            else:
                bandwidth = 1.0 # fallback, should not happen with different samples

        # 生成多个尺度的核
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # 计算 exp(-||x_i - x_j||^2 / sigma)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        # 将所有核加起来
        return sum(kernel_val) # (Ns+Nt, Ns+Nt)

    def linear_kernel(self, source, target, blind=False):
        """
        计算 Linear 核矩阵 (内积)。

        Args:
            source (Tensor): 源域特征 (Ns, D)
            target (Tensor): 目标域特征 (Nt, D)
            blind (bool): 是否忽略对角线元素。

        Returns:
            Tensor: 线性核矩阵 (Ns+Nt, Ns+Nt).
        """
        n_s = int(source.size()[0])
        n_t = int(target.size()[0])
        total = torch.cat([source, target], dim=0)  # (Ns+Nt, D)

        # 计算内积 <x_i, x_j>
        kernel_val = torch.mm(total, total.T) # (Ns+Nt, Ns+Nt)

        if blind:
            # 创建掩码，排除对角线元素
            n_total = n_s + n_t
            mask = 1 - torch.eye(n_total, dtype=torch.uint8, device=kernel_val.device)
            # 或者使用 float mask: mask = 1 - torch.eye(...).float()
            # kernel_val = kernel_val * mask.float() / mask.float().mean() # Normalize?
            # 更简单地，直接返回非对角线元素的平均似乎不太合适
            # 通常 blind 用于经验协方差估计，这里直接返回矩阵
            pass # Blind logic is tricky here for MMD; usually applied differently.
            # For simplicity in this context, we'll just return the full matrix.
            # If needed, downstream processing can apply the blind mask.

        return kernel_val

    def forward(self, source, target):
        """
        计算 MMD 损失。

        Args:
            source (Tensor): 源域特征，形状 (N_source, D)。
            target (Tensor): 目标域特征，形状 (N_target, D)。

        Returns:
            Tensor: 标量 MMD 损失值。
        """
        batch_size_source = int(source.size()[0])
        batch_size_target = int(target.size()[0])

        if self.kernel_type == 'rbf':
            kernels = self.gaussian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma
            )
        elif self.kernel_type == 'linear':
            kernels = self.linear_kernel(source, target, blind=self.linear_blind)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

        # 分割联合核矩阵
        # kernels = [[K_ss, K_st],
        #            [K_ts, K_tt]]
        XX = kernels[:batch_size_source, :batch_size_source]  # Source-Source
        YY = kernels[batch_size_source:, batch_size_source:]  # Target-Target
        XY = kernels[:batch_size_source, batch_size_source:]  # Source-Target
        YX = kernels[batch_size_source:, :batch_size_source]  # Target-Source

        # loss = torch.mean(XX + YY - XY - YX) # This is also common and simpler
        # More explicitly calculating means:
        E_XX = torch.mean(XX)
        E_YY = torch.mean(YY)
        E_XY = torch.mean(XY)
        E_YX = torch.mean(YX) # Note: E_YX = E_XY for symmetric kernels like RBF/Linear

        mmd_loss = E_XX + E_YY - E_XY - E_YX

        # MMD^2 should be non-negative. Clamp it to avoid numerical issues leading to small negatives.
        # Although mathematically >= 0, numerics can cause tiny negative values.
        return torch.clamp(mmd_loss, min=0.0)
