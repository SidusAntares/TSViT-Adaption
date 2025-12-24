import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from utils.config_files_utils import get_params_values
from copy import deepcopy


def get_loss_da(config, device, method):
    model_config = config['MODEL']
    loss_config = config['SOLVER']
    if method in ['mk-mmd']:
        if method == 'mk-mmd':
            return DANAdapter(model_config['spatial_depth'],
                              kernel_mul=loss_config['mmd_kernel_mul'],
                              kernel_num=loss_config['mmd_kernel_num'],
                              fix_sigma=loss_config['mmd_fix_sigma']).to(device)

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


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MK_MMD_Loss(nn.Module):
    """
    计算 Multi-Kernel Maximum Mean Discrepancy (MK-MMD) 损失。
    基于论文 "Learning Transferable Features with Deep Adaptation Networks" (ICML 2015)。

    此版本加入了最优多核选择机制，以提升性能。
    注意：beta 的更新已被移出 forward 方法，需由外部显式调用 update_beta。
    """

    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None, eps=1e-3):
        """
        初始化 MK-MMD 损失计算器。

        Args:
            kernel_mul (float): RBF 核的带宽乘子，用于生成多个核。
                                默认值 2.0 是论文中常用的设置。
            kernel_num (int): 使用的 RBF 核的数量。默认为 5。
                              论文中使用 5 个不同带宽的高斯核。
            fix_sigma (float, optional): 固定的 RBF 核带宽 σ。
                                         如果为 None（默认），则根据输入数据自动估算。
            eps (float): 用于 QP 优化的小正则化项，防止数值不稳定。默认 1e-3。
        """
        super(MK_MMD_Loss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        self.eps = eps
        # 缓存核权重 beta，初始均匀分布
        # 使用 register_buffer 注册，使其成为模型状态的一部分，但不视为可学习参数
        self.register_buffer('beta', torch.ones(self.kernel_num) / self.kernel_num)

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
            # 根据中位数距离估计带宽 (论文中提到的 median heuristic)
            distances = L2_distance[L2_distance.ne(0)]  # 排除对角线上的 0
            if distances.numel() > 0:
                bandwidth = torch.median(distances)
                bandwidth = bandwidth + 1e-6  # 防止为零
            else:
                bandwidth = 1.0  # fallback

        # 生成多个尺度的核带宽
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # 计算 exp(-||x_i - x_j||^2 / sigma) for each bandwidth
        kernel_val_list = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return kernel_val_list  # List of (Ns+Nt, Ns+Nt) tensors

    def _compute_mk_mmd_unbiased(self, XX, YY, XY):
        """
        计算单个核下的无偏 MMD^2 估计 (U-statistic)。
        排除对角线元素以减少偏差。

        Args:
            XX (Tensor): K(X, X) 子矩阵 (Ns, Ns)
            YY (Tensor): K(Y, Y) 子矩阵 (Nt, Nt)
            XY (Tensor): K(X, Y) 子矩阵 (Ns, Nt)

        Returns:
            Tensor: 该核对应的无偏 MMD^2 估计值 (标量)。
        """
        n = XX.size(0)
        m = YY.size(0)

        # 计算 XX 部分的无偏估计 (排除对角线)
        if n > 1:
            XX_sum = XX.sum() - torch.trace(XX)
            term_XX = XX_sum / (n * (n - 1))
        else:
            term_XX = torch.tensor(0.0, device=XX.device)

        # 计算 YY 部分的无偏估计 (排除对角线)
        if m > 1:
            YY_sum = YY.sum() - torch.trace(YY)
            term_YY = YY_sum / (m * (m - 1))
        else:
            term_YY = torch.tensor(0.0, device=YY.device)

        # 计算 XY 部分 (无需排除对角线)
        if n > 0 and m > 0:
            XY_sum = XY.sum()
            term_XY = XY_sum / (n * m)
        else:
            term_XY = torch.tensor(0.0, device=XY.device)

        mmd2_est = term_XX + term_YY - 2 * term_XY
        return mmd2_est

    def update_beta(self, source, target, max_samples=512):
        """
        根据当前批次数据，更新最优核权重 beta。
        实现论文 Section 3.2 中的 QP 优化 (Eq. 8)。
        应在计算损失之前调用，以确保计算图一致性。

        Args:
            source (Tensor): 源域特征，形状 (N_s, D)
            target (Tensor): 目标域特征，形状 (N_t, D)
            max_samples (int): 为避免 OOM，对 source + target 总 token 数进行采样的上限。
                               默认 512，可根据显存调整。
        """
        # ====== Step 1: 安全采样，防止 OOM ======
        N_s, N_t = source.size(0), target.size(0)
        if N_s + N_t > max_samples:
            n_s_sample = min(N_s, max_samples // 2)
            n_t_sample = min(N_t, max_samples - n_s_sample)
            idx_s = torch.randperm(N_s, device=source.device)[:n_s_sample]
            idx_t = torch.randperm(N_t, device=target.device)[:n_t_sample]
            source = source[idx_s]
            target = target[idx_t]
            N_s, N_t = n_s_sample, n_t_sample

        # 如果采样后样本太少，无法进行 beta 更新
        if N_s < 2 or N_t < 2:
            return  # 保持当前 beta

        # ====== Step 2: 使用采样后的特征进行 beta 更新 ======
        kernels_all = self.gaussian_kernel(
            source, target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma
        )
        batch_size_s = source.size(0)
        batch_size_t = target.size(0)

        m = self.kernel_num
        g_values_per_kernel = [[] for _ in range(m)]

        # Pair up samples for unbiased estimation
        num_pairs_s = batch_size_s // 2
        num_pairs_t = batch_size_t // 2
        num_effective_pairs = min(num_pairs_s, num_pairs_t)

        if num_effective_pairs == 0:
            return  # Keep current beta

        s_indices = list(range(batch_size_s))
        t_indices = list(range(batch_size_t))

        # Calculate d_u and g_ku values needed for Q matrix
        for i in range(num_effective_pairs):
            s_idx1, s_idx2 = s_indices[2 * i], s_indices[2 * i + 1]
            t_idx1, t_idx2 = t_indices[2 * i], t_indices[2 * i + 1]

            hs1 = source[s_idx1:s_idx1 + 1]  # (1, D)
            hs2 = source[s_idx2:s_idx2 + 1]
            ht1 = target[t_idx1:t_idx1 + 1]
            ht2 = target[t_idx2:t_idx2 + 1]

            # ⚠️ 再次对 quad tuple 进行采样？不需要，因为已经很小了
            # 直接调用 gaussian_kernel（此时最多 4 个点，安全）
            kernels_quad_tuple = self.gaussian_kernel(
                torch.cat([hs1, hs2], dim=0),
                torch.cat([ht1, ht2], dim=0),
                kernel_mul=self.kernel_mul,
                kernel_num=self.kernel_num,
                fix_sigma=self.fix_sigma
            )

            for u, k_tensor in enumerate(kernels_quad_tuple):
                # k(hs1, hs2) + k(ht1, ht2) - k(hs1, ht1) - k(hs2, ht2)
                g_ku = k_tensor[0, 1] + k_tensor[2, 3] - k_tensor[0, 2] - k_tensor[1, 3]
                g_values_per_kernel[u].append(g_ku.item())

        # Handle empty g_values (should not happen after sampling check, but safe)
        if all(len(g_vals) == 0 for g_vals in g_values_per_kernel):
            return

        try:
            g_tensors = [torch.tensor(g_vals, dtype=torch.float32, device=source.device) for g_vals in
                         g_values_per_kernel]
        except Exception:
            return

        # Compute d_u = mean of g_ku values
        d_values = torch.stack([torch.mean(g_t) for g_t in g_tensors])  # (m,)

        # Compute Covariance matrix Q (m x m)
        Q = torch.zeros((m, m), dtype=torch.float32, device=source.device)
        for u in range(m):
            for v in range(m):
                if len(g_tensors[u]) > 1 and len(g_tensors[v]) > 1:
                    cov_uv = torch.mean(g_tensors[u] * g_tensors[v]) - torch.mean(g_tensors[u]) * torch.mean(
                        g_tensors[v])
                    Q[u, v] = cov_uv

        # Regularize Q to make it positive definite
        Q = Q + self.eps * torch.eye(m, device=Q.device)

        # Solve Quadratic Program via projected gradient descent
        beta = self.beta.clone().detach()
        lr = 1.0
        max_iter = 100
        tol = 1e-6

        for _ in range(max_iter):
            grad = Q @ beta
            beta_new = beta - lr * grad
            beta_projected = self._project_simplex(beta_new)
            if torch.norm(beta_projected - beta, p=2) < tol:
                break
            beta = beta_projected

        # Update the buffer
        self.beta.data.copy_(beta_projected.data)

    def _project_simplex(self, v):
        """Project vector v onto the probability simplex."""
        mu, indices = torch.sort(v, descending=True)
        cumsum = torch.cumsum(mu, dim=0)
        arange = torch.arange(1, len(v) + 1, dtype=v.dtype, device=v.device)
        cond = mu - (cumsum - 1) / arange > 0
        # Handle potential issue with cond being all False
        valid_cond = torch.where(cond)[0]
        if valid_cond.numel() > 0:
            rho = valid_cond[-1]  # Get last True index
        else:
            rho = torch.tensor(0, device=v.device)  # Fallback
        # Ensure rho is within valid range for indexing
        rho_value = torch.clamp(rho, 0, len(cumsum) - 1)
        theta = (cumsum[rho_value] - 1) / (rho_value + 1)
        projected = torch.clamp(v - theta, min=0)
        # Renormalize to ensure sum is exactly 1 (numerical stability)
        projected_sum = projected.sum()
        if projected_sum > 0:
            projected = projected / (projected_sum + 1e-12)
        else:
            # Handle edge case where projection results in all zeros
            # Reinitialize to uniform distribution or handle gracefully
            projected = torch.ones_like(projected) / len(projected)
        return projected

    def forward(self, source, target, max_samples=512):
        """
        计算 MK-MMD 损失，支持大 N 的特征。

        Args:
            source (Tensor): (N_s, D)
            target (Tensor): (N_t, D)
            max_samples (int): 最大采样 token 数。若 N_s + N_t > max_samples，则随机采样。
        """
        N_s, D = source.shape
        N_t = target.shape[0]

        # ====== 关键：如果总 token 数太大，进行随机采样 ======
        if N_s + N_t > max_samples:
            # 分别对 source 和 target 采样，保持比例（可选）
            # 简单起见：各采样 max_samples // 2
            n_s_sample = min(N_s, max_samples // 2)
            n_t_sample = min(N_t, max_samples - n_s_sample)

            idx_s = torch.randperm(N_s)[:n_s_sample]
            idx_t = torch.randperm(N_t)[:n_t_sample]

            source = source[idx_s]  # (n_s, D)
            target = target[idx_t]  # (n_t, D)

            # 更新 N_s, N_t
            N_s, N_t = n_s_sample, n_t_sample

        # ====== 后续逻辑不变 ======
        batch_size_source = N_s
        kernels = self.gaussian_kernel(
            source, target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma
        )

        weighted_kernels = [self.beta[i] * k for i, k in enumerate(kernels)]
        joint_kernel = sum(weighted_kernels)

        XX = joint_kernel[:batch_size_source, :batch_size_source]
        YY = joint_kernel[batch_size_source:, batch_size_source:]
        XY = joint_kernel[:batch_size_source, batch_size_source:]

        loss = self._compute_mk_mmd_unbiased(XX, YY, XY)
        return torch.clamp(loss, min=0.0)


class DANAdapter(nn.Module):
    def __init__(self, num_layers, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(DANAdapter, self).__init__()
        self.num_layers = num_layers - 1  # Adjust based on your layer counting logic

        # 实例化 MK_MMD_Loss 模块
        self.mmd_loss_fn = MK_MMD_Loss(kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)


    def forward(self, source_features_list, target_features_list):
        """
        计算 DAN 适配损失。

        Args:
            source_features_list (list[Tensor]): 源域各层特征列表。
            target_features_list (list[Tensor]): 目标域各层特征列表。

        Returns:
            tuple: (total_weighted_loss, individual_losses, weights)
                   - total_weighted_loss (Tensor): 总的加权 MMD 损失。
                   - individual_losses (list[Tensor]): 各层的 MMD 损失。
                   - weights (list[Tensor]): 各层对应的 alpha 权重。
        """
        assert len(source_features_list) == len(target_features_list) , \
            f"got {len(source_features_list)} (source) and {len(target_features_list)} (target)."



        # *** 关键修改：在循环前显式更新 beta ***
        # 选择用于更新 beta 的层索引 (例如，最后一个共享层)
        # 这里的索引需要根据你的网络结构和特征列表含义来确定
        # 动态选择用于 beta 更新的特征层
        L = len(source_features_list)
        if L == 0:
            raise ValueError("source_features_list is empty!")

        # 策略：如果有 ≥2 层，用倒数第二层（-2）；否则用最后一层（-1）
        idx_for_beta_update = -2 if L >= 2 else -1

        # 安全取特征（理论上 now always valid）
        src_feat_beta = source_features_list[idx_for_beta_update]
        tgt_feat_beta = target_features_list[idx_for_beta_update]

        # Flatten features if necessary for MMD calculation
        if src_feat_beta.dim() > 2:
            src_feat_beta_flat = src_feat_beta.view(src_feat_beta.size(0), -1)
        else:
            src_feat_beta_flat = src_feat_beta

        if tgt_feat_beta.dim() > 2:
            tgt_feat_beta_flat = tgt_feat_beta.view(tgt_feat_beta.size(0), -1)
        else:
            tgt_feat_beta_flat = tgt_feat_beta

        # 调用 MK_MMD_Loss 的 update_beta 方法
        # 使用 detach() 确保 beta 更新过程不影响主网络梯度
        self.mmd_loss_fn.update_beta(src_feat_beta_flat.detach(), tgt_feat_beta_flat.detach())
        # 计算各层的加权 MMD 损失
        total_weighted_loss = 0.0
        individual_losses = []
        weights = []

        # 遍历所有层的特征 (或根据需要调整范围)
        # 这里假设计算所有层的 MMD
        for i, (src_feat, tgt_feat) in enumerate(zip(source_features_list, target_features_list)):
            # Flatten features if they are multi-dimensional (e.g., conv features)
            if src_feat.dim() > 2:
                src_feat_flat = src_feat.view(src_feat.size(0), -1)
            else:
                src_feat_flat = src_feat

            if tgt_feat.dim() > 2:
                tgt_feat_flat = tgt_feat.view(tgt_feat.size(0), -1)
            else:
                tgt_feat_flat = tgt_feat

            # 调用 MK_MMD_Loss.forward，使用已在循环前更新的固定 beta
            current_loss = self.mmd_loss_fn(src_feat_flat, tgt_feat_flat)
            total_weighted_loss += current_loss
            individual_losses.append(current_loss)
            weights.append(torch.tensor(0))

#=======================================调试代码===========================================
        if torch.rand(1).item() < 0.05:  # 随机打印 5% 的 batch
            with torch.no_grad():
                # 获取用于 beta 更新那一层的带宽（需临时调用 gaussian_kernel）
                src_debug = source_features_list[idx_for_beta_update]
                tgt_debug = target_features_list[idx_for_beta_update]
                src_debug_flat = src_debug.view(src_debug.size(0), -1) if src_debug.dim() > 2 else src_debug
                tgt_debug_flat = tgt_debug.view(tgt_debug.size(0), -1) if tgt_debug.dim() > 2 else tgt_debug
                kernels_debug = self.mmd_loss_fn.gaussian_kernel(
                    src_debug_flat, tgt_debug_flat,
                    kernel_mul=self.mmd_loss_fn.kernel_mul,
                    kernel_num=self.mmd_loss_fn.kernel_num,
                    fix_sigma=self.mmd_loss_fn.fix_sigma
                )
                # 重新估算 bandwidth（与 update_beta 中逻辑一致）
                total_debug = torch.cat([src_debug_flat, tgt_debug_flat], dim=0)
                L2_dist = torch.cdist(total_debug, total_debug, p=2).pow(2)
                distances = L2_dist[L2_dist != 0]
                if distances.numel() > 0:
                    bandwidth_val = torch.median(distances).item() + 1e-6
                else:
                    bandwidth_val = 1.0

                print(f"[DEBUG] Bandwidth (median heuristic): {bandwidth_val:.4f}")
                print(f"[DEBUG] Beta (optimal kernel weights): {self.mmd_loss_fn.beta.detach().cpu().numpy()}")
                print(f"[DEBUG] MMD loss per layer: {[round(l.item(), 6) for l in individual_losses]}")
                print(f"[DEBUG] Alpha weights per layer: {[round(w.item(), 4) for w in weights]}")

# =======================================调试代码===========================================

        return total_weighted_loss, individual_losses, weights

