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

    此版本加入了最优多核选择机制，以提升性能。
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
        # 使用 p=2 并平方，或者直接使用 'sqeuclidean' (如果 PyTorch 版本支持)
        L2_distance = torch.cdist(total, total, p=2).pow(2) # (Ns+Nt, Ns+Nt)

        if fix_sigma is not None:
            bandwidth = fix_sigma
        else:
            # 根据中位数距离估计带宽 (论文中提到的 median heuristic)
            distances = L2_distance[L2_distance.ne(0)] # 排除对角线上的 0
            if distances.numel() > 0:
                bandwidth = torch.median(distances)
                bandwidth = bandwidth + 1e-6  # 防止为零
            else:
                bandwidth = 1.0  # fallback

        # 生成多个尺度的核带宽
        # 论文 Long et al. (2015) footnote 2 提到: gamma_u between 2^-8*gamma and 2^8*gamma
        # 你的原始设置 kernel_mul=2.0, kernel_num=5 会生成大约 2^-4*gamma 到 2^4*gamma
        # 这是一种常见的近似方法
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
            # sum_{i!=j} k(x_i, x_j) = sum(all) - sum(diag)
            XX_sum = XX.sum() - torch.trace(XX)
            # Number of terms: n*(n-1)
            term_XX = XX_sum / (n * (n - 1))
        else:
            # If only one sample, XX term is 0
            term_XX = torch.tensor(0.0, device=XX.device)

        # 计算 YY 部分的无偏估计 (排除对角线)
        if m > 1:
            YY_sum = YY.sum() - torch.trace(YY)
            # Number of terms: m*(m-1)
            term_YY = YY_sum / (m * (m - 1))
        else:
            # If only one sample, YY term is 0
            term_YY = torch.tensor(0.0, device=YY.device)

        # 计算 XY 部分 (无需排除对角线，因为 X!=Y)
        if n > 0 and m > 0:
            XY_sum = XY.sum()
            # Number of terms: n*m
            term_XY = XY_sum / (n * m)
        else:
            term_XY = torch.tensor(0.0, device=XY.device)

        mmd2_est = term_XX + term_YY - 2 * term_XY
        return mmd2_est

    def _update_beta(self, source, target):
        """
        根据当前批次数据，更新最优核权重 beta。
        实现论文 Section 3.2 中的 QP 优化 (Eq. 8)。
        """
        kernels_all = self.gaussian_kernel(
            source, target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma
        )
        batch_size_s = source.size(0)
        batch_size_t = target.size(0)

        # Step 1: Compute d_u (unbiased MMD estimates) and covariance matrix Q for each pair of kernels
        # Following Gretton et al. (2012b) notation used in the paper
        m = self.kernel_num
        d_values = []
        # Store g_ku(z_i) for covariance calculation
        g_values_per_kernel = [[] for _ in range(m)]

        # Pair up samples for unbiased estimation (as suggested by linear-time MMD)
        # Assume even batch sizes for simplicity here. Can be adapted for odd.
        num_pairs_s = batch_size_s // 2
        num_pairs_t = batch_size_t // 2
        num_effective_pairs = min(num_pairs_s, num_pairs_t)

        if num_effective_pairs == 0:
            # Not enough data to compute unbiased estimate, keep beta unchanged or reset?
            # print("Warning: Batch size too small for beta update.")
            return # Keep current beta

        # Sample indices for pairing (simple consecutive pairing)
        s_indices = list(range(batch_size_s))
        t_indices = list(range(batch_size_t))

        # Calculate d_u and g_ku values needed for Q matrix
        for i in range(num_effective_pairs):
            # Indices for source pairs
            s_idx1, s_idx2 = s_indices[2*i], s_indices[2*i + 1]
            # Indices for target pairs
            t_idx1, t_idx2 = t_indices[2*i], t_indices[2*i + 1]

            # Extract features for the quad tuple
            hs1 = source[s_idx1:s_idx1+1] # Keep dims (1, D)
            hs2 = source[s_idx2:s_idx2+1]
            ht1 = target[t_idx1:t_idx1+1]
            ht2 = target[t_idx2:t_idx2+1]

            # Compute kernels for this quad tuple for all u
            kernels_quad_tuple = self.gaussian_kernel(
                torch.cat([hs1, hs2], dim=0),
                torch.cat([ht1, ht2], dim=0),
                kernel_mul=self.kernel_mul,
                kernel_num=self.kernel_num,
                fix_sigma=self.fix_sigma
            ) # Each kernel in list is now (4, 4)

            for u, k_tensor in enumerate(kernels_quad_tuple):
                # Extract relevant parts for g_ku calculation on this quad tuple
                # k(hs1, hs2) + k(ht1, ht2) - k(hs1, ht1) - k(hs2, ht2)
                # Assuming order in total tensor: [hs1, hs2, ht1, ht2]
                g_ku = k_tensor[0, 1] + k_tensor[2, 3] - k_tensor[0, 2] - k_tensor[1, 3]
                g_values_per_kernel[u].append(g_ku.item()) # Store scalar value

        # Convert lists to tensors for easier computation
        try:
            g_tensors = [torch.tensor(g_vals, dtype=torch.float32, device=source.device) for g_vals in g_values_per_kernel]
        except:
            # Handle case where g_values might be empty due to insufficient data
            return

        # Compute d_u = mean of g_ku values
        d_values = torch.stack([torch.mean(g_t) for g_t in g_tensors]) # Shape: (m,)

        # Compute Covariance matrix Q (m x m)
        Q = torch.zeros((m, m), dtype=torch.float32, device=source.device)
        for u in range(m):
            for v in range(m):
                if len(g_tensors[u]) > 1 and len(g_tensors[v]) > 1:
                    # Cov[g_ku, g_kv] = E[g_ku * g_kv] - E[g_ku] * E[g_kv]
                    # Use bessel correction (ddof=1) for sample covariance
                    cov_uv = torch.mean(g_tensors[u] * g_tensors[v]) - torch.mean(g_tensors[u]) * torch.mean(g_tensors[v])
                    Q[u, v] = cov_uv
                # Else leave as zero if not enough data

        # Regularize Q to make it positive definite
        Q = Q + self.eps * torch.eye(m, device=Q.device)

        # Step 2: Solve Quadratic Program (Eq. 8): min_beta beta^T Q beta s.t. sum(beta)=1, beta>=0
        # This is a standard QP problem. We'll use a simple iterative solver or approximate.
        # For simplicity here, let's use a basic projected gradient descent or call a QP solver if available.
        # PyTorch doesn't have a built-in QP solver, so we'll implement a basic version.
        # Alternatively, one could use cvxpy or scipy.optimize, but that breaks pure pytorch dependency.

        # Simple Gradient Descent approach for QP (projected onto simplex)
        # Minimize f(beta) = 0.5 * beta^T Q beta
        # Subject to A_eq * beta = b_eq (sum(beta) = 1) and beta >= 0

        # Initialize beta (already done in __init__ and stored in buffer)
        beta = self.beta.clone().detach() # Start from current beta
        lr = 1.0 # Learning rate - needs tuning
        max_iter = 100
        tol = 1e-6

        for _ in range(max_iter):
            grad = Q @ beta # Gradient of 0.5 * beta^T Q beta
            beta_new = beta - lr * grad

            # Project onto simplex (sum=1, beta>=0) - Simplex projection algorithm
            beta_projected = self._project_simplex(beta_new)

            # Check convergence
            if torch.norm(beta_projected - beta, p=2) < tol:
                break
            beta = beta_projected

        # Update the stored beta buffer
        with torch.no_grad(): # Ensure no gradients flow through this update
            self.beta.copy_(beta_projected)


    def _project_simplex(self, v):
        """Project vector v onto the probability simplex."""
        # Sort v in descending order
        mu, indices = torch.sort(v, descending=True)
        # Compute cumulative sum
        cumsum = torch.cumsum(mu, dim=0)
        # Find rho (largest index satisfying the condition)
        arange = torch.arange(1, len(v) + 1, dtype=v.dtype, device=v.device)
        cond = mu - (cumsum - 1) / arange > 0
        rho = torch.where(cond)[0][-1] if torch.any(cond) else 0 # Get last True index or 0
        # Compute theta
        theta = (cumsum[rho] - 1) / (rho + 1)
        # Apply thresholding
        projected = torch.clamp(v - theta, min=0)
        # Renormalize to ensure sum is exactly 1 (numerical stability)
        projected = projected / (projected.sum() + 1e-12)
        return projected

    def forward(self, source, target, update_beta=True):
        """
        计算 MK-MMD 损失。

        Args:
            source (Tensor): 源域特征，形状 (N_source, D)。
            target (Tensor): 目标域特征，形状 (N_target, D)。
            update_beta (bool): 是否在此前向传播中更新最优核权重 beta。
                                通常在训练过程中启用，在验证/测试时禁用。

        Returns:
            Tensor: 标量 MK-MMD 损失值。
        """
        batch_size_source = int(source.size()[0])

        # 可选：更新 beta 权重
        if self.training and update_beta:
            self._update_beta(source.detach(), target.detach()) # Detach to avoid gradients flowing back through beta update

        # 获取所有核的 Gram 矩阵
        kernels = self.gaussian_kernel(
            source, target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma
        )

        # 使用当前存储的 beta 权重对核进行加权组合
        # joint_kernel = sum(beta_u * K_u)
        weighted_kernels = [self.beta[i] * k for i, k in enumerate(kernels)]
        joint_kernel = sum(weighted_kernels)

        # 分割联合核矩阵
        XX = joint_kernel[:batch_size_source, :batch_size_source]  # Source-Source
        YY = joint_kernel[batch_size_source:, batch_size_source:]  # Target-Target
        XY = joint_kernel[:batch_size_source, batch_size_source:]  # Source-Target

        # 计算 MK-MMD^2 (使用无偏估计)
        # loss = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y')]
        # loss = torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY) # Biased V-statistic
        # 使用无偏估计 (更符合论文)
        loss = self._compute_mk_mmd_unbiased(XX, YY, XY)

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