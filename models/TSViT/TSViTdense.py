import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.TSViT.module import Attention, PreNorm, FeedForward
import numpy as np
from utils.config_files_utils import get_params_values

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.channel = channel
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, K, _, _ = x.size()
        y = self.squeeze(x).view(B, K)
        y = self.excitation(y)
        return y.view(B, K, 1, 1)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class TSViT(nn.Module):
    """
    Temporal-Spatial ViT5 (used in main results, section 4.3)
    For improved training speed, this implementation uses a (365 x dim) temporal position encodings indexed for
    each day of the year. Use TSViT_lookup for a slower, yet more general implementation of lookup position encodings
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2)
        )
        # --- 新增 DA 特定模块 ---
        self.da_se_block = SEBlock(channel=self.num_classes, reduction=16)
        # 缓存常用变量
        self.K = self.num_classes
        self.N_patch = num_patches

    # def forward(self, x):
    #     x = x.permute(0, 1, 4, 2, 3)
    #     B, T, C, H, W = x.shape
    #
    #     xt = x[:, :, -1, 0, 0]
    #     x = x[:, :, :-1]
    #     xt = (xt * 365.0001).to(torch.int64)
    #     xt = F.one_hot(xt, num_classes=366).to(torch.float32)
    #
    #     xt = xt.reshape(-1, 366)
    #     temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
    #     x = self.to_patch_embedding(x)
    #     x = x.reshape(B, -1, T, self.dim)
    #     x += temporal_pos_embedding.unsqueeze(1)
    #     x = x.reshape(-1, T, self.dim)
    #     cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
    #     x = torch.cat((cls_temporal_tokens, x), dim=1)
    #     x = self.temporal_transformer(x)
    #     x = x[:, :self.num_classes]
    #     x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
    #     x += self.space_pos_embedding#[:, :, :(n + 1)]
    #     x = self.dropout(x)
    #     x = self.space_transformer(x)
    #     x = self.mlp_head(x.reshape(-1, self.dim))
    #     x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
    #     x = x.reshape(B, H, W, self.num_classes)
    #     x = x.permute(0, 3, 1, 2)
    #     return x

    def forward(self, x, return_da_features=False):
        """
        Args:
            x (Tensor): 输入张量，形状为 [B, T, C+1, H, W]。最后一个通道是时间戳。
            return_da_features (bool): 是否返回用于域适应的特征。
        Returns:
            logits (Tensor): 分割 logits，形状 [B, num_classes, H, W].
            da_features (Tensor, optional): 域适应特征，形状 [B, N_patch, dim]. 仅当 return_da_features=True 时返回.
        """
        B, T, C, H, W = x.shape
        # ... (时间戳处理、Patch Embedding、Temporal Transformer 保持不变) ...
        xt = x[:, :, -1, 0, 0] # [B, T]
        x = x[:, :, :-1]       # [B, T, C-1, H, W]
        # ... (时间位置编码等) ...
        x = self.to_patch_embedding(x) # [(B*H'*W'), T, dim]
        x = x.reshape(B, -1, T, self.dim)
        # ... (Temporal Transformer 处理) ...
        x = x.reshape(B, self.num_patches_1d ** 2, self.K, self.dim).permute(0, 2, 1, 3) # [B, K, N_patch, dim]

        # ... (空间位置编码) ...
        x = x + self.space_pos_embedding.unsqueeze(1) # [B, K, N_patch, dim]
        x = self.dropout(x)
        x_for_spatial_transformer = x.permute(0, 2, 1, 3).reshape(B * self.K, self.N_patch, self.dim) # [B*K, N_patch, dim]

        # --- Spatial Transformer ---
        x = self.space_transformer(x_for_spatial_transformer) # [B*K, N_patch, dim]

        # --- 提取 DA Features ---
        da_features = None
        if return_da_features:
            x_grouped = x.reshape(B, self.K, self.N_patch, self.dim) # [B, K, N_patch, dim]
            se_weights = self.da_se_block(x_grouped) # [B, K, 1, 1]
            x_weighted = x_grouped * se_weights # [B, K, N_patch, dim]
            da_features = torch.sum(x_weighted, dim=1) # [B, N_patch, dim]

        # --- Prediction Head ---
        x = self.mlp_head(x.reshape(-1, self.dim)) # [B*K*N_patch, patch_size^2]
        x = x.reshape(B, self.K, self.N_patch, self.patch_size ** 2).permute(0, 2, 3, 1) # [B, N_patch, ps^2, K]
        x = x.reshape(B, self.num_patches_1d, self.num_patches_1d, self.patch_size, self.patch_size, self.K)
        logits = rearrange(x, 'b h1 w1 p1 p2 k -> b k (h1 p1) (w1 p2)') # [B, K, H, W]

        if return_da_features:
            return logits, da_features
        else:
            return logits





