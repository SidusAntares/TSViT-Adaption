import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.TSViT.module import Attention, PreNorm, FeedForward
import numpy as np
from utils.config_files_utils import get_params_values

class SEBlock(nn.Module):
    def __init__(self, channel,reduction=6):
        super(SEBlock, self).__init__()

        self.channel = channel
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.reduction=reduction
        self.excitation = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, K, P, D = x.shape
        # 1. Permute to group patches: [B, P, K, D]
        x_permuted = x.permute(0, 2, 1, 3).contiguous() # (B, P, K, D)
        # 2. Reshape for processing each (K, D) group independently: [B*P, K, D]
        x_reshaped = x_permuted.view(B * P, K, D) # (B*P, K, D)

        # 3. Squeeze: 对每个 D 维特征做平均，得到每个 Token 的响应强度: [B*P, K, 1]
        squeezed = self.pool(x_reshaped) # (B*P, K, 1)
        squeezed = squeezed.view(B * P, K) # (B*P, K)

        # 4. Excitation: 学习 K 个 Token 的权重: [B*P, K]
        weights = self.excitation(squeezed) # (B*P, K)

        # 5. Reshape weights back to match original dimensions for broadcasting
        # We want to apply these weights to the original x [B, K, P, D]
        # So we need weights to be [B, K, P, 1]
        weights = weights.view(B, P, K).permute(0, 2, 1).contiguous() # (B, K, P)
        weights = weights.unsqueeze(-1) # (B, K, P, 1)

        return weights

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
        self.se_block = SEBlock(channel=self.num_classes, reduction=16)


    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape

        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)

        xt = xt.reshape(-1, 366)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)
        x = self.space_transformer(x) # (B×num_classes, num_patches_1d^2 , dim)

        da_features = x.reshape(B,self.num_classes, self.num_patches_1d**2 , self.dim)
        se_weights = self.se_block(x) # [B, K,  num_patches, 1]
        da_features = da_features * se_weights # [B, K, num_patches, dim]
        da_features = torch.sum(da_features, dim=1) # [B, num_patches, dim]

        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x , da_features








