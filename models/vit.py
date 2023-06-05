import torch
import torch.nn as nn
import torch.functional as F
import math

class PatchEmbedding(nn.Module):
    def __init__(self, hidden_size, num_channels=3, image_size=224, patch_size=32):
        super().__init__()

        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches = image_size ** 2 // (patch_size ** 2)

        self.linear = nn.Linear(self.num_channels * (self.patch_size ** 2), hidden_size)

    def forward(self, imgs):
        imgs = imgs.view(-1, self.num_patches, self.num_channels * (self.patch_size ** 2))
        return self.linear(imgs)

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()

        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)

        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=dropout)

    def forward(self, x):
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        return self.multihead_attn(q, k, v)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, mlp_size):
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.multihead_attn = MultiheadAttention(hidden_size, num_heads=num_heads, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.multihead_attn(self.layer_norm1(x))[0]
        return x + self.mlp(self.layer_norm2(x))

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_blocks, num_heads, dropout, mlp_size):
        super().__init__()

        blocks = [TransformerBlock(hidden_size, num_heads, dropout, mlp_size)] * num_blocks
        self.encoder = nn.Sequential(*blocks)

    def forward(self, x):
        return self.encoder(x)

class ViT(nn.Module):
    def __init__(self, hidden_size=768, num_blocks=12, num_heads=12, dropout=0.0, mlp_size=3072):
        super().__init__()
        self.hidden_size = hidden_size

        self.patch_embedding = PatchEmbedding(hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1 + self.patch_embedding.num_patches, hidden_size))
        self.class_token = nn.Parameter(torch.randn((1, hidden_size)))

        self.transformer_encoder = TransformerEncoder(hidden_size, num_blocks, num_heads, dropout, mlp_size)

        self.classification_head = nn.Linear(hidden_size, 1000)

    def forward(self, batch):
        x = self.patch_embedding(batch["img"])
        b, n, _ = x.shape
        class_tokens = torch.stack([self.class_token] * b, dim=0)

        x = torch.cat([class_tokens, x], dim=1)
        x += self.pos_encoding[:, :(n + 1)]

        x = self.transformer_encoder(x)

        return self.classification_head(x[:, 0])
