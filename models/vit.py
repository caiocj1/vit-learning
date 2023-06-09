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

        self.num_patches = (image_size // patch_size) ** 2

        self.layer_norm1 = nn.LayerNorm(self.num_channels * (self.patch_size ** 2))
        self.linear = nn.Linear(self.num_channels * (self.patch_size ** 2), hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, imgs):
        imgs = imgs.view(-1, self.num_patches, self.num_channels * (self.patch_size ** 2))
        return self.layer_norm2(self.linear(self.layer_norm1(imgs)))

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, head_size=64):
        super().__init__()
        self.num_heads = num_heads
        self.inner_dim = num_heads * head_size
        self.scale = head_size ** (- 0.5)

        self.to_qkv = nn.Linear(hidden_size, 3 * self.inner_dim, bias=False)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.out = nn.Sequential(
            nn.Linear(self.inner_dim, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        b, n, hd = q.shape

        q = q.reshape(b, n, self.num_heads, -1).transpose(1, 2)
        k = k.reshape(b, n, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(b, n, self.num_heads, -1).transpose(1, 2)

        out = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        out = self.dropout(self.attend(out))

        out = torch.matmul(out, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.out(out)

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
        x = x + self.multihead_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_blocks, num_heads, dropout, mlp_size):
        super().__init__()

        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(TransformerBlock(hidden_size, num_heads, dropout, mlp_size))

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x

class ViT(nn.Module):
    def __init__(self, hidden_size=768, num_blocks=12, num_heads=12, dropout=0.0, mlp_size=3072):
        super().__init__()
        self.hidden_size = hidden_size

        self.patch_embedding = PatchEmbedding(hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1 + self.patch_embedding.num_patches, hidden_size))
        self.class_token = nn.Parameter(torch.randn((1, hidden_size)))

        self.transformer_encoder = TransformerEncoder(hidden_size, num_blocks, num_heads, dropout, mlp_size)

        self.classification_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1000)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        b, n, _ = x.shape
        class_tokens = torch.stack([self.class_token] * b, dim=0)

        x = torch.cat([class_tokens, x], dim=1)
        x += self.pos_encoding[:, :(n + 1)]

        x = self.transformer_encoder(x)

        return self.classification_head(x[:, 0])
