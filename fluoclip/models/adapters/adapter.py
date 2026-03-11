import torch
import torch.nn as nn
from .builder import ADAPTERS


@ADAPTERS.register_module()
class AttentionAdapter(nn.Module):
    def __init__(self, input_dim, num_heads=8, bottleneck_dim_ratio=0.25):
        super().__init__()        
        # Self-Attention Layer, changed to batch_first=False (the default)
        self.attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=False)
        self.norm1 = nn.LayerNorm(input_dim)
        
        # Feed-Forward MLP Layer
        bottleneck_dim = int(input_dim * bottleneck_dim_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, input_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x starts as (Batch_size, Feature_dim)
        
        # Reshape to (Batch_size, 1, Feature_dim) to match the expected
        # (Sequence, Batch, Feature) format where the batch is the sequence.
        x_seq = x.unsqueeze(1)
        
        # 1. Self-Attention
        residual_1 = x_seq
        attn_output, _ = self.attn(x_seq, x_seq, x_seq) # Q, K, V
        x_seq = self.norm1(residual_1 + attn_output)
        
        # 2. MLP
        residual_2 = x_seq
        mlp_output = self.mlp(x_seq)
        x_seq = self.norm2(residual_2 + mlp_output)
        
        # Reshape back to (Batch_size, Feature_dim) by squeezing the batch dimension
        return x_seq.squeeze(1)