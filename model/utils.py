import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps)"""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # gamma, learnable

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


def precompute_rope_freqs(head_dim, context_length, theta=10000.0, device="cpu"):
    """Precompute cos/sin for RoPE (Rotary Positional Embedding)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(context_length, device=device).float()
    angles = torch.outer(positions, freqs)

    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Apply RoPE to input tensor x using precomputed cos/sin."""
    x1, x2 = x.chunk(2, dim=-1)  # (batch, n_head, seq_len, head_dim/2)

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    # (x1 + i*x2)(cos + i*sin) = (x1*cos - x2*sin) + i*(x1*sin + x2*cos)
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    return torch.cat([rotated_x1, rotated_x2], dim=-1)