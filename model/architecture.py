import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import RMSNorm, precompute_rope_freqs, apply_rope


class CausalSelfAttention(nn.Module):
    """Causal self-attention layer with RoPE and RMSNorm."""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        # Q, K, V are calculated with a single linear layer for efficiency
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape  # batch, seq_len, d_model

        qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)  # each: (B, T, d_model)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])
        # v is not rotated because it is 'content' and does not need positional information

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)  # (B, T, d_model)
    

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x
    

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight
 
        # Precompute RoPE cos/sin for efficiency, store as buffers (not parameters)
        cos, sin = precompute_rope_freqs(
            config.head_dim, config.context_length, config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
 
        self.apply(self._init_weights)
 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
 
    def forward(self, idx, targets=None):
        """
        idx: (batch, seq_len) — token ids input
        targets: (batch, seq_len) — token ids target (shifted by 1 position), None during inference
        """
        B, T = idx.shape
        assert T <= self.config.context_length, \
            f"Sequence length {T} exceeds context_length {self.config.context_length}"
 
        x = self.token_embedding(idx)  # (B, T, d_model)
 
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)
 
        for block in self.blocks:
            x = block(x, cos, sin)
 
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
 
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss
 
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
 

if __name__ == "__main__":
    from model.config import ModelConfig
    config = ModelConfig()
    model = GPTModel(config)
    print(f"Model has {model.num_parameters() / 1e6:.2f}M parameters.")