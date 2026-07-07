from dataclasses import dataclass
import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 384
    n_layer: int = 6
    n_head: int = 6
    d_ff: int = 1536
    context_length: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1.0e-5

    def __post_init__(self):
        assert self.d_model % self.n_head == 0, \
            f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
        self.head_dim = self.d_model // self.n_head

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)["model"]
        return cls(**cfg)

    def num_params(self):
        emb = self.vocab_size * self.d_model
        attn_per_layer = 4 * self.d_model * self.d_model  # Q,K,V,O projections
        ffn_per_layer = 2 * self.d_model * self.d_ff      # 2 linear layers in FFN
        per_layer = attn_per_layer + ffn_per_layer
        total = emb + self.n_layer * per_layer
        return total