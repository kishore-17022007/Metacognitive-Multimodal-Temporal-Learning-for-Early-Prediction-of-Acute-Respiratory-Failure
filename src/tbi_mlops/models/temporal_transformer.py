"""PyTorch temporal Transformer over fixed-length, LEFT-padded windows. [UNTESTED in build sandbox: torch unavailable]
Windows contain only observations <= prediction index; the representation is read at the LAST position (the prediction index)."""
import math
import torch
import torch.nn as nn
from ..data.schema import N_F

TIME_CH = 3 * N_F          # index of time_since_admission channel (see preprocessing.pipeline.channel_names)


class TemporalTransformer(nn.Module):
    def __init__(self, in_dim: int, window_length: int = 12, d_model: int = 32, n_heads: int = 2, n_layers: int = 2, dim_ff: int = 64, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model); self.pos = nn.Embedding(window_length, d_model); self.time_proj = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers); self.norm = nn.LayerNorm(d_model); self.d_model = d_model

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x (B,L,C) float, valid (B,L) bool. Returns (B,d_model) representation at the last step."""
        B, L, _ = x.shape
        h = self.proj(x) + self.pos(torch.arange(L, device=x.device))[None] + self.time_proj(x[:, :, TIME_CH:TIME_CH + 1])   # positional + continuous-time encoding
        h = self.enc(h, src_key_padding_mask=~valid)            # padded steps are masked out of attention
        return self.norm(h[:, -1])                              # last step is always valid (left padding)
