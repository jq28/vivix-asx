"""Pre-norm transformer encoder shared by ViViX and ASX (vit-pytorch `Transformer`)."""

import torch
from torch import nn

from vivix_asx.attention import build_attention


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, attention: nn.Module, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attention
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.norm1(x)) + x
        return self.ff(self.norm2(x)) + x


class Encoder(nn.Module):
    """Stack of pre-norm layers with a final LayerNorm; only the attention kind varies."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        seq_len: int,
        attention: str = "T",
        attention_kwargs: dict | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        kwargs = attention_kwargs or {}
        self.layers = nn.ModuleList(
            EncoderLayer(dim, build_attention(attention, dim, heads, dim_head, seq_len, dropout, **kwargs), mlp_dim, dropout)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)

    @property
    def budget(self) -> int | None:
        """Tokens the attention compresses this stream to; None for kinds without a budget."""
        return self.layers[0].attn.budget

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
