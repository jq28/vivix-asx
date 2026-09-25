"""ASX: the AST (Gong et al. 2021) patch and positional layout on a small DeiT-style encoder."""

import torch
from torch import nn

from vivix_asx.attention import stream_kwargs
from vivix_asx.transformer import Encoder


class ASX(nn.Module):
    """Overlapping 16x16 patches of a log-mel spectrogram, cls + distillation tokens, encoder.

    A sequence budget in attention_kwargs above half the token count is cut to half, as in ViViX;
    the result is readable as stream_budgets.
    """

    def __init__(
        self,
        attention: str = "T",
        attention_kwargs: dict | None = None,
        n_mels: int = 128,
        n_frames: int = 512,
        patch_size: int = 16,
        stride: int = 10,
        dim: int = 192,
        heads: int = 3,
        dim_head: int = 64,
        mlp_dim: int = 512,
        depth: int = 4,
        num_outputs: int | None = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.dim = dim
        f_dim = (n_mels - patch_size) // stride + 1
        t_dim = (n_frames - patch_size) // stride + 1
        num_patches = f_dim * t_dim

        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch_size, stride=stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, dim))
        for token in (self.cls_token, self.dist_token, self.pos_embed):
            nn.init.trunc_normal_(token, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.encoder = Encoder(
            dim, depth, heads, dim_head, mlp_dim, seq_len=num_patches + 2, attention=attention,
            attention_kwargs=stream_kwargs(attention, attention_kwargs, num_patches + 2), dropout=dropout,
        )
        # None: feature extractor only, as inside MultimodalModel.
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_outputs)) if num_outputs else None

    @property
    def stream_budgets(self) -> dict[str, int | None]:
        return {"audio": self.encoder.budget}

    def extra_repr(self) -> str:
        return f"stream_budgets={self.stream_budgets}"

    def features(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """(B, 1, n_mels, T) -> (B, dim): the mean of the cls and distillation tokens, as in AST."""
        if tuple(spectrogram.shape[1:]) != (1, self.n_mels, self.n_frames):
            raise ValueError(f"expected (B, 1, {self.n_mels}, {self.n_frames}), got {tuple(spectrogram.shape)}")
        b = spectrogram.shape[0]
        x = self.patch_embed(spectrogram).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(b, -1, -1), self.dist_token.expand(b, -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        return (x[:, 0] + x[:, 1]) / 2

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if self.mlp_head is None:
            raise RuntimeError("built with num_outputs=None: call features() instead")
        return torch.sigmoid(self.mlp_head(self.features(spectrogram)))


class AST(ASX):
    """ASX with standard attention."""

    def __init__(self, **kwargs):
        super().__init__(attention="T", **kwargs)


class ASL(ASX):
    """ASX with Linformer attention."""

    def __init__(self, k: int = 64, one_kv_head: bool = True, **kwargs):
        super().__init__(attention="L", attention_kwargs={"k": k, "one_kv_head": one_kv_head}, **kwargs)


class ASN(ASX):
    """ASX with Nystromformer attention."""

    def __init__(self, num_landmarks: int = 64, **kwargs):
        super().__init__(attention="N", attention_kwargs={"num_landmarks": num_landmarks}, **kwargs)


class ASP(ASX):
    """ASX with Performer attention."""

    def __init__(self, nb_features: int = 256, redraw_interval: int | None = 1000, **kwargs):
        super().__init__(
            attention="P", attention_kwargs={"nb_features": nb_features, "redraw_interval": redraw_interval}, **kwargs
        )
