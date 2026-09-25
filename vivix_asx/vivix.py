"""ViViX: the ViViT factorised encoder (Arnab et al. 2021, model 2) with attention X."""

import torch
from torch import nn

from vivix_asx.attention import stream_kwargs
from vivix_asx.transformer import Encoder


class ViViX(nn.Module):
    """Tubelet embedding, a spatial encoder per tubelet, a temporal encoder over their cls tokens.

    attention_kwargs is written for the spatial stream (patches + 1 tokens). A sequence budget in
    it (Linformer k, Nystrom landmarks) above half a stream's token count is cut to half for that
    stream, so it also fits the short temporal stream (tubelets + 1 tokens); temporal_attention_kwargs
    replaces the derived temporal kwargs when given. The result is readable as stream_budgets.
    """

    def __init__(
        self,
        attention: str = "T",
        attention_kwargs: dict | None = None,
        temporal_attention_kwargs: dict | None = None,
        frames: int = 30,
        image_size: int = 224,
        patch_size: int = 16,
        tubelet_frames: int = 2,
        channels: int = 3,
        dim: int = 192,
        heads: int = 3,
        dim_head: int = 64,
        mlp_dim: int = 512,
        spatial_depth: int = 2,
        temporal_depth: int = 2,
        num_outputs: int | None = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if image_size % patch_size or frames % tubelet_frames:
            raise ValueError("image_size must be divisible by patch_size and frames by tubelet_frames")
        self.frames = frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.tubelet_frames = tubelet_frames
        self.dim = dim
        num_patches = (image_size // patch_size) ** 2
        num_tubelets = frames // tubelet_frames
        patch_dim = channels * patch_size * patch_size * tubelet_frames

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tubelets, num_patches, dim))
        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        encoder_kwargs = dict(dim=dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, attention=attention, dropout=dropout)
        spatial_kwargs = stream_kwargs(attention, attention_kwargs, num_patches + 1)
        temporal_kwargs = stream_kwargs(attention, attention_kwargs, num_tubelets + 1)
        if temporal_attention_kwargs is not None:
            temporal_kwargs = temporal_attention_kwargs
        self.spatial_transformer = Encoder(
            depth=spatial_depth, seq_len=num_patches + 1, attention_kwargs=spatial_kwargs, **encoder_kwargs
        )
        self.temporal_transformer = Encoder(
            depth=temporal_depth, seq_len=num_tubelets + 1, attention_kwargs=temporal_kwargs, **encoder_kwargs
        )
        # None: feature extractor only, as inside MultimodalModel.
        self.mlp_head = nn.Linear(dim, num_outputs) if num_outputs else None

    @property
    def stream_budgets(self) -> dict[str, int | None]:
        return {"spatial": self.spatial_transformer.budget, "temporal": self.temporal_transformer.budget}

    def extra_repr(self) -> str:
        return f"stream_budgets={self.stream_budgets}"

    def tubelets(self, video: torch.Tensor) -> torch.Tensor:
        """(B, F, C, H, W) -> (B, tubelets, patches, tubelet_frames * p * p * C) in vit-pytorch order."""
        b, f, c, h, w = video.shape
        pf, p = self.tubelet_frames, self.patch_size
        x = video.view(b, f // pf, pf, c, h // p, p, w // p, p)
        x = x.permute(0, 1, 4, 6, 2, 5, 7, 3)
        return x.reshape(b, f // pf, (h // p) * (w // p), pf * p * p * c)

    def features(self, video: torch.Tensor) -> torch.Tensor:
        """(B, F, 3, H, W) -> (B, dim): the temporal cls token after the final LayerNorm."""
        expected = (self.frames, self.image_size, self.image_size)
        if (video.shape[1], *video.shape[-2:]) != expected:
            raise ValueError(f"expected (B, {expected[0]}, C, {expected[1]}, {expected[2]}), got {tuple(video.shape)}")
        x = self.to_patch_embedding(self.tubelets(video))
        b, num_tubelets, _, _ = x.shape
        x = x + self.pos_embedding
        x = torch.cat((self.spatial_cls_token.expand(b, num_tubelets, -1, -1), x), dim=2)
        x = self.dropout(x)

        x = self.spatial_transformer(x.flatten(0, 1))[:, 0].view(b, num_tubelets, -1)
        x = torch.cat((self.temporal_cls_token.expand(b, -1, -1), x), dim=1)
        return self.temporal_transformer(x)[:, 0]

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if self.mlp_head is None:
            raise RuntimeError("built with num_outputs=None: call features() instead")
        return torch.sigmoid(self.mlp_head(self.features(video)))


class ViViT(ViViX):
    """ViViX with standard attention."""

    def __init__(self, **kwargs):
        super().__init__(attention="T", **kwargs)


class ViViL(ViViX):
    """ViViX with Linformer attention."""

    def __init__(self, k: int = 64, one_kv_head: bool = True, **kwargs):
        super().__init__(attention="L", attention_kwargs={"k": k, "one_kv_head": one_kv_head}, **kwargs)


class ViViN(ViViX):
    """ViViX with Nystromformer attention."""

    def __init__(self, num_landmarks: int = 64, **kwargs):
        super().__init__(attention="N", attention_kwargs={"num_landmarks": num_landmarks}, **kwargs)


class ViViP(ViViX):
    """ViViX with Performer attention."""

    def __init__(self, nb_features: int = 256, redraw_interval: int | None = 1000, **kwargs):
        super().__init__(
            attention="P", attention_kwargs={"nb_features": nb_features, "redraw_interval": redraw_interval}, **kwargs
        )
