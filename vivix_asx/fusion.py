"""Modality fusion: the proposed learnable-scalar concatenation (Eq. 11-13) and the Table 3 baselines."""

import inspect

import torch
import torch.nn.functional as F
from torch import nn

from vivix_asx.attention import stream_kwargs
from vivix_asx.transformer import Encoder


class LearnableScalarConcatFusion(nn.Module):
    """Eq. 11-13: h_m = ReLU(W_m f_m + b_m); z = concat(h_v p_v, h_a p_a); y = sigmoid(W_out z + b_out)."""

    def __init__(
        self,
        dim_v: int,
        dim_a: int,
        unified_dim: int = 64,
        num_outputs: int = 1,
        learnable_scalars: bool = True,
    ):
        super().__init__()
        self.project_video = nn.Linear(dim_v, unified_dim)
        self.project_audio = nn.Linear(dim_a, unified_dim)
        # TODO(unspecified in paper): the scalars start at 1.0, i.e. at plain concatenation.
        self.p_v = nn.Parameter(torch.ones(())) if learnable_scalars else None
        self.p_a = nn.Parameter(torch.ones(())) if learnable_scalars else None
        self.out = nn.Linear(2 * unified_dim, num_outputs)

    def forward(self, f_video: torch.Tensor, f_audio: torch.Tensor) -> torch.Tensor:
        h_v = F.relu(self.project_video(f_video))
        h_a = F.relu(self.project_audio(f_audio))
        if self.p_v is not None:
            h_v = h_v * self.p_v
            h_a = h_a * self.p_a
        return torch.sigmoid(self.out(torch.cat((h_v, h_a), dim=-1)))


class ConcatFusion(LearnableScalarConcatFusion):
    """Table 3 baseline: Eq. 11 and 13 without the scalars of Eq. 12."""

    def __init__(self, dim_v: int, dim_a: int, unified_dim: int = 64, num_outputs: int = 1):
        super().__init__(dim_v, dim_a, unified_dim, num_outputs, learnable_scalars=False)


class TransformerFusion(nn.Module):
    """Table 3 baseline: a 1-D conv tokeniser, then a 3-layer, 3-head, dim-64 transformer encoder."""

    def __init__(
        self,
        dim_v: int,
        dim_a: int,
        num_outputs: int = 1,
        *,
        attention: str,
        attention_kwargs: dict | None = None,
        fusion_attention_kwargs: dict | None = None,
        embed_dim: int = 64,
        depth: int = 3,
        heads: int = 3,
        dim_head: int = 64,
        conv_kernel: int = 16,
        mlp_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        total = dim_v + dim_a
        if total % conv_kernel:
            raise ValueError("dim_v + dim_a must be divisible by conv_kernel")
        # TODO(unspecified in paper): the token layout below is invented.
        self.conv = nn.Conv1d(1, embed_dim, kernel_size=conv_kernel, stride=conv_kernel)
        self.pos_embedding = nn.Parameter(torch.zeros(1, total // conv_kernel, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        # The budget rule of ViViX applies to these few tokens unless fusion_attention_kwargs is given.
        kwargs = stream_kwargs(attention, attention_kwargs, total // conv_kernel)
        if fusion_attention_kwargs is not None:
            kwargs = fusion_attention_kwargs
        self.encoder = Encoder(
            embed_dim, depth, heads, dim_head, mlp_dim, seq_len=total // conv_kernel,
            attention=attention, attention_kwargs=kwargs, dropout=dropout,
        )
        self.out = nn.Linear(embed_dim, num_outputs)

    @property
    def stream_budgets(self) -> dict[str, int | None]:
        return {"fusion": self.encoder.budget}

    def extra_repr(self) -> str:
        return f"stream_budgets={self.stream_budgets}"

    def forward(self, f_video: torch.Tensor, f_audio: torch.Tensor) -> torch.Tensor:
        signal = torch.cat((f_video, f_audio), dim=-1).unsqueeze(1)
        tokens = self.conv(signal).transpose(1, 2) + self.pos_embedding
        pooled = self.encoder(tokens).mean(dim=1)
        return torch.sigmoid(self.out(pooled))


class LowRankTensorFusion(nn.Module):
    """Liu et al. 2018 (LMF): y = sum_r w_r prod_m ([f_m; 1] W_m^(r)) + b, rank 10 in Table 3."""

    def __init__(self, dim_v: int, dim_a: int, num_outputs: int = 1, rank: int = 10):
        super().__init__()
        self.factors = nn.ParameterList(
            nn.Parameter(torch.empty(rank, d + 1, num_outputs)) for d in (dim_v, dim_a)
        )
        self.fusion_weights = nn.Parameter(torch.empty(1, rank))
        self.fusion_bias = nn.Parameter(torch.zeros(1, num_outputs))
        for factor in self.factors:
            nn.init.xavier_normal_(factor)
        nn.init.xavier_normal_(self.fusion_weights)

    def forward(self, f_video: torch.Tensor, f_audio: torch.Tensor) -> torch.Tensor:
        fused = None
        for f, factor in zip((f_video, f_audio), self.factors):
            z = torch.cat((f, f.new_ones(f.shape[0], 1)), dim=-1)
            projected = torch.einsum("bd,rdo->rbo", z, factor)
            fused = projected if fused is None else fused * projected
        out = torch.einsum("kr,rbo->bo", self.fusion_weights, fused) + self.fusion_bias
        return torch.sigmoid(out)


FUSIONS: dict[str, type[nn.Module]] = {
    "learnable_concat": LearnableScalarConcatFusion,
    "concat": ConcatFusion,
    "transformer": TransformerFusion,
    "low_rank": LowRankTensorFusion,
}


def build_fusion(
    kind: str,
    dim_v: int,
    dim_a: int,
    num_outputs: int = 1,
    attention: str | None = None,
    attention_kwargs: dict | None = None,
    **kwargs,
) -> nn.Module:
    """Dict lookup on FUSIONS; the attention kind is forwarded only to fusions that take one."""
    try:
        cls = FUSIONS[kind]
    except KeyError:
        raise ValueError(f"unknown fusion kind {kind!r}; known kinds: {sorted(FUSIONS)}") from None
    if "attention" in inspect.signature(cls).parameters:
        kwargs.update(attention=attention, attention_kwargs=attention_kwargs)
    return cls(dim_v, dim_a, num_outputs=num_outputs, **kwargs)
