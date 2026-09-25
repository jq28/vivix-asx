"""Four attention modules behind one interface, selected by "T" | "L" | "N" | "P"."""

import inspect
import math

import torch
from torch import nn


class Attention(nn.Module):
    """Shared q/k/v and output projections; subclasses implement attend() on (B, h, n, d)."""

    # Tokens a variant compresses the sequence to (k, landmarks); None when it does not,
    # and the name of the constructor argument that sets it.
    budget: int | None = None
    budget_kwarg: str | None = None

    def __init__(
        self,
        dim: int,
        heads: int = 3,
        dim_head: int = 64,
        seq_len: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.seq_len = seq_len
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, -1, self.dim_head).transpose(1, 2)

    def project(self, x: torch.Tensor):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        return self.split_heads(q), self.split_heads(k), self.split_heads(v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        out = self.attend(*self.project(x))
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.dropout(self.to_out(out))

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def redraw(self, step: int) -> None:
        """Hook for variants with random state, called by train.py after every optimiser step."""


class StandardAttention(Attention):
    """Eq. 3: softmax(Q K^T / sqrt(d)) V, materialising the n x n matrix."""

    def __init__(
        self,
        dim: int,
        heads: int = 3,
        dim_head: int = 64,
        seq_len: int | None = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__(dim, heads, dim_head, seq_len, dropout)
        # lucidrains drops attention probabilities at the output-dropout rate; off by default.
        self.attn_dropout = nn.Dropout(attn_dropout)

    def attend(self, q, k, v):
        attn = (q @ k.transpose(-1, -2) * self.scale).softmax(dim=-1)
        return self.attn_dropout(attn) @ v


def segment_mean_matrix(seq_len: int, num_segments: int) -> torch.Tensor:
    """(seq_len, num_segments) matrix averaging each of torch.tensor_split's contiguous segments."""
    # float64 so a module cast to double keeps exact segment weights like 1/3.
    matrix = torch.zeros(seq_len, num_segments, dtype=torch.float64)
    for j, index in enumerate(torch.tensor_split(torch.arange(seq_len), num_segments)):
        matrix[index, j] = 1.0 / index.numel()
    return matrix


class LinformerAttention(Attention):
    """Eq. 7: softmax(Q (E_k K)^T / sqrt(d)) (E_v V) with learnable E_k, E_v in R^{n x k}."""

    budget_kwarg = "k"

    def __init__(
        self,
        dim: int,
        heads: int = 3,
        dim_head: int = 64,
        seq_len: int | None = None,
        dropout: float = 0.0,
        k: int = 64,
        one_kv_head: bool = True,
        share_kv: bool = False,
        attn_dropout: float = 0.0,
    ):
        if seq_len is None:
            raise ValueError("LinformerAttention needs seq_len to size E_k and E_v")
        super().__init__(dim, heads, dim_head, seq_len, dropout)
        del self.to_qkv
        inner_dim = heads * dim_head
        kv_dim = dim_head if one_kv_head else inner_dim
        self.budget = k
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.to_v = None if share_kv else nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(self._init_projection())
        self.proj_v = None if share_kv else nn.Parameter(self._init_projection())
        self.attn_dropout = nn.Dropout(attn_dropout)

    def _init_projection(self) -> torch.Tensor:
        # lucidrains init_: uniform on (-1/sqrt(k), 1/sqrt(k)), i.e. std 1/sqrt(3k).
        bound = 1.0 / math.sqrt(self.budget)
        return torch.empty(self.seq_len, self.budget).uniform_(-bound, bound)

    def project(self, x):
        q = self.split_heads(self.to_q(x))
        k = self.split_heads(self.to_k(x))
        v = k if self.to_v is None else self.split_heads(self.to_v(x))
        return q, k, v

    def attend(self, q, k, v):
        n = k.shape[-2]
        if n != self.seq_len:
            raise ValueError(f"Linformer was built for seq_len {self.seq_len}, got {n}")
        proj_v = self.proj_k if self.proj_v is None else self.proj_v
        k = torch.einsum("bhnd,nk->bhkd", k, self.proj_k)
        v = torch.einsum("bhnd,nk->bhkd", v, proj_v)
        attn = (q @ k.transpose(-1, -2) * self.scale).softmax(dim=-1)
        return self.attn_dropout(attn) @ v


def moore_penrose_pinv(x: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """Iterative Moore-Penrose pseudo-inverse used by Nystromformer (Razavi et al.)."""
    abs_x = x.abs()
    # Scaled per matrix rather than by the batch-wide maximum as nystrom-attention does,
    # so a sample's output never depends on its batch mates; identical once converged.
    inf_norm = abs_x.sum(dim=-1).amax(dim=-1)
    one_norm = abs_x.sum(dim=-2).amax(dim=-1)
    z = x.transpose(-1, -2) / (one_norm * inf_norm)[..., None, None]
    eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * eye - (xz @ (15 * eye - (xz @ (7 * eye - xz)))))
    return z


class NystromAttention(Attention):
    """Eq. 4-6: softmax(Q K~^T) (softmax(Q~ K~^T))^+ softmax(Q~ K^T) V over m landmarks."""

    budget_kwarg = "num_landmarks"

    def __init__(
        self,
        dim: int,
        heads: int = 3,
        dim_head: int = 64,
        seq_len: int | None = None,
        dropout: float = 0.0,
        num_landmarks: int = 64,
        pinv_iterations: int = 6,
        residual_conv: bool = True,
        residual_conv_kernel: int = 33,
    ):
        if seq_len is None:
            raise ValueError("NystromAttention needs seq_len to lay out its landmark segments")
        super().__init__(dim, heads, dim_head, seq_len, dropout)
        self.budget = num_landmarks
        self.pinv_iterations = pinv_iterations
        # Landmarks are means of contiguous segments of the real tokens (Sec. 3.2); the
        # segments differ in length by at most one, so no landmark is built from padding.
        self.register_buffer("segment_means", segment_mean_matrix(seq_len, num_landmarks), persistent=False)
        self.res_conv = None
        if residual_conv:
            self.res_conv = nn.Conv2d(
                heads,
                heads,
                (residual_conv_kernel, 1),
                padding=(residual_conv_kernel // 2, 0),
                groups=heads,
                bias=False,
            )

    def attend(self, q, k, v):
        n = k.shape[-2]
        if n != self.seq_len:
            raise ValueError(f"Nystrom attention was built for seq_len {self.seq_len}, got {n}")
        q = q * self.scale
        segment_means = self.segment_means.to(q.dtype)
        q_landmarks = torch.einsum("nm,bhnd->bhmd", segment_means, q)
        k_landmarks = torch.einsum("nm,bhnd->bhmd", segment_means, k)

        kernel_1 = (q @ k_landmarks.transpose(-1, -2)).softmax(dim=-1)
        kernel_2 = (q_landmarks @ k_landmarks.transpose(-1, -2)).softmax(dim=-1)
        kernel_3 = (q_landmarks @ k.transpose(-1, -2)).softmax(dim=-1)
        out = (kernel_1 @ moore_penrose_pinv(kernel_2, self.pinv_iterations)) @ (kernel_3 @ v)
        if self.res_conv is not None:
            out = out + self.res_conv(v)
        return out


def orthogonal_gaussian_random_matrix(rows: int, cols: int, sign_correction: bool = True) -> torch.Tensor:
    """FAVOR+ projection (Sec. 2.4): orthogonalised Gaussian blocks with Gaussian row norms."""
    q, r = torch.linalg.qr(torch.randn(math.ceil(rows / cols), cols, cols))
    # Fixing the sign of R's diagonal makes QR equal to Gram-Schmidt, so Q is uniform on
    # the orthogonal group and every feature keeps a Gaussian marginal (Lemma 1); the
    # lucidrains and Google reference code skip this and bias the first coordinates.
    if sign_correction:
        q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    matrix = q.transpose(-1, -2).reshape(-1, cols)[:rows]
    multiplier = torch.randn(rows, cols).norm(dim=1)
    return multiplier[:, None] * matrix


def softmax_kernel(x: torch.Tensor, projection: torch.Tensor, is_query: bool, eps: float = 1e-6):
    """FAVOR+ positive features exp(w^T x - |x|^2 / 2) / sqrt(r) (Choromanski et al. 2021, Eq. 5)."""
    # d^-1/4 on both q and k gives the 1/sqrt(d) of Eq. 3 inside the exponent.
    normalizer = x.shape[-1] ** -0.25
    ratio = projection.shape[0] ** -0.5
    projected = torch.einsum("bhnd,rd->bhnr", x * normalizer, projection)
    # The |x|^2 / 2 term that the paper's Eq. 8 omits.
    diag = (x**2).sum(dim=-1, keepdim=True) / 2 * normalizer**2
    if is_query:
        stabilizer = projected.amax(dim=-1, keepdim=True).detach()
    else:
        stabilizer = projected.amax(dim=(-1, -2), keepdim=True).detach()
    return ratio * (torch.exp(projected - diag - stabilizer) + eps)


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Eq. 9-10: phi(Q) (phi(K)^T V), row-normalised, never forming an n x n matrix."""
    k_sum = k.sum(dim=-2)
    d_inv = 1.0 / torch.einsum("bhnr,bhr->bhn", q, k_sum)
    context = torch.einsum("bhnr,bhnd->bhrd", k, v)
    return torch.einsum("bhrd,bhnr,bhn->bhnd", context, q, d_inv)


class PerformerAttention(Attention):
    """Eq. 8-10: FAVOR+ with positive orthogonal random features (Choromanski et al.)."""

    def __init__(
        self,
        dim: int,
        heads: int = 3,
        dim_head: int = 64,
        seq_len: int | None = None,
        dropout: float = 0.0,
        nb_features: int = 256,
        redraw_interval: int | None = 1000,
        eps: float = 1e-6,
        orthogonal_sign_correction: bool = True,
    ):
        super().__init__(dim, heads, dim_head, seq_len, dropout)
        self.nb_features = nb_features
        self.redraw_interval = redraw_interval
        self.eps = eps
        self.orthogonal_sign_correction = orthogonal_sign_correction
        self.register_buffer("projection_matrix", self.draw_projection_matrix())

    def draw_projection_matrix(self) -> torch.Tensor:
        return orthogonal_gaussian_random_matrix(self.nb_features, self.dim_head, self.orthogonal_sign_correction)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.projection_matrix.copy_(self.draw_projection_matrix().to(self.projection_matrix))

    def redraw(self, step: int) -> None:
        """Sec. 4.2: the random features are redrawn periodically during training."""
        if self.redraw_interval and step % self.redraw_interval == 0:
            self.redraw_projection_matrix()

    def attend(self, q, k, v):
        q = softmax_kernel(q, self.projection_matrix, is_query=True, eps=self.eps)
        k = softmax_kernel(k, self.projection_matrix, is_query=False, eps=self.eps)
        return linear_attention(q, k, v)


KINDS: dict[str, type[Attention]] = {
    "T": StandardAttention,
    "L": LinformerAttention,
    "N": NystromAttention,
    "P": PerformerAttention,
}


def build_attention(kind: str, dim: int, heads: int, dim_head: int, seq_len: int, dropout: float = 0.0, **kwargs) -> Attention:
    """Dict lookup on KINDS; `kwargs` are the constructor arguments specific to that kind."""
    try:
        cls = KINDS[kind]
    except KeyError:
        raise ValueError(f"unknown attention kind {kind!r}; known kinds: {sorted(KINDS)}") from None
    attention = cls(dim=dim, heads=heads, dim_head=dim_head, seq_len=seq_len, dropout=dropout, **kwargs)
    if attention.budget is not None and attention.budget >= seq_len:
        raise ValueError(f"attention {kind!r}: budget {attention.budget} must be below seq_len {seq_len}")
    return attention


def stream_kwargs(kind: str, kwargs: dict | None, seq_len: int) -> dict:
    """Kind-specific kwargs for one token stream: a budget above half the stream is cut to half."""
    kwargs = dict(kwargs or {})
    try:
        cls = KINDS[kind]
    except KeyError:
        raise ValueError(f"unknown attention kind {kind!r}; known kinds: {sorted(KINDS)}") from None
    name = cls.budget_kwarg
    if name is not None:
        requested = kwargs.get(name, inspect.signature(cls).parameters[name].default)
        kwargs[name] = min(requested, seq_len // 2)
    return kwargs
