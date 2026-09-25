"""Fidelity: with identical weights, each attention kind matches the upstream implementation it was rewritten from."""

import re

import pytest
import torch
from torch import nn

from vivix_asx.attention import build_attention
from vivix_asx.transformer import EncoderLayer
from vivix_asx.vivix import ViViX

N, DIM, HEADS, DIM_HEAD, BATCH = 64, 32, 2, 16, 2
TOL = 1e-5
MODEL_TOL = 1e-6


def reference_standard(ours, heads, settings):
    """vit-pytorch vivit.Attention wraps its own LayerNorm, so ours is fed the same normed input."""
    vivit = pytest.importorskip("vit_pytorch.vivit")
    ref = vivit.Attention(DIM, heads=heads, dim_head=DIM_HEAD, use_flash_attn=False, **settings)
    state = ref.state_dict()
    ours.load_state_dict(
        {
            "to_qkv.weight": state["to_qkv.weight"],
            "to_out.weight": state["to_out.0.weight"],
            "to_out.bias": state["to_out.0.bias"],
        }
    )
    norm = nn.LayerNorm(DIM)
    norm.load_state_dict({"weight": state["norm.weight"], "bias": state["norm.bias"]})
    return ref, nn.Sequential(norm, ours)


def reference_linformer(ours, heads, settings):
    linformer = pytest.importorskip("linformer")
    ref = linformer.LinformerSelfAttention(DIM, N, heads=heads, dim_head=DIM_HEAD, **settings)
    ours.load_state_dict(ref.state_dict())
    return ref, ours


def reference_nystrom(ours, heads, settings):
    nystrom = pytest.importorskip("nystrom_attention")
    ref = nystrom.NystromAttention(DIM, dim_head=DIM_HEAD, heads=heads, **settings)
    ours.load_state_dict({key.replace("to_out.0.", "to_out."): value for key, value in ref.state_dict().items()})
    return ref, ours


# Our compatibility flags, set to the lucidrains values in the Performer variant; the
# upstream constructor does not take them (it hard-codes eps = 1e-4 and plain QR).
COMPATIBILITY_FLAGS = {"eps", "orthogonal_sign_correction"}


def reference_performer(ours, heads, settings):
    performer = pytest.importorskip("performer_pytorch")
    upstream = {key: value for key, value in settings.items() if key not in COMPATIBILITY_FLAGS}
    ref = performer.SelfAttention(DIM, heads=heads, dim_head=DIM_HEAD, **upstream)
    state = ref.state_dict()
    ours.load_state_dict(
        {
            "to_qkv.weight": torch.cat([state["to_q.weight"], state["to_k.weight"], state["to_v.weight"]]),
            "to_out.weight": state["to_out.weight"],
            "to_out.bias": state["to_out.bias"],
            "projection_matrix": state["fast_attention.projection_matrix"],
        }
    )
    return ref, ours


# Per kind: the builder that constructs the upstream module and copies its weights into
# ours, the settings (shared by both) to compare under, and the heads and batch size.
# Nystrom is compared at a landmark count dividing n, one matrix at a time (one head,
# batch 1): upstream scales its pseudo-inverse iteration by the batch-wide maximum while
# ours scales per matrix, and the two coincide only then.
REFERENCES = {
    "T": (reference_standard, [{}], HEADS, BATCH),
    "L": (
        reference_linformer,
        [{"k": 16, "one_kv_head": False}, {"k": 16, "one_kv_head": True}, {"k": 16, "one_kv_head": False, "share_kv": True}],
        HEADS,
        BATCH,
    ),
    "N": (reference_nystrom, [{"num_landmarks": 16}], 1, 1),
    "P": (reference_performer, [{"nb_features": 32, "eps": 1e-4, "orthogonal_sign_correction": False}], HEADS, BATCH),
}


def test_matches_upstream_implementation(kind):
    if kind not in REFERENCES:
        pytest.skip(f"no upstream reference registered for kind {kind!r}")
    build_reference, variants, heads, batch = REFERENCES[kind]
    for settings in variants:
        torch.manual_seed(0)
        ours = build_attention(kind, DIM, heads, DIM_HEAD, seq_len=N, **settings)
        ref, ours = build_reference(ours, heads, settings)
        ref.double().eval()
        ours.double().eval()

        x = torch.randn(batch, N, DIM, dtype=torch.float64)
        with torch.no_grad():
            expected = ref(x)
            actual = ours(x)
        difference = (expected - actual).abs().max().item()
        name = type(ref)
        print(f"\n[{kind}] {settings}: max abs difference from {name.__module__}.{name.__name__}: {difference:.2e}")
        assert actual.shape == expected.shape
        assert difference < TOL, settings


# vit-pytorch parameter names to ours. The Sequential indices shift because vit-pytorch
# keeps the LayerNorm inside its Attention and FeedForward while here it is EncoderLayer's.
VIVIT_RENAMES = [
    (r"^to_patch_embedding\.1\.", "to_patch_embedding.0."),
    (r"^to_patch_embedding\.2\.", "to_patch_embedding.1."),
    (r"^to_patch_embedding\.3\.", "to_patch_embedding.2."),
    (r"(_transformer\.layers\.\d+)\.0\.norm\.", r"\1.norm1."),
    (r"(_transformer\.layers\.\d+)\.0\.to_qkv\.", r"\1.attn.to_qkv."),
    (r"(_transformer\.layers\.\d+)\.0\.to_out\.0\.", r"\1.attn.to_out."),
    (r"(_transformer\.layers\.\d+)\.1\.net\.0\.", r"\1.norm2."),
    (r"(_transformer\.layers\.\d+)\.1\.net\.1\.", r"\1.ff.net.0."),
    (r"(_transformer\.layers\.\d+)\.1\.net\.4\.", r"\1.ff.net.3."),
]


def test_vivix_matches_vit_pytorch_vivit():
    """The whole video model against the vit-pytorch factorised-encoder ViT, every weight copied."""
    vivit = pytest.importorskip("vit_pytorch.vivit")
    torch.manual_seed(0)
    ref = vivit.ViViT(
        image_size=224, image_patch_size=16, frames=30, frame_patch_size=2, num_classes=1, dim=192,
        spatial_depth=2, temporal_depth=2, heads=3, dim_head=64, mlp_dim=512,
        variant="factorized_encoder", use_flash_attn=False,
    )
    ours = ViViX(
        attention="T", frames=30, image_size=224, patch_size=16, tubelet_frames=2, dim=192, heads=3,
        dim_head=64, mlp_dim=512, spatial_depth=2, temporal_depth=2, num_outputs=1, dropout=0.0,
    )
    state = {}
    for key, value in ref.state_dict().items():
        for pattern, replacement in VIVIT_RENAMES:
            key = re.sub(pattern, replacement, key)
        state[key] = value
    assert set(state) == set(ours.state_dict()), set(state) ^ set(ours.state_dict())
    state["spatial_cls_token"] = state["spatial_cls_token"].reshape(1, 1, 1, -1)
    ours.load_state_dict(state)
    ref.double().eval()
    ours.double().eval()

    clip = torch.randn(1, 30, 3, 224, 224, dtype=torch.float64)
    with torch.no_grad():
        expected = ref(clip.permute(0, 2, 1, 3, 4))
        actual = ours.mlp_head(ours.features(clip))
    difference = (expected - actual).abs().max().item()
    print(f"\n[ViViX-T] max abs logit difference from vit_pytorch.vivit.ViViT: {difference:.2e}")
    assert difference < MODEL_TOL


def test_asx_block_matches_timm_deit_block():
    """One ASX encoder layer against the timm DeiT Block that YuanGongND's AST is built from."""
    # The depth-4 ASTModel is not constructible: it only offers the fixed DeiT sizes and
    # requires timm 0.4.5, which no longer imports on current torch. DeiT blocks also carry
    # a q/k/v bias and LayerNorm eps 1e-6, which ours does not, so the reference block is
    # built without them; the mlp_ratio rounds to exactly 512 hidden units.
    vit = pytest.importorskip("timm.models.vision_transformer")
    dim, heads, mlp_dim = 192, 3, 512
    torch.manual_seed(0)
    ref = vit.Block(dim, heads, mlp_ratio=(mlp_dim + 0.5) / dim, qkv_bias=False, norm_layer=nn.LayerNorm)
    ours = EncoderLayer(dim, build_attention("T", dim, heads, dim // heads, seq_len=602), mlp_dim)
    assert ref.norm1.eps == ours.norm1.eps and ref.mlp.fc1.out_features == mlp_dim
    renames = {"attn.qkv.": "attn.to_qkv.", "attn.proj.": "attn.to_out.", "mlp.fc1.": "ff.net.0.", "mlp.fc2.": "ff.net.3."}
    state = {}
    for key, value in ref.state_dict().items():
        for old, new in renames.items():
            key = key.replace(old, new)
        state[key] = value
    assert set(state) == set(ours.state_dict()), set(state) ^ set(ours.state_dict())
    ours.load_state_dict(state)
    ref.double().eval()
    ours.double().eval()

    x = torch.randn(2, 602, dim, dtype=torch.float64)
    with torch.no_grad():
        difference = (ref(x) - ours(x)).abs().max().item()
    print(f"\n[ASX-T block] max abs difference from timm.models.vision_transformer.Block: {difference:.2e}")
    assert difference < MODEL_TOL
