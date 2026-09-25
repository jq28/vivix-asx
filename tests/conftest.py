import pytest
import torch

from vivix_asx.attention import KINDS

# Small enough that every variant runs in well under a second on CPU; the encoders fit
# each budget to their streams (spatial 17, temporal 5, audio 12, fusion 4 tokens).
SMALL_KWARGS = {"L": {"k": 8}, "N": {"num_landmarks": 8}, "P": {"nb_features": 32}}


def small_video_config(kind: str) -> dict:
    return dict(
        attention=kind,
        attention_kwargs=SMALL_KWARGS.get(kind, {}),
        frames=8,
        image_size=32,
        patch_size=8,
        tubelet_frames=2,
        dim=32,
        heads=2,
        dim_head=16,
        mlp_dim=64,
        spatial_depth=1,
        temporal_depth=1,
        dropout=0.1,
    )


def small_audio_config(kind: str) -> dict:
    return dict(
        attention=kind,
        attention_kwargs=SMALL_KWARGS.get(kind, {}),
        n_mels=32,
        n_frames=64,
        patch_size=16,
        stride=10,
        dim=32,
        heads=2,
        dim_head=16,
        mlp_dim=64,
        depth=1,
        dropout=0.1,
    )


def small_inputs(batch: int = 2):
    return torch.rand(batch, 8, 3, 32, 32), torch.randn(batch, 1, 32, 64)


@pytest.fixture(params=sorted(KINDS), ids=lambda k: f"kind={k}")
def kind(request):
    return request.param


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
