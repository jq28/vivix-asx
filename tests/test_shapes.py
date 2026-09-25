import pytest
import torch

from conftest import SMALL_KWARGS, small_audio_config, small_inputs, small_video_config
from vivix_asx.asx import ASX
from vivix_asx.attention import KINDS, build_attention, stream_kwargs
from vivix_asx.fusion import FUSIONS, build_fusion
from vivix_asx.multimodal import MultimodalModel
from vivix_asx.vivix import ViViX


def assert_probabilities(out, shape):
    assert out.shape == shape
    assert torch.isfinite(out).all()
    assert ((out >= 0) & (out <= 1)).all()


def test_attention_keeps_shape(kind):
    attn = build_attention(kind, dim=32, heads=2, dim_head=16, seq_len=20, **SMALL_KWARGS.get(kind, {}))
    out = attn(torch.randn(2, 20, 32))
    assert out.shape == (2, 20, 32)
    assert torch.isfinite(out).all()


def test_unknown_kind_raises(kind):
    with pytest.raises(ValueError, match="unknown attention kind"):
        build_attention(kind.lower(), dim=32, heads=2, dim_head=16, seq_len=20)


def test_budget_must_be_below_seq_len(kind):
    name = KINDS[kind].budget_kwarg
    if name is None:
        build_attention(kind, dim=32, heads=2, dim_head=16, seq_len=4)
        return
    with pytest.raises(ValueError, match="budget"):
        build_attention(kind, dim=32, heads=2, dim_head=16, seq_len=20, **{name: 20})
    build_attention(kind, dim=32, heads=2, dim_head=16, seq_len=20, **{name: 19})


def test_stream_budget_is_cut_to_half_the_stream(kind):
    name = KINDS[kind].budget_kwarg
    if name is None:
        assert stream_kwargs(kind, {}, 10) == {}
        return
    assert stream_kwargs(kind, {name: 64}, 10)[name] == 5
    assert stream_kwargs(kind, {name: 3}, 10)[name] == 3
    assert stream_kwargs(kind, {}, 10)[name] == 5


def test_stream_budgets_are_derived_and_overridable(kind):
    name = KINDS[kind].budget_kwarg
    derived = ViViX(**small_video_config(kind))
    fusion = build_fusion("transformer", 32, 32, attention=kind, attention_kwargs=SMALL_KWARGS.get(kind, {}))
    if name is None:
        assert derived.stream_budgets == {"spatial": None, "temporal": None}
        assert fusion.stream_budgets == {"fusion": None}
        return
    assert derived.stream_budgets == {"spatial": 8, "temporal": 2}
    assert fusion.stream_budgets == {"fusion": 2}
    overridden = ViViX(temporal_attention_kwargs={name: 3}, **small_video_config(kind))
    assert overridden.stream_budgets == {"spatial": 8, "temporal": 3}
    assert "stream_budgets" in repr(overridden)
    fusion = build_fusion(
        "transformer", 32, 32, attention=kind, attention_kwargs=SMALL_KWARGS[kind], fusion_attention_kwargs={name: 1}
    )
    assert fusion.stream_budgets == {"fusion": 1}


def test_vivix_shapes(kind):
    model = ViViX(**small_video_config(kind))
    video, _ = small_inputs()
    assert model.features(video).shape == (2, 32)
    assert_probabilities(model(video), (2, 1))


def test_asx_shapes(kind):
    model = ASX(**small_audio_config(kind))
    _, audio = small_inputs()
    assert model.features(audio).shape == (2, 32)
    assert_probabilities(model(audio), (2, 1))


@pytest.mark.parametrize("num_outputs", [1, 2])
def test_num_outputs(kind, num_outputs):
    video, audio = small_inputs()
    assert ViViX(num_outputs=num_outputs, **small_video_config(kind))(video).shape == (2, num_outputs)
    assert ASX(num_outputs=num_outputs, **small_audio_config(kind))(audio).shape == (2, num_outputs)


@pytest.mark.parametrize("fusion_kind", sorted(FUSIONS))
def test_multimodal_shapes(kind, fusion_kind):
    fusion = build_fusion(fusion_kind, 32, 32, attention=kind, attention_kwargs=SMALL_KWARGS.get(kind, {}))
    model = MultimodalModel(ViViX(**small_video_config(kind)), ASX(**small_audio_config(kind)), fusion)
    assert model.video.mlp_head is None and model.audio.mlp_head is None
    assert_probabilities(model(*small_inputs()), (2, 1))


def test_headless_encoder_only_yields_features(kind):
    video, audio = small_inputs()
    assert ViViX(num_outputs=None, **small_video_config(kind)).features(video).shape == (2, 32)
    with pytest.raises(RuntimeError, match="features"):
        ASX(num_outputs=None, **small_audio_config(kind))(audio)


def test_wrong_input_shape_raises(kind):
    video, audio = small_inputs()
    with pytest.raises(ValueError, match="expected"):
        ViViX(**small_video_config(kind))(video[:, :4])
    with pytest.raises(ValueError, match="expected"):
        ASX(**small_audio_config(kind))(audio[..., :32])
