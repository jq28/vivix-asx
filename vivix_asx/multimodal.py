"""ViViX + ASX + fusion, built directly or from a yaml config."""

import yaml
from torch import nn

from vivix_asx.asx import ASX
from vivix_asx.fusion import build_fusion
from vivix_asx.vivix import ViViX


class MultimodalModel(nn.Module):
    """Eq. 1-2 feature extractors feeding the fusion module of Sec. 3.3."""

    def __init__(self, video: ViViX, audio: ASX, fusion: nn.Module):
        super().__init__()
        self.video = video
        self.audio = audio
        self.fusion = fusion
        # The fusion module classifies, so the encoders' own heads would never receive a gradient.
        self.video.mlp_head = None
        self.audio.mlp_head = None

    def forward(self, video, audio):
        return self.fusion(self.video.features(video), self.audio.features(audio))


def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config) -> nn.Module:
    """Video-only, audio-only or multimodal, depending on which sections the config has."""
    if not isinstance(config, dict):
        config = load_config(config)
    num_outputs = config.get("num_outputs", 1)

    def section(name):
        """The section's kwargs and its attention kind and kwargs, top-level unless it overrides them."""
        settings = dict(config.get(name) or {})
        kind = settings.pop("attention", config["attention"])
        # attention_kwargs is keyed by kind in the yaml so that switching `attention` needs no other edit.
        kwargs = settings.pop("attention_kwargs", config.get("attention_kwargs") or {}).get(kind, {})
        return kind, kwargs, settings

    video = audio = None
    if "video" in config:
        kind, kwargs, settings = section("video")
        video = ViViX(attention=kind, attention_kwargs=kwargs, num_outputs=num_outputs, **settings)
    if "audio" in config:
        kind, kwargs, settings = section("audio")
        audio = ASX(attention=kind, attention_kwargs=kwargs, num_outputs=num_outputs, **settings)
    if video is None and audio is None:
        raise ValueError("config needs a 'video' and/or an 'audio' section")
    if video is None or audio is None:
        return audio if video is None else video
    kind, kwargs, settings = section("fusion")
    fusion_kind = settings.pop("kind", "learnable_concat")
    fusion = build_fusion(fusion_kind, video.dim, audio.dim, num_outputs, attention=kind, attention_kwargs=kwargs, **settings)
    return MultimodalModel(video, audio, fusion)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
