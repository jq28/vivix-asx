from pathlib import Path

import pytest
import torch

from vivix_asx.multimodal import build_model, load_config

CONFIGS = sorted(Path(__file__).resolve().parents[1].joinpath("configs").glob("*.yaml"))


def random_inputs(config: dict, batch: int):
    inputs = []
    if "video" in config:
        video = config["video"]
        inputs.append(torch.rand(batch, video["frames"], 3, video["image_size"], video["image_size"]))
    if "audio" in config:
        audio = config["audio"]
        inputs.append(torch.randn(batch, 1, audio["n_mels"], audio["n_frames"]))
    return inputs


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_forward_backward(path):
    config = load_config(path)
    model = build_model(config).train()
    out = model(*random_inputs(config, batch=2))
    assert out.shape == (2, config.get("num_outputs", 1))
    loss = torch.nn.BCELoss()(out, torch.randint(0, 2, out.shape).float())
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
