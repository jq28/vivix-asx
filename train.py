"""Training skeleton on random tensors: BCE (Eq. 14), Adam, per-modality learning rates (Sec. 4.2)."""

import argparse

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vivix_asx.asx import ASX
from vivix_asx.attention import Attention
from vivix_asx.multimodal import MultimodalModel, build_model, count_parameters, load_config

SAMPLE_RATE = 16000
CLIP_SECONDS = 5
TARGETS = ("arousal", "valence")


class KEmoConDataset(Dataset):
    """Interface to fill in; no dataset code, data or features ship with this repo."""

    def __init__(self, root: str, split: str):
        raise NotImplementedError("fill in the K-EmoCon loader; see __getitem__ for the item contract")

    def __getitem__(self, index: int) -> dict:
        """One 5 s clip: frames (30, 3, H, W) in [0, 1], waveform (1, 80000) at 16 kHz, arousal and valence as 0. or 1."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class RandomClipDataset(Dataset):
    """Random tensors with the KEmoConDataset item contract, for smoke-testing the loop."""

    def __init__(self, config: dict, length: int):
        self.config = config
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        item = {target: float(torch.randint(0, 2, ())) for target in TARGETS}
        if "video" in self.config:
            video = self.config["video"]
            item["frames"] = torch.rand(video["frames"], 3, video["image_size"], video["image_size"])
        if "audio" in self.config:
            item["waveform"] = 0.1 * torch.randn(1, SAMPLE_RATE * CLIP_SECONDS)
        return item


class LogMel:
    """Kaldi-style log-mel filterbank as in the AST dataloader, padded or cropped to n_frames."""

    def __init__(self, n_mels: int, n_frames: int):
        self.n_mels = n_mels
        self.n_frames = n_frames

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # TODO(unspecified in paper): AST also normalises by dataset mean and std after
        # this; that belongs in the Dataset once the statistics exist.
        spectrograms = []
        for clip in waveform:
            fbank = torchaudio.compliance.kaldi.fbank(
                clip - clip.mean(),
                htk_compat=True,
                sample_frequency=SAMPLE_RATE,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=self.n_mels,
                dither=0.0,
                frame_shift=10,
            )[: self.n_frames]
            fbank = F.pad(fbank, (0, 0, 0, self.n_frames - fbank.shape[0]))
            spectrograms.append(fbank.t())
        return torch.stack(spectrograms).unsqueeze(1)


# TODO(unspecified in paper): every augmentation strength below, and that they are applied
# to every sample; Sec. 4.2 names the five augmentations only.
def video_noise(frames: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    return frames + std * torch.randn_like(frames)


def video_hflip(frames: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    flip = torch.rand(frames.shape[0], device=frames.device) < p
    return torch.where(flip[:, None, None, None, None], frames.flip(-1), frames)


def video_brightness(frames: torch.Tensor, max_delta: float = 0.2) -> torch.Tensor:
    delta = (torch.rand(frames.shape[0], 1, 1, 1, 1, device=frames.device) * 2 - 1) * max_delta
    return (frames + delta).clamp(0, 1)


def audio_noise(waveform: torch.Tensor, std: float = 0.005) -> torch.Tensor:
    return waveform + std * torch.randn_like(waveform)


def audio_pitch_shift(waveform: torch.Tensor, max_steps: int = 2) -> torch.Tensor:
    steps = int(torch.randint(-max_steps, max_steps + 1, ()))
    if steps == 0:
        return waveform
    return torchaudio.functional.pitch_shift(waveform, SAMPLE_RATE, steps)


AUGMENTATIONS = {
    "frames": [video_noise, video_hflip, video_brightness],
    "waveform": [audio_noise, audio_pitch_shift],
}


def parameter_groups(model: nn.Module, train_config: dict) -> list:
    lrs = {
        "video": train_config.get("lr_video", 1e-3),
        "audio": train_config.get("lr_audio", 1e-4),
        "fusion": train_config.get("lr_fusion", 1e-3),
    }
    if isinstance(model, MultimodalModel):
        return [{"params": getattr(model, name).parameters(), "lr": lr} for name, lr in lrs.items()]
    return [{"params": model.parameters(), "lr": lrs["audio" if isinstance(model, ASX) else "video"]}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target", choices=TARGETS, default="arousal", help="label used when num_outputs is 1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = load_config(args.config)
    train_config = config.get("train", {})
    batch_size = args.batch_size or train_config.get("batch_size", 4)
    num_outputs = config.get("num_outputs", 1)
    if num_outputs not in (1, 2):
        raise ValueError("num_outputs must be 1 (--target picks the label) or 2 (arousal and valence)")
    targets = TARGETS if num_outputs == 2 else (args.target,)

    model = build_model(config).to(args.device).train()
    print(f"{args.config}: {count_parameters(model) / 1e6:.3f}M parameters, targets {targets}")
    loader = DataLoader(RandomClipDataset(config, batch_size * args.steps), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(parameter_groups(model, train_config))
    loss_fn = nn.BCELoss()
    log_mel = LogMel(config["audio"]["n_mels"], config["audio"]["n_frames"]) if "audio" in config else None

    for step, batch in enumerate(loader, 1):
        inputs = []
        for key in ("frames", "waveform"):
            if key not in batch:
                continue
            tensor = batch[key].to(args.device)
            if not args.no_augment:
                for augment in AUGMENTATIONS[key]:
                    tensor = augment(tensor)
            inputs.append(log_mel(tensor) if key == "waveform" else tensor)
        labels = torch.stack([batch[t] for t in targets], dim=1).float().to(args.device)

        loss = loss_fn(model(*inputs), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for module in model.modules():
            if isinstance(module, Attention):
                module.redraw(step)
        print(f"step {step} loss {loss.item():.4f}")
        if step >= args.steps:
            break


if __name__ == "__main__":
    main()
