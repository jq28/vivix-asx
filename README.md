# vivix-asx

PyTorch implementation of ViViX and ASX, video and audio transformers with swappable linear attention, from [Are You Paying Attention? Multimodal Linear Attention Transformers for Affect Prediction in Video Conversations](https://doi.org/10.1145/3689092.3689409). Model-only reimplementation: no dataset code, data or checkpoints.

## Install

```bash
pip install git+https://github.com/jq28/vivix-asx.git
```

## Usage

### ViViX

```python
import torch
from vivix_asx import ViViX

model = ViViX(
    attention="L",                                    # "T" standard, "L" Linformer, "N" Nystromformer, "P" Performer
    attention_kwargs={"k": 128},                      # kind-specific: k, one_kv_head, share_kv, num_landmarks,
                                                      # pinv_iterations, nb_features, redraw_interval, eps;
                                                      # dim_head here overrides the encoder's. A landmark
                                                      # count above half a stream's token count is cut to
                                                      # half for that stream (64 stays 64 on the 785-token
                                                      # spatial stream, becomes 8 on the 16-token temporal
                                                      # stream); k is kept on every stream. See model.stream_budgets.
    temporal_attention_kwargs=None,                   # replaces the derived temporal kwargs when given
    frames=30,
    image_size=224,
    patch_size=8,
    tubelet_frames=2,
    channels=3,
    dim=128,
    heads=2,
    dim_head=64,
    mlp_dim=512,
    spatial_depth=2,
    temporal_depth=2,
    num_outputs=1,
    dropout=0.1,
)
video = torch.rand(2, 30, 3, 224, 224)   # (B, frames, C, H, W)
probabilities = model(video)             # (B, 1)
features = model.features(video)         # (B, dim)
```

### ViViT, ViViL, ViViN, ViViP

```python
from vivix_asx import ViViT, ViViL, ViViN, ViViP

model = ViViT()                               # two layers per stream
model = ViViL(k=128)                          # the three linear models default to one layer per stream
model = ViViN(num_landmarks=64)
model = ViViP(nb_features=110, dim_head=32)   # all four take the ViViX keyword arguments too
```

### ASX

```python
import torch
from vivix_asx import ASX

model = ASX(
    attention="L",
    attention_kwargs={"k": 128, "share_kv": True},    # kind-specific, as for ViViX (qkv_bias for "T"); heads and
                                                      # dim_head here override the encoder's. Same landmark rule
                                                      # as ViViX; see model.stream_budgets
    n_mels=128,
    n_frames=1024,
    patch_size=16,
    stride=10,
    dim=192,
    heads=3,
    dim_head=64,
    mlp_dim=768,
    depth=2,
    num_outputs=1,
    dropout=0.1,
)
spectrogram = torch.randn(2, 1, 128, 1024)  # (B, 1, mel bins, frames), log-mel at 16 kHz
probabilities = model(spectrogram)          # (B, 1)
features = model.features(spectrogram)      # (B, dim)
```

### AST, ASL, ASN, ASP

```python
from vivix_asx import AST, ASL, ASN, ASP

model = AST()                                         # two DeiT-tiny blocks with q/k/v bias
model = ASL(k=128, share_kv=True, heads=2)
model = ASN(num_landmarks=64, heads=2)
model = ASP(nb_features=110, heads=1, dim_head=32)   # all four take the ASX keyword arguments too
```

### Multimodal

```python
import torch
from vivix_asx import ASL, LearnableScalarConcatFusion, MultimodalModel, ViViN

model = MultimodalModel(ViViN(), ASL(), LearnableScalarConcatFusion(128, 192, unified_dim=64))
video = torch.rand(2, 30, 3, 224, 224)
spectrogram = torch.randn(2, 1, 128, 512)
probabilities = model(video, spectrogram)   # (B, 1)
```

### From a config

```python
from vivix_asx import build_model, load_config
model = build_model(load_config("configs/asl_vivil.yaml"))
```

## Tests

```bash
pytest -q
```

## Profiling

`benchmarks/efficiency.py` builds the model a config describes, runs it on random inputs of the config's shape, and reports its parameter count, training-step and inference time, and peak GPU memory. No data is needed.

```bash
python benchmarks/efficiency.py configs/vivil.yaml --batch-size 8
```

Several configs at once, saved to a CSV:

```bash
python benchmarks/efficiency.py "configs/*.yaml" --batch-size 8 --csv results.csv
```

The largest batch whose training step fits in, say, 10 GB of GPU memory:

```bash
python benchmarks/efficiency.py configs/asl_vivil.yaml --fit 10
```

To profile your own clip length, frame size or audio duration without editing the yaml, pass `--frames`, `--image-size`, `--n-mels` or `--n-frames`. Without a CUDA device the timings are still reported and memory is n/a.

Columns: `params`; `train_ms` and `infer_ms`, the median time of one training step (forward, BCE, backward, Adam) and one inference forward at the given batch; `samples_per_s`; `peak_alloc_gb`, the memory the training step needs, and `peak_reserved_gb`, what `nvidia-smi` shows; `batch_fitting_gb` with `--fit`.

## Citations

```bibtex
@inproceedings{poh2024attention,
  author    = {Poh, Jia Qing and See, John and El Gayar, Neamat and Wong, Lai-Kuan},
  title     = {Are You Paying Attention? Multimodal Linear Attention Transformers for Affect Prediction in Video Conversations},
  booktitle = {Proceedings of the 2nd International Workshop on Multimodal and Responsible Affective Computing (MRAC '24)},
  year      = {2024},
  publisher = {ACM},
  address   = {New York, NY, USA},
  doi       = {10.1145/3689092.3689409}
}
```

## Acknowledgements

- https://github.com/lucidrains/vit-pytorch
- https://github.com/lucidrains/linformer
- https://github.com/lucidrains/nystrom-attention
- https://github.com/lucidrains/performer-pytorch
- https://github.com/YuanGongND/ast

Licence notices for the reference implementations are in THIRD_PARTY_NOTICES.md.
