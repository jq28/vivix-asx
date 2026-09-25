import math
import statistics
import time
import warnings

import pytest
import torch
from torch.overrides import TorchFunctionMode

from vivix_asx.attention import build_attention

SETTINGS = {"L": {"k": 32}, "N": {"num_landmarks": 64}, "P": {"nb_features": 64}}
# Kinds expected to materialise an n x n tensor; every other kind, including any
# added later, must not.
QUADRATIC = {"T"}


class ShapeRecorder(TorchFunctionMode):
    """Records the shape of every tensor any torch function returns while active."""

    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.record(out)
        return out

    def record(self, out):
        if isinstance(out, torch.Tensor):
            self.shapes.append(tuple(out.shape))
        elif isinstance(out, (tuple, list)):
            for item in out:
                self.record(item)


def test_no_quadratic_tensor(kind):
    # 500 is not a multiple of the 64 landmarks, so Nystrom's uneven segments are exercised too.
    n = 500
    attn = build_attention(kind, dim=64, heads=2, dim_head=32, seq_len=n, **SETTINGS.get(kind, {})).eval()
    with ShapeRecorder() as recorder:
        attn(torch.randn(1, n, 64))
    # Any tensor with two dimensions of at least n, wherever they sit, counts as quadratic.
    quadratic = [s for s in recorder.shapes if sum(size >= n for size in s) >= 2]
    if kind in QUADRATIC:
        assert quadratic, "standard attention should form an n x n matrix"
    else:
        assert not quadratic, f"linear attention formed n x n tensors: {quadratic}"


def log_slope(xs, ys):
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    mx, my = statistics.mean(lx), statistics.mean(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


@pytest.mark.slow
def test_empirical_scaling(kind):
    seq_lens = [128, 256, 512, 1024]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    times, memories = [], []
    for n in seq_lens:
        attn = build_attention(kind, dim=64, heads=4, dim_head=64, seq_len=n, **SETTINGS.get(kind, {})).to(device)
        x = torch.randn(8, n, 64, device=device, requires_grad=True)

        def step():
            attn(x).sum().backward()
            if device.type == "cuda":
                torch.cuda.synchronize()

        for _ in range(2):
            step()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            step()
            samples.append(time.perf_counter() - start)
        times.append(statistics.median(samples))
        if device.type == "cuda":
            memories.append(torch.cuda.max_memory_allocated(device))

    time_slope = log_slope(seq_lens, times)
    print(f"\n[{kind}] time slope {time_slope:.2f} on {device.type}; ms per step: "
          + ", ".join(f"{n}: {t * 1e3:.1f}" for n, t in zip(seq_lens, times)))
    expect_quadratic = kind in QUADRATIC
    if device.type == "cuda":
        memory_slope = log_slope(seq_lens, memories)
        print(f"[{kind}] memory slope {memory_slope:.2f}")
        assert (memory_slope > 1.5) == expect_quadratic, memory_slope
    else:
        warnings.warn("CUDA unavailable: memory scaling not checked and CPU timings are noisy")
    assert (time_slope > 1.5) == expect_quadratic, time_slope
