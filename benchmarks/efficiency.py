"""Parameters, step time and peak GPU memory of the model a config builds, on random inputs of its shape."""

import argparse
import csv
import functools
import glob
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Runnable as a script from anywhere without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vivix_asx.multimodal import build_model, count_parameters, load_config  # noqa: E402

OVERRIDES = {"frames": "video", "image_size": "video", "n_mels": "audio", "n_frames": "audio"}


def random_inputs(config, batch, device):
    inputs = []
    if "video" in config:
        video = config["video"]
        inputs.append(torch.rand(batch, video["frames"], 3, video["image_size"], video["image_size"], device=device))
    if "audio" in config:
        audio = config["audio"]
        inputs.append(torch.randn(batch, 1, audio["n_mels"], audio["n_frames"], device=device))
    return inputs


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device, warmup, steps):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(steps):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def profile(config, batch, device, warmup, steps):
    """One row: a training step (forward, BCE, backward, Adam) and an inference forward at this batch."""
    torch.manual_seed(0)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = random_inputs(config, batch, device)
    targets = torch.randint(0, 2, (batch, config.get("num_outputs", 1)), device=device).float()

    def train_step():
        model.train()
        loss = F.binary_cross_entropy(model(*inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    def infer_step():
        model.eval()
        with torch.no_grad():
            model(*inputs)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    row = {"params": count_parameters(model), "batch": batch}
    row["train_ms"] = timed(train_step, device, warmup, steps)
    row["samples_per_s"] = batch / (row["train_ms"] / 1e3)
    row["infer_ms"] = timed(infer_step, device, warmup, steps)
    if device.type == "cuda":
        row["peak_alloc_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
        row["peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 2**30
        del model, optimizer, inputs
        torch.cuda.empty_cache()
    return row


def fits(config, batch, device, budget_gb):
    try:
        return profile(config, batch, device, warmup=1, steps=1)["peak_alloc_gb"] <= budget_gb
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False


def largest_batch(config, device, budget_gb):
    """Doubling then bisection on the largest batch whose training step stays within the budget."""
    fit = functools.partial(fits, config, device=device, budget_gb=budget_gb)
    if not fit(1):
        return 0
    low = 1
    while fit(low * 2):
        low *= 2
    high = low * 2
    while high - low > 1:
        mid = (low + high) // 2
        low, high = (mid, high) if fit(mid) else (low, mid)
    return low


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="yaml files; shell globs like configs/*.yaml work")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10, help="timed steps; the median is reported")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fit", type=float, metavar="GB", help="also find the largest batch whose training step fits GB")
    for name, section in OVERRIDES.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, help=f"override {section}.{name} in every config")
    parser.add_argument("--csv", help="also write the table to this file")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.fit is not None and device.type != "cuda":
        parser.error("--fit needs a CUDA device")

    rows = []
    for pattern in args.configs:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            config = load_config(path)
            for name, section in OVERRIDES.items():
                value = getattr(args, name)
                if value is not None and section in config:
                    config[section][name] = value
            row = {"config": Path(path).as_posix(), **profile(config, args.batch_size, device, args.warmup, args.steps)}
            if args.fit is not None:
                row["batch_fitting_gb"] = largest_batch(config, device, args.fit)
            rows.append(row)
            print(row, flush=True)

    columns = list(rows[0])
    formats = {"params": "{:,}", "train_ms": "{:.1f}", "infer_ms": "{:.1f}", "samples_per_s": "{:.1f}",
               "peak_alloc_gb": "{:.2f}", "peak_reserved_gb": "{:.2f}"}
    print("\n| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for row in rows:
        print("| " + " | ".join(formats.get(key, "{}").format(row[key]) for key in columns) + " |")
    if device.type != "cuda":
        print("\npeak memory: n/a (no CUDA device)")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
