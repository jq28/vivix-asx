"""Forward+backward time, peak memory and parameter count per attention kind and sequence length."""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

# Runnable as a script from anywhere without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vivix_asx.attention import KINDS, build_attention
from vivix_asx.multimodal import count_parameters

# Budgets of 32 so every kind builds at the shortest sequence length, 64.
SETTINGS = {"L": {"k": 32, "one_kv_head": True}, "N": {"num_landmarks": 32}, "P": {"nb_features": 256}}
COLUMNS = ["kind", "seq_len", "median_ms", "peak_memory_mb", "params"]


def benchmark(kind, seq_len, args, device):
    torch.manual_seed(0)
    attn = build_attention(kind, args.dim, args.heads, args.dim_head, seq_len=seq_len, **SETTINGS.get(kind, {})).to(device)
    x = torch.randn(args.batch, seq_len, args.dim, device=device, requires_grad=True)
    cuda = device.type == "cuda"

    def step():
        attn(x).sum().backward()
        if cuda:
            torch.cuda.synchronize(device)

    for _ in range(args.warmup):
        step()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(args.runs):
        start = time.perf_counter()
        step()
        samples.append(time.perf_counter() - start)
    peak = f"{torch.cuda.max_memory_allocated(device) / 2**20:.1f}" if cuda else "n/a"
    return {
        "kind": kind,
        "seq_len": seq_len,
        "median_ms": f"{statistics.median(samples) * 1e3:.2f}",
        "peak_memory_mb": peak,
        "params": count_parameters(attn),
    }


def markdown_table(rows):
    header = "| " + " | ".join(COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = ["| " + " | ".join(str(row[c]) for c in COLUMNS) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--kinds", nargs="+", default=sorted(KINDS))
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--dim-head", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--quick", action="store_true", help="seq_lens 64 and 256, 3 runs")
    args = parser.parse_args()
    if args.quick:
        args.seq_lens, args.runs = [64, 256], 3

    device = torch.device(args.device)
    rows = [benchmark(kind, n, args, device) for kind in args.kinds for n in args.seq_lens]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "efficiency.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    table = markdown_table(rows)
    (out / "efficiency.md").write_text(table + "\n")
    print(f"device {device.type}, batch {args.batch}, dim {args.dim}, heads {args.heads} x {args.dim_head}, settings {SETTINGS}")
    print(table)


if __name__ == "__main__":
    main()
