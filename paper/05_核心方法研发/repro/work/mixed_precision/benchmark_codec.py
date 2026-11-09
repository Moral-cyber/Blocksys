"""Small reproducible CPU codec benchmark; never starts training."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import statistics
import time

import torch

from codec import decode, encode


def main() -> None:
    numel = 136874
    budget = 64 + 8 * int(0.05 * numel)
    generator = torch.Generator().manual_seed(20261104)
    update = torch.randn(numel, generator=generator)
    importance = torch.randn(numel, generator=generator).square()
    report = {"numel": numel, "budget_bytes": budget, "warmups": 1, "repeats": 7,
              "python": platform.python_version(), "torch": str(torch.__version__),
              "device": "cpu", "encode_includes_decode": True, "modes": {}}
    for mode in ("uniform3", "mixed_mse", "mixed_sensitive"):
        kwargs = {"importance": importance} if mode == "mixed_sensitive" else {}
        result = encode(update, budget, mode=mode, **kwargs)
        encode_times, decode_times = [], []
        for _ in range(report["repeats"]):
            start = time.perf_counter()
            result = encode(update, budget, mode=mode, **kwargs)
            encode_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            decode(result.wire)
            decode_times.append(time.perf_counter() - start)
        report["modes"][mode] = {
            "wire_bytes": len(result.wire), "bitwidth_counts": result.diagnostics["bitwidth_counts"],
            "encode_seconds": encode_times, "decode_seconds": decode_times,
            "encode_median_seconds": statistics.median(encode_times),
            "encode_max_seconds": max(encode_times),
            "decode_median_seconds": statistics.median(decode_times),
        }
    destination = Path(__file__).with_name("CODEC_BENCHMARK.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
