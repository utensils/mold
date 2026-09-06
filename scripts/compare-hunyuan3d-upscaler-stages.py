#!/usr/bin/env python3
"""Compare retained Torch and Candle upscaler stages without changing evidence.

This is diagnostic evidence, not a replacement for the Rust qualification gate.
Requires NumPy and safetensors in the retained oracle venv; no GPU is used.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch", type=Path, required=True)
    parser.add_argument("--candle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = load_file(args.torch)
    rows = {}
    for name in ["conv_first", "body.0", "body.11", "body.22", "conv_body",
                 "conv_up1", "conv_up2", "conv_hr", "conv_last"]:
        source = args.candle / f"{name}.safetensors"
        actual = load_file(source)["value"]
        expected = reference[name]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"shape/dtype mismatch for {name}")
        a, b = actual.reshape(-1), expected.reshape(-1)
        maximum, squared, different, nonfinite, different_bytes = 0.0, 0.0, 0, 0, 0
        for offset in range(0, a.size, 1 << 20):
            delta = a[offset:offset + (1 << 20)].astype(np.float64) - b[offset:offset + (1 << 20)]
            nonfinite += int(np.count_nonzero(~np.isfinite(delta)))
            maximum = max(maximum, float(np.abs(delta).max()))
            squared += float(np.sum(delta * delta))
            different += int(np.count_nonzero(delta))
            different_bytes += int(np.count_nonzero(
                a[offset:offset + (1 << 20)].view(np.uint8)
                != b[offset:offset + (1 << 20)].view(np.uint8)))
        rows[name] = {"max": maximum if nonfinite == 0 else None,
                      "rms": (squared / a.size) ** 0.5 if nonfinite == 0 else None,
                      "different": different, "nonfinite": nonfinite,
                      "different_bytes": different_bytes,
                      "elements": int(a.size), "candle_sha256": digest(source)}
    report = {"torch": str(args.torch.resolve()), "torch_sha256": digest(args.torch),
              "candle": str(args.candle.resolve()), "stages": rows}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
