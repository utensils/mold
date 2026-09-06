#!/usr/bin/env python3
"""Compare retained real-mesh conditioning PNGs without rewriting either set."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    results = []
    for index in range(12):
        name = f"condition-{index:02d}.png"
        actual = np.asarray(Image.open(args.actual / name).convert("RGB"))
        reference = np.asarray(Image.open(args.reference / name).convert("RGB"))
        assert actual.shape == reference.shape
        delta = np.abs(actual.astype(np.int16) - reference.astype(np.int16))
        mse = float(np.mean(delta.astype(np.float64) ** 2))
        psnr = float(10 * np.log10(255 ** 2 / mse)) if mse else None
        a_mask = np.any(actual != 255, axis=-1)
        b_mask = np.any(reference != 255, axis=-1)
        union = int(np.count_nonzero(a_mask | b_mask))
        iou = float(np.count_nonzero(a_mask & b_mask) / union) if union else 1.
        within_one = float(np.mean(delta <= 1))
        results.append(dict(name=name, max_channel_error=int(delta.max()), psnr=psnr,
                            mask_iou=iou, channels_within_one=within_one,
                            passed=iou >= .999 and within_one >= .995 and (psnr is None or psnr >= 40)))
    report = dict(thresholds=dict(mask_iou=.999, channels_within_one=.995, psnr=40),
                  actual=str(args.actual.resolve()), reference=str(args.reference.resolve()),
                  passed=all(result["passed"] for result in results), results=results)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
