#!/usr/bin/env python3
"""Metal-vs-CUDA A/B summary for one LTX-2.5 render pair.

Both files are decoded to rgb24 frames through ffmpeg, then compared frame by
frame: PSNR over RGB and a block SSIM (8x8 luma windows, K1=0.01, K2=0.03)
whose per-frame value is the mean over blocks. Pure Python on purpose — the
qualification host runs it under whichever interpreter the harness names, and
the CI contract runs it under a bare python3 with neither numpy nor PIL.

Exact pixels are NOT expected across backends; every number here is recorded
provenance and never a gate. Geometry parity (frames, width, height) is
reported beside the metrics, and mismatched geometry yields null metrics
rather than a failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

METHOD = (
    "ffmpeg rgb24 frames; PSNR over RGB; block SSIM over 8x8 luma windows "
    "(K1=0.01, K2=0.03, L=255), per-frame mean over blocks; recorded, never gated"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> tuple[int, int, int]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-show_entries",
            "stream=codec_type,width,height,nb_frames,nb_read_frames", "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(result.stdout)["streams"]
    video = next(stream for stream in streams if stream.get("codec_type") == "video")
    frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    return int(video["width"]), int(video["height"]), frames


def frames_rgb24(path: Path, width: int, height: int) -> list[bytes]:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    stride = width * height * 3
    return [raw[offset:offset + stride] for offset in range(0, len(raw) - stride + 1, stride)]


def psnr(a: bytes, b: bytes) -> float:
    total = 0
    for x, y in zip(a, b):
        d = x - y
        total += d * d
    if total == 0:
        return math.inf
    mse = total / len(a)
    return 10.0 * math.log10(255.0 * 255.0 / mse)


def luma(frame: bytes, width: int, height: int) -> list[list[float]]:
    rows = []
    for y in range(height):
        base = y * width * 3
        row = []
        for x in range(width):
            i = base + x * 3
            row.append(0.299 * frame[i] + 0.587 * frame[i + 1] + 0.114 * frame[i + 2])
        rows.append(row)
    return rows


def block_ssim(a: bytes, b: bytes, width: int, height: int, block: int = 8) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    la = luma(a, width, height)
    lb = luma(b, width, height)
    scores = []
    n = block * block
    for by in range(0, height - block + 1, block):
        for bx in range(0, width - block + 1, block):
            sum_a = sum_b = sum_aa = sum_bb = sum_ab = 0.0
            for y in range(by, by + block):
                ra = la[y]
                rb = lb[y]
                for x in range(bx, bx + block):
                    va = ra[x]
                    vb = rb[x]
                    sum_a += va
                    sum_b += vb
                    sum_aa += va * va
                    sum_bb += vb * vb
                    sum_ab += va * vb
            mean_a = sum_a / n
            mean_b = sum_b / n
            var_a = sum_aa / n - mean_a * mean_a
            var_b = sum_bb / n - mean_b * mean_b
            cov = sum_ab / n - mean_a * mean_b
            scores.append(
                ((2 * mean_a * mean_b + c1) * (2 * cov + c2))
                / ((mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2))
            )
    return sum(scores) / len(scores) if scores else float("nan")


def summary(values: list[float]) -> dict:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "per_frame": [None if math.isinf(value) else round(value, 4) for value in values],
        "min": round(min(finite), 4) if finite else None,
        "mean": round(sum(finite) / len(finite), 4) if finite else None,
        "max": round(max(finite), 4) if finite else None,
        "infinite_frames": sum(1 for value in values if math.isinf(value)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    ref_w, ref_h, ref_n = probe(args.reference)
    cand_w, cand_h, cand_n = probe(args.candidate)
    parity = {
        "frames": {"reference": ref_n, "candidate": cand_n, "equal": ref_n == cand_n},
        "width": {"reference": ref_w, "candidate": cand_w, "equal": ref_w == cand_w},
        "height": {"reference": ref_h, "candidate": cand_h, "equal": ref_h == cand_h},
    }
    result = {
        "method": METHOD,
        "reference_file": str(args.reference),
        "reference_sha256": sha256_file(args.reference),
        "candidate_file": str(args.candidate),
        "candidate_sha256": sha256_file(args.candidate),
        "parity": parity,
        "frames_compared": 0,
        "psnr_db": None,
        "ssim": None,
        "identical": False,
    }
    if ref_w == cand_w and ref_h == cand_h:
        ref_frames = frames_rgb24(args.reference, ref_w, ref_h)
        cand_frames = frames_rgb24(args.candidate, cand_w, cand_h)
        count = min(len(ref_frames), len(cand_frames))
        psnrs = [psnr(ref_frames[i], cand_frames[i]) for i in range(count)]
        ssims = [block_ssim(ref_frames[i], cand_frames[i], ref_w, ref_h) for i in range(count)]
        result["frames_compared"] = count
        result["psnr_db"] = summary(psnrs)
        result["ssim"] = summary(ssims)
        result["identical"] = (
            count > 0 and len(ref_frames) == len(cand_frames) and all(math.isinf(v) for v in psnrs)
        )
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames_compared": result["frames_compared"], "identical": result["identical"]}))


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, OSError, ValueError, KeyError, StopIteration) as error:
        print(f"ltx25-metal-ab: {error}", file=sys.stderr)
        raise SystemExit(1)
