#!/usr/bin/env python3
"""Capture the OpenCV Canny call used by Tencent paint's depth sketch."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from safetensors.numpy import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.opencv), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == "71d3237a093b60a27601c20e9ee6c3e52154e8b1"
    assert cv2.__version__ == "4.10.0"
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(15111496)
    y, x = np.mgrid[:49, :65]
    cases = {
        "single": np.array([[255]], dtype=np.uint8),
        "row": np.array([[0, 0, 8, 8, 20, 20, 255, 255, 0]], dtype=np.uint8),
        "column": np.array([[0, 8, 20, 40, 255, 0]], dtype=np.uint8).T.copy(),
        "noise": rng.integers(0, 256, (63, 67), dtype=np.uint8),
        "weak": rng.integers(0, 24, (33, 31), dtype=np.uint8),
        "directions": ((x * 3 + y * 7 + (x > y) * 80) % 256).astype(np.uint8),
        "ties": (((x // 5 + y // 7) % 2) * 20).astype(np.uint8),
    }
    # Weak nonmaximal-suppressed edges connected to a strong segment must
    # survive hysteresis; detached weak components must not.
    bridge = np.zeros((40, 60), dtype=np.uint8)
    bridge[5:35, 8:18] = 12
    bridge[5:10, 8:18] = 64
    bridge[5:35, 40:50] = 12
    cases["bridge"] = bridge
    tensors = {}
    for name, pixels in cases.items():
        tensors[f"{name}.pixels"] = pixels
        tensors[f"{name}.edges"] = cv2.Canny(pixels, 30, 80)
    save_file(tensors, str(args.output / "paint-edges.safetensors"))
    (args.output / "paint-edges.json").write_text(json.dumps({
        "opencv": cv2.__version__, "revision": revision,
        "thresholds": [30, 80], "aperture": 3, "l2_gradient": False,
        "cases": list(cases),
        "reference": "Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer/MeshRender.py:1105-1107",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
