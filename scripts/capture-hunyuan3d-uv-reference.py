#!/usr/bin/env python3
"""Capture xatlas 0.0.9 UV output for a synthetic tetrahedron; oracle only."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import xatlas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    version = importlib.metadata.version("xatlas")
    if version != "0.0.9":
        raise ValueError(f"expected xatlas 0.0.9, got {version}")
    vertices = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
    mapping, indices, uv = xatlas.parametrize(vertices, faces)
    record = dict(xatlas_version=version,
                  xatlas_revision="f700c7790aaa030e794b52ba7791a05c085faf0c",
                  vertices=vertices.tolist(), faces=faces.tolist(),
                  mapping=mapping.tolist(), indices=indices.tolist(), uv=uv.tolist())
    with args.output.open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
