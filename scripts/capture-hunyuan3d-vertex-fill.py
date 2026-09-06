#!/usr/bin/env python3
"""Capture Tencent's default mesh vertex smoothing before Navier–Stokes fill."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == '82920d643c0dc2f7bfd7255f45f62d386edfe60c'
    spec = importlib.util.spec_from_file_location('mesh_inpaint_processor', args.extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(15111496)
    positions = rng.uniform(-1, 1, (9, 3)).astype(np.float32)
    positions[2] = positions[1]  # coincident vertices exercise minimum distance
    uv = np.array([[0,0],[.25,0],[.5,0],[.75,0],[1,0],[0,1],[.25,1],[.5,1],[1,1],[.5,.5]], dtype=np.float32)
    faces = np.array([[0,1,2],[2,3,0],[3,4,5],[5,0,3],[6,7,8]], dtype=np.int32)
    uv_faces = faces.copy()
    uv_faces[1,2] = 9  # split UV for the same position
    cases=[]
    for name, known in [('empty', []), ('full', list(range(10))), ('seam', [0,3]), ('disconnected', [1,7])]:
        texture = rng.random((7,9,3), dtype=np.float32)
        mask = np.zeros((7,9), dtype=np.uint8)
        for index in known:
            u,v=uv[index]
            mask[int(np.floor((1-float(v))*6+.5)),int(np.floor(float(u)*8+.5))]=255
        result, result_mask = module.meshVerticeInpaint(texture, mask, positions, uv, faces, uv_faces)
        cases.append(dict(name=name,width=9,height=7,texture=texture.reshape(-1,3).tolist(),mask=mask.ravel().tolist(),positions=positions.tolist(),uv=uv.tolist(),faces=faces.tolist(),uv_faces=uv_faces.tolist(),expected=result.reshape(-1,3).tolist(),expected_mask=result_mask.ravel().tolist()))
    source=args.upstream/'hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor.cpp'
    (args.output/'vertex-fill.json').write_text(json.dumps(dict(revision=revision,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),extension_sha256=hashlib.sha256(args.extension.read_bytes()).hexdigest(),cases=cases),indent=2)+'\n')


if __name__ == '__main__':
    main()
