#!/usr/bin/env python3
"""Execute Tencent's unchanged depth/angle/morphology reliability statements."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    assert revision == "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
    args.output.mkdir(parents=True, exist_ok=False)
    import cv2
    import numpy as np
    import torch
    from safetensors.torch import save_file
    source = args.reference / "hy3dpaint/DifferentiableRenderer/MeshRender.py"
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "back_project")
    start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) and n.targets[0].elts[0].id == "depth_max")
    end = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "method")
    wrapper = ast.parse("def reliability(self, depth, visible_mask, normal):\n    pass").body[0]
    wrapper.body = method.body[start:end] + ast.parse("return visible_mask, cos_image, sketch_image").body
    sketch = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "render_sketch_from_depth")
    context = dict(torch=torch, F=torch.nn.functional, np=np, cv2=cv2)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper, sketch], type_ignores=[])), str(source), "exec"), context)
    tensors = {}
    cases = []
    for case, radius in [("full", 2), ("hole", 1), ("step", 0), ("step_dilated", 2), ("flat", 8), ("angle", 0)]:
        size = 16
        depth = torch.linspace(-1.7, -1.2, size*size).reshape(size, size, 1)
        visible = torch.ones(size, size, 1)
        normal = torch.zeros(size, size, 3)
        normal[..., 2] = -1
        if case == "hole":
            visible[8, 8] = 0
            depth[8, 8] = 0
        if case.startswith("step"):
            depth[:, 8:] += .4
            visible[:2] = 0
            depth[:2] = 0
        if case == "flat":
            depth[:] = -1.5
        if case == "angle":
            threshold = np.float32(np.cos(75 / 180 * np.pi))
            cosine = torch.tensor([0., .2, np.nextafter(threshold, np.float32(0)), threshold,
                                   np.nextafter(threshold, np.float32(1)), .5, 1., -1.] * 2)
            normal[..., 0] = torch.sqrt(1 - cosine.square())
            normal[..., 2] = -cosine
            normal[0] = 0
        renderer = types.SimpleNamespace(device="cpu", bake_angle_thres=75, bake_unreliable_kernel_size=radius)
        renderer.render_sketch_from_depth = types.MethodType(context["render_sketch_from_depth"], renderer)
        tensors.update({f"{case}.depth":depth.clone(), f"{case}.visible":visible.to(torch.uint8), f"{case}.normal":normal.clone()})
        reliable, cosine, boundary = context["reliability"](renderer, depth, visible, normal)
        tensors.update({f"{case}.reliable":reliable.to(torch.uint8), f"{case}.cosine":cosine, f"{case}.boundary":boundary.to(torch.uint8)})
        cases.append(dict(name=case, size=size, radius=radius))
    save_file(tensors, str(args.output / "paint-reliability.safetensors"))
    (args.output / "paint-reliability.json").write_text(json.dumps(dict(revision=revision, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), torch=torch.__version__, opencv=cv2.__version__, cases=cases), indent=2)+"\n")


if __name__ == "__main__":
    main()
