#!/usr/bin/env python3
"""Capture Tencent's camera geometry and reliability masks on an actual mesh."""
import argparse
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=[64, 1024, 2048], default=2048)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    assert revision == "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.reference.resolve() / "hy3dpaint"))
    sys.modules["bpy"] = types.ModuleType("bpy")
    import DifferentiableRenderer
    DifferentiableRenderer.__path__.append(str(args.native.resolve()))
    module = importlib.import_module("DifferentiableRenderer.MeshRender")
    import numpy as np
    import torch
    from safetensors.torch import save_file
    source = args.reference / "hy3dpaint/DifferentiableRenderer/MeshRender.py"
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "back_project")
    start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) and n.targets[0].elts[0].id == "depth_max")
    end = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "method")
    geometry = copy.deepcopy(method)
    geometry.name = "geometry"
    geometry.body = geometry.body[:start] + ast.parse("return depth, normal, visible_mask, rast_out, pos_camera, pos_clip").body
    reliability = ast.parse("def reliability(self, depth, visible_mask, normal):\n    pass").body[0]
    reliability.body = method.body[start:end] + ast.parse("return visible_mask, cos_image, sketch_image").body
    context = dict(vars(module))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[geometry, reliability], type_ignores=[])), str(source), "exec"), context)
    with np.load(args.mesh) as mesh:
        vertices = mesh["vertices"].astype(np.float32)
        faces = mesh["faces"].astype(np.int32)
        uv = mesh["uv"].astype(np.float32)
    save_file({"vertices":torch.from_numpy(vertices), "faces":torch.from_numpy(faces), "uv":torch.from_numpy(uv)}, str(args.output / "mesh.safetensors"))
    renderer = module.MeshRender(default_resolution=args.size, texture_size=32, device="cuda")
    renderer.set_mesh(vertices.copy(), faces.copy(), uv.copy(), faces.copy())
    views = [(0,0), (0,90), (0,180), (0,270), (90,0), (-90,180)]
    record = dict(revision=revision, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  torch=torch.__version__, size=args.size, radius=renderer.bake_unreliable_kernel_size,
                  views=views, mesh=str(args.mesh), mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
                  device=torch.cuda.get_device_name(), results=[])
    (args.output / "invocation.json").write_text(json.dumps(record, indent=2)+"\n")
    image = torch.zeros(args.size, args.size, 3, device="cuda")
    for index, (elev, azim) in enumerate(views):
        start_time = time.monotonic()
        depth, normal, visible, raster, camera, clip = context["geometry"](renderer, image, elev, azim)
        fields = dict(depth=depth, normal=normal, visible=visible.to(torch.uint8), raster=raster, camera=camera, clip=clip)
        reliable, cosine, boundary = context["reliability"](renderer, depth, visible, normal)
        fields.update(reliable=reliable.to(torch.uint8), cosine=cosine, boundary=boundary.to(torch.uint8))
        save_file({name:value.detach().cpu().contiguous() for name,value in fields.items()}, str(args.output / f"view.{index}.safetensors"))
        record["results"].append(dict(index=index, seconds=time.monotonic()-start_time, visible=int(visible.sum()), reliable=int(reliable.sum())))
        print(record["results"][-1], flush=True)
    record["peak_cuda_allocated"] = torch.cuda.max_memory_allocated()
    (args.output / "completed.json").write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main()
