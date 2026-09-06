#!/usr/bin/env python3
"""Capture Tencent's actual UV raster geometry and paint-camera projections."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--size", type=int, choices=[32,1024,2048,4096], default=32)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    assert revision == "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.reference.resolve() / "hy3dpaint"
    sys.path.insert(0, str(root))
    sys.modules["bpy"] = types.ModuleType("bpy")
    import DifferentiableRenderer
    DifferentiableRenderer.__path__.append(str(args.native.resolve()))
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from DifferentiableRenderer.MeshRender import MeshRender
    from DifferentiableRenderer.camera_utils import get_mv_matrix
    vertices = np.array([[2,3,4],[4,3.2,4],[2.1,6,4.2],[2.3,3.1,5.5]], dtype=np.float32)
    faces = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]], dtype=np.int32)
    vertices = np.concatenate([vertices[faces].reshape(-1,3),np.array([[2,3,4],[3,3,4],[4,3,4]],dtype=np.float32)])
    faces = np.arange(15,dtype=np.int32).reshape(-1,3)
    uv = []
    for index in range(5):
        x = (index % 2) * .5 + .03
        y = (index // 2) / 3 + .025
        uv.extend([[x,y],[x+.42,y+.015],[x+.05,y+.26]])
    uv = np.array(uv,dtype=np.float32)
    if args.mesh:
        with np.load(args.mesh) as mesh:
            vertices = mesh["vertices"].astype(np.float32)
            faces = mesh["faces"].astype(np.int32)
            uv = mesh["uv"].astype(np.float32)
    tensors = dict(vertices=torch.from_numpy(vertices.copy()),faces=torch.from_numpy(faces.copy()),uv=torch.from_numpy(uv.copy()))
    renderer = MeshRender(default_resolution=32,texture_size=args.size,device="cuda")
    renderer.set_mesh(vertices.copy(),faces.copy(),uv.copy(),faces.copy())
    tensors.update(positions=renderer.tex_position[:,:3],normals=renderer.tex_normal,texels=renderer.tex_grid[:,0]*args.size+renderer.tex_grid[:,1])
    views = [(0,0),(0,90),(0,180),(0,270),(90,0),(-90,180)]
    for index,(elev,azim) in enumerate(views):
        matrix = torch.from_numpy(get_mv_matrix(elev,azim,renderer.camera_distance)).to(renderer.tex_position)
        projection = torch.diag(torch.tensor([renderer.camera_proj_mat[0,0],renderer.camera_proj_mat[1,1],1,1],device="cuda",dtype=torch.float32))
        tensors[f"projected.{index}"] = (renderer.tex_position @ matrix.T @ projection)[:,:3]
    save_file({name:value.detach().cpu().contiguous() for name,value in tensors.items()},str(args.output/"paint-uv.safetensors"))
    (args.output/"paint-uv.json").write_text(json.dumps(dict(revision=revision,torch=torch.__version__,size=args.size,views=views,mesh=str(args.mesh) if args.mesh else None),indent=2)+"\n")


if __name__ == "__main__":
    main()
