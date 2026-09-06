#!/usr/bin/env python3
"""Execute Tencent's unchanged back_sample branch on boundary/visibility cases."""
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
    import torch
    import numpy as np
    from safetensors.torch import save_file
    source = args.reference / "hy3dpaint/DifferentiableRenderer/MeshRender.py"
    method = next(node for node in ast.walk(ast.parse(source.read_text())) if isinstance(node, ast.FunctionDef) and node.name == "back_project")
    branch = next(node for node in ast.walk(method) if isinstance(node, ast.If) and ast.unparse(node.test) == "method == 'back_sample'")
    xyz = torch.tensor([
        [-1.,-1.,-1.], [1.,1.,-1.], [-1.0001,0.,-1.], [1.0001,0.,-1.],
        [0.,-1.0001,-1.], [0.,1.0001,-1.], [-.6,-.6,-1.], [0.,0.,-1.],
        [-.9,-.9,-1.], [.1,.1,-1.], [.2,.2,-1.], [.7,.8,-1.],
        [-.8,-1.,-1.], [.8,-.8,-.997001], [.8,-.8,-.996999], [.3,.9,-1.],
    ])
    image = torch.arange(75).float().reshape(5,5,3)/74
    depth = torch.full((5,5,1), -1.)
    depth[2,2] = -1.1
    visible = torch.ones(5,5,1)
    visible[1,1] = 0
    cosine = torch.arange(25).float().reshape(5,5,1)/24
    cosine[0,1] = 0
    edges = torch.zeros(5,5,1)
    edges[4,4] = 1
    grid = torch.tensor([[i//4,i%4] for i in range(16)])
    renderer = types.SimpleNamespace(tex_position=torch.cat([xyz,torch.ones(16,1)],1), tex_grid=grid, texture_size=(4,4), device="cpu")
    context = dict(torch=torch,np=np,self=renderer,proj=np.eye(4,dtype=np.float32),r_mv=np.eye(4,dtype=np.float32),
                   resolution=(5,5),channel=3,image=image,depth=depth,visible_mask=visible,cos_image=cosine,sketch_image=edges)
    exec(compile(ast.Module(body=branch.body,type_ignores=[]),str(source),"exec"),context)
    tensors = dict(projected=xyz,image=image,depth=depth,visible=visible,cosine=cosine,edges=edges,
                   texture=context["texture"], output_cosine=context["cos_map"], output_edges=context["boundary_map"])
    save_file(tensors, str(args.output/"back-sample.safetensors"))
    (args.output/"back-sample.json").write_text(json.dumps(dict(revision=revision,torch=torch.__version__,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2)+"\n")


if __name__ == "__main__":
    main()
