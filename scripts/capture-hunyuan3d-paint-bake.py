#!/usr/bin/env python3
"""Capture Tencent's unchanged weighted merge, overlap skip and trust policy."""
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
    from safetensors.torch import save_file
    context = dict(torch=torch)
    sources = {}
    for path, name in [("hy3dpaint/DifferentiableRenderer/MeshRender.py", "fast_bake_texture"),
                       ("hy3dpaint/utils/pipeline_utils.py", "bake_from_multiview")]:
        source = args.reference / path
        method = next(node for node in ast.walk(ast.parse(source.read_text())) if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), context)
        sources[path] = hashlib.sha256(source.read_bytes()).hexdigest()
    tensors = {}
    for case in ["overlap", "tiny"]:
        colors = torch.arange(4*100*3).reshape(4,10,10,3).float().remainder(251)/250
        cosine = torch.ones(4,10,10,1)
        weights = torch.tensor([1., .1, .5, .05])
        if case == "overlap":
            cosine[0].flatten()[-1] = 0
            cosine[3] = 0
        else:
            cosine[0].flatten()[::4] = 0
            cosine[0].flatten()[1::4] = .001
            cosine[0].flatten()[2::4] = .01
            cosine[0].flatten()[3::4] = .010001
            cosine[1].flatten()[::4] = 0
            cosine[2] = 0
        renderer = types.SimpleNamespace(texture_size=(10,10), device="cpu")
        renderer.fast_bake_texture = types.MethodType(context["fast_bake_texture"], renderer)
        renderer.back_project = lambda index, elev, azim: (colors[index], cosine[index], torch.zeros(10,10,1))
        processor = types.SimpleNamespace(render=renderer, config=types.SimpleNamespace(bake_exp=4))
        tensors[case+".colors"] = colors
        tensors[case+".cosine"] = cosine
        tensors[case+".weights"] = weights
        for count in range(1,5):
            texture, trust = context["bake_from_multiview"](processor, list(range(count)), [0]*count, [0]*count, weights[:count].tolist())
            tensors[f"{case}.texture.{count}"] = texture
            tensors[f"{case}.trust.{count}"] = trust.to(torch.uint8)
    save_file(tensors, str(args.output / "paint-bake.safetensors"))
    (args.output / "paint-bake.json").write_text(json.dumps(dict(revision=revision, torch=torch.__version__, sources=sources), indent=2)+"\n")


if __name__ == "__main__":
    main()
