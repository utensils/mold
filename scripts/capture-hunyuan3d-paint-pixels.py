#!/usr/bin/env python3
"""Capture Tencent view normalization and Diffusers material image conversion."""
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
    revision = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
    assert subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip() == revision
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import diffusers
    import numpy as np
    from einops import rearrange
    from diffusers.image_processor import VaeImageProcessor
    from safetensors.torch import save_file
    assert diffusers.__version__ == "0.30.0"
    source = args.reference / "hy3dpaint/hunyuanpaintpbr/pipeline.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HunyuanPaintPipeline")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "encode_images")
    context = dict(torch=torch,rearrange=rearrange)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),"exec"),context)
    class Captured(Exception):
        def __init__(self,pixels):
            self.pixels = pixels
    class EncoderProbe:
        def __init__(self,dtype):
            self.dtype = dtype
        def parameters(self):
            return iter([torch.zeros((),dtype=self.dtype)])
        def encode(self,pixels):
            raise Captured(pixels)
    tensors = {}
    for name,input_dtype,model_dtype in [("f32",torch.float32,torch.float32),("f16",torch.float16,torch.float16),("f16_to_f32",torch.float16,torch.float32)]:
        pixels = ((torch.arange(2*3*64*64).float()%256)/255).reshape(1,2,3,64,64).to(input_dtype)
        tensors[name+".input"] = pixels
        try:
            context["encode_images"](types.SimpleNamespace(vae=EncoderProbe(model_dtype)),pixels)
        except Captured as capture:
            tensors[name+".normalized"] = capture.pixels.contiguous()
        else:
            raise RuntimeError("upstream encode_images no longer calls VAE encode")
    processor = VaeImageProcessor()
    for name,dtype in [("f32",torch.float32),("f16",torch.float16)]:
        pixels = torch.linspace(-1.1,1.1,4*3*8*16).reshape(4,3,8,16).to(dtype)
        pixels.flatten()[:3] = torch.tensor([-1.,0.,1.],dtype=dtype)
        tensors[name+".decoded"] = pixels
        images = processor.postprocess(pixels,output_type="pil")
        tensors[name+".rgb"] = torch.from_numpy(np.stack([np.array(image) for image in images]))
    save_file(tensors,str(args.output/"paint-pixels.safetensors"))
    (args.output/"paint-pixels.json").write_text(json.dumps(dict(revision=revision,torch=torch.__version__,diffusers=diffusers.__version__,pipeline_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2)+"\n")


if __name__ == "__main__":
    main()
