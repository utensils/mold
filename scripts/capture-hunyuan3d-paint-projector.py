#!/usr/bin/env python3
"""Capture Tencent's unchanged paint image projector for Rust parity tests."""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

REVISION = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dino-fixture", type=Path)
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.dino_fixture):
        parser.error("checkpoint and DINO fixture must be supplied together")
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        parser.error("reference revision differs from the qualified Tencent source")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from einops import rearrange
    from safetensors.torch import load_file, save_file
    source = args.reference / "hy3dpaint/hunyuanpaintpbr/unet/modules.py"
    tree = ast.parse(source.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ImageProjModel")
    namespace = {"torch": torch, "rearrange": rearrange}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    cls = namespace["ImageProjModel"]
    torch.manual_seed(25026)
    torch.backends.cuda.matmul.allow_tf32 = False
    tiny = cls(cross_attention_dim=6, clip_embeddings_dim=8, clip_extra_context_tokens=4).eval()
    tensors = {"weights." + name: value for name, value in tiny.state_dict().items()}
    with torch.inference_mode():
        for label, shape in [("pooled", (2,8)), ("tokens", (2,3,8))]:
            count = 1
            for dimension in shape: count *= dimension
            value = (torch.arange(count).float()*.17).sin().reshape(shape)
            tensors[label] = value
            tensors[label + "_expected"] = tiny(value)
    save_file(tensors, str(args.output / "paint-projector-tiny.safetensors"))
    metadata = dict(revision=revision,source_sha256=digest(source),torch=torch.__version__,seed=25026)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        prefix = "unet.image_proj_model_dino."
        weights = {key[len(prefix):]: value.contiguous() for key,value in state.items() if key.startswith(prefix)}
        model = cls(cross_attention_dim=1024, clip_embeddings_dim=1536, clip_extra_context_tokens=4).eval()
        model.load_state_dict(weights, strict=True)
        save_file(weights, str(args.output / "paint-projector-weights.safetensors"))
        dino = load_file(str(args.dino_fixture))
        fixture = {}
        with torch.inference_mode():
            for name, dtype in [("f32",torch.float32),("f16",torch.float16)]:
                model = model.to("cuda",dtype=dtype)
                value = dino["expected_"+name].to("cuda",dtype=dtype)
                fixture["input_"+name] = value.cpu()
                fixture["expected_"+name] = model(value).cpu()
        save_file(fixture, str(args.output / "paint-projector-pretrained.safetensors"))
        metadata.update(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=digest(args.checkpoint),
            dino_fixture_sha256=digest(args.dino_fixture),gpu=torch.cuda.get_device_name())
    (args.output / "paint-projector.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__ == "__main__":
    main()
