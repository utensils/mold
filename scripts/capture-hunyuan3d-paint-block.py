#!/usr/bin/env python3
"""Capture Tencent's unchanged complete paint transformer block and reference cache."""
import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import typing
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
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        parser.error("reference revision differs from the qualified Tencent source")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from diffusers.models.attention import BasicTransformerBlock
    from diffusers.models.attention_processor import Attention
    from einops import rearrange
    from safetensors.torch import save_file
    source_dir = args.reference / "hy3dpaint/hunyuanpaintpbr/unet"
    attention_source = source_dir / "attn_processor.py"
    spec = importlib.util.spec_from_file_location("tencent_paint_attention", attention_source)
    attention = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attention)
    source = source_dir / "modules.py"
    definition = next(node for node in ast.parse(source.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == "Basic2p5DTransformerBlock")
    namespace = dict(torch=torch, rearrange=rearrange, BasicTransformerBlock=BasicTransformerBlock, Attention=Attention,
                     SelfAttnProcessor2_0=attention.SelfAttnProcessor2_0, RefAttnProcessor2_0=attention.RefAttnProcessor2_0,
                     PoseRoPEAttnProcessor2_0=attention.PoseRoPEAttnProcessor2_0,
                     Optional=typing.Optional, Dict=typing.Dict, Any=typing.Any)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    cls = namespace["Basic2p5DTransformerBlock"]
    torch.manual_seed(25026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu", mmap=True) if args.checkpoint else None
    width, context, head_dim = (320, 1024, 64) if state is not None else (80, 12, 16)
    tensors = {}

    def signal(shape, offset=0):
        count = 1
        for dimension in shape:
            count *= dimension
        return (torch.arange(count, dtype=torch.float32) * .17 + offset).sin().reshape(shape)

    with torch.inference_mode():
        for branch in ("main", "dual"):
            main_branch = branch == "main"
            pbr = 2 if main_branch else 1
            base = BasicTransformerBlock(dim=width, num_attention_heads=width // head_dim,
                                         attention_head_dim=head_dim, cross_attention_dim=context,
                                         activation_fn="geglu", norm_eps=1e-5)
            model = cls(base, "fixture", use_ma=main_branch, use_ra=main_branch, use_mda=main_branch,
                        use_dino=main_branch, pbr_setting=["albedo", "mr"] if main_branch else None).eval()
            if state is not None:
                prefix = ("unet" if main_branch else "unet_dual") + ".down_blocks.0.attentions.0.transformer_blocks.0."
                model.load_state_dict({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}, strict=True)
            else:
                # Every residual branch is nonzero, unlike the wrapper's zero initialization.
                for key, value in model.named_parameters():
                    value.copy_(torch.randn_like(value) * .05 + (1 if "norm" in key and key.endswith("weight") else 0))
            for key, value in model.state_dict().items():
                tensors[f"{branch}.weights.{key}"] = value.cpu().clone()
            for name, dtype in (("f32", torch.float32), ("f16", torch.float16)):
                model.to("cuda", dtype=dtype)
                max_views = 6 if state is not None else 3
                cases = [(max_views, 2, ""), (1, 2, "")]
                if main_branch:
                    cases.append((max_views, 3, ".cfg3"))
                for views, batch, suffix in cases:
                    spatial = 64 if state is not None else 4
                    label = f"{branch}.{name}.views{views}{suffix}"
                    hidden = signal((batch*pbr*views, spatial, width)).to("cuda", dtype=dtype)
                    encoder = signal((batch*pbr*views, 77 if state is not None else 7, context), .3).to("cuda", dtype=dtype)
                    dino = signal((batch, 1028 if state is not None else 5, context), .6).to("cuda", dtype=dtype)
                    reference = signal((batch, 256 if state is not None else 9, width), .9).to("cuda", dtype=dtype)
                    indices = (torch.arange(batch*views*spatial*3).reshape(batch, views*spatial, 3)*13 % 512).long().cuda()
                    cache = {"fixture": reference} if main_branch else {}
                    ref_scale = torch.tensor([0., 1., 1.] if suffix else [.3, 1.1], device="cuda", dtype=dtype)
                    kwargs = dict(num_in_batch=views, mode="r" if main_branch else "w", mva_scale=.7,
                                  ref_scale=ref_scale, condition_embed_dict=cache, dino_hidden_states=dino,
                                  position_voxel_indices={views*spatial: {"voxel_indices": indices, "voxel_resolution": 512}})
                    output = model(hidden, encoder_hidden_states=encoder, cross_attention_kwargs=kwargs)
                    for key, value in dict(input=hidden, encoder=encoder, dino=dino, reference=reference,
                                           positions=indices, ref_scale=ref_scale, expected=output).items():
                        tensors[label + "." + key] = value.cpu()
                    if not main_branch:
                        tensors[label + ".cache_expected"] = cache["fixture"].cpu()
                    else:
                        zero_scale = torch.tensor([0.] + [1.1]*(batch-1), device="cuda", dtype=dtype)
                        for scale_name, scale in (("scalar", .6), ("zero_cfg", zero_scale)):
                            scaled_kwargs = dict(kwargs, ref_scale=scale)
                            tensors[label + "." + scale_name + "_expected"] = model(hidden, encoder_hidden_states=encoder,
                                cross_attention_kwargs=scaled_kwargs).cpu()
                        tensors[label + ".zero_cfg_scale"] = zero_scale.cpu()
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(args.output / "paint-block.safetensors"))
    metadata = dict(revision=revision, source_sha256=digest(source), attention_source_sha256=digest(attention_source),
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(), seed=25026, width=width,
                    context=context, head_dim=head_dim, main_branch_scales=dict(multiview=.7, reference=[.3, 1.1]))
    if args.checkpoint:
        metadata.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint))
    (args.output / "paint-block.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
