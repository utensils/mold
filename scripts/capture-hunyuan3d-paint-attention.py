#!/usr/bin/env python3
"""Capture unchanged Tencent paint attention processors, including PBR head ordering."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

REVISION = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--block-prefix", default="unet.down_blocks.0.attentions.0.transformer_blocks.0")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        parser.error("reference revision differs from the qualified Tencent source")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
    from safetensors.torch import save_file
    source = args.reference / "hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py"
    spec = importlib.util.spec_from_file_location("tencent_paint_attention", source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    torch.manual_seed(25026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tensors = {}
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True) if args.checkpoint else None
    width = state[args.block_prefix + ".attn_multiview.to_q.weight"].shape[0] if state is not None else 80
    head_dim = 64 if state is not None else 16
    heads = width // head_dim
    views, spatial_tokens = (6, 64) if state is not None else (3, 4)
    tokens = views * spatial_tokens
    cross_width = 1024 if state is not None else 12

    def signal(shape, offset=0):
        count = 1
        for dimension in shape:
            count *= dimension
        return (torch.arange(count, dtype=torch.float32) * .17 + offset).sin().reshape(shape)

    with torch.inference_mode():
        for label in ("self", "ref", "pose", "plain", "cross"):
            context = cross_width if label == "cross" else None
            kwargs = dict(query_dim=width, heads=heads, dim_head=head_dim, bias=False, out_bias=True, cross_attention_dim=context)
            processor = {
                "self": lambda: upstream.SelfAttnProcessor2_0(**kwargs),
                "ref": lambda: upstream.RefAttnProcessor2_0(**kwargs),
                "pose": upstream.PoseRoPEAttnProcessor2_0,
                "plain": AttnProcessor2_0,
                "cross": AttnProcessor2_0,
            }[label]()
            model = Attention(**kwargs, processor=processor).eval().cuda()
            if state is not None:
                leaf = {"self": "transformer.attn1", "ref": "attn_refview", "pose": "attn_multiview",
                        "plain": "attn_multiview", "cross": "attn_dino"}[label]
                prefix = args.block_prefix + "." + leaf + "."
                model.load_state_dict({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}, strict=True)
            # Retain original float32 parameters before the model is cast in place.
            for key, value in model.state_dict().items():
                tensors[f"{label}.weights.{key}"] = value.cpu().clone()
            hidden = signal((2, 2, views, spatial_tokens, width) if label == "self" else (2, tokens, width))
            context_tokens = (1028 if label == "cross" else 256) if state is not None else 7
            encoder = signal((2, context_tokens, context or width), .6) if label in ("ref", "cross") else None
            positions = (torch.arange(2 * tokens * 3).reshape(2, tokens, 3) * 13 % 512).long()
            if label == "pose":
                tensors["positions"] = positions
                cos, sin = upstream.RotaryEmbedding.get_3d_rotary_pos_embed(positions, head_dim, 512)
                tensors["rope_cos"], tensors["rope_sin"] = cos, sin
            for dtype_name, dtype in (("f32", torch.float32), ("f16", torch.float16)):
                model.to(dtype=dtype)
                value = hidden.to("cuda", dtype=dtype)
                options = {}
                if encoder is not None:
                    options["encoder_hidden_states"] = encoder.to("cuda", dtype=dtype)
                if label == "pose":
                    options["position_indices"] = {"voxel_indices": positions.cuda(), "voxel_resolution": 512}
                tensors[f"{label}.{dtype_name}.input"] = value.cpu()
                if encoder is not None:
                    tensors[f"{label}.{dtype_name}.encoder"] = options["encoder_hidden_states"].cpu()
                tensors[f"{label}.{dtype_name}.expected"] = model(value, **options).cpu()
        # Explicit query rotation checks the float32 arithmetic and half output boundary.
        for name, dtype in (("f32", torch.float32), ("f16", torch.float16)):
            query = signal((2, heads, tokens, head_dim)).to(dtype)
            tensors[f"rope_{name}_input"] = query
            tensors[f"rope_{name}_expected"] = upstream.RotaryEmbedding.apply_rotary_emb(query, (cos, sin))
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(args.output / "paint-attention.safetensors"))
    metadata = dict(revision=revision, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(), seed=25026,
                    heads=heads, head_dim=head_dim, batch=2, materials=2, views=views,
                    reference_value_order="concatenate materials before head reshape; split after attention")
    if args.checkpoint:
        digest = hashlib.sha256()
        with args.checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        metadata.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest.hexdigest(), block_prefix=args.block_prefix)
    (args.output / "paint-attention.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
