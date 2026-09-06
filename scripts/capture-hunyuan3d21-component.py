#!/usr/bin/env python3
"""Export a tiny executable Tencent 2.1 DiT fixture; never used by mold inference.

The two namespace packages bypass upstream's eager pymeshlab postprocessor
import, whose wheel is not usable on this Nix host. All model computations and
parameters are from the unmodified upstream module. The adapter only converts
the external [B,C,L]/sigma convention to Tencent's [B,L,C]/time convention.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import types

import torch
from safetensors.torch import save_file
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Retained full repack; exports only inputs and expected output")
    args = parser.parse_args()
    pin = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
    actual = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != pin:
        raise ValueError(f"expected Tencent revision {pin}, found {actual}")
    subprocess.run(["git", "-C", str(args.upstream), "diff", "--exit-code"], check=True)
    root = args.upstream.resolve() / "hy3dshape/hy3dshape"
    for name, path in [("hy3dshape", root), ("hy3dshape.models", root / "models")]:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain

    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(15111496)
    cfg = dict(in_channels=4, hidden_size=32, context_dim=16, depth=3,
               num_heads=2, num_moe_layers=1, num_experts=3, moe_top_k=2,
               qk_norm=True, qkv_bias=False, use_attention_pooling=False,
               use_pos_emb=False, with_decoupled_ca=False)
    if args.checkpoint:
        cfg.update(in_channels=64, hidden_size=2048, context_dim=1024, depth=21,
                   num_heads=16, num_moe_layers=6, num_experts=8)
        with torch.device("meta"):
            model = HunYuanDiTPlain(**cfg).eval()
        with safe_open(args.checkpoint, framework="pt", device="cpu") as weights:
            state = {key.removeprefix("model."): weights.get_tensor(key)
                     for key in weights.keys() if key.startswith("model.")}
        model.load_state_dict(state, strict=True, assign=True)
        model = model.to(device="cuda", dtype=torch.float32)
        del state
    else:
        model = HunYuanDiTPlain(**cfg).eval().cuda()
    x = torch.randn(1, cfg["in_channels"], 4096 if args.checkpoint else 5, device="cuda")
    sigma = torch.tensor([0.3], device="cuda")
    context = torch.randn(1, 1370 if args.checkpoint else 7, cfg["context_dim"], device="cuda")
    with torch.no_grad():
        result = -model(x.transpose(1, 2), 1 - sigma, {"main": context}).transpose(1, 2)
    tensors = {} if args.checkpoint else {"model." + k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    for name, tensor in [("input", x), ("sigma", sigma), ("context", context), ("expected", result)]:
        tensors[name] = tensor.detach().cpu().contiguous()
    save_file(tensors, args.output / "transformer21.safetensors")
    metadata = {"upstream": pin, "config": cfg, "seed": 15111496,
                "torch": torch.__version__, "device": torch.cuda.get_device_name(),
                "dtype": "float32", "atol": 0.00005,
                "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                "note": "Pretrained full transformer; synthetic inputs." if args.checkpoint else "Random synthetic weights; not a pretrained checkpoint. Tencent CUDA attention unmodified."}
    (args.output / "transformer21.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
