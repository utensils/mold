#!/usr/bin/env python3
"""Capture the UNCONDITIONAL PuLID identity embedding from upstream.

Issue https://github.com/utensils/mold/issues/1226.

PuLID's true classifier-free guidance runs a second transformer forward per
step over the negative prompt AND an *unconditional* identity embedding
(`PuLID/flux/sampling.py:136-149`). That embedding is not a zero tensor: it is
the IDFormer evaluated on all-zero conditioning,

    id_uncond          = torch.zeros_like(id_cond)
    id_vit_hidden_unc  = [torch.zeros_like(h) for h in id_vit_hidden]
    uncond_id_embedding = self.pulid_encoder(id_uncond, id_vit_hidden_unc)

(`PuLID/pulid/pipeline_flux.py:188-192`). Because every input is zero it
depends on no photograph at all — it is a pure function of the adapter weights,
which is also why `cubiq/PuLID_ComfyUI`'s per-image uncond is the same tensor
for every image at `noise == 0` (`pulid.py:396-407`) and its mean is that
tensor (`pulid.py:416-419`).

That is exactly the property this fixture pins: one committed 256 KB tensor is
the whole answer, and mold's port must reproduce it from the same checkpoint.

Committed **as documentation of provenance only**. Nothing in mold's build,
test, or runtime path executes it, and mold ships no Python. Re-run by hand:

    python3 -m venv /tmp/pulid-venv
    /tmp/pulid-venv/bin/pip install torch numpy safetensors einops timm
    git clone https://github.com/ToTheBeginning/PuLID /tmp/PuLID
    /tmp/pulid-venv/bin/python capture_true_cfg_goldens.py \
      --pulid-repo /tmp/PuLID \
      --adapter /path/to/pulid_flux_v0.9.1.safetensors \
      --out .
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# `pipeline_flux.py:181` — cat([arcface_512, clip_768]).
ID_COND_DIM = 512 + 768
# The EVA02-CLIP-L-14-336 tap shape: 5 scales of [1, 577, 1024].
VIT_SCALES = 5
VIT_TOKENS = 577
VIT_DIM = 1024


def stats(tensor: torch.Tensor) -> dict:
    flat = tensor.detach().float().reshape(-1)
    return {
        "shape": list(tensor.shape),
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "l2": flat.norm().item(),
    }


def stats_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """The same five numbers as a tensor, plus the peak absolute value.

    Mirrors `capture_eva_goldens.py`'s `stats_tensor` so the Rust side's
    `GoldenStats` loader reads this file the same way it reads that one.
    """
    flat = tensor.detach().float().reshape(-1)
    return torch.tensor(
        [
            flat.mean().item(),
            flat.std(unbiased=False).item(),
            flat.min().item(),
            flat.max().item(),
            flat.abs().max().item(),
        ],
        dtype=torch.float32,
    )


def capture(pulid_repo: Path, adapter: Path) -> tuple[dict, dict]:
    sys.path.insert(0, str(pulid_repo))
    from pulid.encoders_transformer import IDFormer

    model = IDFormer().eval().float()
    state = load_file(adapter)
    prefix = "pulid_encoder."
    encoder_state = {
        k[len(prefix) :]: v.float() for k, v in state.items() if k.startswith(prefix)
    }
    model.load_state_dict(encoder_state, strict=True)

    # `pipeline_flux.py:188-192`, verbatim: zeros of the conditioning's own
    # shape, and one zeroed hidden state per scale.
    id_uncond = torch.zeros(1, ID_COND_DIM, dtype=torch.float32)
    vit_hidden_uncond = [
        torch.zeros(1, VIT_TOKENS, VIT_DIM, dtype=torch.float32)
        for _ in range(VIT_SCALES)
    ]

    with torch.no_grad():
        uncond = model(id_uncond, vit_hidden_uncond)
    assert tuple(uncond.shape) == (1, 32, 2048), uncond.shape

    arrays = {
        "idformer.uncond": uncond[0].contiguous(),
        "idformer.uncond.stats": stats_tensor(uncond),
    }
    meta = {
        "uncond": {
            "source": "PuLID/pulid/pipeline_flux.py:188-192",
            "id_cond_shape": [1, ID_COND_DIM],
            "vit_hidden_shape": [VIT_SCALES, 1, VIT_TOKENS, VIT_DIM],
            "inputs": "all zeros — the unconditional embedding depends on no photograph",
            "output": stats(uncond),
        }
    }
    return arrays, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pulid-repo", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()

    torch.manual_seed(0)
    arrays, meta = capture(args.pulid_repo, args.adapter)
    save_file(
        {k: v.contiguous() for k, v in arrays.items()},
        str(args.out / "true_cfg_goldens.safetensors"),
    )
    (args.out / "true_cfg_goldens.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name, tensor in sorted(arrays.items()):
        print(f"{name:32s} {tuple(tensor.shape)} {tensor.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
