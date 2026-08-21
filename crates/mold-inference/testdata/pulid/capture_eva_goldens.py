#!/usr/bin/env python3
"""Capture the PuLID parity goldens committed next to this script.

PROVENANCE ONLY. Nothing in mold's build or test path ever runs this file; it
is committed so a maintainer can reproduce or refresh the goldens and audit how
they were produced. Mold's shipped code is pure Rust/candle.

Upstream reference: https://github.com/ToTheBeginning/PuLID
  commit 1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa
Modules read: eva_clip/eva_vit_model.py, eva_clip/rope.py, eva_clip/model.py,
  eva_clip/constants.py, pulid/encoders_transformer.py, pulid/pipeline_flux.py

Checkpoints (SHA-256 pinned in crates/mold-core/src/manifest.rs):
  EVA02_CLIP_L_336_psz14_s6B.pt
    84c3a17a228c567a155259b2245b0b59072bf7da510260a0a02ec54de6d50b05
  pulid_flux_v0.9.1.safetensors
    92c41c3af322b02e58e1b32842e4601e08c8f16ec1fe80089dbe957df510f51d

Everything is captured on CPU in float32 from fixed, procedurally generated
inputs, so no image or tensor of unknown provenance is committed.

Usage (scratch venv; torch CPU + torchvision + safetensors + pillow + timm):
    python capture_goldens.py \
        --pulid-repo /path/to/PuLID \
        --eva /path/to/EVA02_CLIP_L_336_psz14_s6B.pt \
        --adapter /path/to/pulid_flux_v0.9.1.safetensors \
        --out .
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file, save_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

# --------------------------------------------------------------------------
# The deterministic value stream. Mold reimplements this bit-for-bit in
# `crates/mold-inference/src/encoders/eva_clip_vision.rs::DeterministicStream`,
# so a fixture input never has to be committed as bytes.
#
#   xorshift64*: x ^= x >> 12; x ^= x << 25; x ^= x >> 27; out = x * M
# with M = 0x2545F4914F6CDD1D, all in wrapping u64 arithmetic. A sample is
#   ((out >> 11) as f64 / 2^53) * 2 - 1
# rendered as f32.
# --------------------------------------------------------------------------

MULT = 0x2545F4914F6CDD1D
MASK = (1 << 64) - 1


class DeterministicStream:
    def __init__(self, seed: int) -> None:
        assert seed != 0
        self.state = seed & MASK

    def next_u64(self) -> int:
        x = self.state
        x ^= x >> 12
        x &= MASK
        x ^= (x << 25) & MASK
        x ^= x >> 27
        self.state = x & MASK
        return (x * MULT) & MASK

    def next_unit(self) -> float:
        return np.float32(((self.next_u64() >> 11) / float(1 << 53)) * 2.0 - 1.0)

    def tensor(self, *shape: int) -> torch.Tensor:
        n = math.prod(shape)
        flat = np.fromiter((self.next_unit() for _ in range(n)), np.float32, n)
        return torch.from_numpy(flat.reshape(shape))

    def indices(self, count: int, modulo: int) -> np.ndarray:
        return np.fromiter(
            (self.next_u64() % modulo for _ in range(count)), np.int64, count
        )


# Seeds. Each fixture draws from its own stream so adding one never shifts
# another's values.
SEED_TOWER_INPUT = 0x50554C49_44544F57  # "PULIDTOW"
SEED_TOWER_PROBE = 0x50554C49_44505242  # "PULIDPRB"
SEED_IDFORMER_ID = 0x50554C49_44494446  # "PULIDIDF"
SEED_IDFORMER_VIT = 0x50554C49_44564954  # "PULIDVIT"
SEED_IMAGE = 0x50554C49_44494D47  # "PULIDIMG"

PROBE_COUNT = 512

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def stats(t: torch.Tensor) -> dict:
    f = t.double().flatten()
    return {
        "shape": list(t.shape),
        "mean": float(f.mean()),
        "std": float(f.std(unbiased=False)),
        "min": float(f.min()),
        "max": float(f.max()),
        "l2": float(f.norm(2)),
    }


# Order of the `*.stats` golden arrays. Mold reads the same five slots, and the
# peak magnitude (slot 4) is the scale every hidden-state tolerance is quoted
# against.
STAT_SLOTS = ("mean", "std", "min", "max", "peak")


def stats_tensor(t: torch.Tensor) -> torch.Tensor:
    f = t.double().flatten()
    return torch.tensor(
        [
            float(f.mean()),
            float(f.std(unbiased=False)),
            float(f.min()),
            float(f.max()),
            float(f.abs().max()),
        ],
        dtype=torch.float32,
    )


# --------------------------------------------------------------------------
# The fixed input image: a deterministic procedural RGB pattern, so the
# committed PNG carries no third-party provenance. 512x512 matches PuLID's
# aligned-face size (`pipeline_flux.py:50`, `face_size=512`), so the resize
# under test is the same 512 -> 336 downscale the real pipeline performs.
# --------------------------------------------------------------------------


def build_input_image() -> Image.Image:
    size = 512
    stream = DeterministicStream(SEED_IMAGE)
    yy, xx = np.meshgrid(
        np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64),
        indexing="ij",
    )
    u = xx / (size - 1)
    v = yy / (size - 1)
    r = 0.5 + 0.5 * np.sin(9.0 * u + 3.0 * v)
    g = 0.5 + 0.5 * np.sin(5.0 * v - 2.0 * u * u * 7.0)
    b = 0.5 + 0.5 * np.cos(11.0 * (u - 0.5) * (v - 0.5) * 4.0)
    # A little deterministic high-frequency energy so an antialiasing
    # difference in the resize is actually visible in the fixture.
    noise = stream.tensor(size, size).numpy().astype(np.float64) * 0.08
    rgb = np.stack([r + noise, g - noise, b + noise], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), mode="RGB")


# --------------------------------------------------------------------------
# EVA02-CLIP-L-14-336 vision tower, constructed exactly as
# `eva_clip/model.py:110-131` `_build_vision_tower` does for this config
# (`eva_clip/model_configs/EVA02-CLIP-L-14-336.json`). Without apex,
# `FusedLayerNorm` is eva_clip's own `LayerNorm` (`model.py:25-27`), i.e. plain
# `nn.LayerNorm(eps=1e-6)` in float32.
# --------------------------------------------------------------------------


def build_tower(pulid_repo: Path, eva_ckpt: Path):
    sys.path.insert(0, str(pulid_repo))
    from eva_clip.eva_vit_model import EVAVisionTransformer

    visual = EVAVisionTransformer(
        img_size=336,
        patch_size=14,
        num_classes=768,  # embed_dim
        use_mean_pooling=False,  # global_average_pool
        init_values=None,  # ls_init_value
        patch_dropout=0.0,
        embed_dim=1024,
        depth=24,
        num_heads=1024 // 64,
        mlp_ratio=2.6667,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        xattn=False,
        rope=True,
        postnorm=False,
        pt_hw_seq_len=16,
        intp_freq=True,
        naiveswiglu=True,
        subln=True,
    )
    state = torch.load(eva_ckpt, map_location="cpu", weights_only=True)
    visual_state = {
        k[len("visual.") :]: v.float()
        for k, v in state.items()
        if k.startswith("visual.")
    }
    missing, unexpected = visual.load_state_dict(visual_state, strict=False)
    assert not [m for m in missing], f"missing: {missing}"
    assert not unexpected, f"unexpected: {unexpected}"
    return visual.eval().float()


def capture_tower(visual, out: Path, arrays: dict, meta: dict) -> None:
    pixels = DeterministicStream(SEED_TOWER_INPUT).tensor(1, 3, 336, 336)
    with torch.no_grad():
        cls_proj, hidden = visual(pixels, return_all_features=False, return_hidden=True)
    assert len(hidden) == 5, len(hidden)

    # The pipeline L2-normalizes the projection along dim 1
    # (`pulid/pipeline_flux.py:178-179`).
    cls_norm = cls_proj / torch.norm(cls_proj, 2, 1, True)

    arrays["tower.cls_projection"] = cls_proj[0].contiguous()
    arrays["tower.cls_projection_normalized"] = cls_norm[0].contiguous()

    hidden_meta = []
    for i, h in enumerate(hidden):
        assert tuple(h.shape) == (1, 577, 1024), h.shape
        flat = h.reshape(-1)
        probe = DeterministicStream(SEED_TOWER_PROBE + i)
        idx = probe.indices(PROBE_COUNT, flat.numel())
        arrays[f"tower.hidden_{i}.probe"] = flat[torch.from_numpy(idx)].contiguous()
        arrays[f"tower.hidden_{i}.stats"] = stats_tensor(h)
        hidden_meta.append(stats(h))
    meta["tower"] = {
        "stat_slots": list(STAT_SLOTS),
        "hidden_state_block_indices": [4, 8, 12, 16, 20],
        "hidden_states": hidden_meta,
        "cls_projection": stats(cls_proj),
        "cls_projection_normalized": stats(cls_norm),
        "probe_count": PROBE_COUNT,
        "probe_seed_base": SEED_TOWER_PROBE,
        "input_seed": SEED_TOWER_INPUT,
    }

    # The RoPE tables the checkpoint itself carries, sliced so the committed
    # golden stays tiny. Row r of the 576-row table is patch (r // 24, r % 24).
    rows = [0, 1, 23, 24, 300, 575]
    freqs = {
        "cos": visual.rope.freqs_cos.detach().float(),
        "sin": visual.rope.freqs_sin.detach().float(),
    }
    for name, table in freqs.items():
        assert tuple(table.shape) == (576, 64), table.shape
        arrays[f"rope.freqs_{name}.rows"] = table[rows].contiguous()
    meta["rope"] = {"rows": rows, "table_shape": [576, 64]}


def capture_preprocess(visual_image_size: int, out: Path, arrays: dict, meta: dict):
    image = build_input_image()
    image.save(out / "input_pattern.png", optimize=True)

    # `pipeline_flux.py:161` builds the tensor as CHW float in [0, 1];
    # `:173-174` resizes bicubic to the tower's image size and normalizes with
    # the OpenAI CLIP statistics.
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    resized = resize(tensor, visual_image_size, InterpolationMode.BICUBIC)
    normalized = normalize(resized, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
    assert tuple(normalized.shape) == (1, 3, 336, 336), normalized.shape

    flat = normalized.reshape(-1)
    idx = DeterministicStream(SEED_IMAGE ^ 0x1).indices(PROBE_COUNT, flat.numel())
    arrays["preprocess.probe"] = flat[torch.from_numpy(idx)].contiguous()
    arrays["preprocess.probe_indices"] = torch.from_numpy(idx)
    # A full row through the middle of the green channel: cheap, and a
    # channel-order or transpose mistake cannot survive it.
    arrays["preprocess.row_g_168"] = normalized[0, 1, 168, :].contiguous()
    meta["preprocess"] = {
        "source": "input_pattern.png",
        "source_size": [512, 512],
        "target_size": [336, 336],
        "interpolation": "bicubic (torchvision, antialias=True)",
        "mean": list(OPENAI_DATASET_MEAN),
        "std": list(OPENAI_DATASET_STD),
        "resized": stats(resized),
        "normalized": stats(normalized),
        "probe_count": PROBE_COUNT,
    }


# --------------------------------------------------------------------------
# IDFormer (`pulid/encoders_transformer.py:122-209`), loaded from the
# `pulid_encoder.*` tensors of pulid_flux_v0.9.1.safetensors
# (`pipeline_flux.py:99-109` splits the checkpoint by leading module name).
# --------------------------------------------------------------------------


def capture_idformer(pulid_repo: Path, adapter: Path, arrays: dict, meta: dict):
    sys.path.insert(0, str(pulid_repo))
    from pulid.encoders_transformer import IDFormer

    model = IDFormer().eval().float()
    state = load_file(adapter)
    prefix = "pulid_encoder."
    encoder_state = {
        k[len(prefix) :]: v.float() for k, v in state.items() if k.startswith(prefix)
    }
    model.load_state_dict(encoder_state, strict=True)

    # `pipeline_flux.py:181` — cat([arcface_512, clip_768]) -> [1, 1280].
    id_stream = DeterministicStream(SEED_IDFORMER_ID)
    arcface = id_stream.tensor(1, 512)
    clip = id_stream.tensor(1, 768)
    id_cond = torch.cat([arcface, clip], dim=-1)
    vit_hidden = [
        DeterministicStream(SEED_IDFORMER_VIT + i).tensor(1, 577, 1024)
        for i in range(5)
    ]

    with torch.no_grad():
        output = model(id_cond, vit_hidden)
    assert tuple(output.shape) == (1, 32, 2048), output.shape

    arrays["idformer.output"] = output[0].contiguous()
    arrays["idformer.output.stats"] = stats_tensor(output)
    meta["idformer"] = {
        "id_cond_seed": SEED_IDFORMER_ID,
        "vit_hidden_seed_base": SEED_IDFORMER_VIT,
        "id_cond": stats(id_cond),
        "output": stats(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pulid-repo", type=Path, required=True)
    parser.add_argument("--eva", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()

    torch.manual_seed(0)
    arrays: dict[str, torch.Tensor] = {}
    meta: dict = {}

    capture_preprocess(336, args.out, arrays, meta)
    visual = build_tower(args.pulid_repo, args.eva)
    capture_tower(visual, args.out, arrays, meta)
    del visual
    capture_idformer(args.pulid_repo, args.adapter, arrays, meta)

    save_file(
        {k: v.contiguous() for k, v in arrays.items()},
        str(args.out / "goldens.safetensors"),
    )
    (args.out / "goldens.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name, tensor in sorted(arrays.items()):
        print(f"{name:44s} {tuple(tensor.shape)} {tensor.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
