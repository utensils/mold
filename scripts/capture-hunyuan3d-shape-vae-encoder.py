#!/usr/bin/env python3
"""Capture the pinned Tencent Hunyuan3D 2.1 shape-VAE encoder.

This imports the upstream attention implementation directly from a pinned
checkout while bypassing the package's unrelated mesh postprocessor imports.
The production checkpoint weights and every input/output tensor are retained;
the Rust port never invokes this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import pathlib
import subprocess
import sys
import types

import numpy as np
import torch
from safetensors.torch import load_file, save_file


PINNED_TENCENT_COMMIT = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_upstream(source: pathlib.Path):
    package_paths = {
        "hy3dshape": source / "hy3dshape",
        "hy3dshape.models": source / "hy3dshape" / "models",
        "hy3dshape.models.autoencoders": source
        / "hy3dshape"
        / "models"
        / "autoencoders",
    }
    for name, path in package_paths.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    utilities = types.ModuleType("hy3dshape.utils")
    utilities.logger = logging.getLogger("hunyuan3d-shape-vae-oracle")
    sys.modules["hy3dshape.utils"] = utilities
    autoencoders = package_paths["hy3dshape.models.autoencoders"]
    load_module(
        "hy3dshape.models.autoencoders.attention_processors",
        autoencoders / "attention_processors.py",
    )
    return load_module(
        "hy3dshape.models.autoencoders.attention_blocks",
        autoencoders / "attention_blocks.py",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=130013)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--latents", type=int, default=64)
    args = parser.parse_args()

    source_commit = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != PINNED_TENCENT_COMMIT:
        raise RuntimeError(
            f"expected Tencent source {PINNED_TENCENT_COMMIT}, found {source_commit}"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    upstream = load_upstream(args.source)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16

    fourier = upstream.FourierEmbedder(
        num_freqs=8, input_dim=3, include_input=True, include_pi=False
    )
    encoder = upstream.PointCrossAttentionEncoder(
        num_latents=args.latents,
        downsample_ratio=max(1, args.points // args.latents),
        pc_size=args.points,
        pc_sharpedge_size=0,
        fourier_embedder=fourier,
        point_feats=4,
        width=1024,
        heads=16,
        layers=8,
        qkv_bias=False,
        use_ln_post=True,
        qk_norm=True,
    )
    pre_kl = torch.nn.Linear(1024, 128)

    checkpoint = load_file(str(args.checkpoint), device="cpu")
    encoder_state = {
        key.removeprefix("vae.encoder."): value
        for key, value in checkpoint.items()
        if key.startswith("vae.encoder.")
    }
    pre_kl_state = {
        key.removeprefix("vae.pre_kl."): value
        for key, value in checkpoint.items()
        if key.startswith("vae.pre_kl.")
    }
    encoder.load_state_dict(encoder_state, strict=True)
    pre_kl.load_state_dict(pre_kl_state, strict=True)
    del checkpoint, encoder_state, pre_kl_state
    encoder = encoder.to(device=device, dtype=dtype).eval()
    pre_kl = pre_kl.to(device=device, dtype=dtype).eval()

    rng = np.random.default_rng(args.seed)
    points = rng.uniform(-0.9999, 0.9999, (1, args.points, 3)).astype(np.float32)
    normals = rng.normal(size=(1, args.points, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    features = np.concatenate(
        [normals, np.zeros((1, args.points, 1), dtype=np.float32)], axis=-1
    )
    selected = np.linspace(0, args.points - 1, args.latents, dtype=np.int64)

    points_t = torch.from_numpy(points).to(device=device, dtype=dtype)
    features_t = torch.from_numpy(features).to(device=device, dtype=dtype)
    selected_t = torch.from_numpy(selected).to(device=device)
    query_points = points_t.index_select(1, selected_t)
    query_features = features_t.index_select(1, selected_t)

    with torch.inference_mode(), torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ):
        data_input = torch.cat([fourier(points_t), features_t], dim=-1)
        query_input = torch.cat([fourier(query_points), query_features], dim=-1)
        data_projected = encoder.input_proj(data_input)
        query_projected = encoder.input_proj(query_input)
        cross = encoder.cross_attn(query_projected, data_projected)
        hidden = encoder.self_attn(cross)
        hidden = encoder.ln_post(hidden)
        moments = pre_kl(hidden)
        mean, logvar = moments.chunk(2, dim=-1)
        logvar = logvar.clamp(-30.0, 20.0)

    tensors = {
        "points": torch.from_numpy(points),
        "features": torch.from_numpy(features),
        "selected": torch.from_numpy(selected),
        "data_projected": data_projected.float().cpu(),
        "query_projected": query_projected.float().cpu(),
        "cross": cross.float().cpu(),
        "hidden": hidden.float().cpu(),
        "mean": mean.float().cpu(),
        "logvar": logvar.float().cpu(),
    }
    tensor_path = args.output / "encoder-oracle.safetensors"
    save_file({name: value.contiguous() for name, value in tensors.items()}, tensor_path)
    metadata = {
        "schema": "mold.hunyuan3d.shape-vae-encoder-oracle.v1",
        "upstream_commit": source_commit,
        "upstream_source": str(args.source.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "output_sha256": sha256(tensor_path),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "seed": args.seed,
        "points": args.points,
        "latents": args.latents,
        "attention": "torch SDPA math",
        "encoder_tensor_count": 142,
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
