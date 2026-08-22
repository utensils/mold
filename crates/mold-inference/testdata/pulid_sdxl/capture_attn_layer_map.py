#!/usr/bin/env python3
"""Capture the SDXL / SD1.5 UNet attn-processor layer map for PuLID v1.1.

PROVENANCE ONLY. Nothing in mold's build or test path executes this file, and
mold ships no Python; this script exists so the ordering it records can be
independently reproduced and audited.

PuLID v1.1's `PuLIDPipeline.hack_unet_attn_layers`
(`PuLID/pulid/pipeline_v1_1.py:129-149`) walks `unet.attn_processors.items()`
in whatever order that dict iterates (Python dict order == insertion order ==
`named_modules()` traversal order for a freshly constructed model) and installs
one `IDAttnProcessor2_0` per cross-attention ("attn2") entry and a bare
`AttnProcessor()` per self-attention ("attn1") entry — `cross_attention_dim`
is `None` exactly when the processor name ends `attn1.processor`. `pulid_v1.1
.safetensors`'s `id_adapter_attn_layers.<i>.id_to_k.weight` /
`.id_to_v.weight` tensors are keyed by that SAME positional index, because
upstream saves `nn.ModuleList(unet.attn_processors.values())` whole
(`load_pretrain`, `pipeline_v1_1.py:151-163`) — attn1 entries hold an
`AttnProcessor()` with no parameters, so they simply contribute nothing to the
checkpoint. A Rust port that wants to attach the i'th `id_to_k`/`id_to_v` pair
to the right cross-attention module needs exactly this map.

The UNet is constructed on the `meta` device from its published config only —
no weights are downloaded, so this script does not need (and does not fetch)
the multi-GB `diffusers` checkpoint, only its small `config.json`.

Usage (scratch venv; torch CPU + diffusers + safetensors + accelerate):

    python capture_attn_layer_map.py \
        --pulid-adapter /path/to/pulid_v1.1.safetensors \
        --out .

`--pulid-adapter` is optional; when given, out-features and index coverage
are cross-checked against the real checkpoint and asserted, and
`attn_layer_map.json` records the checkpoint tensor shapes. Without it, the
script still emits both maps (no assertions, no checkpoint shapes) — useful
for `attn_layer_map_sd15.json`, since no PuLID checkpoint exists for SD1.5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel


def build_meta_unet(repo: str, subfolder: str = "unet") -> UNet2DConditionModel:
    """Construct the UNet's real module graph on `Device::Meta` — config only,
    no weight download, so `attn_processors` iterates in the exact order the
    real (weight-loaded) model would."""
    cfg = UNet2DConditionModel.load_config(repo, subfolder=subfolder)
    with torch.device("meta"):
        unet = UNet2DConditionModel.from_config(cfg)
    return unet


def enumerate_attn_layers(unet: UNet2DConditionModel) -> list[dict]:
    """Mirror `hack_unet_attn_layers`'s traversal and classification exactly.

    `unet.attn_processors` is an (ordered) dict; enumeration order over it IS
    `processor_index`, matching `nn.ModuleList(unet.attn_processors.values())`
    on the upstream side.
    """
    entries: list[dict] = []
    attn2_ordinal = 0
    for processor_index, proc_name in enumerate(unet.attn_processors.keys()):
        assert proc_name.endswith(".processor"), proc_name
        module_name = proc_name[: -len(".processor")]
        module = unet.get_submodule(module_name)

        is_attn2 = not proc_name.endswith("attn1.processor")
        kind = "attn2" if is_attn2 else "attn1"

        hidden_size = module.to_q.in_features
        heads = module.heads
        inner_dim = module.to_q.out_features
        assert inner_dim % heads == 0, (module_name, inner_dim, heads)
        dim_head = inner_dim // heads
        cross_attention_dim = module.to_k.in_features if is_attn2 else None

        entry = {
            "processor_index": processor_index,
            "module_name": module_name,
            "kind": kind,
            "hidden_size": hidden_size,
            "cross_attention_dim": cross_attention_dim,
            "heads": heads,
            "dim_head": dim_head,
        }
        if is_attn2:
            entry["attn2_ordinal"] = attn2_ordinal
            attn2_ordinal += 1
        entries.append(entry)
    return entries


def cross_check_against_checkpoint(entries: list[dict], adapter_path: Path) -> None:
    """Assert the checkpoint agrees with the traversal: every attn2 index has
    `id_to_k`/`id_to_v` weights, every attn1 index has none, and each
    `id_to_k` out-features equals that layer's own `hidden_size`."""
    from safetensors import safe_open

    with safe_open(str(adapter_path), framework="pt") as f:
        keys = set(f.keys())
        shapes = {k: f.get_slice(k).get_shape() for k in keys}

    prefix = "id_adapter_attn_layers."
    weighted_indices = set()
    for k in keys:
        if not k.startswith(prefix):
            continue
        idx = int(k[len(prefix) :].split(".", 1)[0])
        weighted_indices.add(idx)

    for entry in entries:
        idx = entry["processor_index"]
        id_to_k_key = f"{prefix}{idx}.id_to_k.weight"
        id_to_v_key = f"{prefix}{idx}.id_to_v.weight"
        has_weights = idx in weighted_indices

        if entry["kind"] == "attn2":
            assert has_weights, f"attn2 processor {idx} ({entry['module_name']}) has no checkpoint weights"
            assert id_to_k_key in keys, id_to_k_key
            assert id_to_v_key in keys, id_to_v_key
            k_shape = list(shapes[id_to_k_key])
            v_shape = list(shapes[id_to_v_key])
            # nn.Linear(cross_attention_dim, hidden_size, bias=False).weight
            # is stored [out_features, in_features].
            assert k_shape[0] == entry["hidden_size"], (entry, k_shape)
            assert k_shape[1] == entry["cross_attention_dim"], (entry, k_shape)
            assert v_shape == k_shape, (entry, k_shape, v_shape)
            entry["checkpoint"] = {
                "id_to_k.weight": k_shape,
                "id_to_v.weight": v_shape,
            }
        else:
            assert not has_weights, f"attn1 processor {idx} ({entry['module_name']}) unexpectedly has checkpoint weights"
            assert id_to_k_key not in keys, id_to_k_key
            assert id_to_v_key not in keys, id_to_v_key

    # Every checkpoint index must correspond to a real attn2 processor — no
    # orphaned weights the traversal didn't visit.
    attn2_indices = {e["processor_index"] for e in entries if e["kind"] == "attn2"}
    assert weighted_indices == attn2_indices, (
        weighted_indices - attn2_indices,
        attn2_indices - weighted_indices,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdxl-repo",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HF repo id whose unet/config.json is fetched (config only, no weights)",
    )
    parser.add_argument(
        "--sd15-repo",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="HF repo id for the SD1.5 unet config",
    )
    parser.add_argument(
        "--pulid-adapter",
        type=Path,
        default=None,
        help="pulid_v1.1.safetensors; when given, cross-checks + records checkpoint shapes",
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    # --- SDXL (checkpoint-asserted) ---
    sdxl_unet = build_meta_unet(args.sdxl_repo)
    sdxl_entries = enumerate_attn_layers(sdxl_unet)
    attn2_count = sum(1 for e in sdxl_entries if e["kind"] == "attn2")
    attn1_count = sum(1 for e in sdxl_entries if e["kind"] == "attn1")

    checkpoint_meta = None
    if args.pulid_adapter is not None:
        cross_check_against_checkpoint(sdxl_entries, args.pulid_adapter)
        import hashlib

        digest = hashlib.sha256(args.pulid_adapter.read_bytes()).hexdigest()
        checkpoint_meta = {
            "file": args.pulid_adapter.name,
            "sha256": digest,
            "assertions": [
                "every attn2 processor_index has id_to_k/id_to_v weights",
                "every attn1 processor_index has none",
                "id_to_k out_features == hidden_size",
                "checkpoint weighted indices == traversal attn2 indices (no orphans)",
            ],
        }

    sdxl_doc = {
        "source": "diffusers UNet2DConditionModel.attn_processors traversal, "
        "mirroring PuLID/pulid/pipeline_v1_1.py:129-149 hack_unet_attn_layers",
        "unet_repo": args.sdxl_repo,
        "unet_subfolder": "unet",
        "cross_attention_dim": sdxl_unet.config.cross_attention_dim,
        "block_out_channels": list(sdxl_unet.config.block_out_channels),
        "total_processors": len(sdxl_entries),
        "attn1_count": attn1_count,
        "attn2_count": attn2_count,
        "checkpoint": checkpoint_meta,
        "layers": sdxl_entries,
    }
    (args.out / "attn_layer_map.json").write_text(json.dumps(sdxl_doc, indent=2) + "\n")
    print(
        f"SDXL: {len(sdxl_entries)} processors "
        f"({attn1_count} attn1, {attn2_count} attn2)"
    )

    # --- SD1.5 (traversal only, no checkpoint — no PuLID-SD1.5 exists) ---
    sd15_unet = build_meta_unet(args.sd15_repo)
    sd15_entries = enumerate_attn_layers(sd15_unet)
    sd15_attn2 = sum(1 for e in sd15_entries if e["kind"] == "attn2")
    sd15_attn1 = sum(1 for e in sd15_entries if e["kind"] == "attn1")

    sd15_doc = {
        "source": "diffusers UNet2DConditionModel.attn_processors traversal, "
        "mirroring PuLID/pulid/pipeline_v1_1.py:129-149 hack_unet_attn_layers "
        "(no PuLID-SD1.5 checkpoint exists; this pins the traversal shape only)",
        "unet_repo": args.sd15_repo,
        "unet_subfolder": "unet",
        "cross_attention_dim": sd15_unet.config.cross_attention_dim,
        "block_out_channels": list(sd15_unet.config.block_out_channels),
        "total_processors": len(sd15_entries),
        "attn1_count": sd15_attn1,
        "attn2_count": sd15_attn2,
        "checkpoint": None,
        "layers": sd15_entries,
    }
    (args.out / "attn_layer_map_sd15.json").write_text(json.dumps(sd15_doc, indent=2) + "\n")
    print(
        f"SD1.5: {len(sd15_entries)} processors "
        f"({sd15_attn1} attn1, {sd15_attn2} attn2)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
