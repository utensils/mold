#!/usr/bin/env python3
"""Capture PuLID v1.1 (SDXL) `IDAttnProcessor2_0` ID-branch goldens.

PROVENANCE ONLY. Nothing in mold's build or test path executes this file, and
mold ships no Python.

Replicates the identity-conditioning half of upstream
`pulid/attention_processor.py::IDAttnProcessor2_0.__call__` (the branch
guarded by `if id_embedding is not None:`, lines ~299-333) EXACTLY, with the
real `id_to_k` / `id_to_v` weights for three representative cross-attention
("attn2") layers named in `attn_layer_map.json`. The upstream globals this
module reads are `NUM_ZERO = 0` (no zero-token padding branch) and
`ORTHO = ORTHO_v2 = False` (plain additive combination) — this script asserts
those are still the values in the cloned upstream before running, so a
future upstream default change cannot silently invalidate the goldens.

What is captured vs. what upstream computes: the full `__call__` also derives
`query`/the pre-id `hidden_states` ("attended") from `attn.to_q` /
`attn.to_k` / `attn.to_v` applied to real UNet activations, which requires
building a whole `diffusers.Attention` module this fixture has no use for.
Instead — as `capture_ca_goldens.py` does for `PerceiverAttentionCA` — this
script supplies `query` and `attended` as synthetic tensors of the shapes
those intermediates actually have at that point in `__call__`:

  query    = attn.to_q(hidden_states)              # BEFORE the head reshape
  attended = <post scaled_dot_product_attention text branch, reshaped,
              BEFORE `attn.to_out`>

and then performs, verbatim, the `id_key`/`id_value` projection, the head
reshape (`head_dim = inner_dim // attn.heads` where `inner_dim` is the
projected width, i.e. `hidden_size`), the `scaled_dot_product_attention` over
`(query, id_key, id_value)`, and the final `attended + id_scale *
id_hidden_states` combination. This is line-for-line the code path with
`NUM_ZERO == 0` and `not ORTHO and not ORTHO_v2`.

IMPORTANT — `id_embedding` carries 32 tokens, not 37. It is exactly the
`IDFormer` forward's own output (`pipeline_v1_1.py`'s
`id_embedding = self.id_adapter(id_cond, id_vit_hidden)`, shape
`[batch, 32, 2048]` — `num_queries=32`, sliced from the 37 internal latents
`torch.cat([32 query latents, 5 id tokens])` INSIDE `IDFormer.forward`
before the `latents[:, :self.num_queries]` line, `encoders_transformer.py`'s
last few lines). Nothing outside `IDFormer` ever sees a 37-token identity
tensor; `IDAttnProcessor2_0` receives the already-sliced 32.

Usage (scratch venv; torch CPU + safetensors):

    python capture_attn_goldens.py \
        --pulid-repo /path/to/PuLID \
        --adapter /path/to/pulid_v1.1.safetensors \
        --attn-layer-map attn_layer_map.json \
        --out .
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

MULT = 0x2545F4914F6CDD1D
MASK = (1 << 64) - 1


class DeterministicStream:
    """`xorshift64*` — bit-identical to the sibling capture scripts and to
    `crates/mold-inference/src/pulid_fixtures.rs::DeterministicStream`."""

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
        n = 1
        for d in shape:
            n *= d
        flat = np.fromiter((self.next_unit() for _ in range(n)), np.float32, n)
        return torch.from_numpy(flat.reshape(shape))

    def indices(self, count: int, modulo: int) -> np.ndarray:
        return np.fromiter((self.next_u64() % modulo for _ in range(count)), np.int64, count)


# ASCII spells "PULIDSX*"; distinct from capture_idformer_goldens.py's seeds
# (different fixture, collision-free is simpler to audit than shared).
SEED_QUERY = 0x50554C4944535851  # "PULIDSXQ"
SEED_ATTENDED = 0x50554C4944535841  # "PULIDSXA"
SEED_ID_EMBEDDING = 0x50554C4944535845  # "PULIDSXE"
SEED_PROBE = 0x50554C4944535850  # "PULIDSXP"

BATCH = 2
SEQ = 64
ID_TOKENS = 32
CROSS_ATTENTION_DIM = 2048
ID_SCALES = (1.0, 0.7)
PROBE_COUNT = 512

STAT_SLOTS = ("mean", "std", "min", "max", "peak")


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


def assert_upstream_flags_unchanged(pulid_repo: Path) -> None:
    path = pulid_repo / "pulid" / "attention_processor.py"
    spec = importlib.util.spec_from_file_location("pulid_attention_processor", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.NUM_ZERO == 0, "upstream NUM_ZERO changed — this script's math no longer matches"
    assert module.ORTHO is False, "upstream ORTHO changed — additive combination assumption invalid"
    assert module.ORTHO_v2 is False, "upstream ORTHO_v2 changed — additive combination assumption invalid"


def id_branch_forward(
    id_to_k: torch.nn.Linear,
    id_to_v: torch.nn.Linear,
    query_reshaped: torch.Tensor,  # [batch, heads, seq, head_dim] — already the real code's `query`
    attended: torch.Tensor,  # [batch, seq, hidden_size] — pre-`to_out` text branch output
    id_embedding: torch.Tensor,  # [batch, id_tokens, cross_attention_dim]
    heads: int,
    head_dim: int,
    id_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`IDAttnProcessor2_0.__call__`'s `if id_embedding is not None:` branch,
    verbatim, for `NUM_ZERO == 0` and `not ORTHO and not ORTHO_v2`
    (`pulid/attention_processor.py:299-333`)."""
    batch_size = attended.shape[0]

    id_key = id_to_k(id_embedding).to(query_reshaped.dtype)
    id_value = id_to_v(id_embedding).to(query_reshaped.dtype)

    id_key = id_key.view(batch_size, -1, heads, head_dim).transpose(1, 2)
    id_value = id_value.view(batch_size, -1, heads, head_dim).transpose(1, 2)

    id_hidden_states = F.scaled_dot_product_attention(
        query_reshaped, id_key, id_value, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    id_hidden_states = id_hidden_states.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)
    id_hidden_states = id_hidden_states.to(query_reshaped.dtype)

    # NUM_ZERO == 0, ORTHO == ORTHO_v2 == False, asserted above.
    combined = attended + id_scale * id_hidden_states
    return id_hidden_states, combined


def cast_round_trip_f16(t: torch.Tensor) -> torch.Tensor:
    return t.half().float()


def capture_layer(
    pulid_repo: Path,
    weights: dict[str, torch.Tensor],
    layer: dict,
    cast_f16_inputs: bool = False,
) -> tuple[dict, dict]:
    sys.path.insert(0, str(pulid_repo))
    from pulid.attention_processor import IDAttnProcessor2_0

    idx = layer["processor_index"]
    hidden_size = layer["hidden_size"]
    heads = layer["heads"]
    head_dim = layer["dim_head"]
    assert layer["cross_attention_dim"] == CROSS_ATTENTION_DIM, layer

    # Build the real module so id_to_k/id_to_v are constructed exactly as
    # upstream constructs them, then load the pinned checkpoint weights.
    proc = IDAttnProcessor2_0(hidden_size=hidden_size, cross_attention_dim=CROSS_ATTENTION_DIM)
    prefix = f"id_adapter_attn_layers.{idx}."
    state = {
        k[len(prefix):]: v.float()
        for k, v in weights.items()
        if k.startswith(prefix)
    }
    missing = proc.load_state_dict(state, strict=True)
    assert not missing.missing_keys and not missing.unexpected_keys, (idx, missing)
    proc.eval()

    query_flat = DeterministicStream(SEED_QUERY + idx).tensor(BATCH, SEQ, hidden_size)
    attended = DeterministicStream(SEED_ATTENDED + idx).tensor(BATCH, SEQ, hidden_size)
    id_embedding = DeterministicStream(SEED_ID_EMBEDDING + idx).tensor(BATCH, ID_TOKENS, CROSS_ATTENTION_DIM)

    if cast_f16_inputs:
        query_flat = cast_round_trip_f16(query_flat)
        attended = cast_round_trip_f16(attended)
        id_embedding = cast_round_trip_f16(id_embedding)

    query_reshaped = query_flat.view(BATCH, -1, heads, head_dim).transpose(1, 2)

    arrays: dict[str, torch.Tensor] = {}
    layer_meta: dict = {
        "processor_index": idx,
        "module_name": layer["module_name"],
        "hidden_size": hidden_size,
        "heads": heads,
        "dim_head": head_dim,
        "cross_attention_dim": CROSS_ATTENTION_DIM,
    }

    with torch.no_grad():
        id_hidden_states = None
        for scale in ID_SCALES:
            id_hidden_states, combined = id_branch_forward(
                proc.id_to_k,
                proc.id_to_v,
                query_reshaped,
                attended,
                id_embedding,
                heads,
                head_dim,
                scale,
            )
            assert tuple(combined.shape) == (BATCH, SEQ, hidden_size), combined.shape

            scale_tag = f"{scale:.1f}".replace(".", "p")
            flat = combined.reshape(-1)
            probe_idx = DeterministicStream(SEED_PROBE + idx).indices(PROBE_COUNT, flat.numel())
            arrays[f"attn{idx}.combined_s{scale_tag}.probe"] = flat[torch.from_numpy(probe_idx)].contiguous()
            arrays[f"attn{idx}.combined_s{scale_tag}.stats"] = stats_tensor(combined)
            layer_meta[f"combined_s{scale_tag}"] = stats(combined)

        # id_hidden_states does not depend on id_scale — record once, from
        # the last iteration (identical every time).
        assert tuple(id_hidden_states.shape) == (BATCH, SEQ, hidden_size), id_hidden_states.shape
        flat = id_hidden_states.reshape(-1)
        probe_idx = DeterministicStream(SEED_PROBE + idx + 1).indices(PROBE_COUNT, flat.numel())
        arrays[f"attn{idx}.id_hidden_states.probe"] = flat[torch.from_numpy(probe_idx)].contiguous()
        arrays[f"attn{idx}.id_hidden_states.stats"] = stats_tensor(id_hidden_states)
        layer_meta["id_hidden_states"] = stats(id_hidden_states)

    return arrays, layer_meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pulid-repo", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--attn-layer-map",
        type=Path,
        default=Path(__file__).resolve().parent / "attn_layer_map.json",
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--layer-indices",
        type=int,
        nargs=3,
        default=None,
        help="override the three processor_index values (default: one "
        "hidden_size=640 down_blocks layer, one hidden_size=1280 mid_block "
        "layer, one hidden_size=1280 up_blocks layer)",
    )
    parser.add_argument(
        "--f16-tolerance-check",
        action="store_true",
        help="also run every layer with query/attended/id_embedding cast "
        "through f16 and back, printing the max abs diff against the f32 "
        "goldens (for the README's tolerance table); writes nothing extra "
        "to disk",
    )
    args = parser.parse_args()

    assert_upstream_flags_unchanged(args.pulid_repo)

    layer_map = json.loads(args.attn_layer_map.read_text())
    by_index = {entry["processor_index"]: entry for entry in layer_map["layers"]}

    if args.layer_indices is not None:
        chosen_indices = list(args.layer_indices)
    else:
        # Chosen once, from a fresh SDXL traversal: 1 (down_blocks, 640),
        # 121 (mid_block, 1280), 49 (up_blocks, 1280).
        chosen_indices = [1, 121, 49]

    for idx in chosen_indices:
        entry = by_index[idx]
        assert entry["kind"] == "attn2", entry

    weights = load_file(str(args.adapter))
    digest = hashlib.sha256(args.adapter.read_bytes()).hexdigest()

    arrays: dict[str, torch.Tensor] = {}
    layers_meta = []
    for idx in chosen_indices:
        layer_arrays, layer_meta = capture_layer(args.pulid_repo, weights, by_index[idx])
        arrays.update(layer_arrays)
        layers_meta.append(layer_meta)

    meta = {
        "source": "ToTheBeginning/PuLID pulid/attention_processor.py::IDAttnProcessor2_0 "
        "(id_embedding branch only, NUM_ZERO=0, ORTHO=ORTHO_v2=False)",
        "adapter_file": args.adapter.name,
        "adapter_sha256": digest,
        "torch": torch.__version__,
        "dtype": "float32",
        "batch": BATCH,
        "seq": SEQ,
        "id_tokens": ID_TOKENS,
        "cross_attention_dim": CROSS_ATTENTION_DIM,
        "id_scales": list(ID_SCALES),
        "probe_count": PROBE_COUNT,
        "stat_slots": list(STAT_SLOTS),
        "seeds": {
            "query_base": hex(SEED_QUERY),
            "attended_base": hex(SEED_ATTENDED),
            "id_embedding_base": hex(SEED_ID_EMBEDDING),
            "probe_base": hex(SEED_PROBE),
            "note": "each layer adds its own processor_index to every base seed",
        },
        "chosen_processor_indices": chosen_indices,
        "layers": layers_meta,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in arrays.items()},
        str(args.out / "attn_goldens.safetensors"),
    )
    (args.out / "attn_goldens.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name, tensor in sorted(arrays.items()):
        print(f"{name:40s} {tuple(tensor.shape)} {tensor.dtype}")

    if args.f16_tolerance_check:
        print("\nf32 vs f16-round-tripped-inputs, max abs diff (combined, s=1.0):")
        for idx in chosen_indices:
            f16_arrays, _ = capture_layer(args.pulid_repo, weights, by_index[idx], cast_f16_inputs=True)
            a = arrays[f"attn{idx}.combined_s1p0.probe"]
            b = f16_arrays[f"attn{idx}.combined_s1p0.probe"]
            diff = (a - b).abs().max().item()
            peak = arrays[f"attn{idx}.combined_s1p0.stats"][4].item()
            print(f"  attn{idx:<4d} max_abs={diff:.6e}  (peak {peak:.3f}, rel {diff / peak:.3e})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
