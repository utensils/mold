#!/usr/bin/env python3
"""Capture PuLID `PerceiverAttentionCA` goldens from upstream PyTorch.

Runs upstream `ToTheBeginning/PuLID`'s own
`pulid/encoders_transformer.py::PerceiverAttentionCA` with the real
`pulid_ca.{i}.*` weights out of `pulid_flux_v0.9.1.safetensors`, on a
deterministic synthetic image-token block, and writes the probe values and
summary statistics `crates/mold-inference/tests/pulid_adapter_parity.rs`
compares mold's port against.

The *inputs* are never committed: both sides generate them from the same
`xorshift64*` stream, so a fixture of any size costs nothing in the repository.
That stream is deliberately the same one
`crates/mold-inference/testdata/pulid/capture_goldens.py` (#1229) uses; if both
land, the two scripts should share it.

Usage (from the repository root):

    /Volumes/ExternalStorage/pulid-dev/venv/bin/python \\
      crates/mold-inference/testdata/pulid/capture_ca_goldens.py \\
      --pulid-weights /path/to/pulid_flux_v0.9.1.safetensors \\
      --pulid-repo tmp/PuLID

Requires only `torch` and `safetensors`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

import torch
from safetensors.torch import load_file, save_file

MULTIPLIER = 0x2545F4914F6CDD1D

# ASCII, so the constant says what it seeds: "PULIDCAI", "PULIDCAD", "PULIDCAP".
SEED_CA_IMAGE = 0x50554C4944434149
SEED_CA_ID = 0x50554C4944434144
SEED_CA_PROBE = 0x50554C4944434150

PROBE_COUNT = 512

# PuLID-FLUX's trained geometry (`pulid/encoders_transformer.py:30`).
DIM = 3072
DIM_HEAD = 128
HEADS = 16
KV_DIM = 2048
ID_TOKENS = 32
IMAGE_TOKENS = 64

# The boundaries of the two index ranges plus one interior module each: for
# FLUX.1 the double-stream loop consumes 0-9 and the single-stream loop 10-19.
MODULES = (0, 5, 9, 10, 15, 19)


class DeterministicStream:
    """`xorshift64*` — four lines in both Python and Rust, so no library
    version can move the numbers underneath a golden."""

    def __init__(self, seed: int) -> None:
        assert seed != 0, "xorshift64* has a fixed point at zero"
        self.state = seed

    def next_u64(self) -> int:
        x = self.state
        x ^= x >> 12
        x = (x ^ (x << 25)) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 27
        self.state = x
        return (x * MULTIPLIER) & 0xFFFFFFFFFFFFFFFF

    def next_unit(self) -> float:
        mantissa = self.next_u64() >> 11
        return (mantissa / float(1 << 53)) * 2.0 - 1.0

    def values(self, count: int) -> list[float]:
        return [self.next_unit() for _ in range(count)]

    def tensor(self, shape: tuple[int, ...]) -> torch.Tensor:
        total = 1
        for dim in shape:
            total *= dim
        # float32 the whole way: the Rust side builds the same values from f32
        # samples, so a float64 intermediate here would not match.
        flat = torch.tensor(self.values(total), dtype=torch.float64)
        return flat.to(torch.float32).reshape(shape)

    def indices(self, count: int, modulo: int) -> list[int]:
        return [self.next_u64() % modulo for _ in range(count)]


def load_upstream_module(repo: pathlib.Path):
    """Import upstream's `PerceiverAttentionCA` rather than restating it.

    A reimplementation in the capture script would be the same guess mold's
    port is, which is exactly what the golden exists to falsify.
    """
    path = repo / "pulid" / "encoders_transformer.py"
    if not path.is_file():
        raise SystemExit(f"upstream PuLID not found at {path}")
    spec = importlib.util.spec_from_file_location("pulid_encoders_transformer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PerceiverAttentionCA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pulid-weights", required=True, type=pathlib.Path)
    parser.add_argument("--pulid-repo", required=True, type=pathlib.Path)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent,
    )
    args = parser.parse_args()

    perceiver_cls = load_upstream_module(args.pulid_repo)
    weights = load_file(str(args.pulid_weights))

    digest = hashlib.sha256(args.pulid_weights.read_bytes()).hexdigest()
    print(f"pulid weights sha256 {digest}", file=sys.stderr)

    image_tokens = DeterministicStream(SEED_CA_IMAGE).tensor((1, IMAGE_TOKENS, DIM))
    id_embeds = DeterministicStream(SEED_CA_ID).tensor((1, ID_TOKENS, KV_DIM))

    goldens: dict[str, torch.Tensor] = {}
    manifest: dict[str, object] = {
        "source": "ToTheBeginning/PuLID pulid/encoders_transformer.py::PerceiverAttentionCA",
        "weights_file": args.pulid_weights.name,
        "weights_sha256": digest,
        "torch": torch.__version__,
        "dtype": "float32",
        "dim": DIM,
        "dim_head": DIM_HEAD,
        "heads": HEADS,
        "kv_dim": KV_DIM,
        "id_tokens": ID_TOKENS,
        "image_tokens": IMAGE_TOKENS,
        "seed_ca_image": hex(SEED_CA_IMAGE),
        "seed_ca_id": hex(SEED_CA_ID),
        "seed_ca_probe": hex(SEED_CA_PROBE),
        "probe_count": PROBE_COUNT,
        "modules": list(MODULES),
    }

    total_elements = IMAGE_TOKENS * DIM
    probe_indices = DeterministicStream(SEED_CA_PROBE).indices(PROBE_COUNT, total_elements)

    with torch.no_grad():
        for index in MODULES:
            prefix = f"pulid_ca.{index}."
            module = perceiver_cls(dim=DIM, dim_head=DIM_HEAD, heads=HEADS, kv_dim=KV_DIM)
            state = {
                key[len(prefix) :]: value.to(torch.float32)
                for key, value in weights.items()
                if key.startswith(prefix)
            }
            missing = module.load_state_dict(state, strict=True)
            assert not missing.missing_keys, missing
            module.eval().to(torch.float32)

            out = module(id_embeds, image_tokens)
            assert tuple(out.shape) == (1, IMAGE_TOKENS, DIM), out.shape
            flat = out.reshape(-1)
            goldens[f"ca{index}.probe"] = flat[probe_indices].contiguous()
            goldens[f"ca{index}.stats"] = torch.tensor(
                [
                    float(flat.mean()),
                    float(flat.std()),
                    float(flat.abs().max()),
                ],
                dtype=torch.float32,
            )
            print(
                f"pulid_ca.{index}: mean={float(flat.mean()):+.6f} "
                f"std={float(flat.std()):.6f} absmax={float(flat.abs().max()):.6f}",
                file=sys.stderr,
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_file(goldens, str(args.out_dir / "ca_goldens.safetensors"))
    (args.out_dir / "ca_goldens.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out_dir / 'ca_goldens.safetensors'}", file=sys.stderr)


if __name__ == "__main__":
    main()
