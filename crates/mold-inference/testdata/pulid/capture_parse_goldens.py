#!/usr/bin/env python3
"""Capture the #1225 face-parsing goldens from facexlib's BiSeNet.

PROVENANCE ONLY. Nothing in mold's build, test, or runtime path runs this
file; mold ships no Python. It is committed so the fixtures beside it can be
regenerated and audited.

It takes the ALIGNED 512 crops #1222 already committed
(`faces/<stem>.eva512.png`) rather than re-detecting, so this script measures
exactly one thing: what upstream's parser and mask do to a crop both
implementations agree on.

    python3 capture_parse_goldens.py \\
        --facexlib-repo /path/to/facexlib \\
        --weights       /path/to/facexlib/weights \\
        --faces         crates/mold-inference/testdata/pulid/faces \\
        --out           crates/mold-inference/testdata/pulid

`--weights` is a directory holding facexlib's `parsing_bisenet.pth`
(sha256 468e13ca...26567, the pin in `crates/mold-core/src/manifest.rs`).

Needs a scratch venv with `torch torchvision opencv-python-headless numpy
scipy safetensors`, plus a checkout of facexlib on `--facexlib-repo`
(`tmp/` is gitignored and is where mold's upstream clones live). A source
checkout needs a `facexlib/version.py`; `pip install facexlib` generates one,
a bare `git clone` does not:

    printf "__version__='0.3.0'\\n__gitsha__='260620ae'\\n" > facexlib/version.py

Writes, per face:

* `faces/<stem>.parsed512.png` — the masked crop the EVA tower is fed, i.e.
  `face_features_image` at `PuLID/pulid/pipeline_flux.py:169`, rounded to u8.
* into `parse_goldens.safetensors`:
  * `<stem>.labels.probe`     512 scattered labels of the argmax map
  * `<stem>.labels.histogram` per-class pixel counts, 19 wide
  * `<stem>.masked.probe`     512 scattered values of the masked f32 crop
  * `<stem>.preprocess.probe` 512 scattered values of the 336 tensor the tower
                              actually receives (masked, bicubic, CLIP-normalized)

With `--eva` and `--adapter` it additionally runs the rest of upstream's
pipeline on that tensor — the EVA02-CLIP-L-14-336 tower and the IDFormer,
concatenated with the RAW ArcFace embedding #1222 already committed in
`faces/<stem>.golden.json` — and records the FINAL identity:

  * `<stem>.identity.probe`   512 scattered values of the `[1, 32, 2048]` output
  * `<stem>.identity.stats`   [mean, std, min, max, peak] over the whole tensor

That is the end-to-end pin: it is the only fixture in the tree that compares
mold's whole extraction, mask included, against upstream's on a real
photograph. It needs `--pulid-repo` as well, for upstream's own
`eva_clip` and `pulid` modules.

Probe indices come from the same `xorshift64*` stream the encoder goldens use
(`capture_eva_goldens.py`), so nothing has to be committed to describe them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

# --- the shared value stream (see capture_eva_goldens.py) -------------------
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

    def indices(self, count: int, modulo: int) -> np.ndarray:
        return np.fromiter(
            (self.next_u64() % modulo for _ in range(count)), np.int64, count
        )


# "PULIDPRS" — this issue's own seed, so adding it shifts no existing fixture.
SEED_PARSE_PROBE = 0x50554C49_44505253
PROBE_COUNT = 512

# `pipeline_flux.py:163`
PARSE_MEAN = (0.485, 0.456, 0.406)
PARSE_STD = (0.229, 0.224, 0.225)
# `eva_clip/constants.py:1-2`
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# `pipeline_flux.py:166`
BACKGROUND_LABELS = [0, 16, 18, 7, 8, 9, 14, 15]
NUM_CLASSES = 19


def to_gray(img: torch.Tensor) -> torch.Tensor:
    """`pipeline_flux.py:113-116`."""
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    return x.repeat(1, 3, 1, 1)


def build_identity_stack(pulid_repo: Path, eva: Path, adapter: Path):
    """Upstream's tower and IDFormer, exactly as `capture_eva_goldens.py` builds them."""
    from functools import partial

    import torch.nn as nn
    from safetensors.torch import load_file

    sys.path.insert(0, str(pulid_repo))
    from eva_clip.eva_vit_model import EVAVisionTransformer
    from pulid.encoders_transformer import IDFormer

    visual = EVAVisionTransformer(
        img_size=336, patch_size=14, num_classes=768, use_mean_pooling=False,
        init_values=None, patch_dropout=0.0, embed_dim=1024, depth=24,
        num_heads=16, mlp_ratio=2.6667, qkv_bias=True, drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), xattn=False, rope=True,
        postnorm=False, pt_hw_seq_len=16, intp_freq=True, naiveswiglu=True,
        subln=True,
    )
    state = torch.load(eva, map_location="cpu", weights_only=True)
    _, unexpected = visual.load_state_dict(
        {k[len("visual."):]: v.float() for k, v in state.items() if k.startswith("visual.")},
        strict=False,
    )
    assert not unexpected, unexpected

    encoder = IDFormer().eval().float()
    adapter_state = load_file(adapter)
    prefix = "pulid_encoder."
    encoder.load_state_dict(
        {k[len(prefix):]: v.float() for k, v in adapter_state.items() if k.startswith(prefix)},
        strict=True,
    )
    return visual.eval().float(), encoder


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facexlib-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--faces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pulid-repo", type=Path)
    parser.add_argument("--eva", type=Path)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    end_to_end = bool(args.eva and args.adapter and args.pulid_repo)

    sys.path.insert(0, str(args.facexlib_repo))
    from facexlib.parsing import init_parsing_model
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import normalize, resize
    from PIL import Image

    tower = idformer = None
    if end_to_end:
        tower, idformer = build_identity_stack(
            args.pulid_repo, args.eva, args.adapter
        )

    device = torch.device("cpu")
    face_parse = init_parsing_model(
        model_name="bisenet", device=device, model_rootpath=str(args.weights)
    )

    sources = json.loads((args.faces / "sources.json").read_text())
    arrays: dict[str, torch.Tensor] = {}
    meta: dict[str, dict] = {}

    for entry in sources:
        stem = Path(entry["file"]).stem
        crop_path = args.faces / f"{stem}.eva512.png"
        # The committed crop is RGB; `pipeline_flux.py:161` reads BGR from cv2
        # and converts, so starting from RGB is the same tensor.
        rgb = np.asarray(Image.open(crop_path).convert("RGB"), dtype=np.float32) / 255.0
        crop = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()

        with torch.no_grad():
            out = face_parse(normalize(crop.clone(), PARSE_MEAN, PARSE_STD))[0]
            labels = out.argmax(dim=1, keepdim=True)
            background = sum(labels == label for label in BACKGROUND_LABELS).bool()
            masked = torch.where(background, torch.ones_like(crop), to_gray(crop))

        label_map = labels[0, 0].to(torch.uint8)
        u8 = (masked[0].permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255)
        Image.fromarray(u8.astype(np.uint8)).save(
            args.faces / f"{stem}.parsed512.png", optimize=True
        )

        preprocessed = normalize(
            resize(masked, 336, InterpolationMode.BICUBIC), CLIP_MEAN, CLIP_STD
        )

        flat_labels = label_map.reshape(-1)
        stream = DeterministicStream(SEED_PARSE_PROBE)
        label_idx = stream.indices(PROBE_COUNT, flat_labels.numel())
        flat_masked = masked.reshape(-1)
        masked_idx = stream.indices(PROBE_COUNT, flat_masked.numel())
        flat_pre = preprocessed.reshape(-1)
        pre_idx = stream.indices(PROBE_COUNT, flat_pre.numel())

        arrays[f"{stem}.labels.probe"] = flat_labels[
            torch.from_numpy(label_idx)
        ].contiguous()
        arrays[f"{stem}.labels.histogram"] = torch.bincount(
            flat_labels.to(torch.int64), minlength=NUM_CLASSES
        ).to(torch.int64)
        arrays[f"{stem}.masked.probe"] = flat_masked[
            torch.from_numpy(masked_idx)
        ].contiguous()
        arrays[f"{stem}.preprocess.probe"] = flat_pre[
            torch.from_numpy(pre_idx)
        ].contiguous()

        if end_to_end:
            golden = json.loads(
                (args.faces / f"{stem}.golden.json").read_text()
            )
            arcface = torch.tensor(golden["embedding"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                # `pipeline_flux.py:175-181`
                cls, hidden = tower(
                    preprocessed, return_all_features=False, return_hidden=True,
                    shuffle=False,
                )
                cls = cls / torch.norm(cls, 2, 1, True)
                identity = idformer(torch.cat([arcface, cls], dim=-1), hidden)
            flat_identity = identity.reshape(-1)
            identity_idx = stream.indices(PROBE_COUNT, flat_identity.numel())
            arrays[f"{stem}.identity.probe"] = flat_identity[
                torch.from_numpy(identity_idx)
            ].contiguous()
            arrays[f"{stem}.identity.stats"] = torch.tensor(
                [
                    float(flat_identity.mean()),
                    float(flat_identity.std(unbiased=False)),
                    float(flat_identity.min()),
                    float(flat_identity.max()),
                    float(flat_identity.abs().max()),
                ],
                dtype=torch.float32,
            )

        present = sorted(int(v) for v in torch.unique(flat_labels))
        meta[stem] = {
            "labels_present": present,
            "background_fraction": float(background.float().mean()),
            "masked": {
                "mean": float(masked.mean()),
                "min": float(masked.min()),
                "max": float(masked.max()),
            },
            "preprocess": {
                "mean": float(preprocessed.mean()),
                "min": float(preprocessed.min()),
                "max": float(preprocessed.max()),
            },
        }
        print(stem, meta[stem]["background_fraction"], present, flush=True)

    save_file(arrays, str(args.out / "parse_goldens.safetensors"))
    (args.out / "parse_goldens.json").write_text(
        json.dumps(
            {
                "upstream": {
                    "pulid": "ToTheBeginning/PuLID pipeline_flux.py:161-170",
                    "facexlib": "xinntao/facexlib 260620ae bisenet.py + resnet.py",
                },
                "checkpoint": {
                    "file": "parsing_bisenet.pth",
                    "sha256": "468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567",
                    "license": "MIT (facexlib)",
                },
                "probe_seed": SEED_PARSE_PROBE,
                "probe_count": PROBE_COUNT,
                "background_labels": BACKGROUND_LABELS,
                "faces": meta,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
