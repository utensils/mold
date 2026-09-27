#!/usr/bin/env python3
"""Capture Qwen-Image 2.1 parity fixtures from the upstream diffusers oracle.

Provenance only: nothing in mold's build, test, or runtime path executes this
file, and mold ships no Python. See README.md in this directory for the venv,
the pinned revisions, and how to re-run it.

Every fixture is produced by calling upstream code (diffusers e0abab83b and
transformers' Qwen3-VL) directly. Where a value is not returned by upstream
(RoPE position indices, per-block activations) it is recorded with a forward
hook, or recomputed by a verbatim copy that is asserted equal to upstream's
own output before it is written.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import save_file

HF_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
DIFFUSERS_COMMIT = "e0abab83b5df05de9e7abd788643c1a7c1e42e28"
VIGGLE_REVISION = "bb26a0f38e5fe6c124aaccc9187a87eed5d9ed13"
VIGGLE_R128 = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
VIGGLE_R256 = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
TURBO_SIGMAS = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]

# The model card's transparency recipe (Qwen/Qwen-Image-2.1 README at
# b3179ad, "Transparent Image Generation (RGBA)"; GitHub README line 118:
# `This is an RGBA image with transparency. <your description>. The image has
# alpha channel and the background is transparent.`).
RGBA_PREFIX = "This is an RGBA image with transparency. "
RGBA_SUFFIX = ". The image has alpha channel and the background is transparent."

P6_PROMPT = "Put the red apple from image 2 on the white sign in image 1, keep everything else unchanged."
NEG_PROMPT = "blurry, lowres, watermark"
P8_PROMPT = "Change the sky to a warm sunset with orange clouds, keep the house and the sign unchanged."
P6_TARGET = (512, 512)  # (width, height)
P8_TARGET = (512, 512)
P6_SIGMA_T = (900.0, 600.0)  # scheduler-scale timesteps for the two P6 steps
NOISE_SEEDS = {"p6_x_a": 21, "p6_x_b": 22, "p8_noise": 1234}

ALPHA_PROMPTS = [
    ("bakery", 'A cozy bakery storefront at dawn with a hand-painted sign that reads "MOLD & FLOUR", warm light spilling onto a cobblestone street.'),
    ("fox", "A red fox standing in fresh snow, soft overcast light, shallow depth of field."),
    ("lake", "Minimalist flat vector illustration of a mountain lake at sunset, pastel colors."),
    ("teapot", "A studio product photo of a ceramic teapot on a plain white seamless background."),
]
ALPHA_RGBA_DESCRIPTIONS = [
    ("dragon", "A cute cartoon dragon sticker"),
    ("juice", "A glass of orange juice with ice cubes and a slice of orange"),
]
ALPHA_SEED = 42

HERE = Path(__file__).resolve().parent
STATE: dict = {}


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def log(*a):
    print("[capture]", *a, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def save_st(path: Path, tensors: dict, meta: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}
    # safetensors serializes its metadata map in hash order, so several keys
    # make the file bytes (and its SHA-256) vary run to run. One key holding a
    # sorted JSON document keeps a re-capture byte-identical.
    md = {"capture": json.dumps(meta or {}, sort_keys=True, default=str)}
    save_file(out, str(path), metadata=md)
    log("wrote", path, f"{path.stat().st_size / 1e6:.2f} MB")


def load_st(path: Path) -> dict:
    from safetensors.torch import load_file

    return load_file(str(path))


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")
    log("wrote", path)


def stats(t: torch.Tensor) -> dict:
    f = t.detach().float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "mean": f.mean().item(),
        "std": f.std().item() if f.numel() > 1 else 0.0,
        "min": f.min().item(),
        "max": f.max().item(),
        "abs_mean": f.abs().mean().item(),
    }


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().cpu(), b.double().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def randn(seed: int, shape) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32)


def rgba_recipe(description: str) -> str:
    return RGBA_PREFIX + description + RGBA_SUFFIX


# --------------------------------------------------------------------------------------------
# deterministic reference images (committed)
# --------------------------------------------------------------------------------------------


def make_opaque_reference() -> Image.Image:
    """1536x1024 RGB scene: sky gradient, sun, a house, and a lettered sign."""
    w, h = 1536, 1024
    y = np.arange(h, dtype=np.float64)[:, None]
    x = np.arange(w, dtype=np.float64)[None, :]
    horizon = 640.0
    t = np.clip(y / horizon, 0, 1)
    sky = np.stack([70 + 130 * t, 130 + 95 * t, 200 + 50 * t], -1) + 0 * x[..., None]
    g = np.clip((y - horizon) / (h - horizon), 0, 1)
    ground = np.stack([60 + 30 * g, 140 - 50 * g, 50 + 10 * g], -1) + 0 * x[..., None]
    arr = np.where((y >= horizon)[..., None], ground, sky)
    img = Image.fromarray(np.round(arr).clip(0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    d.ellipse((1060, 110, 1280, 330), fill=(250, 210, 60))
    d.rectangle((300, 420, 700, 760), fill=(180, 40, 40))
    d.polygon([(260, 420), (500, 250), (740, 420)], fill=(90, 50, 30))
    d.rectangle((460, 590, 560, 760), fill=(60, 35, 20))
    d.rectangle((340, 480, 430, 560), fill=(200, 230, 250))
    d.rectangle((850, 560, 1400, 760), fill=(255, 255, 255), outline=(20, 20, 20), width=6)
    d.rectangle((1110, 760, 1140, 900), fill=(110, 80, 50))
    font = ImageFont.load_default(size=110)
    d.text((1125, 660), "MOLD 2.1", fill=(20, 30, 120), font=font, anchor="mm")
    return img


def make_rgba_reference() -> Image.Image:
    """640x800 RGBA apple sticker with soft edges and a translucent glass pane.

    Fully transparent pixels carry non-zero colour (magenta) on purpose: Pillow
    resizes RGBA premultiplied, so that colour must vanish after the resize.
    """
    w, h = 640, 800
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    rgb = np.zeros((h, w, 3)) + np.array([255.0, 0.0, 255.0])
    a = np.zeros((h, w))
    # apple body: radial soft edge over the outer 14 px
    cx, cy, r = 320.0, 470.0, 230.0
    dist = np.hypot(xx - cx, yy - cy)
    body = np.clip((r - dist) / 14.0, 0, 1)
    shade = np.clip(1 - (dist / r) * 0.45 - (xx - cx) / (4 * r), 0, 1)
    apple = np.stack([200 * shade + 40, 20 + 25 * shade, 30 + 20 * shade], -1)
    rgb = np.where(body[..., None] > 0, apple, rgb)
    a = np.maximum(a, body)
    # stem (opaque)
    stem = (np.abs(xx - 330) < 12) & (yy > 170) & (yy < 260)
    rgb[stem] = [90, 55, 25]
    a[stem] = 1.0
    # leaf (alpha ~0.78)
    leaf = ((xx - 400) / 80) ** 2 + ((yy - 200) / 36) ** 2 <= 1
    rgb[leaf] = [40, 150, 50]
    a[leaf] = np.maximum(a[leaf], 200 / 255)
    # translucent glass pane (alpha 96) overlapping the lower left
    pane = (xx > 60) & (xx < 300) & (yy > 560) & (yy < 760)
    rgb[pane] = rgb[pane] * 0.3 + np.array([60, 120, 255]) * 0.7
    a[pane] = np.maximum(a[pane], 96 / 255)
    out = np.concatenate([rgb, a[..., None] * 255], -1)
    return Image.fromarray(np.round(out).clip(0, 255).astype(np.uint8), "RGBA")


def stage_images(args):
    op, ra = make_opaque_reference(), make_rgba_reference()
    op.save(args.committed / "ref_opaque.png", optimize=True)
    ra.save(args.committed / "ref_rgba.png", optimize=True)
    log("reference images", op.size, op.mode, ra.size, ra.mode)


def reference_images(args):
    op = Image.open(args.committed / "ref_opaque.png")
    ra = Image.open(args.committed / "ref_rgba.png")
    op.load()
    ra.load()
    assert op.mode == "RGB" and ra.mode == "RGBA"
    return op, ra


# --------------------------------------------------------------------------------------------
# Pillow fixtures (U8): premultiplied RGBA LANCZOS resize + white composite
# --------------------------------------------------------------------------------------------


def white_composite(img: Image.Image) -> Image.Image:
    # Verbatim from pipeline_qwenimage21.py:266-271.
    white = Image.new("RGB", img.size, (255, 255, 255))
    white.paste(img, mask=img.getchannel("A"))
    return white


def stage_pillow(args):
    from diffusers.image_processor import VaeImageProcessor

    rng = np.random.default_rng(20260926)
    src_rgba = rng.integers(0, 256, (23, 37, 4), dtype=np.uint8)
    # force the alpha extremes and a transparent pixel with loud colour
    src_rgba[0, :, 3] = 0
    src_rgba[1, :, 3] = 255
    src_rgba[2, :5] = [255, 0, 255, 0]
    src_rgb = rng.integers(0, 256, (23, 37, 3), dtype=np.uint8)
    im_rgba = Image.fromarray(src_rgba, "RGBA")
    im_rgb = Image.fromarray(src_rgb, "RGB")
    proc = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
    assert proc.config.resample == "lanczos" and proc.config.reducing_gap is None
    tensors = {"src_rgba": torch.from_numpy(src_rgba), "src_rgb": torch.from_numpy(src_rgb)}
    cases = {}
    for tag, (w, h) in {"up": (64, 40), "down": (16, 10), "odd": (29, 19)}.items():
        # The pipeline's own call: VaeImageProcessor.resize -> Image.resize(LANCZOS, reducing_gap=None).
        r = proc.resize(im_rgba, height=h, width=w)
        assert r.mode == "RGBA"
        tensors[f"rgba_{tag}"] = torch.from_numpy(np.array(r))
        tensors[f"rgba_{tag}_white"] = torch.from_numpy(np.array(white_composite(r)))
        # non-RGBA inputs are converted to RGBA first (pipeline:653-654)
        r2 = proc.resize(im_rgb.convert("RGBA"), height=h, width=w)
        tensors[f"rgb_as_rgba_{tag}"] = torch.from_numpy(np.array(r2))
        tensors[f"rgb_{tag}"] = torch.from_numpy(np.array(proc.resize(im_rgb, height=h, width=w)))
        cases[tag] = [w, h]
    # the premultiplied path explicitly, for the record
    assert np.array_equal(
        np.array(im_rgba.resize((64, 40), Image.Resampling.LANCZOS)),
        np.array(im_rgba.convert("RGBa").resize((64, 40), Image.Resampling.LANCZOS).convert("RGBA")),
    )
    meta = {
        "pillow_version": PIL.__version__,
        "resample": "LANCZOS (Image.Resampling.LANCZOS=1), reducing_gap=None, via diffusers VaeImageProcessor.resize",
        "cases_wh": cases,
        "layout": "uint8 HxWxC",
        "notes": "RGBA resize is premultiplied inside Pillow (RGBA->RGBa->resize->RGBA); *_white is white.paste(img, mask=A)",
    }
    save_st(args.committed / "pillow_resize.safetensors", tensors, meta)
    # full-size: the two references exactly as the pipeline hands them to the VL copy / VAE
    op, ra = reference_images(args)
    big = {}
    for name, img in (("opaque", op), ("rgba", ra)):
        rgba = img if img.mode == "RGBA" else img.convert("RGBA")
        w, h, _ = calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
        r = proc.resize(rgba, height=h, width=w)
        big[f"{name}_resized_rgba"] = torch.from_numpy(np.array(r))
        big[f"{name}_resized_white"] = torch.from_numpy(np.array(white_composite(r)))
        r.save(args.large / f"ref_{name}_resized_{w}x{h}.png")
        white_composite(r).save(args.large / f"ref_{name}_resized_{w}x{h}_white.png")
    save_st(args.large / "pillow_reference_resize.safetensors", big, {"pillow_version": PIL.__version__})


# --------------------------------------------------------------------------------------------
# CPU-only: calculate_dimensions, schedules
# --------------------------------------------------------------------------------------------


def calculate_dimensions(target_area, ratio):
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions as up

    return up(target_area, ratio)


def stage_cpu(args):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

    # calculate_dimensions (U1), including exact .5 ties where Python round() goes to even.
    rows = []
    area = 1024 * 1024
    ratios = [1.0, 1.5, 2 / 3, 4 / 3, 3 / 4, 16 / 9, 9 / 16, 0.8, 1536 / 1024, 640 / 800, 21 / 9, 1 / 3, 3.0]
    # w = 32*(n+0.5) exactly: ratio = (w/1024)^2 makes sqrt(area*ratio) exact
    for n in (15, 20, 31, 33, 40):
        ratios.append(((32 * n + 16) / 1024) ** 2)
    for r in ratios:
        w, h, _ = calculate_dimensions(area, r)
        wf = math.sqrt(area * r)
        rows.append({"area": area, "ratio": r, "width": w, "height": h, "w_over_32": wf / 32, "h_over_32": wf / r / 32})
    for a2 in (1536 * 1536, 2048 * 2048):
        for r in (1.0, 4 / 3, 16 / 9, 0.8):
            w, h, _ = calculate_dimensions(a2, r)
            rows.append({"area": a2, "ratio": r, "width": w, "height": h})
    save_json(args.committed / "calculate_dimensions.json", {"source": "pipeline_qwenimage21.py:149-156 (Python round = half-to-even)", "rows": rows})

    # U14: the pinned revision's small config files, verbatim, in one place.
    cfgs = {}
    for rel in ("model_index.json", "scheduler/scheduler_config.json", "transformer/config.json",
                "vae/config.json", "text_encoder/config.json", "processor/preprocessor_config.json"):
        raw = (args.model_root / rel).read_bytes()
        cfgs[rel] = {"sha256": hashlib.sha256(raw).hexdigest(), "json": json.loads(raw)}
    save_json(args.committed / "hf_configs.json", {"hf_repo": "Qwen/Qwen-Image-2.1", "revision": HF_REVISION, "files": cfgs})

    # Schedules (U11): base config and the turbo recipe (shift_terminal None).
    cfg_path = args.model_root / "scheduler" / "scheduler_config.json"
    base = FlowMatchEulerDiscreteScheduler.from_pretrained(str(args.model_root), subfolder="scheduler")
    turbo = FlowMatchEulerDiscreteScheduler.from_config(base.config, shift_terminal=None)
    cfg = json.loads(cfg_path.read_text())
    out = {"scheduler_config": cfg, "turbo_overrides": {"shift_terminal": None}, "cases": []}
    for name, sched, steps, sigmas_in, wh in [
        ("base_40_1024sq", base, 40, None, (1024, 1024)),
        ("base_40_512sq", base, 40, None, (512, 512)),
        ("base_4_512sq", base, 4, None, (512, 512)),
        ("base_40_1344x768", base, 40, None, (1344, 768)),
        ("base_40_2048sq", base, 40, None, (2048, 2048)),
        ("base_40_2752x1536", base, 40, None, (2752, 1536)),
        ("turbo_6_512sq", turbo, 6, TURBO_SIGMAS, (512, 512)),
        ("turbo_6_1024sq", turbo, 6, TURBO_SIGMAS, (1024, 1024)),
        ("turbo_6_2048sq", turbo, 6, TURBO_SIGMAS, (2048, 2048)),
    ]:
        seq = (wh[0] // 16) * (wh[1] // 16)
        sig = np.linspace(1.0, 1 / steps, steps) if sigmas_in is None else sigmas_in
        # the pipeline's call (pipeline_qwenimage21.py:723-733)
        mu = calculate_shift(
            seq,
            sched.config.get("base_image_seq_len", 256),
            sched.config.get("max_image_seq_len", 4096),
            sched.config.get("base_shift", 0.5),
            sched.config.get("max_shift", 1.15),
        )
        sched.set_timesteps(num_inference_steps=steps, device="cpu", sigmas=sig, mu=mu)
        out["cases"].append(
            {
                "name": name,
                "width": wh[0],
                "height": wh[1],
                "target_tokens": seq,
                "steps": steps,
                "input_sigmas": [float(s) for s in sig],
                "mu": mu,
                "sigmas": [float(s) for s in sched.sigmas.tolist()],
                "timesteps": [float(s) for s in sched.timesteps.tolist()],
                "sigmas_dtype": str(sched.sigmas.dtype),
                # What the bf16 transformer receives: t cast to the latent
                # dtype, then divided by 1000 (pipeline_qwenimage21.py:769,773).
                "transformer_timesteps_bf16": [
                    float(x) for x in (sched.timesteps.to(torch.bfloat16) / 1000).float().tolist()
                ],
            }
        )
    save_json(args.committed / "schedules.json", out)


# --------------------------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------------------------


def assert_bundled_cudnn():
    """Refuse to capture with a foreign cuDNN mixed into the process.

    A Nix devshell's LD_LIBRARY_PATH carries its own cuDNN engine libraries;
    torch then loads its bundled libcudnn.so.9 beside a different release's
    engine libraries, convolutions still run, and only
    `torch.backends.cudnn.version()` notices. Run with
    `LD_LIBRARY_PATH=/run/opengl-driver/lib:<libstdc++ dir>` (see README).
    """
    x = torch.zeros(1, 1, 4, 4, device="cuda")
    torch.nn.functional.conv2d(x, torch.zeros(1, 1, 3, 3, device="cuda"))
    torch.cuda.synchronize()
    libs = sorted({l.split()[-1] for l in open("/proc/self/maps") if "libcudnn" in l})
    torch_dir = str(Path(torch.__file__).resolve().parent.parent)
    foreign = [l for l in libs if not l.startswith(torch_dir)]
    assert not foreign, f"foreign cuDNN libraries loaded: {foreign}"
    STATE["cudnn"] = torch.backends.cudnn.version()


def set_precision_flags():
    # A true fp32 oracle: no TF32 anywhere (cuDNN defaults to TF32 for convolutions).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def load_text_encoder(args, dtype):
    from transformers import Qwen3VLForConditionalGeneration

    te = STATE.get("te")
    if te is None:
        log("loading text encoder", dtype, "->", args.te_device)
        te = Qwen3VLForConditionalGeneration.from_pretrained(str(args.model_root / "text_encoder"), dtype=dtype)
        # The pipeline reads hidden states only; the LM head's logits are
        # computed and discarded. At fp32 its 2.5 GB weight plus a
        # [L, 151936] logits tensor is what pushes a single 46 GB card over,
        # so it is replaced by an identity. No hidden state depends on it.
        te.lm_head = torch.nn.Identity()
        te = te.to(args.te_device).eval()
        log("text encoder resident", f"{torch.cuda.memory_allocated(args.te_device) / 2**30:.2f} GiB")
        STATE["te"] = te
    elif next(te.parameters()).dtype != dtype:
        te.to(dtype)
    return te


def load_vae(args, dtype, device=None):
    from diffusers import AutoencoderKLQwenImage21

    vae = STATE.get("vae")
    if vae is None:
        log("loading vae", dtype)
        vae = AutoencoderKLQwenImage21.from_pretrained(str(args.model_root), subfolder="vae", torch_dtype=dtype).eval()
        STATE["vae"] = vae
    vae.to(device or args.dit_device, dtype)
    return vae


def load_transformer(args, dtype):
    from diffusers import QwenImage21Transformer2DModel

    dit = STATE.get("dit")
    if dit is None:
        log("loading transformer", dtype, "->", args.dit_device)
        dit = QwenImage21Transformer2DModel.from_pretrained(str(args.model_root), subfolder="transformer", torch_dtype=dtype)
        dit = dit.to(args.dit_device).eval()
        STATE["dit"] = dit
    elif next(dit.parameters()).dtype != dtype:
        dit.to(dtype)
    return dit


def build_pipeline(args, dtype, with_te=True, with_dit=True):
    """Assemble QwenImage21Pipeline from separately loaded components.

    `_get_qwen_prompt_embeds` runs on `--te-device` and its outputs are moved to
    the caller's device, so the text encoder may live on another GPU (or be
    parked on the CPU between stages). Nothing else about the upstream call
    path changes. On a single GPU both devices are the same.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from transformers import AutoProcessor

    processor = STATE.get("processor") or AutoProcessor.from_pretrained(str(args.model_root / "processor"))
    STATE["processor"] = processor
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(args.model_root), subfolder="scheduler")
    vae = load_vae(args, dtype)
    te = load_text_encoder(args, dtype) if with_te else None
    dit = load_transformer(args, dtype) if with_dit else None
    pipe = QwenImage21Pipeline(scheduler=scheduler, vae=vae, text_encoder=te, processor=processor, transformer=dit)
    pipe.set_progress_bar_config(disable=True)
    orig = pipe._get_qwen_prompt_embeds
    te_dev = torch.device(args.te_device)

    def on_te(prompt=None, image=None, device=None):
        outs = orig(prompt, image, te_dev)
        return tuple(o.to(device) if o is not None else None for o in outs)

    pipe._get_qwen_prompt_embeds = on_te
    return pipe


def install_processor_recorder(processor):
    base = type(processor)
    if getattr(base, "_mold_recorder", False):
        return

    class Recorder(base):
        _mold_recorder = True

        def __call__(self, *a, **k):
            out = super().__call__(*a, **k)
            STATE["proc_calls"].append({"kwargs": k, "out": out})
            return out

    processor.__class__ = Recorder
    STATE["proc_calls"] = []


def pipeline_condition_images(pipe, images, output_resolution=1024):
    """Step 1 of QwenImage21Pipeline.__call__ (pipeline_qwenimage21.py:648-663), verbatim."""
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions as cd

    input_image_sizes, input_images, vae_images = [], [], []
    for img in images:
        if hasattr(img, "mode") and img.mode != "RGBA":
            img = img.convert("RGBA")
        image_width, image_height = img.size
        input_width, input_height, _ = cd(output_resolution * output_resolution, image_width / image_height)
        input_image_sizes.append((input_width, input_height))
        input_images.append(pipe.image_processor.resize(img, width=input_width, height=input_height))
        vae_images.append(pipe.image_processor.preprocess(img, width=input_width, height=input_height).unsqueeze(2))
    return input_image_sizes, input_images, vae_images


# --------------------------------------------------------------------------------------------
# P1-P3: processor, vision tower, trimmed hidden states
# --------------------------------------------------------------------------------------------


class EncoderTaps:
    """Forward hooks on the Qwen3-VL text encoder for P2/P3 internals."""

    HIDDEN_LAYERS = (0, 1, 2, 3, 4, 18, 35, 36)

    def __init__(self, te):
        self.te = te
        self.data = {}
        self.handles = [
            te.model.visual.register_forward_hook(self._vision),
            te.model.language_model.register_forward_pre_hook(self._lm_pre, with_kwargs=True),
            te.register_forward_hook(self._outer),
        ]

    def _vision(self, mod, inp, out):
        # copy before get_image_features splits pooler_output in place
        self.data["vision_last_hidden_state"] = out.last_hidden_state.detach().float().cpu()
        self.data["vision_merger"] = out.pooler_output.detach().float().cpu()
        for i, d in enumerate(out.deepstack_features):
            self.data[f"vision_deepstack_{i}"] = d.detach().float().cpu()

    def _lm_pre(self, mod, a, k):
        if k.get("position_ids") is not None:
            self.data["lm_position_ids"] = k["position_ids"].detach().cpu()
        if k.get("visual_pos_masks") is not None:
            self.data["lm_visual_pos_masks"] = k["visual_pos_masks"].detach().cpu()

    def _outer(self, mod, inp, out):
        hs = out.hidden_states
        self.data["hidden_state_count"] = len(hs)
        for i in self.HIDDEN_LAYERS:
            self.data[f"lm_hidden_{i}"] = hs[i].detach().float().cpu()

    def remove(self):
        for h in self.handles:
            h.remove()


def encode_with_taps(pipe, prompt, input_images):
    te = pipe.text_encoder
    taps = EncoderTaps(te)
    STATE["proc_calls"] = []
    try:
        pe, pm, ipm = pipe.encode_prompt(prompt=prompt, image=input_images, device=torch.device(STATE["dit_device"]))
    finally:
        taps.remove()
    call = STATE["proc_calls"][-1]
    return pe, pm, ipm, taps.data, call


def stage_encode(args, dtype=torch.float32, tag="fp32"):
    set_precision_flags()
    pipe = build_pipeline(args, dtype, with_te=True, with_dit=False)
    # One 46 GB card holds the fp32 text encoder or the VAE work, not both at
    # peak: park the VAE until the text-encoder captures are done.
    pipe.vae.to("cpu")
    install_processor_recorder(pipe.processor)
    op, ra = reference_images(args)

    sizes2, imgs2, vae2 = pipeline_condition_images(pipe, [op, ra])
    sizes1, imgs1, vae1 = pipeline_condition_images(pipe, [op])
    log("reference sizes", sizes2)

    record = {"reference_sizes_wh": sizes2, "drop_idx": pipe._drop_idx, "img_token_id": pipe._img_token_id}
    enc = {}
    for name, prompt, imgs in (
        ("p6_pos", P6_PROMPT, imgs2),
        ("p6_neg", NEG_PROMPT, imgs2),
        ("p8_pos", P8_PROMPT, imgs1),
        ("t2i_p8", P8_PROMPT, None),
    ):
        pe, pm, ipm, taps, call = encode_with_taps(pipe, prompt, imgs)
        enc[name] = (pe, pm, ipm)
        mi = call["out"]
        if tag == "fp32" and name == "p6_pos":
            # P1: processor output for 2 references, exactly as the pipeline called it
            p1 = {
                "input_ids": mi["input_ids"],
                "attention_mask": mi["attention_mask"],
                "image_grid_thw": mi["image_grid_thw"],
            }
            if "mm_token_type_ids" in mi:
                p1["mm_token_type_ids"] = mi["mm_token_type_ids"]
            save_st(args.committed / "p1_processor_ids.safetensors", p1, {"prompt": prompt, "text": call["kwargs"]["text"], "padding_side": call["kwargs"].get("padding_side")})
            save_st(
                args.large / "p1_processor_pixels.safetensors",
                {"pixel_values": mi["pixel_values"], "image_grid_thw": mi["image_grid_thw"]},
                {"formula": "(x - 127.5) / 127.5 in f32 (torchvision backend), patch 16 / temporal 2 / merge 2"},
            )
            # P2: vision merger + deepstack
            save_st(
                args.large / "p2_vision_fp32.safetensors",
                {k: v for k, v in taps.items() if k.startswith("vision_")},
                {"deepstack_visual_indexes": [8, 16, 24], "image_grid_thw": mi["image_grid_thw"].tolist()},
            )
            record["p1"] = {k: list(v.shape) for k, v in p1.items()}
            record["p1_processor_text"] = call["kwargs"]["text"]
        if tag == "fp32" or name != "t2i_p8":
            lm = {k: v for k, v in taps.items() if k.startswith("lm_") and isinstance(v, torch.Tensor)}
            if lm:
                save_st(args.large / f"p3_{name}_lm_internals_{tag}.safetensors", lm, {"hidden_state_count": taps["hidden_state_count"], "note": "lm_hidden_36 is PRE final norm (pipeline hook), lm_hidden_0 is inputs_embeds after the image scatter"})
        t = {"prompt_embeds": pe, "image_pad_mask": ipm}
        if pm is not None:
            t["prompt_embeds_mask"] = pm
        save_st(args.large / f"p3_{name}_{tag}.safetensors", t, {"prompt": prompt, "drop_idx": pipe._drop_idx, "mask_is_none": pm is None})
        record[f"p3_{name}"] = {"prompt_embeds": stats(pe), "image_pad_true": int(ipm.sum()), "len": pe.shape[1]}
        # committed copy of the (small) mask
        if tag == "fp32":
            save_st(args.committed / f"p3_{name}_image_pad_mask.safetensors", {"image_pad_mask": ipm.to(torch.uint8)})

    if tag == "fp32":
        # Tokenize-check: templates for 1/2/3 references (needs the text encoder)
        stage_templates(args, pipe, record)
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        dev = torch.device(args.dit_device)
        pipe.vae.to(dev)
        # P4: VAE encode for the opaque and the transparent reference
        vae = pipe.vae
        p4 = {}
        for i, (name, (w, h)) in enumerate(zip(("opaque", "rgba"), sizes2)):
            x = vae2[i].to(dev, dtype)
            raw = vae.encode(x).latent_dist.mode()
            norm = pipe._encode_vae_image(x, generator=None)
            packed = pipe._pack_latents(norm, 1, 64, norm.shape[3], norm.shape[4])
            p4[f"{name}_input"] = x
            p4[f"{name}_mode"] = raw
            p4[f"{name}_normalized"] = norm
            p4[f"{name}_packed"] = packed
            dec = vae.decode(raw, return_dict=False)[0][:, :, 0]
            p4[f"{name}_roundtrip_decoded"] = dec
            npimg = pipe.image_processor.postprocess(dec, output_type="np")
            pipe.image_processor.numpy_to_pil(npimg)[0].save(args.large / f"p4_{name}_roundtrip.png")
            record[f"p4_{name}"] = {"size_wh": [w, h], "packed": stats(packed), "roundtrip_mode": pipe.image_processor.numpy_to_pil(npimg)[0].mode}
        save_st(args.large / "p4_vae_encode_fp32.safetensors", p4, {"latents_mean_std": "vae/config.json latents_mean/latents_std; normalized = (mode - mean) / std; packed = view(B,64,H*W).transpose(1,2)"})
        # Small VAE encoder internals (64x96 crop of the resized RGBA reference), per module.
        crop = imgs2[1].crop((400, 500, 496, 564))
        xs = pipe.image_processor.preprocess(crop, width=96, height=64).unsqueeze(2).to(dev, dtype)
        taps = {}
        hooks = []
        names = ["conv_in", "mid_block", "norm_out", "conv_out"]
        for i, blk in enumerate(vae.encoder.down_blocks):
            names.append(f"down_blocks.{i}")
            if hasattr(blk, "avg_shortcut") and blk.avg_shortcut is not None:
                names.append(f"down_blocks.{i}.avg_shortcut")
        mods = dict(vae.encoder.named_modules())
        for n in names:
            if n in mods:
                hooks.append(mods[n].register_forward_hook(lambda m, a, o, n=n: taps.__setitem__(n, o.detach().float().cpu())))
        hooks.append(vae.quant_conv.register_forward_hook(lambda m, a, o: taps.__setitem__("quant_conv", o.detach().float().cpu())))
        mode = vae.encode(xs).latent_dist.mode()
        for h in hooks:
            h.remove()
        taps = {("encoder." + k if k != "quant_conv" else k): v for k, v in taps.items()}
        taps["input"] = xs
        taps["mode"] = mode
        save_st(args.large / "p4_vae_encoder_internals_64x96_fp32.safetensors", taps, {"crop_box_of_resized_rgba": [400, 500, 496, 564]})
        record["p4_internals_modules"] = sorted(taps)

        # P5: joint layout from upstream internals, for every encoded case
        stage_layout(args, enc, sizes2, sizes1, record)

    save_json(args.large / f"encode_record_{tag}.json", record)
    STATE["encode_record_" + tag] = record
    return pipe


def stage_templates(args, pipe, record):
    """Record the exact template strings the pipeline builds and hands the processor."""
    colors = [(200, 30, 30), (30, 160, 60), (40, 60, 200)]
    small = [Image.new("RGB", (256, 256), c) for c in colors]
    out = {"system_prompt": pipe.sys_prompt, "drop_idx": pipe._drop_idx, "img_token_id": pipe._img_token_id,
           "template_t2i": pipe.prompt_template_t2i, "template_ti2i": pipe.prompt_template_ti2i, "cases": []}
    for n in (0, 1, 2, 3):
        STATE["proc_calls"] = []
        imgs = small[:n] if n else None
        pipe.encode_prompt(prompt="a test prompt", image=imgs, device=torch.device(args.dit_device))
        call = STATE["proc_calls"][-1]
        text = call["kwargs"]["text"][0]
        ids = call["out"]["input_ids"][0].tolist()
        case = {
            "references": n,
            "processor_called_with_raw_text": True,
            "processor_kwargs": sorted(k for k in call["kwargs"] if k != "images"),
            "text": text,
            "text_utf8_hex": text.encode("utf-8").hex(),
            "input_ids": ids,
            "image_grid_thw": call["out"]["image_grid_thw"].tolist() if n else None,
        }
        if n:
            # For contrast: what apply_chat_template would have produced for the same content.
            content = [{"type": "image"} for _ in range(n)] + [{"type": "text", "text": "a test prompt"}]
            msgs = [
                {"role": "system", "content": [{"type": "text", "text": pipe.sys_prompt}]},
                {"role": "user", "content": content},
            ]
            chat = pipe.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            case["apply_chat_template_text"] = chat
            case["apply_chat_template_equal"] = chat == text
        out["cases"].append(case)
    save_json(args.committed / "templates.json", out)
    record["templates"] = [c["text"] for c in out["cases"]]


def rope_indices(img_shapes, image_pad_mask):
    """Verbatim index computation of QwenImage21Rope.forward (transformer_qwenimage21.py:682-708)."""
    frame_index, image_height_index, image_width_index = [], [], []
    cursor, position = 0, 0
    total_len = image_pad_mask.shape[-1]
    is_image_token = image_pad_mask.tolist()
    for _, height, width in img_shapes:
        block_start = is_image_token.index(True, cursor)
        text_len = block_start - cursor
        frame_index.extend(range(position, position + text_len))
        position += text_len
        cursor = block_start + height * width
        frame_index.extend([position] * (height * width))
        position += max(height, width)
        image_height_index.extend([h for h in range(-(height - height // 2), height // 2) for _ in range(width)])
        image_width_index.extend([w for _ in range(height) for w in range(-(width - width // 2), width // 2)])
    if cursor < total_len:
        frame_index.extend(range(position, position + total_len - cursor))
    frame_index = torch.tensor(frame_index, dtype=torch.long)
    height_index = frame_index.clone()
    width_index = frame_index.clone()
    height_index[image_pad_mask] = torch.tensor(image_height_index, dtype=torch.long)
    width_index[image_pad_mask] = torch.tensor(image_width_index, dtype=torch.long)
    return frame_index, height_index, width_index


def joint_layout(img_mask_vl: torch.Tensor, img_shapes):
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21Rope,
        QwenImage21Transformer2DModel,
        _qwenimage21_prefix_segments,
    )

    img_mask_vl = img_mask_vl.cpu().bool()
    # transformer_qwenimage21.py:911-912
    repeats = torch.where(img_mask_vl, 4, 1)[0]
    image_pad_mask = torch.repeat_interleave(img_mask_vl[0], repeats)
    image_ids, target = QwenImage21Transformer2DModel.build_token_metadata(image_pad_mask, img_shapes)
    prefix_len = int((~target).sum())
    segments = _qwenimage21_prefix_segments(image_ids, prefix_len)
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    freqs = rope(img_shapes, image_pad_mask, device="cpu")
    f, h, w = rope_indices(img_shapes, image_pad_mask)
    replica = torch.cat([rope.freqs[0][f], rope.freqs[1][h], rope.freqs[2][w]], dim=-1)
    assert torch.equal(replica, freqs), "RoPE index replica disagrees with upstream pos_embed"
    return {
        "image_pad_mask": image_pad_mask,
        "image_ids": image_ids,
        "target_token_mask": target,
        "prefix_len": prefix_len,
        "segments": segments,
        "rope_frame": f,
        "rope_height": h,
        "rope_width": w,
        "rope_freqs": freqs,
    }


def append_target_slots(mask, target_tokens):
    # pipeline_qwenimage21.py:740-741
    return torch.cat([mask, mask.new_ones(mask.shape[0], target_tokens // 4)], dim=1)


def img_shapes_for(sizes_wh, target_wh):
    # pipeline_qwenimage21.py:712-720
    return [(1, h // 16, w // 16) for w, h in sizes_wh] + [(1, target_wh[1] // 16, target_wh[0] // 16)]


def stage_layout(args, enc, sizes2, sizes1, record):
    cases = {
        "p6_pos": (enc["p6_pos"][2], img_shapes_for(sizes2, P6_TARGET)),
        "p6_neg": (enc["p6_neg"][2], img_shapes_for(sizes2, P6_TARGET)),
        "p8_pos": (enc["p8_pos"][2], img_shapes_for(sizes1, P8_TARGET)),
        "t2i_p8": (enc["t2i_p8"][2], img_shapes_for([], P8_TARGET)),
    }
    small, big, js = {}, {}, {}
    for name, (ipm, shapes) in cases.items():
        tt = shapes[-1][1] * shapes[-1][2]
        vl = append_target_slots(ipm.cpu(), tt)
        L = joint_layout(vl, shapes)
        for k in ("image_pad_mask", "target_token_mask"):
            small[f"{name}.{k}"] = L[k].to(torch.uint8)
        for k in ("image_ids", "rope_frame", "rope_height", "rope_width"):
            small[f"{name}.{k}"] = L[k]
        small[f"{name}.vl_img_mask"] = vl[0].to(torch.uint8)
        big[f"{name}.rope_freqs"] = torch.view_as_real(L["rope_freqs"]).contiguous()
        ids = L["image_ids"].tolist()
        blocks = []
        for b in range(len(shapes)):
            pos = [i for i, v in enumerate(ids) if v == b]
            blocks.append({"id": b, "start": pos[0], "end": pos[-1] + 1, "shape_fhw": list(shapes[b])})
        js[name] = {
            "img_shapes": [list(s) for s in shapes],
            "vl_len": int(vl.shape[1]),
            "text_len_after_drop": int(ipm.shape[1]),
            "joint_len": int(L["image_pad_mask"].numel()),
            "prefix_len": L["prefix_len"],
            "target_tokens": tt,
            "image_blocks": blocks,
            "segments": [[s, e, bool(t)] for s, e, t in L["segments"]],
            "rope_position_after": int(L["rope_frame"][-1]),
        }
    save_st(args.committed / "p5_layout.safetensors", small, {"source": "build_token_metadata, _qwenimage21_prefix_segments, QwenImage21Rope (upstream), rope indices via asserted-equal replica"})
    save_st(args.large / "p5_rope_freqs.safetensors", big, {"layout": "view_as_real(complex64 [L, 64]) -> [L, 64, 2]; axes (16,56,56) -> 8+28+28 complex pairs, theta 10000"})
    save_json(args.committed / "p5_layout.json", js)
    record["p5"] = js


# --------------------------------------------------------------------------------------------
# P6 / P7: transformer forwards
# --------------------------------------------------------------------------------------------


def p6_inputs(args, dtype):
    p3 = load_st(args.large / "p3_p6_pos_fp32.safetensors")
    p4 = load_st(args.large / "p4_vae_encode_fp32.safetensors")
    rec = json.loads((args.large / "encode_record_fp32.json").read_text())
    sizes2 = [tuple(s) for s in rec["reference_sizes_wh"]]
    shapes = img_shapes_for(sizes2, P6_TARGET)
    tt = shapes[-1][1] * shapes[-1][2]
    cond = torch.cat([p4["opaque_packed"], p4["rgba_packed"]], dim=1)
    x_a = randn(NOISE_SEEDS["p6_x_a"], (1, tt, 64))
    x_b = randn(NOISE_SEEDS["p6_x_b"], (1, tt, 64))
    # pipeline_qwenimage21.py:770,775: timestep = t.to(latents.dtype) / 1000
    ts = [torch.tensor([t], dtype=torch.float32).to(dtype) / 1000 for t in P6_SIGMA_T]
    img_mask = append_target_slots(p3["image_pad_mask"].bool(), tt)
    return {
        "prompt_embeds": p3["prompt_embeds"].to(dtype),
        "prompt_embeds_mask": p3.get("prompt_embeds_mask"),
        "img_mask": img_mask,
        "cond_latents": cond.to(dtype),
        "x_a": x_a.to(dtype),
        "x_b": x_b.to(dtype),
        "timestep_a": ts[0],
        "timestep_b": ts[1],
        "img_shapes": [shapes],
    }


def dit_forward(dit, inp, x, t, kv_cache=None, kv_cache_mode=None):
    dev = next(dit.parameters()).device
    mask = inp["prompt_embeds_mask"]
    return dit(
        hidden_states=torch.cat([inp["cond_latents"], x], dim=1).to(dev),
        encoder_hidden_states=inp["prompt_embeds"].to(dev),
        encoder_hidden_states_mask=None if mask is None else mask.to(dev),
        timestep=t.to(dev),
        img_shapes=inp["img_shapes"],
        img_mask=inp["img_mask"].to(dev),
        kv_cache=kv_cache,
        kv_cache_mode=kv_cache_mode,
        return_dict=False,
    )[0]


def stage_transformer(args, dtype, tag):
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    set_precision_flags()
    dit = load_transformer(args, dtype)
    inp = p6_inputs(args, dtype)
    if tag == "fp32":
        save_st(
            args.large / "p6_inputs.safetensors",
            {k: v for k, v in inp.items() if isinstance(v, torch.Tensor) and k not in ("timestep_a", "timestep_b")},
            {"img_shapes": inp["img_shapes"], "noise_seeds": NOISE_SEEDS, "timesteps_scheduler_scale": P6_SIGMA_T,
             "note": "cond_latents = cat(P4 opaque_packed, P4 rgba_packed); x_a/x_b = torch.randn(CPU generator seed) fp32; prompt_embeds = P3 p6_pos fp32"},
        )
    taps, hooks = {}, []
    blocks = dit.transformer_blocks
    with torch.no_grad():
        hooks.append(blocks[0].register_forward_pre_hook(lambda m, a, k: taps.__setitem__("block0_in", k["hidden_states"].detach().cpu()), with_kwargs=True))
        hooks.append(blocks[0].register_forward_hook(lambda m, a, o: taps.__setitem__("block0_out", o.detach().cpu())))
        hooks.append(blocks[-1].register_forward_hook(lambda m, a, o: taps.__setitem__("block31_out", o.detach().cpu())))
        hooks.append(dit.modulation.register_forward_hook(lambda m, a, o: taps.__setitem__("modulation", o.detach().cpu())))
        hooks.append(dit.time_text_embed.register_forward_hook(lambda m, a, o: taps.__setitem__("temb", o.detach().cpu())))
        full_a = dit_forward(dit, inp, inp["x_a"], inp["timestep_a"])
        for h in hooks:
            h.remove()
        cache = QwenImage21KVCache(len(blocks))
        extract_a = dit_forward(dit, inp, inp["x_a"], inp["timestep_a"], cache, "extract")
        kv = {}
        for li in (0, len(blocks) - 1):
            k, v = cache.get_layer(li).get()
            kv[f"layer{li}.k"], kv[f"layer{li}.v"] = k, v
        cached_b = dit_forward(dit, inp, inp["x_b"], inp["timestep_b"], cache, "cached")
        del cache
        full_b = dit_forward(dit, inp, inp["x_b"], inp["timestep_b"])
    tt = inp["x_a"].shape[1]
    outs = {
        "timestep_a": inp["timestep_a"],
        "timestep_b": inp["timestep_b"],
        "full_a": full_a,
        "extract_a": extract_a,
        "cached_b": cached_b,
        "full_b": full_b,
    }
    save_st(args.large / f"p6_outputs_{tag}.safetensors", outs, {"note": "full_*/extract_a are [1, joint_len, 64] (every token); cached_b is target-only [1, target, 64]; pipeline keeps [:, -target:]"})
    save_st(args.large / f"p6_internals_{tag}.safetensors", {**taps, **kv}, {"note": "block activations of full_a; K/V (post-norm, post-RoPE, [B, prefix, heads, head_dim]) from the extract pass"})
    rec = {
        "full_a_target": stats(full_a[:, -tt:]),
        "extract_vs_full_a_target_rel": rel_err(extract_a[:, -tt:], full_a[:, -tt:]),
        "extract_vs_full_a_all_rel": rel_err(extract_a, full_a),
        "cached_b_vs_full_b_target_rel": rel_err(cached_b, full_b[:, -tt:]),
        "timestep_a": inp["timestep_a"].float().item(),
        "timestep_b": inp["timestep_b"].float().item(),
    }
    log("P6", tag, rec)
    STATE[f"p6_{tag}"] = {"inp": inp, "full_a": full_a}
    return rec


def stage_lora(args, dtype, tag):
    """P7: one full forward with the Viggle r128 LoRA applied by PEFT (unmerged)."""
    set_precision_flags()
    pipe = build_pipeline(args, dtype, with_te=False, with_dit=True)
    dit = pipe.transformer
    inp = p6_inputs(args, dtype)
    pipe.load_lora_weights(str(args.viggle_dir), weight_name=VIGGLE_R128, adapter_name="viggle_r128")
    n_lora = sum(1 for n, _ in dit.named_modules() if n.endswith("lora_A.viggle_r128"))
    merged = any(getattr(m, "merged", False) for m in dit.modules() if hasattr(m, "merged_adapters"))
    scaling = {n: m.scaling.get("viggle_r128") for n, m in dit.named_modules() if hasattr(m, "scaling") and "viggle_r128" in getattr(m, "scaling", {})}
    lora_dtype = {str(p.dtype) for n, p in dit.named_parameters() if "lora_" in n}
    with torch.no_grad():
        with_lora = dit_forward(dit, inp, inp["x_a"], inp["timestep_a"])
    pipe.delete_adapters("viggle_r128")
    with torch.no_grad():
        without = dit_forward(dit, inp, inp["x_a"], inp["timestep_a"])
    tt = inp["x_a"].shape[1]
    save_st(
        args.large / f"p7_lora_r128_{tag}.safetensors",
        {"timestep_a": inp["timestep_a"], "full_a_lora": with_lora, "full_a_no_lora": without},
        {"lora": VIGGLE_R128, "lora_sha256": "bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b", "revision": VIGGLE_REVISION,
         "peft_lora_layers": n_lora, "merged": merged, "scaling_values": sorted({float(v) for v in scaling.values()}), "lora_param_dtypes": sorted(lora_dtype),
         "inputs": "p6_inputs.safetensors (x_a, timestep_a)"},
    )
    rec = {
        "peft_lora_layers": n_lora,
        "merged": merged,
        "scaling": sorted({float(v) for v in scaling.values()}),
        "lora_param_dtypes": sorted(lora_dtype),
        "lora_effect_rel_target": rel_err(with_lora[:, -tt:], without[:, -tt:]),
        "no_lora_matches_p6": rel_err(without, STATE[f"p6_{tag}"]["full_a"].cpu()) if f"p6_{tag}" in STATE else None,
    }
    log("P7", tag, rec)
    return rec


# --------------------------------------------------------------------------------------------
# P8: end-to-end with injected noise
# --------------------------------------------------------------------------------------------


def stage_e2e(args, dtype, tag):
    from diffusers import FlowMatchEulerDiscreteScheduler

    set_precision_flags()
    # The fp32 text encoder and fp32 transformer do not fit one 46 GB card
    # together. The prompt was already encoded by this very function in the
    # encode stage (p3_p8_pos_<tag>); hand those exact tensors back in place of
    # the text-encoder call, so the rest of __call__ runs unmodified.
    pipe = build_pipeline(args, dtype, with_te=False, with_dit=True)
    cached = load_st(args.large / f"p3_p8_pos_{tag}.safetensors")

    def cached_embeds(prompt=None, image=None, device=None):
        assert prompt == [P8_PROMPT] and image is not None and len(image) == 1, (prompt, image)
        pe = cached["prompt_embeds"]
        # _get_qwen_prompt_embeds returns an all-ones mask; encode_prompt turns it into None
        pm = cached.get("prompt_embeds_mask", torch.ones(pe.shape[:2], dtype=torch.long))
        return pe.to(device), pm.to(device), cached["image_pad_mask"].to(device)

    pipe._get_qwen_prompt_embeds = cached_embeds
    assert str(pipe._execution_device) == str(torch.device(args.dit_device)), pipe._execution_device
    op, _ = reference_images(args)
    tt = (P8_TARGET[0] // 16) * (P8_TARGET[1] // 16)
    noise = randn(NOISE_SEEDS["p8_noise"], (1, tt, 64))
    if tag == "fp32":
        save_st(args.large / "p8_noise.safetensors", {"latents": noise}, {"seed": NOISE_SEEDS["p8_noise"], "layout": "packed [B, H*W, 64] = view(B,64,H*W).transpose(1,2) of [B,64,H,W]; H=W=32"})
    base_sched = pipe.scheduler
    recs = {}
    for variant in ("base4", "turbo6"):
        steps_lat = []

        def cb(p, i, t, kw):
            steps_lat.append((float(t), kw["latents"].detach().float().cpu()))
            return {}

        kwargs = dict(
            prompt=P8_PROMPT,
            image=[op],
            width=P8_TARGET[0],
            height=P8_TARGET[1],
            latents=noise.clone(),
            output_type="np",
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        if variant == "base4":
            pipe.scheduler = base_sched
            kwargs.update(num_inference_steps=4)
        else:
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_sched.config, shift_terminal=None)
            pipe.load_lora_weights(str(args.viggle_dir), weight_name=VIGGLE_R256, adapter_name="viggle_r256")
            kwargs.update(num_inference_steps=6, sigmas=TURBO_SIGMAS, true_cfg_scale=1.0)
        with torch.no_grad():
            img = pipe(**kwargs).images  # np float [1, H, W, 4] in [0,1]
        sigmas = pipe.scheduler.sigmas.tolist()
        timesteps = pipe.scheduler.timesteps.tolist()
        if variant == "turbo6":
            pipe.delete_adapters("viggle_r256")
            pipe.scheduler = base_sched
        t = {f"step{i}_latents": l for i, (_, l) in enumerate(steps_lat)}
        t["final_latents"] = steps_lat[-1][1].clone()
        t["decoded_rgba_float"] = torch.from_numpy(img[0])
        pil = pipe.image_processor.numpy_to_pil(img)[0]
        png = args.large / f"p8_{variant}_{tag}.png"
        pil.save(png)
        save_st(args.large / f"p8_{variant}_{tag}.safetensors", t, {
            "prompt": P8_PROMPT, "reference": "ref_opaque.png", "target_wh": list(P8_TARGET), "output_resolution": 1024,
            "sigmas": sigmas, "timesteps": timesteps, "use_kv_cache": True, "true_cfg_scale": 1.0,
            "lora": VIGGLE_R256 if variant == "turbo6" else None, "shift_terminal": None if variant == "turbo6" else 0.02,
            "noise": "p8_noise.safetensors"})
        a = np.array(pil)[..., 3]
        recs[variant] = {"sigmas": sigmas, "timesteps": timesteps, "final": stats(t["final_latents"]),
                         "png": png.name, "png_mode": pil.mode, "alpha_min": int(a.min()), "alpha_ne_255": int((a != 255).sum())}
        log("P8", variant, tag, recs[variant])
    return recs


# --------------------------------------------------------------------------------------------
# t2i alpha histograms (bf16, the model card's default setup)
# --------------------------------------------------------------------------------------------


def stage_alpha(args):
    set_precision_flags()
    pipe = build_pipeline(args, torch.bfloat16, with_te=True, with_dit=True)
    jobs = [(n, p, False) for n, p in ALPHA_PROMPTS] + [(n, rgba_recipe(d), True) for n, d in ALPHA_RGBA_DESCRIPTIONS]
    results = []
    for name, prompt, recipe in jobs:
        with torch.no_grad():
            img = pipe(prompt=prompt, width=1024, height=1024, num_inference_steps=40, output_type="np",
                       generator=torch.Generator(args.dit_device).manual_seed(ALPHA_SEED)).images
        pil = pipe.image_processor.numpy_to_pil(img)[0]
        path = args.large / f"alpha_t2i_{name}.png"
        pil.save(path)
        af = img[0][..., 3]
        a8 = np.array(pil)[..., 3]
        hist = np.bincount(a8.ravel(), minlength=256)
        r = {
            "name": name, "prompt": prompt, "transparency_recipe": recipe, "png": path.name, "mode": pil.mode,
            "alpha_float_min": float(af.min()), "alpha_float_max": float(af.max()),
            "alpha_u8_min": int(a8.min()), "alpha_u8_ne_255": int((a8 != 255).sum()),
            "alpha_u8_eq_0": int((a8 == 0).sum()), "alpha_u8_fraction_lt_128": float((a8 < 128).mean()),
            "alpha_u8_hist_nonzero": {str(i): int(c) for i, c in enumerate(hist) if c},
        }
        results.append(r)
        log("alpha", name, {k: r[k] for k in ("alpha_u8_min", "alpha_u8_ne_255", "alpha_u8_eq_0", "alpha_float_min")})
    save_json(args.committed / "alpha_histograms.json", {
        "setup": {"dtype": "bf16 (all components)", "size": "1024x1024", "steps": 40, "true_cfg_scale": 1.0, "seed": ALPHA_SEED,
                  "generator": f"torch.Generator('{args.dit_device}')", "byte_rule": "numpy_to_pil: round(clamp(x/2+0.5,0,1)*255)"},
        "renders": results})
    return results


# --------------------------------------------------------------------------------------------
# Viggle LoRA key layout (headers only)
# --------------------------------------------------------------------------------------------

VIGGLE_FILES = {
    "r128": (VIGGLE_R128, "bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b", 679604800),
    "r256": (VIGGLE_R256, "2a0148f5c73abbed5f97da5ea356e439318aadb281d01fce4af39cdf43728803", 1359147904),
}


def stage_viggle(args):
    import collections
    import re
    import struct

    def header(path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(n))

    out = {
        "repo": "Viggle/Qwen-Image-2.1-viggle-turbo",
        "revision": VIGGLE_REVISION,
        "license": {
            "name": "qwen-research",
            "text_file": "LICENSE (byte-identical to Qwen/Qwen-Image-2.1 LICENSE, sha256 8dc973f024ff95966bea25866efa443fd16776dcb1001e681e3d467ea572b28d)",
            "summary": "Qwen RESEARCH LICENSE AGREEMENT (2026-09-20): non-commercial research/evaluation use only; derivative of Qwen-Image-2.1; NOTICE says 'Built with Qwen.'",
        },
        "recipe": {"steps": 6, "sigmas_raw": TURBO_SIGMAS, "true_cfg_scale": 1.0, "negative_prompt": None,
                   "shift_terminal": None, "lora_scale": 1.0, "apply": "unmerged: W x + B (A x) * alpha/rank"},
        "files": {},
    }
    for tag, (fn, sha, size) in VIGGLE_FILES.items():
        path = args.viggle_dir / fn
        assert path.stat().st_size == size, (fn, path.stat().st_size)
        assert sha256_file(path) == sha, fn
        h = header(path)
        meta = h.pop("__metadata__", None)
        mods, ranks = {}, set()
        for k, v in sorted(h.items()):
            m = re.match(r"^(transformer\.)?(.*)\.(lora_A|lora_B)\.weight$", k)
            assert m, f"unexpected key {k}"
            mods.setdefault(m.group(2), {})[m.group(3)] = v["shape"]
            if m.group(3) == "lora_A":
                ranks.add(v["shape"][0])
        gen = collections.Counter(re.sub(r"transformer_blocks\.\d+\.", "transformer_blocks.{i}.", m) for m in mods)
        shapes = {}
        for m, e in mods.items():
            shapes.setdefault(re.sub(r"transformer_blocks\.\d+\.", "transformer_blocks.{i}.", m), e)
        blocks = sorted({int(x) for x in re.findall(r"transformer_blocks\.(\d+)\.", " ".join(mods))})
        out["files"][tag] = {
            "file": fn, "sha256": sha, "size_bytes": size, "tensor_count": len(h),
            "dtypes": dict(collections.Counter(v["dtype"] for v in h.values())),
            "__metadata__": meta,
            "alpha_tensors": sum(1 for k in h if k.endswith(".alpha")),
            "top_level_prefixes": dict(collections.Counter(k.split(".")[0] for k in h)),
            "key_pattern": "transformer.<module>.lora_A.weight [r, in] / transformer.<module>.lora_B.weight [out, r]",
            "sample_keys": sorted(h)[:6],
            "ranks": sorted(ranks),
            "block_indices": [blocks[0], blocks[-1], len(blocks)],
            "modules": {g: {"count": c, "shapes": shapes[g]} for g, c in sorted(gen.items())},
        }
    save_json(args.committed / "viggle_lora_layout.json", out)


# --------------------------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------------------------

DESCRIPTIONS = {
    "ref_opaque.png": "Opaque 1536x1024 RGB reference (deterministic, make_opaque_reference)",
    "ref_rgba.png": "640x800 RGBA reference with soft alpha, translucent pane, magenta under alpha 0 (make_rgba_reference)",
    "pillow_resize.safetensors": "U8: Pillow LANCZOS resize of small RGBA/RGB inputs (premultiplied) + white composite, uint8",
    "calculate_dimensions.json": "U1: calculate_dimensions rows incl. half-to-even ties",
    "schedules.json": "U11: FlowMatchEuler sigmas/timesteps/mu for base and turbo (shift_terminal None) cases, plus the bf16 transformer timesteps t.to(bf16)/1000",
    "templates.json": "U2: exact template strings, bytes, token ids for 0-3 references; apply_chat_template contrast",
    "p1_processor_ids.safetensors": "P1: processor input_ids/attention_mask/mm_token_type_ids/image_grid_thw for 2 references",
    "p3_p6_pos_image_pad_mask.safetensors": "P3: image_pad_mask after drop_idx, 2 refs, positive prompt",
    "p3_p6_neg_image_pad_mask.safetensors": "P3: image_pad_mask after drop_idx, 2 refs, negative prompt",
    "p3_p8_pos_image_pad_mask.safetensors": "P3: image_pad_mask after drop_idx, 1 ref (P8 prompt)",
    "p3_t2i_p8_image_pad_mask.safetensors": "P3: image_pad_mask (all false), t2i P8 prompt",
    "p5_layout.safetensors": "P5: joint image mask, image_ids, target mask, RoPE indices per case",
    "p5_layout.json": "P5: segments, prefix_len, image blocks per case",
    "alpha_histograms.json": "t2i alpha statistics (bf16, 1024^2, 40 steps) incl. transparency recipe renders",
    "viggle_lora_layout.json": "Viggle v0.2.1 LoRA key layout, rank, alpha source, licence",
    "p1_processor_pixels.safetensors": "P1: pixel_values for 2 references (f32)",
    "p2_vision_fp32.safetensors": "P2: vision last hidden, merger output, 3 deepstack outputs (fp32)",
    "p4_vae_encode_fp32.safetensors": "P4: VAE input, mode, normalized, packed latents + decode roundtrip, opaque and RGBA refs",
    "p4_vae_encoder_internals_64x96_fp32.safetensors": "P4 debug: per-module VAE encoder outputs on a 64x96 RGBA crop",
    "p5_rope_freqs.safetensors": "P5: complex RoPE tables per case (view_as_real)",
    "p6_inputs.safetensors": "P6: transformer inputs (prompt embeds, img mask, cond latents, noise)",
    "p8_noise.safetensors": "P8: injected initial latents (packed)",
    "pillow_reference_resize.safetensors": "Full-size resized references (RGBA + white composite) as the pipeline sees them",
}


def describe(name: str) -> str:
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    for pre, d in [
        ("p3_", "P3: trimmed pre-norm hidden states (+ LM internals when *_lm_internals_*)"),
        ("p6_outputs_", "P6: full/extract/cached transformer outputs"),
        ("p6_internals_", "P6 debug: block 0/31 activations, modulation, temb, K/V layers 0 and 31"),
        ("p7_", "P7: full forward with Viggle r128 via PEFT (unmerged) and without"),
        ("p8_", "P8: end-to-end per-step latents, final latents, decoded RGBA (and PNG)"),
        ("p4_", "P4: VAE decode roundtrip PNG"),
        ("alpha_t2i_", "t2i alpha-histogram render"),
        ("ref_", "Resized reference image as fed to VL/VAE"),
        ("encode_record_", "capture record (stats, reference sizes)"),
    ]:
        if name.startswith(pre):
            return d
    return ""


def environment(args) -> dict:
    import diffusers
    import peft
    import safetensors
    import transformers

    try:
        drv = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).split()[0]
    except Exception:
        drv = None
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": STATE.get("cudnn"),
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "peft": peft.__version__,
        "safetensors": safetensors.__version__,
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "nvidia_driver": drv,
        "gpus": gpus,
        "te_device": args.te_device,
        "dit_device": args.dit_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "allow_tf32": {"matmul": False, "cudnn": False},
        "sdpa": "torch native (diffusers QwenImage21AttnProcessor per-segment; transformers Qwen3-VL attn 'sdpa')",
    }


def stage_manifest(args):
    entries = []
    skip = {"README.md", "capture.py", "manifest.json"}
    for p in sorted(args.committed.iterdir()):
        if p.is_file() and p.name not in skip:
            entries.append({"file": p.name, "location": "committed", "sha256": sha256_file(p), "bytes": p.stat().st_size, "description": describe(p.name)})
    for p in sorted(args.large.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            entries.append({"file": p.name, "location": "large", "sha256": sha256_file(p), "bytes": p.stat().st_size, "description": describe(p.name)})
    old = {}
    mf = args.committed / "manifest.json"
    if mf.exists():
        old = json.loads(mf.read_text())
    manifest = {
        "schema": "mold.qwen_image21.fixtures.v1",
        "captured": _dt.date.today().isoformat(),
        "large_dir": str(args.large),
        "upstream": {
            "hf_repo": "Qwen/Qwen-Image-2.1",
            "hf_revision": HF_REVISION,
            "diffusers": "https://github.com/huggingface/diffusers",
            "diffusers_commit": DIFFUSERS_COMMIT,
            "viggle_repo": "Viggle/Qwen-Image-2.1-viggle-turbo",
            "viggle_revision": VIGGLE_REVISION,
        },
        "environment": environment(args),
        "inputs": {
            "p6_prompt": P6_PROMPT, "negative_prompt": NEG_PROMPT, "p8_prompt": P8_PROMPT,
            "p6_target_wh": list(P6_TARGET), "p8_target_wh": list(P8_TARGET), "p6_timesteps": list(P6_SIGMA_T),
            "noise_seeds": NOISE_SEEDS, "turbo_sigmas": TURBO_SIGMAS,
            "rgba_recipe": {"prefix": RGBA_PREFIX, "suffix": RGBA_SUFFIX, "example": rgba_recipe("A cute cartoon dragon sticker")},
        },
        "results": STATE.get("results", {}),
        "files": entries,
    }
    save_json(mf, manifest)


# --------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-root", type=Path, default=Path("/storage/mold/fixtures/qwen_image21/model-b3179ad"))
    ap.add_argument("--viggle-dir", type=Path, default=Path("/storage/mold/fixtures/qwen_image21/viggle"))
    ap.add_argument("--committed", type=Path, default=HERE)
    ap.add_argument("--large", type=Path, default=Path("/storage/mold/fixtures/qwen_image21/captures"))
    ap.add_argument("--te-device", default="cuda:0")
    ap.add_argument("--dit-device", default="cuda:0")
    ap.add_argument("--stages", default="images,viggle,pillow,cpu,encode,transformer,lora,e2e,bf16,alpha,manifest")
    args = ap.parse_args()
    args.large.mkdir(parents=True, exist_ok=True)
    STATE["dit_device"] = args.dit_device
    # Inference only. encode_prompt / vae.encode / the transformer are not
    # decorated with no_grad upstream (only __call__ is), and autograd's saved
    # activations alone overflow the card at fp32.
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        assert_bundled_cudnn()
    stages = args.stages.split(",")
    STATE["results"] = {}
    R = STATE["results"]
    if "images" in stages:
        stage_images(args)
    if "viggle" in stages:
        stage_viggle(args)
    if "pillow" in stages:
        stage_pillow(args)
    if "cpu" in stages:
        stage_cpu(args)
    if "encode" in stages:
        stage_encode(args, torch.float32, "fp32")
        R["encode_fp32"] = {k: v for k, v in STATE["encode_record_fp32"].items() if k.startswith(("p3_", "p4_o", "p4_r", "reference", "drop", "img_tok"))}
    if "transformer" in stages:
        # the VAE and text encoder are not needed here; keep the DiT GPU for the fp32 prefix cache
        if "vae" in STATE:
            STATE["vae"].to("cpu")
        R["p6_fp32"] = stage_transformer(args, torch.float32, "fp32")
    if "lora" in stages:
        R["p7_fp32"] = stage_lora(args, torch.float32, "fp32")
    if "e2e" in stages:
        R["p8_fp32"] = stage_e2e(args, torch.float32, "fp32")
    if "bf16" in stages:
        stage_encode(args, torch.bfloat16, "bf16")
        STATE["te"].to("cpu")
        torch.cuda.empty_cache()
        R["p6_bf16"] = stage_transformer(args, torch.bfloat16, "bf16")
        R["p7_bf16"] = stage_lora(args, torch.bfloat16, "bf16")
        R["p8_bf16"] = stage_e2e(args, torch.bfloat16, "bf16")
    if "alpha" in stages:
        R["alpha"] = [{k: r[k] for k in ("name", "transparency_recipe", "alpha_u8_min", "alpha_u8_ne_255", "alpha_u8_eq_0", "alpha_float_min")} for r in stage_alpha(args)]
    res_path = args.large / "results.json"
    merged = json.loads(res_path.read_text()) if res_path.exists() else {}
    merged.update(R)
    res_path.write_text(json.dumps(merged, indent=1) + "\n")
    STATE["results"] = merged
    if "manifest" in stages:
        stage_manifest(args)


if __name__ == "__main__":
    main()
