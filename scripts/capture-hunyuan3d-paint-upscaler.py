#!/usr/bin/env python3
"""Capture unchanged Tencent/RealESRGAN inference and FP16 scalar boundaries.

Reference-only: run in the retained oracle venv inside the Nix CUDA shell.
Output directories must be new; weights and source images are never modified.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--pth", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int)
    args = parser.parse_args()
    root = args.upstream.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != "82920d643c0dc2f7bfd7255f45f62d386edfe60c":
        raise ValueError(f"unexpected Tencent revision: {revision}")
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code"], check=True)
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(root), str(root / "hy3dpaint")]
    from torchvision_fix import fix_torchvision_functional_tensor
    fix_torchvision_functional_tensor()
    import numpy as np
    from PIL import Image
    import torch
    from safetensors.torch import load_file, save_file
    from utils.image_super_utils import imageSuperNet
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    installed = load_file(str(args.weights))
    oracle = torch.load(args.pth, map_location="cpu", weights_only=True)["params_ema"]
    if installed.keys() != oracle.keys() or any(
        installed[key].dtype != oracle[key].dtype
        or not torch.equal(installed[key], oracle[key]) for key in installed
    ):
        raise ValueError("oracle params_ema differs from installed safetensors")
    metadata = {
        "argv": sys.argv, "revision": revision, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "tensor_count": len(installed),
        "weights_sha256": sha256(args.weights), "pth_sha256": sha256(args.pth),
        "image_sha256": sha256(args.image),
        "sources": {inspect.getfile(cls): sha256(inspect.getfile(cls))
                    for cls in (imageSuperNet, RRDBNet, RealESRGANer)},
    }
    del installed, oracle
    (args.output / "invocation.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # Exhaust all finite half values, including subnormals and signed zero.
    half_values = np.arange(65536, dtype=np.uint16).view(np.float16)
    values = torch.from_numpy(half_values[np.isfinite(half_values)].copy()).cuda()
    save_file({
        "input": values.cpu(),
        "scaled": (values * 0.2).cpu(),
        "leaky_relu": torch.nn.functional.leaky_relu(values, 0.2).cpu(),
    }, str(args.output / "scalars.safetensors"))
    pixels = values.float().clamp(0, 1).cpu().numpy() * np.float32(255)
    metadata["half_output_rounding_disagreements"] = int(
        np.count_nonzero(np.round(pixels) != np.floor(pixels + np.float32(0.5)))
    )
    image = Image.open(args.image).convert("RGB")
    if args.crop_size is not None:
        if not 1 <= args.crop_size <= min(image.size):
            raise ValueError("crop must fit the input image")
        image = image.crop((0, 0, args.crop_size, args.crop_size))
    image.save(args.output / "input.png")
    network = imageSuperNet(SimpleNamespace(realesrgan_ckpt_path=str(args.pth)))
    captured = {}

    def hook(name):
        def capture(module, inputs, output):
            captured[name] = output.detach().cpu().contiguous().clone()
            if name == "conv_first":
                captured["input"] = inputs[0].detach().cpu().contiguous().clone()
        return capture

    handles = []
    for name, module in network.upsampler.model.named_modules():
        if name in {"conv_first", "body.0", "body.11", "body.22", "conv_body",
                    "conv_up1", "conv_up2", "conv_hr", "conv_last"}:
            handles.append(module.register_forward_hook(hook(name)))
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with torch.inference_mode():
        result = network(image)
    torch.cuda.synchronize()
    metadata["seconds"] = time.monotonic() - start
    metadata["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    metadata["input_size"] = image.size
    result.save(args.output / "expected.png")
    save_file(captured, str(args.output / "stages.safetensors"))
    for handle in handles:
        handle.remove()
    metadata["artifacts"] = {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / "completed.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
