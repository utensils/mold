#!/usr/bin/env python3
"""Run Tencent's actual multiview paint pipeline and retain every boundary."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--appearance", type=Path, required=True)
    parser.add_argument("--appearance-mode", choices=["preserve", "rgb", "rgba"], default="preserve")
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=[64, 128, 256, 512], default=512)
    parser.add_argument("--views", type=int, choices=[1, 2, 6], default=6)
    args = parser.parse_args()
    root = args.reference.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert revision == "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code"], check=True)
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(root), str(root / "hy3dpaint")]
    import torch
    import diffusers
    import huggingface_hub
    from PIL import Image
    from safetensors.torch import save_file
    from diffusers.models.autoencoders import vae as vae_module
    assert diffusers.__version__ == "0.30.0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def installed_snapshot(repo_id, **kwargs):
        if repo_id != "tencent/Hunyuan3D-2.1":
            raise ValueError(f"unexpected download {repo_id}")
        return str(args.cache.resolve() / "paint-layout")

    huggingface_hub.snapshot_download = installed_snapshot
    # Resolve the checkpoint's legacy flat module name to unchanged Tencent
    # code, bypassing the package's unrelated training-only initializer.
    package = types.ModuleType("hunyuanpaintpbr")
    package.__path__ = [str(root / "hy3dpaint/hunyuanpaintpbr")]
    sys.modules["hunyuanpaintpbr"] = package
    sys.modules["modules"] = importlib.import_module("hunyuanpaintpbr.unet.modules")
    from diffusers.pipelines import pipeline_utils
    original_load = pipeline_utils.load_sub_model

    def load_component(*values, **kwargs):
        if kwargs.get("class_name") == "UNet2p5DConditionModel":
            return sys.modules["modules"].UNet2p5DConditionModel.from_pretrained(
                str(args.cache.resolve() / "paint-layout/hunyuan3d-paintpbr-v2-1/unet"),
                torch_dtype=kwargs["torch_dtype"],
            )
        return original_load(*values, **kwargs)

    pipeline_utils.load_sub_model = load_component
    from utils.multiview_utils import multiviewDiffusionNet
    net = multiviewDiffusionNet(types.SimpleNamespace(
        device="cuda", multiview_cfg_path=str(root / "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"),
        multiview_pretrained_path="tencent/Hunyuan3D-2.1",
        dino_ckpt_path=str(args.cache.resolve() / "dinov2-giant"),
    ))
    tensors = {}

    def retain(name, value):
        if name in tensors:
            raise ValueError(f"duplicate boundary {name}")
        value = value.detach().cpu().contiguous().clone()
        assert torch.isfinite(value).all(), name
        tensors[name] = value
        save_file({name: value}, str(args.output / f"{name}.safetensors"))
        print(name, tuple(value.shape), value.dtype, flush=True)

    net.dino_v2.dino_v2.register_forward_pre_hook(lambda model, values: retain("input.appearance", values[0]))
    net.dino_v2.register_forward_hook(lambda model, values, result: retain("expected.appearance", result))
    encode = net.pipeline.encode_images
    roles = iter(["reference", "normal", "position"])
    current = None

    def encode_images(images):
        nonlocal current
        current = next(roles)
        retain(f"input.{current}", images)
        latents = encode(images)
        retain(f"expected.{current}", latents)
        current = None
        return latents

    net.pipeline.encode_images = encode_images
    randn = vae_module.randn_tensor

    def posterior_noise(*values, **kwargs):
        noise = randn(*values, **kwargs)
        assert current is not None, "unidentified posterior sample"
        retain(f"input.{current}_noise", noise.unsqueeze(0))
        return noise

    vae_module.randn_tensor = posterior_noise
    prepare = net.pipeline.prepare_latents

    def prepare_latents(*values, **kwargs):
        latents = prepare(*values, **kwargs)
        retain("input.initial_noise", latents)
        return latents

    net.pipeline.prepare_latents = prepare_latents
    step = net.pipeline.scheduler.step
    step_index = 0

    def scheduler_step(*values, **kwargs):
        nonlocal step_index
        result = step(*values, **kwargs)
        step_index += 1
        retain(f"expected.denoise.{step_index:02}", result[0])
        return result

    net.pipeline.scheduler.step = scheduler_step
    decode = net.pipeline.vae.decode

    def decode_latents(*values, **kwargs):
        result = decode(*values, **kwargs)
        retain("expected.decode", result[0])
        return result

    net.pipeline.vae.decode = decode_latents
    # Preserve textureGenPipeline.py:136-145 ordering for transparent sources.
    appearance = Image.open(args.appearance)
    if args.appearance_mode != "preserve":
        appearance = appearance.convert(args.appearance_mode.upper())
    appearance = appearance.resize((512, 512))
    if appearance.mode == "RGBA":
        background = Image.new("RGB", appearance.size, (255, 255, 255))
        background.paste(appearance, mask=appearance.getchannel("A"))
        appearance = background
    appearance = appearance.convert("RGB")
    conditions = [Image.open(args.conditions / f"condition-{index:02}.png").convert("RGB")
                  for index in list(range(args.views)) + list(range(6, 6 + args.views))]
    metadata = dict(revision=revision, torch=torch.__version__, diffusers=diffusers.__version__,
                    dtype="f16", size=args.size, views=args.views, appearance_mode=args.appearance_mode,
                    source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in [root / "hy3dpaint/utils/multiview_utils.py", root / "hy3dpaint/hunyuanpaintpbr/pipeline.py"]})
    (args.output / "invocation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with torch.inference_mode():
        materials = net(appearance, conditions, custom_view_size=args.size, resize_input=True)
    torch.cuda.synchronize()
    assert step_index == 15
    for role, images in materials.items():
        assert len(images) == args.views
        for index, image in enumerate(images):
            image.save(args.output / f"{role}-{index:02}.png")
    save_file(tensors, str(args.output / "pipeline.safetensors"))
    metadata.update(seconds=time.monotonic() - start,
                    peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
    (args.output / "completed.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
