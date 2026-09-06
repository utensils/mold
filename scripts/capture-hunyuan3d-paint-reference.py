#!/usr/bin/env python3
"""Run pinned Tencent paint using retained mold weights; preserve intermediates.

Run inside the retained oracle venv and Nix CUDA shell, through the generic
capture runner. Python and native Tencent code are reference-only dependencies.
"""

import argparse
import json
import importlib
from pathlib import Path
import subprocess
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--texture-size", type=int, choices=[1024, 2048, 4096], default=1024)
    args = parser.parse_args()
    pin = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
    actual = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != pin:
        raise ValueError(f"expected Tencent revision {pin}, found {actual}")
    subprocess.run(["git", "-C", str(args.upstream), "diff", "--exit-code"], check=True)
    root = args.upstream.resolve()
    cache = args.cache.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    import faulthandler
    faulthandler.enable()
    sys.path[:0] = [str(root), str(root / "hy3dpaint")]
    from torchvision_fix import fix_torchvision_functional_tensor
    fix_torchvision_functional_tensor()
    # mesh_utils imports bpy eagerly, but only its export helpers use it.
    # Those helpers execute unmodified in the native Blender subprocess below.
    sys.modules["bpy"] = types.ModuleType("bpy")
    import DifferentiableRenderer
    DifferentiableRenderer.__path__.append(str(cache / "native"))
    import huggingface_hub
    import numpy as np
    from PIL import Image
    import torch
    from safetensors.torch import load_file

    # Assemble a diffusers-layout view of mold's split storage without copying
    # or replacing its downloaded weights. snapshot_download only resolves the
    # already installed Tencent paint repository; unexpected repos are errors.
    view = cache / "paint-layout"
    for source in [args.models / "hunyuan3d-paint", args.models / "shared/hunyuan3d-paint"]:
        for file in source.rglob("*"):
            if file.is_file() and not file.name.endswith(".sha256-verified"):
                target = view / file.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink():
                    if target.resolve() != file.resolve():
                        raise ValueError(f"conflicting paint view: {target}")
                elif target.exists():
                    raise ValueError(f"paint view must not replace existing file: {target}")
                else:
                    target.symlink_to(file.resolve())

    def installed_snapshot(repo_id, **kwargs):
        if repo_id != "tencent/Hunyuan3D-2.1":
            raise ValueError(f"unexpected oracle download: {repo_id}")
        return str(view)
    huggingface_hub.snapshot_download = installed_snapshot

    upscaler = cache / "realesrgan-x4plus.pth"
    if not upscaler.exists():
        original = args.models / "real-esrgan-x4plus-fp16/diffusion_pytorch_model.fp16.safetensors"
        torch.save({"params_ema": load_file(str(original))}, upscaler)

    # Published model_index uses the original flat module name. Resolve it to
    # the unchanged upstream package, preserving relative attention imports.
    # The package initializer imports the training-only Lightning wrapper.
    # Establish its namespace without executing that unrelated initializer.
    package = types.ModuleType("hunyuanpaintpbr")
    package.__path__ = [str(root / "hy3dpaint/hunyuanpaintpbr")]
    sys.modules["hunyuanpaintpbr"] = package
    sys.modules["modules"] = importlib.import_module("hunyuanpaintpbr.unet.modules")
    # Tencent exposes from_pretrained on nn.Module rather than ModelMixin.
    # Dispatch that published loader explicitly; all other components retain
    # Diffusers loading. No network forward or tensor operation is replaced.
    from diffusers.pipelines import pipeline_utils
    original_load = pipeline_utils.load_sub_model
    def load_component(*positional, **kwargs):
        if kwargs.get("class_name") == "UNet2p5DConditionModel":
            return sys.modules["modules"].UNet2p5DConditionModel.from_pretrained(
                str(view / "hunyuan3d-paintpbr-v2-1/unet"),
                torch_dtype=kwargs["torch_dtype"],
            )
        return original_load(*positional, **kwargs)
    pipeline_utils.load_sub_model = load_component
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
    config = Hunyuan3DPaintConfig(6, 512)
    config.multiview_cfg_path = str(root / "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
    config.dino_ckpt_path = str(cache / "dinov2-giant")
    config.realesrgan_ckpt_path = str(upscaler)
    config.texture_size = args.texture_size
    torch.cuda.reset_peak_memory_stats()
    print("Loading upstream paint components", flush=True)
    pipeline = Hunyuan3DPaintPipeline(config)
    print("Upstream paint components loaded", flush=True)
    import textureGenPipeline
    def export_glb(obj_path, glb_path):
        subprocess.run([str(cache / "native/blender/bin/blender"), "--background",
                        "--factory-startup", "--python-exit-code", "1", "--python",
                        str(Path(__file__).with_name("capture-hunyuan3d-blender-export.py").resolve()),
                        "--", "--upstream", str(root), "--input", obj_path,
                        "--output", glb_path], check=True)
        return True
    textureGenPipeline.convert_obj_to_glb = export_glb
    original_uv = textureGenPipeline.mesh_uv_wrap
    def capture_uv(mesh):
        print("Unwrapping source mesh", flush=True)
        mesh = original_uv(mesh)
        np.savez(args.output / "uv-mesh.npz", vertices=mesh.vertices,
                 faces=mesh.faces, uv=mesh.visual.uv)
        print("UV mesh retained", flush=True)
        return mesh
    textureGenPipeline.mesh_uv_wrap = capture_uv
    original_multiview = pipeline.models["multiview_model"]

    def capture_multiview(images, conditions, **kwargs):
        for index, image in enumerate(conditions):
            image.save(args.output / f"condition-{index:02d}.png")
        print("Rendering upstream material views", flush=True)
        result = original_multiview(images, conditions, **kwargs)
        for role, views in result.items():
            for index, image in enumerate(views):
                image.save(args.output / f"view-{role}-{index:02d}.png")
        return result
    pipeline.models["multiview_model"] = capture_multiview
    original_inpaint = pipeline.view_processor.texture_inpaint
    inpaint_index = 0

    def capture_inpaint(texture, mask, *positional, **kwargs):
        nonlocal inpaint_index
        index = inpaint_index
        inpaint_index += 1
        Image.fromarray(mask).save(args.output / f"trust-{index}.png")
        Image.fromarray(np.clip(texture.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)).save(args.output / f"baked-{index}.png")
        result = original_inpaint(texture, mask, *positional, **kwargs)
        Image.fromarray(np.clip(result.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)).save(args.output / f"inpainted-{index}.png")
        return result
    pipeline.view_processor.texture_inpaint = capture_inpaint
    pipeline(mesh_path=str(args.mesh.resolve()), image_path=str(args.image.resolve()),
             output_mesh_path=str(args.output.resolve() / "textured.obj"),
             use_remesh=False, save_glb=True)
    (args.output / "paint-reference.json").write_text(json.dumps({
        "upstream": pin, "texture_size": args.texture_size, "views": 6,
        "view_size": 512, "render_size": config.render_size, "remesh": False,
        "torch": torch.__version__, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "blender": subprocess.check_output([str(cache / "native/blender/bin/blender"), "--version"], text=True).splitlines()[0],
        "export_note": "Unmodified Tencent exporter in native Blender; embedded bpy teardown crashes on Nix.",
    }, indent=2) + "\n")
    print("Paint artifacts and measurements retained", flush=True)


if __name__ == "__main__":
    main()
