#!/usr/bin/env python3
"""Capture Diffusers 0.30 paint UNet residual and sampling components."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import diffusers
    from diffusers.models.resnet import ResnetBlock2D
    from diffusers.models.downsampling import Downsample2D
    from diffusers.models.upsampling import Upsample2D
    from safetensors.torch import save_file
    if diffusers.__version__ != "0.30.0":
        parser.error("fixture requires pinned Diffusers 0.30.0")
    torch.manual_seed(25026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu", mmap=True) if args.checkpoint else None
    channels, temb, groups = (320, 1280, 32) if state is not None else (32, 128, 4)
    models = {
        "resnet": (ResnetBlock2D(in_channels=channels, out_channels=channels, temb_channels=temb, groups=groups, eps=1e-5),
                   "unet.down_blocks.0.resnets.0."),
        "down": (Downsample2D(channels, use_conv=True, padding=1, name="op"), "unet.down_blocks.0.downsamplers.0."),
        "up": (Upsample2D(channels, use_conv=True), "unet.up_blocks.3.upsamplers.0."),
    }
    # Last up block has no upsampler: select width-640 stage for pretrained sampling.
    if state is not None:
        models["up"] = (Upsample2D(640, use_conv=True), "unet.up_blocks.2.upsamplers.0.")
    else:
        models["shortcut"] = (ResnetBlock2D(in_channels=32, out_channels=64, temb_channels=temb, groups=groups, eps=1e-5), "")
    tensors = {}
    with torch.inference_mode():
        for label, (model, prefix) in models.items():
            if state is not None:
                model.load_state_dict({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}, strict=True)
            for key, value in model.state_dict().items():
                tensors[f"{label}.weights.{key}"] = value.cpu().clone()
            width = model.channels if label in ("up", "down") else model.in_channels
            # Odd sizes expose downsample padding and explicit upsample targets.
            pixels = (torch.arange(2*width*9*7).float()*.013).sin().reshape(2,width,9,7)
            time = (torch.arange(2*temb).float()*.017).cos().reshape(2,temb)
            for name, dtype in (("f32", torch.float32), ("f16", torch.float16)):
                model.to("cuda", dtype=dtype)
                x, t = pixels.to("cuda",dtype=dtype), time.to("cuda",dtype=dtype)
                result = model(x, t) if label in ("resnet", "shortcut") else model(x)
                for key, value in dict(input=x, time=t, expected=result).items():
                    tensors[f"{label}.{name}.{key}"] = value.cpu()
                if label == "up":
                    tensors[f"{label}.{name}.explicit_expected"] = model(x, output_size=(17,13)).cpu()
        for epsilon_name, epsilon in (("vae", 1e-6), ("unet", 1e-5)):
            x = (torch.arange(2*32*8*8, device="cuda").float()*.137).sin().reshape(2,32,8,8).half() * .003
            weight = torch.linspace(.73,1.43,32,device="cuda").half()
            bias = torch.linspace(-.27,.33,32,device="cuda").half()
            result = torch.nn.functional.group_norm(x,4,weight,bias,epsilon)
            for key, value in dict(input=x,weight=weight,bias=bias,expected=result).items():
                tensors[f"norm.{epsilon_name}.{key}"] = value.cpu()
    save_file({key:value.contiguous() for key,value in tensors.items()},str(args.output/"paint-conv.safetensors"))
    import inspect
    sources = {str(Path(inspect.getfile(cls)).resolve()):hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
               for cls in (ResnetBlock2D, Downsample2D, Upsample2D)}
    metadata = dict(torch=torch.__version__,diffusers=diffusers.__version__,gpu=torch.cuda.get_device_name(),
                    sources=sources,seed=25026,groups=groups,channels=channels,time_width=temb,epsilon=1e-5)
    if args.checkpoint:
        digest = hashlib.sha256()
        with args.checkpoint.open("rb") as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b""): digest.update(chunk)
        metadata.update(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=digest.hexdigest())
    (args.output/"paint-conv.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__ == "__main__":
    main()
