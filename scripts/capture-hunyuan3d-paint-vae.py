#!/usr/bin/env python3
"""Capture Diffusers paint VAE posterior, explicit-noise sampling and decode."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--tiny-only", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    parser.add_argument("--encoder-trace", action="store_true")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--size", type=int, choices=[64,128,256,512], default=64)
    parser.add_argument("--attention-backend", choices=["default","math"], default="default")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import diffusers
    from diffusers import AutoencoderKL
    from safetensors.torch import save_file
    from contextlib import nullcontext
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    if len(args.image) > 6:
        parser.error("at most six conditioning images per capture")
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    dtype = torch.float32 if args.dtype == "f32" else torch.float16
    # Isolate CUDA GroupNorm saved-statistics rounding and SiLU opmath from
    # convolution/attention backend differences in the complete VAE.
    if args.device == "cuda":
        x = (torch.arange(1024, device="cuda").float() * .137).sin().reshape(2,8,8,8).half()
        weight = torch.linspace(.73, 1.43, 8, device="cuda").half()
        bias = torch.linspace(-.27, .33, 8, device="cuda").half()
        with torch.inference_mode():
            normalized = torch.nn.functional.group_norm(x, 4, weight, bias, 1e-6)
            activated = torch.nn.functional.silu(x)
        opmath = dict(input=x,weight=weight,bias=bias,normalized=normalized,activated=activated)
        for suffix, shape in [("large",(1,8,32,32)),("spatial1",(2,8,1,1))]:
            count = 1
            for dimension in shape: count *= dimension
            value = (torch.arange(count,device="cuda").float()*.137).sin().reshape(shape).half()
            opmath[f"input_{suffix}"] = value
            opmath[f"normalized_{suffix}"] = torch.nn.functional.group_norm(value,4,weight,bias,1e-6)
        linear_input = torch.linspace(-1,1,64,device="cuda").reshape(2,4,8).half()
        linear_weight = (torch.arange(64,device="cuda").float()*.17).sin().reshape(8,8).half()
        opmath.update(linear_input=linear_input,linear_weight=linear_weight,
            linear_output=torch.nn.functional.linear(linear_input,linear_weight,bias))
        save_file({k:v.cpu().contiguous() for k,v in opmath.items()},str(args.output/"paint-vae-opmath.safetensors"))
    torch.manual_seed(25026)
    tiny = AutoencoderKL(block_out_channels=(8,16), down_block_types=("DownEncoderBlock2D",)*2,
                         up_block_types=("UpDecoderBlock2D",)*2, layers_per_block=1,
                         norm_num_groups=4, latent_channels=4, sample_size=16).eval()
    save_file(tiny.state_dict(),str(args.output/"paint-vae-tiny-weights.safetensors"))
    torch.save(tiny.state_dict(),args.output/"paint-vae-tiny.bin")
    models = [("tiny",tiny,16)]
    if not args.tiny_only:
        models.append(("pretrained",AutoencoderKL.from_pretrained(str(args.checkpoint.resolve())),args.size))
    measurements = {}
    for name, model, size in models:
        model = model.eval().to(args.device,dtype=dtype)
        trace = {}
        hooks = []
        if args.encoder_trace:
            def observe(key):
                def hook(module, inputs, output):
                    trace[key] = output.detach().cpu().contiguous()
                return hook
            for key, module in model.named_modules():
                if key in {"encoder.conv_in", "encoder.mid_block", "encoder.conv_norm_out", "encoder.conv_out", "quant_conv"} or (key.startswith("encoder.down_blocks.") and key.count(".") == 2):
                    hooks.append(module.register_forward_hook(observe(key)))
        pixels = (torch.arange(3*size*size,device=args.device).float()*0.013).sin().reshape(1,3,size,size).to(dtype)
        if name == "pretrained" and args.image:
            images = []
            for path in args.image:
                with Image.open(path) as image:
                    images.append(to_tensor(image.convert("RGB").resize((size,size))))
            pixels = ((torch.stack(images)-.5)*2).to(args.device,dtype=dtype)
        attention_context = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if args.attention_backend == "math" else nullcontext()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), attention_context:
            posterior = model.encode(pixels).latent_dist
            noise = (torch.arange(posterior.mean.numel(),device=args.device).float()*0.019).cos().reshape(posterior.mean.shape).to(dtype)
            sampled = (posterior.mean+posterior.std*noise)*model.config.scaling_factor
            decoded = model.decode(sampled/model.config.scaling_factor).sample
        if args.device == "cuda":
            measurements[name] = {"peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved()}
        for hook in hooks:
            hook.remove()
        if trace:
            save_file(trace,str(args.output/f"paint-vae-{name}-encoder.safetensors"))
        tensors = dict(pixels=pixels,noise=noise,mean=posterior.mean,std=posterior.std,sampled=sampled,decoded=decoded)
        save_file({key:value.cpu().contiguous() for key,value in tensors.items()},str(args.output/f"paint-vae-{name}.safetensors"))
        model.cpu()
    checkpoint_files = {
        str(path.resolve()): {"size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.checkpoint.glob("*"))
        if path.is_file() and (path.suffix in {".json", ".bin", ".safetensors"})
    }
    (args.output/"paint-vae.json").write_text(json.dumps(dict(torch=torch.__version__,diffusers=diffusers.__version__,
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,device=args.device,
        allow_tf32=args.allow_tf32,dtype=args.dtype,
        checkpoint=str(args.checkpoint.resolve()),checkpoint_files=checkpoint_files,
        encoder_trace=args.encoder_trace,seed=25026,scaling_factor=.18215,
        images=[{"path":str(p.resolve()),"sha256":sha256(p)} for p in args.image],
        size=args.size,attention_backend=args.attention_backend,measurements=measurements),indent=2)+"\n")


if __name__ == "__main__":
    main()
