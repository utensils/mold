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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import diffusers
    from diffusers import AutoencoderKL
    from safetensors.torch import save_file
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    dtype = torch.float32 if args.dtype == "f32" else torch.float16
    torch.manual_seed(25026)
    tiny = AutoencoderKL(block_out_channels=(8,16), down_block_types=("DownEncoderBlock2D",)*2,
                         up_block_types=("UpDecoderBlock2D",)*2, layers_per_block=1,
                         norm_num_groups=4, latent_channels=4, sample_size=16).eval()
    save_file(tiny.state_dict(),str(args.output/"paint-vae-tiny-weights.safetensors"))
    torch.save(tiny.state_dict(),args.output/"paint-vae-tiny.bin")
    models = [("tiny",tiny,16)]
    if not args.tiny_only:
        models.append(("pretrained",AutoencoderKL.from_pretrained(str(args.checkpoint.resolve())),64))
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
        with torch.inference_mode():
            posterior = model.encode(pixels).latent_dist
            noise = (torch.arange(posterior.mean.numel(),device=args.device).float()*0.019).cos().reshape(posterior.mean.shape).to(dtype)
            sampled = (posterior.mean+posterior.std*noise)*model.config.scaling_factor
            decoded = model.decode(sampled/model.config.scaling_factor).sample
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
        encoder_trace=args.encoder_trace,seed=25026,scaling_factor=.18215),indent=2)+"\n")


if __name__ == "__main__":
    main()
