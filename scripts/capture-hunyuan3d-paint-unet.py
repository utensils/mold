#!/usr/bin/env python3
"""Capture the complete unchanged Tencent paint UNet and its reusable conditions."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

REVISION = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="UNet directory containing config.json and .bin")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    parser.add_argument("--latent-size", type=int, choices=[8,16,32,64], default=8)
    parser.add_argument("--views", type=int, choices=[1,2,6], default=2)
    parser.add_argument("--references", type=int, choices=[1,2], default=1)
    args = parser.parse_args()
    if args.tiny == bool(args.checkpoint):
        parser.error("select either --tiny or --checkpoint")
    if args.checkpoint:
        for name in ("config.json", "diffusion_pytorch_model.bin"):
            if not (args.checkpoint/name).is_file():
                parser.error(f"checkpoint directory is missing {name}")
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        parser.error("reference differs from the qualified Tencent revision")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import diffusers
    from safetensors.torch import save_file
    if diffusers.__version__ != "0.30.0":
        parser.error("requires pinned Diffusers 0.30.0")
    folder = args.reference.resolve() / "hy3dpaint/hunyuanpaintpbr/unet"
    package = types.ModuleType("tencent_paint_oracle")
    package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__ + ".modules", folder / "modules.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    torch.manual_seed(25026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.float32 if args.dtype == "f32" else torch.float16
    if args.tiny:
        config = dict(in_channels=4, out_channels=4, sample_size=args.latent_size,
                      down_block_types=("CrossAttnDownBlock2D",)*3+("DownBlock2D",),
                      up_block_types=("UpBlock2D",)+("CrossAttnUpBlock2D",)*3,
                      block_out_channels=(32,64,128,128), layers_per_block=2,
                      cross_attention_dim=1024, attention_head_dim=(2,4,8,8),
                      use_linear_projection=True, norm_num_groups=32, norm_eps=1e-5,
                      flip_sin_to_cos=True, freq_shift=0, downsample_padding=1)
        model = upstream.UNet2p5DConditionModel(diffusers.UNet2DConditionModel(**config))
        model.unet.conv_in = torch.nn.Conv2d(12,32,3,padding=1)
        with torch.inference_mode():
            for key, value in model.named_parameters():
                value.copy_(torch.randn_like(value)*.02 + (1 if "norm" in key and key.endswith("weight") else 0))
        save_file({key:value.detach().contiguous() for key,value in model.state_dict().items()},
                  str(args.output/"paint-unet-tiny-weights.safetensors"))
    else:
        config = json.loads((args.checkpoint/"config.json").read_text())
        model = upstream.UNet2p5DConditionModel.from_pretrained(str(args.checkpoint), torch_dtype=dtype)
    model = model.eval().to("cuda",dtype=dtype)
    invocation_counts = dict(reference=0, dino_projector=0, main=0)
    def counter(name):
        def observe(module, inputs, output):
            invocation_counts[name] += 1
        return observe
    hooks = [model.unet_dual.register_forward_hook(counter("reference")),
             model.unet.image_proj_model_dino.register_forward_hook(counter("dino_projector")),
             model.unet.register_forward_hook(counter("main"))]
    batch, materials, views, size = 3, 2, args.views, args.latent_size

    def signal(shape, offset=0):
        count = 1
        for dimension in shape:
            count *= dimension
        return (torch.arange(count,device="cuda").float()*.013+offset).sin().reshape(shape).to(dtype)

    sample = signal((batch,materials,views,4,size,size))
    normal = signal((batch,views,4,size,size), .3)
    position = signal((batch,views,4,size,size), .6)
    reference = signal((batch,args.references,4,size,size), .9)
    dino = signal((batch,257*args.references,1536), 1.2)
    positions = ((signal((batch,views,3,64,64),1.5).float()+1)*.5).to(dtype)
    positions[:,:,:,:4,:] = 1
    text = torch.stack([model.unet.learned_text_clip_albedo,model.unet.learned_text_clip_mr]).unsqueeze(0).repeat(batch,1,1,1)
    scale = torch.tensor([0.,1.,1.],device="cuda",dtype=dtype)
    # Position quantization mutates its input in Tencent. Retain pre-call bytes.
    tensors = {"input."+key:value.detach().cpu().contiguous() for key,value in
               dict(sample=sample,normal=normal,position=position,reference=reference,dino=dino,
                    position_maps=positions,text=text,reference_scale=scale).items()}
    cache = {}
    conditions = dict(embeds_normal=normal, embeds_position=position, ref_latents=reference,
                      dino_hidden_states=dino, position_maps=positions, ref_scale=scale, mva_scale=1., cache=cache)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for timestep in (500,400):
            result = model(sample,timestep,text,**conditions)[0]
            tensors[f"expected.{timestep}"] = result.cpu().contiguous()
        tensors["cache.dino"] = cache["dino_hidden_states_proj"].cpu().contiguous()
        for name, value in cache["condition_embed_dict"].items():
            tensors["cache.reference."+name] = value.cpu().contiguous()
        for length, values in cache["position_voxel_indices"].items():
            tensors[f"cache.positions.{length}"] = values["voxel_indices"].cpu().contiguous()
    save_file(tensors,str(args.output/"paint-unet.safetensors"))
    for hook in hooks:
        hook.remove()
    if invocation_counts != dict(reference=1, dino_projector=1, main=2):
        raise RuntimeError(f"unexpected upstream cache invocation counts: {invocation_counts}")
    metadata = dict(revision=revision,sources={name:sha256(folder/name) for name in ("modules.py","attn_processor.py")},
                    torch=torch.__version__,diffusers=diffusers.__version__,gpu=torch.cuda.get_device_name(),
                    seed=25026,dtype=args.dtype,tiny=args.tiny,config=config,batch=batch,materials=materials,
                    views=views,references=args.references,latent_size=size,timesteps=[500,400],reference_scale=[0,1,1],
                    invocation_counts=invocation_counts,
                    peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    if args.checkpoint:
        metadata.update(checkpoint=str(args.checkpoint.resolve()),
                        checkpoint_sha256=sha256(args.checkpoint/"diffusion_pytorch_model.bin"))
    (args.output/"paint-unet.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__ == "__main__":
    main()
