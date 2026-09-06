#!/usr/bin/env python3
"""Capture the complete unchanged Tencent paint UNet and its reusable conditions."""
import argparse
import ast
from contextlib import nullcontext
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
    parser.add_argument("--reference-trace", action="store_true", help="capture failing reference spatial stages on their exact inputs")
    parser.add_argument("--guidance-inputs", action="store_true", help="expand one source into the actual three pipeline guidance branches")
    parser.add_argument("--trajectory", action="store_true", help="also capture the fifteen-step guided sampler")
    parser.add_argument("--scheduler-config", type=Path)
    parser.add_argument("--attention-backend", choices=["default", "math"], default="default", help="diagnostic SDPA backend; default preserves upstream selection")
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    parser.add_argument("--latent-size", type=int, choices=[8,16,32,64], default=8)
    parser.add_argument("--views", type=int, choices=[1,2,6], default=2)
    parser.add_argument("--references", type=int, choices=[1,2], default=1)
    args = parser.parse_args()
    if args.trajectory and (not args.guidance_inputs or not args.scheduler_config):
        parser.error("trajectory requires guidance-inputs and scheduler-config")
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
    trace = {}
    trace_layouts = {}
    if args.reference_trace:
        def trace_stage(name):
            def observe(module, inputs, output):
                if name+".input" not in trace:
                    trace_layouts[name] = dict(stride=list(inputs[0].stride()),contiguous=inputs[0].is_contiguous())
                    for key,value in dict(input=inputs[0],output=output,weight=module.weight,bias=module.bias).items():
                        trace[name+"."+key] = value.detach().cpu().contiguous()
            return observe
        for up_index, attention_index in [(1,2),(2,0),(2,1)]:
            spatial = model.unet_dual.up_blocks[up_index].attentions[attention_index]
            name = f"trace.up_{up_index}_{attention_index}_0"
            for stage,module in [("groupnorm",spatial.norm),("projection",spatial.proj_in),
                                 ("layernorm",spatial.transformer_blocks[0].transformer.norm1)]:
                hooks.append(module.register_forward_hook(trace_stage(name+"."+stage)))
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
    base_tensors = {}
    if args.guidance_inputs:
        originals = dict(sample=sample[:1], normal=normal[:1], position=position[:1],
                         reference=reference[:1], dino=dino[:1], position_maps=positions[:1])
        base_tensors = {"base."+key:value.detach().cpu().contiguous() for key,value in originals.items()}
        sample = originals["sample"].repeat(3,1,1,1,1,1)
        normal = originals["normal"].repeat(3,1,1,1,1)
        position = originals["position"].repeat(3,1,1,1,1)
        reference = originals["reference"].repeat(3,1,1,1,1)
        zero_dino = torch.zeros_like(originals["dino"])
        dino = torch.cat([zero_dino,zero_dino,originals["dino"]])
        positions = originals["position_maps"].repeat(3,1,1,1,1)
    text = torch.stack([model.unet.learned_text_clip_albedo,model.unet.learned_text_clip_mr]).unsqueeze(0).repeat(batch,1,1,1)
    scale = torch.tensor([0.,1.,1.],device="cuda",dtype=dtype)
    # Position quantization mutates its input in Tencent. Retain pre-call bytes.
    tensors = {"input."+key:value.detach().cpu().contiguous() for key,value in
               dict(sample=sample,normal=normal,position=position,reference=reference,dino=dino,
                    position_maps=positions,text=text,reference_scale=scale).items()}
    tensors.update(base_tensors)
    cache = {}
    conditions = dict(embeds_normal=normal, embeds_position=position, ref_latents=reference,
                      dino_hidden_states=dino, position_maps=positions, ref_scale=scale, mva_scale=1., cache=cache)
    torch.cuda.reset_peak_memory_stats()
    attention_context = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                         if args.attention_backend == "math" else nullcontext())
    with torch.inference_mode(), attention_context:
        for timestep in (500,400):
            result = model(sample,timestep,text,**conditions)[0]
            tensors[f"expected.{timestep}"] = result.cpu().contiguous()
        tensors["cache.dino"] = cache["dino_hidden_states_proj"].cpu().contiguous()
        for name, value in cache["condition_embed_dict"].items():
            tensors["cache.reference."+name] = value.cpu().contiguous()
        for length, values in cache["position_voxel_indices"].items():
            tensors[f"cache.positions.{length}"] = values["voxel_indices"].cpu().contiguous()
        if args.trajectory:
            import numpy as np
            pipeline_path = args.reference / "hy3dpaint/hunyuanpaintpbr/pipeline.py"
            branches = []
            for node in ast.walk(ast.parse(pipeline_path.read_text())):
                if isinstance(node,ast.If) and node.body and isinstance(node.body[0],ast.Assign):
                    target = node.body[0].targets[0]
                    if isinstance(target,ast.Tuple) and [getattr(n,"id","") for n in target.elts] == ["noise_pred_uncond","noise_pred_ref","noise_pred_full"]:
                        branches.append(node)
            if len(branches) != 1:
                raise RuntimeError("upstream guidance statements changed")
            guidance = compile(ast.Module(body=branches[0].body,type_ignores=[]),str(pipeline_path),"exec")
            scheduler = diffusers.UniPCMultistepScheduler.from_config(json.loads(args.scheduler_config.read_text()),timestep_spacing="trailing")
            scheduler.set_timesteps(15,device="cuda")
            latents = sample[:1].reshape(2*views,4,size,size).clone()
            tensors["trajectory.initial"] = latents.cpu().contiguous()
            tensors["trajectory.timesteps"] = scheduler.timesteps.cpu().contiguous()
            conditions["cache"] = {}
            conditions["position_maps"] = tensors["input.position_maps"].to("cuda").clone()
            for index,timestep in enumerate(scheduler.timesteps):
                model_input = latents.reshape(1,2,views,4,size,size).repeat(3,1,1,1,1,1)
                prediction = model(model_input,timestep,text,**conditions)[0]
                context = dict(torch=torch,np=np,self=types.SimpleNamespace(guidance_scale=3.),
                               kwargs=dict(num_in_batch=views),n_pbr=2,noise_pred=prediction)
                exec(guidance,context)
                tensors[f"trajectory.prediction.{index}"] = prediction.cpu().contiguous()
                latents = scheduler.step(context["noise_pred"],timestep,latents).prev_sample
                tensors[f"trajectory.sample.{index}"] = latents.cpu().contiguous()
                tensors[f"trajectory.x0.{index}"] = scheduler.model_outputs[-1].cpu().contiguous()
    tensors.update(trace)
    save_file(tensors,str(args.output/"paint-unet.safetensors"))
    for hook in hooks:
        hook.remove()
    if invocation_counts != dict(reference=2 if args.trajectory else 1, dino_projector=2 if args.trajectory else 1, main=17 if args.trajectory else 2):
        raise RuntimeError(f"unexpected upstream cache invocation counts: {invocation_counts}")
    metadata = dict(revision=revision,sources={name:sha256(folder/name) for name in ("modules.py","attn_processor.py")},
                    torch=torch.__version__,diffusers=diffusers.__version__,gpu=torch.cuda.get_device_name(),
                    seed=25026,dtype=args.dtype,attention_backend=args.attention_backend,tiny=args.tiny,config=config,batch=batch,materials=materials,
                    views=views,references=args.references,reference_trace=args.reference_trace,trace_layouts=trace_layouts,guidance_inputs=args.guidance_inputs,trajectory=args.trajectory,latent_size=size,timesteps=[500,400],reference_scale=[0,1,1],
                    invocation_counts=invocation_counts,
                    peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    if args.trajectory:
        metadata.update(pipeline_sha256=sha256(pipeline_path),scheduler_config_sha256=sha256(args.scheduler_config),trajectory_steps=15)
    if args.checkpoint:
        metadata.update(checkpoint=str(args.checkpoint.resolve()),
                        checkpoint_sha256=sha256(args.checkpoint/"diffusion_pytorch_model.bin"))
    (args.output/"paint-unet.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__ == "__main__":
    main()
