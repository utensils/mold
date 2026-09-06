#!/usr/bin/env python3
"""Capture pinned Diffusers paint UniPC trajectories, including half boundaries."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
import torch
import diffusers
from diffusers import UniPCMultistepScheduler
from safetensors.torch import save_file
if diffusers.__version__ != '0.30.0':
    raise SystemExit('requires Diffusers 0.30.0')
config = json.loads(args.config.read_text())
tensors = {}
for steps in [1, 2, 3, 15, 30, 48]:
    for label, dtype in [('f32', torch.float32), ('f16', torch.float16)]:
        scheduler = UniPCMultistepScheduler.from_config(config, timestep_spacing='trailing')
        scheduler.set_timesteps(steps, device=args.device)
        prefix = f'{label}.{steps}'
        tensors[prefix+'.timesteps'] = scheduler.timesteps.cpu()
        tensors[prefix+'.sigmas'] = scheduler.sigmas.cpu()
        if steps == 15 and label == 'f32':
            tensors['betas'] = scheduler.betas
            tensors['alphas_cumprod'] = scheduler.alphas_cumprod
        base = torch.arange(120, dtype=torch.float32, device=args.device).reshape(2,4,3,5)
        sample = (base*.031).sin().to(dtype)
        tensors[prefix+'.initial'] = sample.cpu().contiguous()
        for index, timestep in enumerate(scheduler.timesteps):
            model = ((base*.017+index*.31).cos()*.3).to(dtype)
            tensors[prefix+f'.model.{index}'] = model.cpu().contiguous()
            sample = scheduler.step(model, timestep, sample).prev_sample
            tensors[prefix+f'.sample.{index}'] = sample.cpu().contiguous()
            tensors[prefix+f'.x0.{index}'] = scheduler.model_outputs[-1].cpu().contiguous()
            tensors[prefix+f'.corrected.{index}'] = scheduler.last_sample.cpu().contiguous()
        if not torch.isfinite(sample).all():
            raise RuntimeError('nonfinite upstream trajectory')
save_file({key:value.clone() for key,value in tensors.items()}, str(args.output/'paint-sampler.safetensors'))
(args.output/'paint-sampler.json').write_text(json.dumps(dict(
    torch=torch.__version__, diffusers=diffusers.__version__, device=args.device,
    config=config, effective_config=dict(scheduler.config),
    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
    source='diffusers v0.30.0 scheduling_unipc_multistep.py',
    cases=[1,2,3,15,30,48]), indent=2)+'\n')
