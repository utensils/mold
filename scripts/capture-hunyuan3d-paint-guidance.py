#!/usr/bin/env python3
"""Execute the unchanged guidance statements extracted from the pinned pipeline."""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
args = parser.parse_args()
revision = subprocess.check_output(['git','-C',str(args.reference),'rev-parse','HEAD'],text=True).strip()
if revision != '82920d643c0dc2f7bfd7255f45f62d386edfe60c':
    raise SystemExit('wrong Tencent revision')
path = args.reference/'hy3dpaint/hunyuanpaintpbr/pipeline.py'
source = path.read_text()
branches = []
for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.If) and node.body and isinstance(node.body[0],ast.Assign):
        target = node.body[0].targets[0]
        if isinstance(target,ast.Tuple) and [getattr(n,'id','') for n in target.elts] == ['noise_pred_uncond','noise_pred_ref','noise_pred_full']:
            branches.append(node)
if len(branches) != 1:
    raise SystemExit('guidance branch changed')
# Preserve every arithmetic statement verbatim; only supply its surrounding locals.
code = compile(ast.Module(body=branches[0].body,type_ignores=[]),str(path),'exec')
args.output.mkdir(parents=True,exist_ok=False)
import torch
import numpy as np
from safetensors.torch import save_file
tensors = {}
for label,dtype in [('f32',torch.float32),('f16',torch.float16)]:
    for views in [1,2,6]:
        for mode in ['default','azimuth']:
            key = f'{label}.{views}.{mode}'
            pred = (torch.arange(3*2*views*4*3*5,device=args.device).float()*.037).sin().reshape(3*2*views,4,3,5).to(dtype)
            kwargs = dict(num_in_batch=views)
            if mode == 'azimuth':
                kwargs['camera_azims'] = [0,45,90,300,330,359][:views]
            context = dict(torch=torch,np=np,self=SimpleNamespace(guidance_scale=3.),kwargs=kwargs,n_pbr=2,noise_pred=pred)
            exec(code,context)
            tensors[key+'.input'] = pred.cpu().contiguous()
            tensors[key+'.expected'] = context['noise_pred'].cpu().contiguous()
save_file(tensors,str(args.output/'paint-guidance.safetensors'))
(args.output/'paint-guidance.json').write_text(json.dumps(dict(revision=revision,
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    first_line=branches[0].body[0].lineno,last_line=branches[0].body[-1].end_lineno,
    torch=torch.__version__,device=args.device,guidance_scale=3.,
    azimuths=[0,45,90,300,330,359]),indent=2)+'\n')
