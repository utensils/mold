#!/usr/bin/env python3
"""Capture full Tencent projection/bake/fill from qualified upscaled views."""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--native',type=Path,required=True)
p.add_argument('--mesh',type=Path,required=True)
p.add_argument('--campaign',type=Path,required=True)
p.add_argument('--size',type=int,choices=[1024,2048,4096],required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--diagnostic-bake-input',type=Path)
p.add_argument('--trace-projections',action='store_true')
a=p.parse_args()
revision=subprocess.check_output(['git','-C',str(a.reference),'rev-parse','HEAD'],text=True).strip()
assert revision=='82920d643c0dc2f7bfd7255f45f62d386edfe60c'
a.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(a.reference.resolve()/'hy3dpaint'))
sys.modules['bpy']=types.ModuleType('bpy')
import DifferentiableRenderer
DifferentiableRenderer.__path__.append(str(a.native.resolve()))
module=importlib.import_module('DifferentiableRenderer.MeshRender')
import numpy as np
import cv2
assert cv2.__version__ == "4.10.0"
import torch
from PIL import Image
from safetensors.torch import save_file, load_file
source=a.reference/'hy3dpaint/utils/pipeline_utils.py'
method=next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='bake_from_multiview')
context=dict(torch=torch)
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),context)
torch.backends.cuda.matmul.allow_tf32=False
mesh=np.load(a.mesh)
vertices=mesh['vertices'].astype(np.float32);faces=mesh['faces'].astype(np.int32);uv=mesh['uv'].astype(np.float32)
save_file(dict(vertices=torch.from_numpy(vertices),faces=torch.from_numpy(faces),uv=torch.from_numpy(uv)),str(a.output/'mesh.safetensors'))
renderer=module.MeshRender(default_resolution=2048,texture_size=a.size,device='cuda')
renderer.set_mesh(vertices.copy(),faces.copy(),uv.copy(),faces.copy())
original_back_project=renderer.back_project
original_fast_bake=renderer.fast_bake_texture
projection_trace=dict(stream=None,index=0)
def observe_back_project(*args,**kwargs):
    texture,cosine,boundary=original_back_project(*args,**kwargs)
    if a.trace_projections:
        save_file(dict(texture=texture.cpu().contiguous(),cosine=cosine.cpu().contiguous(),boundary=boundary.cpu().contiguous()),str(a.output/f"{projection_trace['stream']}-projection-{projection_trace['index']:02d}.safetensors"))
    projection_trace['index']+=1
    return texture,cosine,boundary
renderer.back_project=observe_back_project
def observe_fast_bake(textures,cosines):
    merged,trust=original_fast_bake(textures,cosines)
    if a.trace_projections:
        index=projection_trace['index']-1
        save_file(dict(weighted=cosines[-1].cpu().contiguous(),merged=merged.cpu().contiguous(),trust=trust.to(torch.uint8).cpu().contiguous()),str(a.output/f"{projection_trace['stream']}-bake-step-{index:02d}.safetensors"))
    return merged,trust
renderer.fast_bake_texture=observe_fast_bake
fill_positions,fill_faces,fill_uv,fill_uv_faces=renderer.get_mesh()
save_file(dict(positions=torch.from_numpy(fill_positions.copy()),faces=torch.from_numpy(fill_faces.copy()),uv=torch.from_numpy(fill_uv.copy()),uv_faces=torch.from_numpy(fill_uv_faces.copy())),str(a.output/'fill-geometry.safetensors'))
original_vertex=module.meshVerticeInpaint
original_ns=module.cv2.inpaint
def observe_vertex(*args,**kwargs):
    colors,trust=original_vertex(*args,**kwargs)
    save_file(dict(colors=torch.from_numpy(colors.copy()),trust=torch.from_numpy(trust.copy())),str(a.output/f'{stream}-vertex.safetensors'))
    return colors,trust
def observe_ns(image,mask,*args,**kwargs):
    Image.fromarray(image).save(a.output/f'{stream}-before-ns.png')
    Image.fromarray(mask).save(a.output/f'{stream}-ns-mask.png')
    return original_ns(image,mask,*args,**kwargs)
module.meshVerticeInpaint=observe_vertex
module.cv2.inpaint=observe_ns
processor=types.SimpleNamespace(render=renderer,config=types.SimpleNamespace(bake_exp=4))
views=[(0,0,1.),(0,90,.1),(0,180,.5),(0,270,.1),(90,0,.05),(-90,180,.05)]
records=[]
for stream in ['albedo','mr']:
    projection_trace.update(stream=stream,index=0)
    images=[];paths=[]
    for i in range(6):
        folder=(a.campaign/f'paint-upscaler-oracle-{stream}00-v1') if i==0 else (a.campaign/f'paint-upscaler-remaining-views-v1/{stream}-{i:02d}-oracle')
        path=folder/'expected.png'
        image=Image.open(path).convert('RGB');assert image.size==(2048,2048)
        images.append(image)
        paths.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    start=time.monotonic()
    if a.diagnostic_bake_input:
        saved=load_file(str(a.diagnostic_bake_input/f'{stream}-bake.safetensors'))
        texture=saved['colors'].cuda();trust=saved['trust'].cuda().bool()
    else:
        texture,trust=context['bake_from_multiview'](processor,images,[v[0] for v in views],[v[1] for v in views],[v[2] for v in views])
    save_file(dict(colors=texture.cpu().contiguous(),trust=trust.to(torch.uint8).cpu().contiguous()),str(a.output/f'{stream}-bake.safetensors'))
    final=renderer.uv_inpaint(texture,trust.squeeze(-1).to(torch.uint8).cpu().numpy()*255)
    Image.fromarray(final).save(a.output/f'{stream}-filled.png')
    record=dict(stream=stream,seconds=time.monotonic()-start,images=paths);records.append(record)
    print(json.dumps(record),flush=True)
(a.output/'completed.json').write_text(json.dumps(dict(revision=revision,size=a.size,views=views,torch=torch.__version__,mesh_sha256=hashlib.sha256(a.mesh.read_bytes()).hexdigest(),records=records,diagnostic_bake_input=str(a.diagnostic_bake_input) if a.diagnostic_bake_input else None,trace_projections=a.trace_projections,opencv=cv2.__version__,native_sha256={str(p.relative_to(a.native)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.native.rglob('*.so'))},source_sha256={str(p.relative_to(a.reference)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,a.reference/'hy3dpaint/DifferentiableRenderer/MeshRender.py',a.reference/'hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor.cpp']}),indent=2)+'\n')
