#!/usr/bin/env python3
"""Capture vertex propagation and NS filling on retained chair-map inputs.

The retained baked PNG is decoded to F32 /255. This qualifies fill composition
on that explicit input; it is not a recapture of the pre-PNG neural/bake floats.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
from PIL import Image
from safetensors.numpy import save_file

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--upstream',type=Path,required=True)
p.add_argument('--extension',type=Path,required=True)
p.add_argument('--capture',type=Path,required=True)
p.add_argument('--size',type=int,choices=[1024,2048,4096],required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
revision=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()
assert revision=='82920d643c0dc2f7bfd7255f45f62d386edfe60c'
assert cv2.__version__=='4.10.0'
spec=importlib.util.spec_from_file_location('mesh_inpaint_processor',a.extension)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
a.output.mkdir(parents=True,exist_ok=False)
mesh=np.load(a.capture/'uv-mesh.npz')
positions=mesh['vertices'].astype(np.float32)
uv=mesh['uv'].astype(np.float32)
faces=mesh['faces'].astype(np.int32)
records=[]
for stream in range(2):
    start=time.monotonic()
    image_path=a.capture/f'baked-{stream}.png'
    mask_path=a.capture/f'trust-{stream}.png'
    image=Image.open(image_path).convert('RGB').resize((a.size,a.size),Image.Resampling.BILINEAR)
    mask=np.array(Image.open(mask_path).convert('L').resize((a.size,a.size),Image.Resampling.NEAREST))
    texture=np.array(image).astype(np.float32)/np.float32(255)
    propagated,trust=module.meshVerticeInpaint(texture,mask,positions,uv,faces,faces)
    before_ns=(propagated*255).astype(np.uint8)
    final=cv2.inpaint(before_ns,255-trust,3,cv2.INPAINT_NS)
    save_file(dict(texture=texture,mask=mask,positions=positions,uv=uv,faces=faces.astype(np.uint32),propagated=propagated,trust=trust,before_ns=before_ns,final=final),str(a.output/f'fill-{stream}.safetensors'))
    Image.fromarray(final).save(a.output/f'expected-{stream}.png')
    records.append(dict(stream=stream,seconds=time.monotonic()-start,image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest()))
    print(json.dumps(records[-1]),flush=True)
(a.output/'completed.json').write_text(json.dumps(dict(revision=revision,opencv=cv2.__version__,size=a.size,input='retained baked PNG decoded to F32/255; bilinear color and nearest mask resize',extension_sha256=hashlib.sha256(a.extension.read_bytes()).hexdigest(),source_sha256=hashlib.sha256((a.upstream/'hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor.cpp').read_bytes()).hexdigest(),records=records),indent=2)+'\n')
