#!/usr/bin/env python3
"""Capture OpenCV 4.10's radius-three RGB Navier–Stokes paint fill."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--opencv', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
revision = subprocess.check_output(['git','-C',str(a.opencv),'rev-parse','HEAD'],text=True).strip()
assert revision == '71d3237a093b60a27601c20e9ee6c3e52154e8b1'
assert cv2.__version__ == '4.10.0'
a.output.mkdir(parents=True, exist_ok=False)
rng = np.random.default_rng(15111496)
cases = []
for name in ['known','missing','island','border','noise','stripes','tiny','two','column','nonbinary']:
    h,w = {'tiny':(3,4), 'two':(2,2), 'column':(11,2)}.get(name,(17,23))
    pixels = rng.integers(0,256,(h,w,3),dtype=np.uint8)
    trust = np.full((h,w),255,dtype=np.uint8)
    if name == 'missing': trust[:] = 0
    elif name == 'island': trust[3:14,5:18] = 0
    elif name == 'border':
        trust[:5,:] = 0
        trust[:, :4] = 0
        trust[-3:, -6:] = 0
    elif name == 'noise': trust[rng.random((h,w)) < .65] = 0
    elif name == 'stripes': trust[:,::2] = 0
    elif name in ['tiny','two','column']: trust[1,1] = 0
    elif name == 'nonbinary': trust[4:13,5:12] = 254
    expected = cv2.inpaint(pixels,255-trust,3,cv2.INPAINT_NS)
    cases.append(dict(name=name,width=w,height=h,pixels=pixels.reshape(-1,3).tolist(),trust=trust.ravel().tolist(),expected=expected.reshape(-1,3).tolist()))
source = a.opencv/'modules/photo/src/inpaint.cpp'
(a.output/'ns-fill.json').write_text(json.dumps(dict(revision=revision,opencv=cv2.__version__,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=cases),indent=2)+'\n')
