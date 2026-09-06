#!/usr/bin/env python3
"""Extract small CPU position fixtures from retained, unmodified Tencent captures."""
import argparse
import hashlib
import json
from pathlib import Path
from safetensors.torch import load_file, save_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--float-capture', type=Path, required=True)
parser.add_argument('--half-capture', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists() or args.output.with_suffix('.json').exists():
    raise SystemExit('refusing to overwrite retained fixture')
tensors, sources = {}, {}
for dtype, root in [('f32', args.float_capture), ('f16', args.half_capture)]:
    source = root / 'paint-unet.safetensors'
    metadata = json.loads((root / 'paint-unet.json').read_text())
    if metadata['latent_size'] != 8 or metadata['dtype'] != dtype:
        raise SystemExit('expected 8x8 capture of matching dtype')
    captured = load_file(str(source))
    for name, tensor in captured.items():
        if name == 'input.position_maps' or name.startswith('cache.positions.'):
            tensors[dtype + '.' + name] = tensor[:1].contiguous()
    sources[dtype] = dict(revision=metadata['revision'], sources=metadata['sources'],
                         sha256=hashlib.sha256(source.read_bytes()).hexdigest())
args.output.parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(args.output))
args.output.with_suffix('.json').write_text(json.dumps(dict(latent_size=8, sources=sources), indent=2)+'\n')
