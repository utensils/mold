#!/usr/bin/env python3
"""Capture Pillow's premultiplied RGBA resize and Tencent's white composition."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import PIL
    from PIL import Image
    import numpy as np
    import torch
    from safetensors.torch import save_file
    assert PIL.__version__ == "12.3.0"
    tensors = {}
    cases = {}
    for name, size, target in [("down", (23, 17), (9, 7)), ("up", (9, 7), (23, 17)), ("same", (11, 13), (11, 13))]:
        width, height = size
        values = np.arange(width * height * 4, dtype=np.uint32).reshape(height, width, 4)
        values = ((values * 71 + 33) % 256).astype(np.uint8)
        values.reshape(-1, 4)[::5, 3] = 0
        values.reshape(-1, 4)[1::5, 3] = 255
        source = Image.fromarray(values, "RGBA")
        resized = source.resize(target)
        expected = Image.new("RGB", target, (255, 255, 255))
        expected.paste(resized, mask=resized.getchannel("A"))
        tensors[name + ".input"] = torch.from_numpy(values)
        tensors[name + ".expected"] = torch.from_numpy(np.array(expected))
        cases[name] = dict(source=size, target=target)
    save_file(tensors, str(args.output / "paint-images.safetensors"))
    (args.output / "paint-images.json").write_text(json.dumps(dict(pillow=PIL.__version__,
        pillow_revision="bb1d8e8ab8d29048624d96e3ee53cecf7c13d13d", cases=cases,
        reference="Tencent 82920d643c0dc2f7bfd7255f45f62d386edfe60c hy3dpaint/textureGenPipeline.py:136-145"), indent=2) + "\n")


if __name__ == "__main__":
    main()
