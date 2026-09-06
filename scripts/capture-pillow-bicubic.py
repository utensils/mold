#!/usr/bin/env python3
"""Capture small deterministic RGB resize fixtures from pinned Pillow."""
import argparse
import json
from pathlib import Path
import PIL
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert PIL.__version__ == "12.3.0"
    cases = []
    for width, height, target_width, target_height in [(5,7,8,4),(11,7,4,9),(1,1,5,2),(17,2,3,11),(5,7,5,7)]:
        source = bytes((x*37+y*19+c*53)%256 for y in range(height) for x in range(width) for c in range(3))
        image = Image.frombytes("RGB",(width,height),source)
        expected = image.resize((target_width,target_height),Image.Resampling.BICUBIC).tobytes()
        cases.append(dict(width=width,height=height,target_width=target_width,target_height=target_height,source=list(source),expected=list(expected)))
    with args.output.open("x") as output:
        json.dump(dict(pillow=PIL.__version__,cases=cases),output,indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
