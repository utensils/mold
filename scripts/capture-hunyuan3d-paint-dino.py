#!/usr/bin/env python3
"""Capture paint's actual DINO wrapper and its position interpolation oracle."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()
    pin = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
    assert subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip() == pin
    subprocess.run(["git", "-C", str(args.upstream), "diff", "--exit-code"], check=True)
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    from PIL import Image
    from safetensors.torch import save_file
    from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings
    # Exercise the exact installed Transformers interpolation with a tiny
    # deterministic position table and nonsquare output grid.
    config = transformers.Dinov2Config(hidden_size=4, image_size=6, patch_size=2)
    embeddings = Dinov2Embeddings(config)
    with torch.no_grad():
        embeddings.position_embeddings.copy_(torch.arange(40).reshape(1,10,4).float().sin())
        expected = embeddings.interpolate_pos_encoding(torch.zeros(1,21,4),10,8)
    save_file({"position_embeddings":embeddings.position_embeddings.detach(),"expected":expected},str(args.output/"paint-dino-position.safetensors"))
    processor = transformers.AutoImageProcessor.from_pretrained(str(args.checkpoint.resolve()))
    raw = bytes((x*37+y*19+c*53)%256 for y in range(11) for x in range(17) for c in range(3))
    synthetic = Image.frombytes("RGB",(17,11),raw)
    save_file({"source":torch.tensor(list(raw),dtype=torch.uint8).reshape(11,17,3),
               "expected":processor(images=[synthetic],return_tensors="pt").pixel_values},
              str(args.output/"paint-dino-preprocess.safetensors"))
    if args.preprocess_only:
        return
    package = types.ModuleType("hunyuanpaintpbr")
    package.__path__ = [str(args.upstream.resolve()/"hy3dpaint/hunyuanpaintpbr")]
    sys.modules["hunyuanpaintpbr"] = package
    module = importlib.import_module("hunyuanpaintpbr.unet.modules")
    model = module.Dino_v2(str(args.checkpoint.resolve())).to("cuda")
    image = Image.open(args.image).resize((512,512))
    if image.mode == "RGBA":
        background = Image.new("RGB",image.size,(255,255,255))
        background.paste(image,mask=image.getchannel("A"))
        image = background
    image.save(args.output/"appearance.png")
    pixels = model.dino_processor(images=[image],return_tensors="pt").pixel_values
    tensors = {"pixels":pixels}
    with torch.inference_mode():
        tensors["expected_f32"] = model([image]).cpu()
        model.to(torch.float16)
        tensors["expected_f16"] = model([image]).cpu()
    save_file({name:value.contiguous() for name,value in tensors.items()},str(args.output/"paint-dino.safetensors"))
    (args.output/"paint-dino.json").write_text(json.dumps(dict(upstream=pin,torch=torch.__version__,transformers=transformers.__version__,
        checkpoint=str(args.checkpoint.resolve()),image=str(args.image.resolve()),gpu=torch.cuda.get_device_name()),indent=2)+"\n")


if __name__ == "__main__":
    main()
