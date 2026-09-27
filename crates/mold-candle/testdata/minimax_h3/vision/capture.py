"""Capture the fp32 upstream oracle for MiniMax H3's Qwen3-VL vision tower.

Runs transformers' `Qwen3VLVisionModel` (the class diffusers' MiniMax H3
modular pipeline instantiates through `Qwen3VLForConditionalGeneration`,
`diffusers/modular_pipelines/minimax_h3/encoders.py:17`) on the `visual.*`
tensors of the installed H3 conditioner, in float32, on a deterministic
synthetic image, and writes the processor's pixels/grid plus the merger and
the three DeepStack maps.

Scratch-venv tooling only (torch + transformers + safetensors + pillow); it is
never shipped. Usage:

    python capture.py <h3 text_encoders/*.safetensors> <text_encoder/config.json> \
        <processor dir> <out.safetensors>
"""

import json
import sys

import numpy as np
import torch
import transformers
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoImageProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


def synthetic_image(width=320, height=256):
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    r = 127.5 + 127.5 * np.sin(x / 13.0) * np.cos(y / 29.0)
    g = 255.0 * x / (width - 1)
    b = 127.5 + 127.5 * np.sin((x + 2 * y) / 7.0)
    rgb = np.stack([r, g, b], axis=-1).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def main():
    weights, config_path, processor_dir, out = sys.argv[1:5]
    with open(config_path) as handle:
        vision_config = json.load(handle)["vision_config"]
    model = Qwen3VLVisionModel._from_config(
        Qwen3VLVisionConfig(**vision_config), dtype=torch.float32
    ).eval()
    state = {}
    with safe_open(weights, "pt") as handle:
        for key in handle.keys():
            if key.startswith("visual."):
                state[key[len("visual.") :]] = handle.get_tensor(key).to(torch.float32)
    model.load_state_dict(state, strict=True)
    processor = AutoImageProcessor.from_pretrained(processor_dir)
    inputs = processor(images=[synthetic_image()], return_tensors="pt")
    pixels = inputs["pixel_values"].to(torch.float32)
    grid = inputs["image_grid_thw"]
    with torch.no_grad():
        output = model(pixels, grid_thw=grid)
    tensors = {
        "pixel_values": pixels.contiguous(),
        "image_grid_thw": grid.to(torch.int64).contiguous(),
        "vision_merger": output.pooler_output.contiguous(),
    }
    for index, feature in enumerate(output.deepstack_features):
        tensors[f"vision_deepstack_{index}"] = feature.contiguous()
    save_file(
        tensors,
        out,
        metadata={"transformers": transformers.__version__, "torch": torch.__version__},
    )
    for key, value in tensors.items():
        print(key, tuple(value.shape), value.dtype)


if __name__ == "__main__":
    main()
