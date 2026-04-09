#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_command(request: dict) -> list[str]:
    module = request["module"]
    cmd = [sys.executable, "-m", module]
    uses_distilled_checkpoint = module in {
        "ltx_pipelines.distilled",
        "ltx_pipelines.ic_lora",
        "ltx_pipelines.retake",
    }
    uses_two_stage = module in {
        "ltx_pipelines.ti2vid_two_stages",
        "ltx_pipelines.ti2vid_two_stages_hq",
        "ltx_pipelines.a2vid_two_stage",
        "ltx_pipelines.keyframe_interpolation",
        "ltx_pipelines.distilled",
        "ltx_pipelines.ic_lora",
    }
    supports_dimensions = module != "ltx_pipelines.retake"

    if request.get("distilled_checkpoint_path") and uses_distilled_checkpoint:
        cmd.extend(["--distilled-checkpoint-path", request["distilled_checkpoint_path"]])
    else:
        cmd.extend(["--checkpoint-path", request["checkpoint_path"]])

    cmd.extend(["--gemma-root", request["gemma_root"]])
    cmd.extend(["--prompt", request["prompt"]])
    cmd.extend(["--output-path", request["output_path"]])
    cmd.extend(["--seed", str(request["seed"])])

    if supports_dimensions and request.get("width") is not None:
        cmd.extend(["--width", str(request["width"])])
    if supports_dimensions and request.get("height") is not None:
        cmd.extend(["--height", str(request["height"])])
    if supports_dimensions and request.get("num_frames") is not None:
        cmd.extend(["--num-frames", str(request["num_frames"])])
    if supports_dimensions and request.get("frame_rate") is not None:
        cmd.extend(["--frame-rate", str(request["frame_rate"])])
    if request.get("num_inference_steps") is not None and module not in {
        "ltx_pipelines.distilled",
        "ltx_pipelines.ic_lora",
        "ltx_pipelines.retake",
    }:
        cmd.extend(["--num-inference-steps", str(request["num_inference_steps"])])
    if request.get("negative_prompt") and module not in {
        "ltx_pipelines.distilled",
        "ltx_pipelines.ic_lora",
        "ltx_pipelines.retake",
    }:
        cmd.extend(["--negative-prompt", request["negative_prompt"]])
    if request.get("quantization"):
        cmd.extend(["--quantization", request["quantization"]])
    if uses_two_stage and request.get("spatial_upsampler_path"):
        cmd.extend(["--spatial-upsampler-path", request["spatial_upsampler_path"]])
    if request.get("distilled_lora_path") and module in {
        "ltx_pipelines.ti2vid_two_stages",
        "ltx_pipelines.ti2vid_two_stages_hq",
        "ltx_pipelines.a2vid_two_stage",
        "ltx_pipelines.keyframe_interpolation",
    }:
        cmd.extend(["--distilled-lora", request["distilled_lora_path"], "1.0"])

    for image in request.get("images", []):
        cmd.extend(["--image", image["path"], str(image["frame"]), str(image["strength"])])

    for lora in request.get("loras", []):
        cmd.extend(["--lora", lora["path"], str(lora["scale"])])

    if module == "ltx_pipelines.a2vid_two_stage":
        cmd.extend(["--audio-path", request["audio_path"]])
    elif module == "ltx_pipelines.retake":
        cmd.extend(["--video-path", request["video_path"]])
        cmd.extend(["--start-time", str(request["retake_start_seconds"])])
        cmd.extend(["--end-time", str(request["retake_end_seconds"])])
    elif module == "ltx_pipelines.ic_lora":
        cmd.extend(["--video-conditioning", request["video_path"], "1.0"])

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    request_path = Path(args.request)
    request = json.loads(request_path.read_text())
    cmd = build_command(request)
    upstream_root = Path(__file__).resolve().parents[1] / "tmp" / "LTX-2-upstream"
    result = subprocess.run(cmd, cwd=upstream_root.resolve(), check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
