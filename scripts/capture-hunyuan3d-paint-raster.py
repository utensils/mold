#!/usr/bin/env python3
"""Capture pinned Tencent paint normalization, cameras and CUDA G-buffers."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pin = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
    assert subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip() == pin
    subprocess.run(["git", "-C", str(args.upstream), "diff", "--exit-code"], check=True)
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.upstream.resolve() / "hy3dpaint"
    sys.path.insert(0, str(root))
    sys.modules["bpy"] = types.ModuleType("bpy")
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from DifferentiableRenderer.MeshRender import MeshRender
    from DifferentiableRenderer.camera_utils import get_mv_matrix
    source = ast.parse((root / "textureGenPipeline.py").read_text())
    config_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "Hunyuan3DPaintConfig")
    namespace = {}
    exec(compile(ast.Module(body=[config_class], type_ignores=[]), str(root / "textureGenPipeline.py"), "exec"), namespace)
    config = namespace["Hunyuan3DPaintConfig"](6, 512)
    views = list(zip(config.candidate_camera_elevs, config.candidate_camera_azims, config.candidate_view_weights))
    # Asymmetric, translated tetrahedron reveals every axis and normalization.
    vertices = np.array([[2, 3, 4], [4, 3.2, 4], [2.1, 6, 4.2], [2.3, 3.1, 5.5]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    renderer = MeshRender(default_resolution=32, texture_size=32, device="cuda")
    renderer.set_mesh(vertices.copy(), faces.copy())
    tensors = dict(vertices=torch.from_numpy(vertices), faces=torch.from_numpy(faces),
                   normalized=renderer.vtx_pos, center=torch.from_numpy(renderer.mesh_normalize_scale_center),
                   scale=torch.tensor([renderer.mesh_normalize_scale_factor], dtype=torch.float32))
    for index, (elev, azim, _) in enumerate(views):
        tensors[f"view.{index}.matrix"] = torch.from_numpy(get_mv_matrix(elev, azim, renderer.camera_distance))
        tensors[f"view.{index}.normal"] = renderer.render_normal(elev, azim, use_abs_coor=True)
        tensors[f"view.{index}.position"] = renderer.render_position(elev, azim)
        tensors[f"view.{index}.face"] = renderer.render_alpha(elev, azim).to(torch.int32)
    save_file({name: value.detach().cpu().contiguous() for name, value in tensors.items()}, str(args.output / "paint-raster.safetensors"))
    (args.output / "paint-raster.json").write_text(json.dumps(dict(upstream=pin, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), resolution=32, views=views,
        source="hy3dpaint/DifferentiableRenderer/MeshRender.py", scale_factor=1.15, ortho_scale=1.2), indent=2) + "\n")


if __name__ == "__main__":
    main()
