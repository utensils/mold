#!/usr/bin/env python3
"""Execute Tencent's unchanged export functions inside the Blender application."""
import argparse
import ast
import math
from pathlib import Path
import subprocess
import sys
import bpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    pin = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
    actual = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != pin or args.output.exists():
        raise ValueError("pinned upstream and a new output file are required")
    subprocess.run(["git", "-C", str(args.upstream), "diff", "--exit-code"], check=True)
    source = args.upstream / "hy3dpaint/DifferentiableRenderer/mesh_utils.py"
    names = {"_setup_blender_scene", "_clear_scene_objects", "_select_mesh_objects",
             "_merge_vertices_if_needed", "_apply_shading", "_apply_auto_smooth", "convert_obj_to_glb"}
    module = ast.parse(source.read_text())
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError("upstream export function inventory changed")
    # Execute the original function ASTs, avoiding the module's unrelated cv2
    # and numpy mesh writers. This is process isolation, not a reimplementation.
    scope = {"bpy": bpy, "math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
    if not scope["convert_obj_to_glb"](str(args.input.resolve()), str(args.output.resolve())):
        raise RuntimeError("Tencent Blender export failed")
    data = args.output.read_bytes()
    if data[:4] != b"glTF" or int.from_bytes(data[8:12], "little") != len(data):
        raise ValueError("export produced an invalid GLB container")


if __name__ == "__main__":
    main()
