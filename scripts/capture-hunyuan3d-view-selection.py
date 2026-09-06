#!/usr/bin/env python3
"""Capture Tencent's view-selection policy on synthetic visibility sets.

The renderer is a fixture provider; the selected ViewProcessor method executes
unchanged. This isolates ordering, coverage thresholds and tie behavior from
the separately qualified rasterizer.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pin = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
    assert subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip() == pin
    subprocess.run(["git", "-C", str(args.upstream), "diff", "--exit-code"], check=True)
    import numpy as np
    import torch
    path = args.upstream / "hy3dpaint/utils/pipeline_utils.py"
    spec = importlib.util.spec_from_file_location("upstream_view_processor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for limit in [6, 7, 10, 30]:
        areas = [50, 20, 10, 7, 5, 4, 2, 1, 1]
        visible = [[0]] * 6 + [[1], [2, 3], [3, 4], [1], [5], [6], [7], [8]] + [[]] * 16

        class Render:
            default_resolution = 512

            def set_default_render_resolution(self, value):
                self.default_resolution = value

            def set_boundary_unreliable_scale(self, value):
                assert value == 2

            def get_face_areas(self, from_one_index):
                assert from_one_index
                return torch.tensor([0] + areas, dtype=torch.float32)

            def render_alpha(self, elev, azim, return_type):
                assert return_type == "np"
                ids = [0] + [face + 1 for face in visible[int(azim)]]
                return np.array(ids, dtype=np.int32).reshape(1, 1, -1, 1)

        render = Render()
        _, selected, _ = module.ViewProcessor(None, render).bake_view_selection(
            [0] * 30, list(range(30)), [1] * 30, limit
        )
        assert render.default_resolution == 512
        cases.append(dict(areas=areas, visible=visible, limit=limit, selected=selected))
    with args.output.open("x") as output:
        json.dump(dict(upstream=pin, source="hy3dpaint/utils/pipeline_utils.py:40-109", cases=cases), output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
