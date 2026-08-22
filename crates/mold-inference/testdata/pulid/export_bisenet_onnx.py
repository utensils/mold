#!/usr/bin/env python3
"""Export facexlib's BiSeNet face parser to ONNX, to reproduce #1225's op gate.

PROVENANCE ONLY. Mold does not ship, download, or execute this ONNX graph —
`identity/parsing.rs` is a candle port, precisely because the gate below
FAILS. The script is committed so the failure can be re-checked rather than
re-argued, e.g. after a candle-onnx release adds the missing arms.

    python3 export_bisenet_onnx.py \\
        --facexlib-repo tmp/facexlib \\
        --weights /path/to/parsing_bisenet.pth \\
        --out /tmp/bisenet_opset11.onnx

    cargo run --release -p mold-ai-inference --features dev-bins,pulid \\
        --bin pulid_face_probe -- gate /tmp/bisenet_opset11.onnx

Needs `torch onnx onnxscript` in a scratch venv. With torch 2.13 and the
weights pinned at sha256 468e13ca...26567 the export lands at

    sha256 176d6ce28f0e37b0d6c71a49bcb54e0c974f9aab1d3690ce947630d91299bfd9
    52598436 bytes

and the gate reports three unsupported attributes: `MaxPool` pads,
`Resize mode=linear`, and `Resize coordinate_transformation_mode=align_corners`.
The exact digest is torch-version dependent; the verdict is not.
"""
import argparse, hashlib, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--facexlib-repo", required=True)
ap.add_argument("--weights", required=True)   # parsing_bisenet.pth
ap.add_argument("--out", required=True)
ap.add_argument("--opset", type=int, default=11)
a = ap.parse_args()
sys.path.insert(0, a.facexlib_repo)
from facexlib.parsing.bisenet import BiSeNet

net = BiSeNet(num_class=19)
net.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True), strict=True)
net.eval()

class ParseOnly(torch.nn.Module):
    """`pipeline_flux.py:164` uses output [0] only."""
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x): return self.net(x)[0]

dummy = torch.zeros(1, 3, 512, 512)
torch.onnx.export(ParseOnly(net), (dummy,), a.out, opset_version=a.opset,
                  input_names=["input"], output_names=["out"], dynamo=False)
print("sha256", hashlib.sha256(open(a.out, "rb").read()).hexdigest())
import os; print("bytes", os.path.getsize(a.out))
