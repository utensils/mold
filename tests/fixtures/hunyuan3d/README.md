# Hunyuan3D component oracle

`transformer21.safetensors` contains synthetic random weights, inputs and the
expected output of Tencent's unchanged Hunyuan3D 2.1 transformer, executed on an
NVIDIA L40S in float32. It contains no pretrained weights. Configuration, seed,
source revision and tolerance are recorded in `transformer21.json`.

Regenerate with `scripts/capture-hunyuan3d21-component.py --help` using the pinned
Tencent checkout and an oracle-only PyTorch environment. Keep the generated
original in the campaign evidence directory before copying a new fixture here.
The CPU Candle test compares the complete forward result, including attention
packing, timestep convention, skip connections, sparse experts and output sign.
A component pass does not establish pretrained or full-pipeline qualification.
