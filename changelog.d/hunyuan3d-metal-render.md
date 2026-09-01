- **Hunyuan3D renders on Apple Silicon, and on CUDA in fp16.** The first real
  render of the image-to-3D family surfaced five engine bugs no CPU test could
  see. Three stopped or degraded the render: the shape transformer's
  affine-less LayerNorms carried a CPU weight tensor that every GPU forward
  tripped over, the two 1.1B tiers fed a 512 px image to a 14 px patch encoder
  that refuses it (both Tencent's `config.yaml` and ComfyUI encode at 518, so
  mold now letterboxes to 512 and resizes to 518 exactly as they do), and the
  timestep embedding was computed in half precision instead of upstream's f32.
  Two were found by comparing the first mesh against ComfyUI on the same
  checkpoint, image and seed: every mesh came out lying on its side, because
  ComfyUI's VAE wrapper applies a channels-last transpose to the voxel grid
  that the port had not reproduced, and every surface was thinner than the
  oracle's with half the triangles, because that same wrapper maps the raw
  logits onto a `[0, 1]` occupancy scale before the mesher thresholds them —
  so `--mesh-threshold 0.6` now means what it means in ComfyUI. The family now
  runs fp16 on CUDA and Metal like ComfyUI does (bf16 quantized the occupancy
  query grid to roughly its own spacing), and the tier is identified from the
  checkpoint's transformer depth rather than its filename. GPU-device forward
  tests and orientation tests guard each of these, `scripts/regression-matrix.sh`
  covers the family, and `scripts/capture-hunyuan3d-metal-uat.sh` plus
  `scripts/capture-hunyuan3d-comfy-metal-reference.sh` reproduce the ComfyUI
  comparison. Metal is now qualified `supported` for `hunyuan3d`; CUDA and CPU
  remain correctness-only until measured
  ([#1496](https://github.com/utensils/mold/issues/1496)).
