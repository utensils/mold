# LTX-2.5 native runtime

Mold adds native LTX-2.5 support, including the official Gemma 4 conditioner,
conventional and diffusion video VAE pack wiring, synchronized audio, duration
prediction, and two-stage latent upscaling. Retained Apple Metal runtime
evidence currently qualifies the compact INT8 ConvRot pack with the
conventional video VAE for explicit-frame synchronized-audio and silent T2V
outputs; the other paths are covered by focused contracts or remain deferred as
listed below.

- The default `ltx-2.5-22b-distilled` alias is the smaller INT8 ConvRot pack
  (about 40.0 GB / 37.2 GiB). BF16 remains available (about 71.4 GB / 66.5 GiB
  for distilled) but is operator-deferred on Metal; dev packs add about 8.9 GB.
- Hosts advertise split-pack readiness. Automatic routing refuses incomplete
  packs instead of discovering a missing component after queue admission.
- Explicit frames remain the default. `--predict-duration` and the Studio
  switch wire the official duration head and preserve omission in metadata;
  duration prediction is not part of the retained Metal runtime row.
- Native multishot is one LTX prompt/clip; Mold Sequence remains separate,
  durable multi-clip authoring.
- CUDA qualification is intentionally completed on a separate NVIDIA host and
  is not claimed by this Apple Metal release campaign.
- NVFP4, LTX-2.5 HDR/EXR, IC-LoRA, Retake, LipDub, Dynamic Frame Rate, and the
  prompt enhancer remain deferred and fail closed.
- Weights are gated, are not redistributed by Mold, and remain governed by the
  [LTX-2.x Community License](../architecture/ltx-2.5-license.md), including its
  USD 10 million commercial-use threshold.

Reference commits and operational details are recorded in the
[LTX-2.5 guide](../ltx-2.5.md).
