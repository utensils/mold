- **Wan and LTX-2 video decode runs on cuDNN convolutions (CUDA Linux builds).**
  The VAE decode is the largest phase of a Wan render outside the denoise, and
  its convolutions are now a measured **4.4x** cheaper: 845 ms to 192 ms per
  latent frame on an RTX 4090 (`wan22-t2v-a14b:q5`, 832x480). Most of that came
  from a defect in candle, not from cuDNN itself — its descriptors were left at
  `CUDNN_DEFAULT_MATH`, which declines tensor cores for bf16 and silently
  admits TF32 for f32, so simply enabling cuDNN captured a third of the gain.
  Image families deliberately stay on the previous path so an archived still
  seed still renders the same bytes; `MOLD_CONV={cudnn,im2col}` overrides the
  per-family default in either direction.
