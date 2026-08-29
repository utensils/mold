- **LTX-2.5's plain `:bf16` diffusion-VAE decode works.** It no longer fails
  with `cannot find tensor decoder.det_stages.0.0.mlp.w_up.bias`: the decoder's
  SwiGLU MLPs now load the checkpoint's bias-free weights, and every rank-5
  projection is flattened before candle's `Linear`, which only accepts rank ≤ 4
  inputs.
