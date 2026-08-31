- **Wan and LTX-2 video decode runs on cuDNN convolutions (CUDA Linux builds).**
  The VAE decode is the largest phase of a Wan render outside the denoise, and
  it is now **2.1x** cheaper — 23.3 s to 11.2 s, taking an 81-frame 832x480
  `wan22-t2v-a14b:q5` render from 105.9 s to 94.9 s on an RTX 4090. Underneath
  that, the convolutions themselves are 4.4x faster (845 ms to 192 ms per latent
  frame), most of which came from a defect in candle rather than from cuDNN:
  its descriptors were left at `CUDNN_DEFAULT_MATH`, which declines tensor cores
  for bf16 and silently admits TF32 for f32, so simply enabling cuDNN captured
  only a third of the available gain. Image families deliberately stay on the
  previous path so an archived still seed still renders the same bytes;
  `MOLD_CONV={cudnn,im2col}` overrides the per-family default in either
  direction.
