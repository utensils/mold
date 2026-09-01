- **Ship the Framewise codec runtime.** Nix and CUDA-container hosts now provide
  `ffmpeg` and `ffprobe` for video upscaling, while other hosts stop advertising
  Framewise upscale when those required tools are unavailable.
