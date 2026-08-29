- **Post-generation upscales no longer stall on CPU while GPUs are available.**
  Mold now keeps the follow-up Real-ESRGAN pass on a viable accelerator, while
  retaining CPU fallback when every accelerator is unavailable or lacks VRAM.
