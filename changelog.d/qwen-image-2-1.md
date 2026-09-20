- **Generate images with Qwen Image 2.1.** Add the official BF16 checkpoint,
  native Qwen3-VL text conditioning, 32-block transformer and 64-channel VAE,
  with request-local prefix KV caching across denoising steps. Text-to-image
  is supported; reference-image editing and block offload remain unavailable.
  Prefixes above 512 tokens render in full without cache retention.
