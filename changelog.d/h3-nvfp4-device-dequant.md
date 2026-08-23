- **MiniMax H3 renders faster: the NVFP4 text encoder now unpacks its weights
  on the GPU.** Every H3 render encodes its prompt through a Qwen3-VL
  conditioner whose ~24.4 G quantized weights were unpacked by a
  single-threaded scalar loop on the CPU and then uploaded as dense 32-bit
  floats — once per render. On CUDA that work now happens on the GPU, from the
  packed bytes themselves, cutting the host-to-device traffic for those weights
  by 8x; measured on an RTX 4090, the widest projection at the sequence length
  a real render encodes drops from 285 ms to 41 ms, a 7.0x speedup. The images
  and videos are unchanged: the two paths are bit-identical, gated by a test
  that compares every element's bits
  ([#1317](https://github.com/utensils/mold/issues/1317)).
