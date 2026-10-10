- **MiniMax H3 Ref2VA no longer crashes the GPU on large reference sets.** The
  Qwen conditioner's text attention ran one dense softmax over the whole
  sequence, and the CUDA softmax kernel's 32-bit index overflowed once the
  sequence passed 5,792 rows — typical of five full-size reference photos — writing
  outside its buffer (NVIDIA Xid 31) and restarting the server. Attention now
  runs in exact query-axis chunks, and the raw cause of a fatal CUDA error on the
  generation, H3, and model-load paths is logged beside the worker quarantine
  ([#1852](https://github.com/utensils/mold/issues/1852)).
