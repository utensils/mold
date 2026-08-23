- **MiniMax H3 needs 4.57 GB less host RAM.** The Qwen3-VL NVFP4 text encoder
  kept the checkpoint's one-byte FP8 block scales in a four-byte host cache,
  and admission charged that expansion, so an H3 print could be refused for
  memory it never actually needed. Mold now retains the scales at their own
  width and widens them through a fixed lookup table at use, which is exactly
  bit-identical arithmetic — the same seed renders the same frames. The
  conditioner's host-resident parameters drop from 19.07 GB to 14.50 GB and
  the H3 admission host floor from 19.87 GB to 15.30 GB, so a host that was
  short by less than that now runs ([#1316](https://github.com/utensils/mold/issues/1316),
  complementing [#1289](https://github.com/utensils/mold/issues/1289))
