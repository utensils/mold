- **Run the LTX-2.5 GGUF tiers natively.** The seven pinned Abiray transformer
  tiers now execute end to end: quantized block linears stay compact at rest
  and dequantize per forward on CUDA (`MOLD_LTX2_QMATMUL=1` opts into candle's
  quantized fast path; Metal keeps `QMatMul`), adaptive residency prices the
  files at their real sizes so Q4_K_M sits fully resident on a 24 GB card, and
  LoRAs apply as a parallel low-rank branch with full-weight `.diff` deltas
  refused by name. The download-only gate and its 501 refusal are gone.
