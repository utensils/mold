- **LTX-2 attention on CUDA now runs the shared BF16 dispatcher.** Unmasked,
  unperturbed self-attention (video and audio) routes through `crate::attention`
  in the checkpoint's own dtype — upstream's arithmetic, measured 6-9x faster
  than the old F32 tiles at stage-2 shapes — while masked cross-attention and
  the STG blend keep the F32 path byte-for-byte. `MOLD_LTX2_ATTN_F32=1`
  restores the F32 path everywhere, and every print records which route
  rendered it as additive `attention_path` provenance
  ([#735](https://github.com/utensils/mold/issues/735)). LTX-2 renders also
  report typed Upscale, AudioDecode, and Mux phases to the scheduler's
  learned-timing model.
