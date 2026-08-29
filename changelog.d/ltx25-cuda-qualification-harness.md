- **LTX-2.5 CUDA qualification harness.** `scripts/capture-ltx25-cuda-verification.sh`
  runs the manifest-driven `scripts/fixtures/ltx25-cuda-matrix.json` (official
  split packs, GGUF tiers, LTX-2/2.3 regressions) against a scratch CUDA
  server and seals a `mold.ltx25.cuda.verification.v1` report whose rows are
  exactly `passed|failed|blocked|not_run`, bound to retained media, Library
  provenance, hashed server-log slices, and 1 Hz VRAM/host samples;
  `scripts/capture-ltx25-comfy-cuda-reference.sh` is the matching ComfyUI CUDA
  oracle for the INT8 ConvRot and GGUF Q4_K_M checkpoints. Both ship with CI
  contract tests; the campaign itself lands separately
  ([#1398](https://github.com/utensils/mold/issues/1398),
  [#1414](https://github.com/utensils/mold/issues/1414)).
