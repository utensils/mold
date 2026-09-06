- **Restore uncached Wan 1.3B and dense 14B rendering.** Refuse residual caching
  on both unqualified execution graphs, including explicit thresholds, to
  prevent noise-only or blurred output while preserving the qualified A14B pair
  ([#1559](https://github.com/utensils/mold/issues/1559)).
- **Carry offload requests to GPU hosts.** Preserve `--offload` through HTTP,
  frozen admission, worker validation, and durable sequence stages
  ([#1462](https://github.com/utensils/mold/issues/1462)).
- **Count retained Wan CUDA context memory once.** After releasing a completed
  Wan engine, certify its retained context separately from live allocations
  release device-owned GGUF scratch buffers, and apply that baseline
  consistently to admission and worker memory checks
  ([#1481](https://github.com/utensils/mold/issues/1481)).
- **Exercise the complete H3 CUDA server library suite.** Isolate test settings
  from the host and run relevant PRs through the existing CUDA toolkit job
  ([#1361](https://github.com/utensils/mold/issues/1361)).
