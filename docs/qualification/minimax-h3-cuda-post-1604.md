# H3 CUDA regression after #1604 — 2026-09-05

**PASS: bounded CUDA attention/INT8 validation.** The user released the CUDA
validation hold after the Metal campaign preparation. Metal UAT remains on
hold. This run loaded no model, started no service, changed no kernel setting,
and qualifies neither default-resolution output nor performance.

Source: `d6096446a74b1eeb70e4fdefff69bc57e3bb128b`, based on merged
`27ed658e` (#1604). The later report-only commits do not change tested source.
Candle: `744ae3b83cfac18db28107a353c449cc9b80d4ec`.
Host: HPE/plato, NVIDIA L40S SM89, driver 595.71.05. GPU 3 UUID
`GPU-6cb1bce8-3699-646a-db9c-c4a9a322a1d9` was selected explicitly.
Immediately before tests it reported 0% utilization and 765 MiB used by an
existing service; the local service activity response contained `items: []`.
Other GPUs had active workloads and were not selected. These timings are
not an exclusive-host performance measurement.

## Results and retained evidence

- [Unit-test transcript](minimax-h3-cuda-post-1604-tests.txt): **29 passed,
  zero failed, zero ignored**, 11.37 seconds. Twenty attention tests include
  BF16 CPU/CUDA parity; nine INT8 tests include native route selection,
  device-resident versus CPU-staged bit identity, bias and device validation.
- [Verbatim FlashAttention report](minimax-h3-cuda-post-1604.json): actual
  CUDA initialization, **257 rows × 56 heads × 128 dimensions**, BF16,
  `FlashAttentionV2`. Maximum absolute CPU-reference difference
  **0.00048828125**, below 0.02. Q/K-swap negative-control difference
  **0.04736328**, above 0.02. Reported kernel time: 10,033 microseconds.
- The standalone probe fails if CUDA initialization fails; it has no CPU
  fallback. Some library tests return early if device initialization fails,
  so their libtest pass count alone is not a no-skip device attestation.
  The separately retained probe confirms CUDA was usable on the same selected
  GPU in this run; it does not instrument each library test's initialization.
- Production row counts 37,296 and 107,856 were **planned only**. The report
  explicitly records `model_artifacts_accessed: false` and
  `runtime_activated: false`. Its release-candidate refusal fields describe
  this synthetic probe's authority, not a new restriction on shipped H3.

The library test build emitted the existing unused re-export warning in
`minimax_h3/comfy_quant.rs`; both builds completed successfully. No source
change or warning suppression was introduced for this qualification.

## Reproduction

Both builds ran in the repository Nix devshell with two Cargo jobs and the
locked, offline dependency graph. The existing task-owned target cache was
reused; concurrent builds in the main checkout used another target directory.
From the tested checkout:

```sh
export CARGO_TARGET_DIR=/home/jamesbrink/.codex/worktrees/20f3/mold/target
export CUDA_COMPUTE_CAP=89
nix develop --command cargo test --locked --offline -j 2 \
  -p mold-ai-candle --features h3,cuda,h3-flash-attn-rc,flash-attn \
  --lib --no-run
nix develop --command cargo build --locked --offline -j 2 \
  -p mold-ai-inference --features dev-bins,h3-cuda \
  --bin h3_attention_qualification
export CUDA_VISIBLE_DEVICES=GPU-6cb1bce8-3699-646a-db9c-c4a9a322a1d9
nix develop --command timeout 120 \
  "$CARGO_TARGET_DIR/debug/deps/mold_candle-acb893fc96c50408" \
  minimax_h3::attention::tests comfy_int8::tests --test-threads=1 --nocapture
nix develop --command timeout 120 \
  "$CARGO_TARGET_DIR/debug/h3_attention_qualification"
```

The libtest filename is the one emitted by this build; another toolchain or
feature graph may emit a different suffix. Hashes of the executed binaries:

| Binary | SHA-256 |
| --- | --- |
| `mold_candle-acb893fc96c50408` | `f086d2ef32fa3fcf6bc0d72fb21308edfe8ce9b32273cb30ea1a1a18d371b9e8` |
| `h3_attention_qualification` | `08f6ba7d3aa2cc52118eec9449ade2effe66548936ffc66a1e569f76974374bb` |

## Remaining scope

The [Metal/default-resolution campaign](minimax-h3-metal-next-campaign.md)
still needs recovered verified allocation/watchdog instrumentation, exact
request budgets, frozen input fixtures, paired full-render evidence and
visual/audio review. This bounded CUDA pass closes the kernel regression
check for the tested source only. It does not close #1164/#1542, lift the
Metal hold, promote `CorrectnessOnly`, or claim a 48 GiB default-resolution fit.
