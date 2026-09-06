# Hunyuan3D campaign qualification ledger

Branch: `work/issues-1511-1496`. Scope and gates:
[`hunyuan3d-complete-campaign-plan.md`](../design/hunyuan3d-complete-campaign-plan.md).
The user requested full implementation with no deferred work. No PR is authorized.

## Capture harness

`scripts/capture-hunyuan3d-cuda.py` creates a unique retained directory beneath
`$MOLD_HOME/output/verification/hunyuan3d/campaign-1511-1496`. It records the
executable digest, source state, model/input digests, argv, exit status and sampled
board VRAM. It preserves stdout, stderr and partial outputs on failure. A zero
process exit without all requested artifacts is a failed capture, and a captured
run is not automatically a parity pass. Use `--help` for its command interface.

The launcher preserves the Nix home/model paths and provides a distinct output
directory and database. It is intended for forced-local renders and reference
processes, not servers sharing production's queue ownership. GPU selection uses
stable UUIDs; board memory includes other processes and is not an allocation peak.

Validation: seven CPU-only capture contract tests pass, after an initial failing
run with the implementation absent. CI and the local contracts route run them.
No model/render qualification is claimed by this checkpoint.

## Persistent environment

- Nix CUDA devshell checked: Rust/Cargo 1.95, compute capability 89.
- Oracle venv: `/storage/mold/cache/hunyuan3d-campaign/oracle-venv`.
- Downloads/cache retained beneath `/storage/mold/cache`; no existing models or
  outputs deleted. Builds now use `/storage/mold/cache/cargo-campaign-1511-1496`;
  another task cleaned the original shared target, so subsequent campaign builds
  retain a dedicated cache and cannot compete for that target lock.
- Initial branch CUDA build log:
  `/storage/mold/output/verification/hunyuan3d/campaign-1511-1496/build-initial.log`.

## 2.1 shape checkpoint

Implemented the 21-block MoE transformer, DINOv2-large selection, 4,096-latent
VAE preset, pinned repack manifest and independent 2.1 licence routing. Admission
now reads the recipe geometry for canvasless requests, fixing the mini encoder
underestimate as well as pricing the larger 2.1 latent sequence.

Evidence retained under the campaign output directory:

- `transformer21-component-v1`: unmodified Tencent CUDA complete-forward oracle,
  synthetic weights. The CPU Candle comparison passes at maximum absolute error
  below 0.00005; the fixture is checked in for deterministic regression coverage.
- `transformer21-pretrained-v1` and `transformer21-pretrained-candle-v1.log`: full
  pretrained 2.1 transformer, 4,096 latents and 1,370 conditioning tokens, same
  synthetic inputs in float32 on CUDA. Maximum absolute error 0.000031471252;
  RMS error 0.000002028392. Inputs, expected output and actual output retained.
- `shape21-engine-green.log`: 139 Hunyuan3D tests passed, one hardware test ignored.
- `shape21-core-green.log`: three targeted core tests passed.
- `core-full-v1.log`: all 1,732 core unit tests passed.
- `shape21-admission-green.log`: seven admission tests passed.
- `shape21-clippy-v2.log`: inference Clippy passed with warnings denied.
- `capture-20260906T040712Z-468b9a749abf`: full branch CUDA render using the
  pinned 2.1 checkpoint, source `219.png`, seed 25026, 30 steps and octree 192.
  Output: 95,165 vertices, 218,448 triangles; generation reports 38.9 seconds.
  Sampled board peak was 7,975 MiB (not an allocator peak).
  This demonstrates execution, not full-pipeline oracle parity.
- `capture-20260906T033601Z-e10082a476b4`: retained installed-binary 2.0 baseline.

The upstream paint reference produced a textured GLB, six albedo/MR view pairs,
conditioning maps, baked maps and inpainted maps in
`capture-20260906T040518Z-d97ee002cc18`. Its embedded Blender then segfaulted at
Python teardown, so the capture remains FAILED despite complete output files.
A clean-shutdown repeat is required. Peak PyTorch allocated bytes: 13,848,856,064;
reserved bytes: 20,789,067,776, at texture size 1,024. All failed loader attempts
and logs remain retained. The adapter resolves metadata through the upstream
loader without replacing network or rendering computations.

## CPU mesh preparation checkpoint

- `mesh-texture` builds pinned xatlas `f700c7790aaa030e794b52ba7791a05c085faf0c`
  through a narrow native ABI. No Python dependency enters the binary. The feature
  is forwarded through CLI/server/TUI/desktop and included in Nix feature recipes.
- The Rust wrapper checks geometry before native entry, polls cancellation and
  remaps positions, normals and colors across seam duplication. It matches the
  executable xatlas-python 0.0.9 tetrahedron oracle at 1e-7 UV tolerance.
- `uv-red-v2.log` records the failing tests before implementation; `uv-green-v2.log`
  records four passing tests with the optimized native build. `uv-clippy-v1.log`
  passes warnings-denied Clippy. The larger retained-mesh comparison is running.
- G-buffers now retain triangle IDs and perspective-correct barycentric weights.
  A test reconstructs every covered world-space point in both camera modes.
  `mesh-preparation-green-v1.log`: 144 tests passed, one hardware test ignored.
- `blender-export-v1.log`: Tencent's original export functions run successfully
  inside native Blender 5.0.1. Two embedded-bpy attempts remain marked failed
  after teardown crashes; the complete reference repeat uses the native exporter.
  This is an export-environment deviation, not a change to model mathematics.

## Static GLB ingestion and oracle export corrections

- GLB geometry ingestion now flattens every triangle primitive in the selected
  scene, composing parent TRS/matrices, inverse-transpose normals and reflected
  winding. Cycles, excessive depth, skinning, morph targets, required extensions,
  external resources, local shear and nonfinite attributes are refused.
- Interleaved float attributes are read with their declared stride. Accessors
  are bounded by their own buffer view as well as the BIN chunk: the new boundary
  regression failed on the previous reader (`glb-view-bounds-red-v1.log`).
- `mesh-import-green-v2.log`: 153 tests pass, one pretrained CUDA test ignored.
  Scene, shear and interleaving red/green logs remain beside that result.
- The retained real-mesh unwrap exposed xatlas's intentional uncharted slivers
  (`xatlas.cpp:9688–9701`, `atlasIndex = -1`, zero UVs). The bridge now preserves
  them just as xatlas-python does. `uv-sliver-red-v1.log` reproduces the refusal;
  `uv-sliver-green-v1.log` passes all six UV tests. The full-mesh repeat succeeds:
  `uv-realmesh-comparison-v2.json` records exact equality with the retained
  upstream arrays for all 192,906 vertices/UVs and 250,396 triangles. CPU unwrap
  took 351.576 seconds (`uv-realmesh-v2.json` preserves the complete result).
- `capture-20260906T044756Z-08905857bcb7` produced 4096-pixel albedo/MR maps and
  `textured.glb`, but remains FAILED: its final Blender version query inherited
  incompatible library paths and prevented measurement metadata publication.
  Export itself succeeded with a clean environment. Both native invocations now
  share that environment, and the version probe runs before model loading.
  The repeat, `capture-20260906T045720Z-89be259ea074`, PASSED with process exit 0
  and no missing outputs: 4096-pixel material maps, 512-pixel views, 2048 render
  resolution, six views, remesh disabled. Wall time 489.318 s, PyTorch peak
  allocated 14,398,047,744 bytes, reserved 21,936,209,920 bytes. Sampled board
  maximum 21,481 MiB is reported separately. All prior failures remain retained.
- The geometry-only OBJ reader preserves independent position/UV/normal indices,
  resolves negative references at each face, ear-clips concave polygons and bounds
  input/corner counts. It never opens referenced material libraries.
  `obj-red-v1.log` reproduces the missing implementation; `obj-green-v2.log` passes
  seven tests. `obj-realmesh-v1.log` imports the retained Tencent OBJ with 192,906
  vertices and 250,396 triangles; `obj-import-poster-v1.png` was visually inspected.
- The paint view selector matches an executable Tencent policy fixture for
  limits 6/7/10/30, preserving the mandatory first six, earliest-candidate ties,
  overlapping coverage and strict one-percent cutoff. Fixture generation uses
  the unchanged upstream method with synthetic renderer visibility only.
  `paint-views-red-v1.log` reproduces the missing selector;
  `paint-views-green-v1.log` passes 162 Hunyuan3D tests (one hardware test ignored).
  Camera/raster parity is still a separate open gate.

## Paint camera and conditioning parity

- `paint-raster-oracle-v1/` captures Tencent CUDA normalization, camera matrices,
  normal/position maps and face IDs for all 30 candidates on an asymmetric mesh.
  The checked-in fixture is 852 KiB. `paint-raster-red-v1.log` records the missing
  implementation; `paint-raster-green-v2.log` passes 164 tests (one hardware test
  ignored), including camera matrices, reversible coordinates, candidate order
  and all-view buffers at the declared 3e-5 interior tolerance. Existing poster
  and turntable tests remain green; their framing and float depth mode are intact.
- `paint-conditions-rust-v1/` contains all twelve 2048-pixel conditioning maps
  produced from the real mesh in 14.968 seconds on CPU. The comparison against
  the successful upstream run passes every preset gate (mask IoU >= .999,
  >= .995 of channels within one level, PSNR >= 40 dB).
  `paint-conditions-comparison-v1.json` records mask IoU exactly 1 for every map;
  positions differ by at most one level. Normal-map PSNR is at least 59.02 dB,
  with more than 99.999% of channels within one level; rare triangle-boundary
  differences can be much larger and are retained in the original maps.
- `paint-raster-clippy-v1.log` passes warnings-denied Clippy. CPU raster buffers
  are allocated one view at a time. The native CUDA renderer is oracle-only.

## Paint DINO component parity

- Paint uses Transformers 4.46 size-based position interpolation, whereas the
  existing shape encoder follows ComfyUI's scale-factor offset of 0.1. The
  separate `paint_giant` config preserves shape's convention. The small executable
  interpolation fixture failed before this change and passes afterward.
- `paint-dino-cuda-red-v1.log`: the pretrained encoder under the shape convention
  diverged (F32 max 4.5694, RMS .0621). After the correction,
  `paint-dino-cuda-green-v1.log` passes: F32 max .000052452087 / RMS .000002648894;
  F16 max .02734375 / RMS .002216486. Both actual outputs remain under
  `paint-dino-candle-v2/`; reference tensors are under `paint-dino-oracle-v1/`.
- Paint's shortest-edge 256 Pillow BICUBIC resize, centered 224 crop and ImageNet
  normalization match the actual processor fixture within 1e-6. The shared Rust
  resampler retains H3's exact LANCZOS pixels and cancellation behavior.
  `paint-dino-preprocess-green-v1.log`: 166 Hunyuan3D tests passed, one hardware
  test ignored. `shared-resize-h3-v1.log`: all 19 H3 pipeline tests passed.
  `pillow-green-v1.log` pins exact pixels for both filters;
  `paint-dino-clippy-v1.log` passes warnings-denied Clippy.
- CUDA component builds now have their own retained target directory,
  `/storage/mold/cache/cargo-campaign-1511-1496-cuda`, separate from the CPU target
  and other tasks' build caches.

## Paint VAE qualification in progress

- The application-owned shared VAE exposes posterior moments, bounded log
  variance and caller-owned sampling noise. Existing SD1.5, SDXL and SD3 consumers
  use that same implementation. Tiny encoder/decoder comparisons against the
  pinned Candle implementation remain bit-identical for both quant-convolution
  configurations; the executable Diffusers fixture also passes.
  `paint-vae-components-v2.log`: four component tests passed;
  `shared-vae-sd-regression-v1.log`: 201 passed, four hardware tests ignored.
- The Rust reader loads the published PyTorch checkpoint directly and rejects
  missing, wrong-shaped and unconsumed tensors. `paint-pth-green-v1.log` passes
  the serialized checkpoint contract. No Python runtime is shipped.
- `paint-vae-cuda-green-v1.log` passes the installed float32 VAE comparison:
  sampled latent maximum error .000027447939, decoded maximum .000026494265.
  The strict oracle disables TF32; the first oracle's default cuDNN TF32 setting
  produced a measurable difference and both captures remain retained.
- **Float16 qualification remains open.** `paint-vae-cuda-f16-v1.log` fails the
  declared latent tolerance: maximum .021087646, RMS .0051730411. The tolerance
  has not been relaxed. Follow-up captures retain posterior moments and decode
  from reference latents to separate encoder and decoder error; a scratch oracle
  probes the effect of half-rounded group-normalization affine operations.
- `paint-vae-half-diagnostic-v1/candle-comparison.json` localizes most error to
  encoding: decoding reference latents has maximum .0056152 / RMS .0008420,
  versus maximum .0234375 / RMS .0034180 after Rust encoding. The half-affine
  probe does not reproduce the Rust tensor, so it is evidence of sensitivity,
  not a confirmed root cause or an accepted substitute reference.

## VAE checkpoint peer review and follow-up

- The independent `review_vae` subagent found that Candle's lenient pickle
  inventory skipped unsupported tensors, weakening the claimed exact-loading
  contract. `paint-pth-review-v1/` reproduces this with an unexpected int32 tensor.
  Strict raw-dictionary validation now rejects skipped/non-tensor entries and
  duplicate names, permitting only PyTorch's defined module-version metadata.
- Follow-up review reproduced a negative-offset panic in Candle's conversion
  arithmetic (`paint-pth-review-v2/`). Offset, dimension, stride and storage-size
  checks now run before conversion. `paint-pth-strict-green-v3.log` passes all
  three loader tests; `paint-pth-strict-clippy-v1.log` passes warnings-denied
  Clippy. The review's installed inventory probe accepts 248 VAE tensors and
  1,747 UNet tensors; this is loader compatibility, not UNet inference proof.
  The final `paint-pth-review-v3/probe.log` rechecks the current parser against
  both installed checkpoints and confirms the negative offset returns an error
  without a panic; the reviewer marks that finding resolved.
- Encoder observation preserves the existing output bit-for-bit and propagates
  observer errors (`paint-vae-observer-green-v1.log`). The fresh installed-weight
  run with strict loading still passes float32 parity
  (`paint-vae-cuda-strict-f32-v1.log`, sampled max .000027447939 / RMS .0000054838;
  decoded max .000026494265 / RMS .0000030068).
- `paint-vae-encoder-comparison-v1.json` records float16 error growth across
  encoder boundaries, with the largest increase at the mid block. Scratch probes
  under `paint-vae-half-diagnostic-v2/` establish sensitivity to half-rounded
  normalization and SiLU; they do not fully reproduce the discrepancy or close
  the original gate. No acceptance tolerance was changed.
- The oracle now records checkpoint/config sizes and SHA-256 digests; the fresh
  `paint-vae-oracle-f32-v3/` capture includes these plus encoder tensors. Earlier
  captures and failures remain retained. Production-size/batched conditioning,
  pipeline RNG sequencing and the full paint network remain separate open gates.

## Paint VAE numerical policy checkpoint

- Peer review of PyTorch 2.5.1 identified CUDA GroupNorm's half-rounded epsilon,
  mean and reciprocal standard deviation, followed by float32 affine arithmetic;
  SiLU uses float32 opmath and one final half conversion. The source files and
  Welford/block-reduction definitions are retained in
  `paint-vae-numerics-review-v1/`. The new `VaeNumerics::Diffusers` path follows
  these boundaries; existing SD callers continue to use Candle's original path.
- The public Candle CUDA normalization operation mirrors the 32/512-thread
  Welford tree, `rsqrtf`, half-rounded saved statistics and fused affine, including
  the separate spatial-one case. `paint-vae-cuda-groupnorm-v1.log` passes exact
  comparison (maximum error zero) in all three cases and the biased-linear check.
  Independent source/safety review found no actionable issue; runtime fixture
  evidence is separate from that review.
- `paint-vae-opmath-components-v2.log` passes five component tests, including
  modern/legacy attention tensor names and 2D/1x1 weight layouts. Existing SD
  encoder/decoder output remains bit-identical to pinned Candle for both quant
  convolution configurations. `paint-vae-opmath-sd-regression-v1.log` passes
  201 inference tests with four hardware tests ignored; CPU warnings-denied
  Clippy passes in `paint-vae-opmath-clippy-v2.log`.
  The native CUDA module also passes warnings-denied Clippy in
  `paint-vae-opmath-cuda-clippy-v1.log`.
- Installed float32 parity still passes (`paint-vae-cuda-opmath-f32-v1.log`):
  sampled max .000025328249 / RMS .000004836021; decoded max .000025980175 /
  RMS .000002742593. The installed checkpoint uses legacy attention names;
  the first numerical-path loading failure remains retained in the v1 log.
- **Float16 maximum-error qualification remains open; no tolerance changed.**
  The synthetic 64-pixel input improves from sampled max .021087646 / RMS
  .005173041 to max .013671875 / RMS .003249301
  (`paint-vae-cuda-opmath-f16-v4.log`). A real 512-pixel conditioning map has
  sampled max .034179688 / RMS .000892346
  (`paint-vae-cuda-real512-v1.log`). Both fail the original .01 maximum bound.
  The exact normalization fixture does not establish whole-VAE parity.
- The oracle now accepts up to six conditioning images, records image hashes,
  resolution, per-model CUDA peaks and attention backend. The separate Torch
  math-vs-default SDPA comparison (`paint-vae-torch-sdpa-comparison-v1.json`) has
  sampled max .001464844 / RMS .000219039, so attention backend choice alone
  does not explain the larger VAE discrepancy. Captures, intermediate tensors
  and failed comparisons remain retained.

## Paint DINO image projector checkpoint

- The Rust projector follows Tencent's `ImageProjModel` at
  `hy3dpaint/hunyuanpaintpbr/unet/modules.py:710-754`, revision
  `82920d643c0dc2f7bfd7255f45f62d386edfe60c`: linear projection, four tokens
  per input token and LayerNorm. Half parameters and intermediate outputs retain
  their rounding boundaries while linear bias and normalization use float32
  arithmetic. Rank, width, dtype and allocation bounds are checked.
- `scripts/capture-hunyuan3d-paint-projector.py` extracts the unchanged upstream
  class for the executable oracle. The checked-in tiny fixture covers pooled
  and token inputs; the initial failing test is retained in
  `paint-projector-red-v1.log`. After implementation, the Hunyuan suite passes
  164 tests with one hardware test ignored (`paint-projector-hunyuan-tests-v1.log`).
- The installed paint checkpoint's four projector tensors and actual DINO
  features pass CUDA comparison in `paint-projector-cuda-v1.log`: float32
  maximum .000005722046 / RMS .000000306899, float16 maximum .00390625 /
  RMS .000043698633. Oracle metadata and tensors are retained in
  `paint-projector-oracle-v1/`; Rust outputs are in
  `paint-projector-candle-v1/`. This qualifies the projector component, not the
  complete paint UNet or its integration into generation.

## Paint attention checkpoint

- `paint_attention.rs` ports Tencent's material self-attention, reference value
  packing, ordinary cross-attention and multiview rotary positions from
  `hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py` at the pinned 2.1 revision.
  Reference values concatenate before head reshape and split after attention;
  independent review's conventional per-material counterexample differs by
  maximum .29798 / RMS .05650 (`paint-attention-review-v1/`). The implementation
  retains float32 arithmetic and the half output boundaries. Query chunks bound
  each score allocation to 64 MiB while retaining every key.
- `paint-attention-red-v1.log` records the initial failing executable-oracle
  comparison. `paint-attention-green-v2.log` passes four tests: all five processor
  kinds in both dtypes, rotary XYZ tables and material repetition, query chunking,
  and allocation rejection before copying indices. The latter was a peer-review
  finding with its own failing test in `paint-attention-bound-red-v1.log`.
  Warnings-denied Clippy passes (`paint-attention-clippy-v2.log`).
- The synthetic CUDA comparison passes (`paint-attention-cuda-v1.log`). The
  installed first down-block's weights also pass all five processors in both
  dtypes at production head width 64 (`paint-attention-pretrained-cuda-v1.log`),
  using two batches, six views with 64 spatial tokens each, 256 reference tokens
  and 1,028 DINO tokens. Predeclared installed bounds are float32 maximum 1e-4 /
  RMS 1e-5 and float16 maximum .01 / RMS .001. The original checkpoint hash,
  upstream hash, tensors and every Rust result are retained in
  `paint-attention-pretrained-oracle-v1/` and
  `paint-attention-pretrained-candle-v1/`. This does not yet qualify the complete
  transformer block, full spatial attention sizes or whole denoiser.

## Complete paint transformer block checkpoint

- `paint_block.rs` ports the complete `Basic2p5DTransformerBlock` from Tencent
  `modules.py:273-707`: material self-attention, albedo-query reference attention,
  multiview attention, text and DINO attention, and GEGLU feed-forward. The first
  three branches share the original norm1; text and DINO share norm2. The dual
  reference network returns its pre-attention norm1 cache. Independent source
  review confirmed the layout and ordering.
- The executable capture uses the unchanged upstream wrapper and Diffusers 0.30
  BasicTransformerBlock. Synthetic weights make all branches nonzero. The
  initial identity baseline failed (`paint-block-red-v1.log`). Review found that
  per-batch reference scales needed the scalar path's finite-value check; its
  failing regression is retained in `paint-block-scales-red-v1.log` and the fix
  rejects NaN and both infinities before attention.
- `paint-block-hunyuan-tests-v2.log` passes 171 Hunyuan tests with one hardware
  test ignored. Coverage includes one/three views, both networks and dtypes,
  scalar and per-batch scales, zero-reference suppression, three CFG batches with
  `[0,1,1]` scales, and cache independence from changed text conditioning.
  Warnings-denied Clippy passes in `paint-block-clippy-v2.log`.
- The installed checkpoint's complete first down-block passes on CUDA in
  `paint-block-pretrained-cuda-v2.log`: one/six views, production head width 64,
  DINO context length 1,028, and the same scaling/CFG cases. Predeclared complete
  block bounds are float32 maximum 1e-4 / RMS 1e-5 and float16 maximum .02 /
  RMS .002. Captures, hashes and outputs are retained in
  `paint-block-pretrained-oracle-v2/` and `paint-block-pretrained-candle-v2/`;
  previous captures and runs remain retained. Wider block widths, full spatial
  sizes, UNet convolution/resampling, scheduling and end-to-end paint remain
  separate open gates.

## Remaining gates

Full-pipeline P0 oracle parity and the remaining P1–P15 implementation/qualification
gates remain open. A successful 2.1 render does not close those gates. This ledger will record measured results as each gate is exercised;
it is not a completion checklist with assumed passes.
