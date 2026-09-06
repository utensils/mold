# Hunyuan3D campaign qualification ledger

Branch: `work/issues-1511-1496`. Scope and gates:
[`hunyuan3d-complete-campaign-plan.md`](../design/hunyuan3d-complete-campaign-plan.md).
The user requested full implementation with no deferred work. Open a PR only after
full implementation, verification and subagent peer review; that gate is not met.

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

## Paint VAE convolution diagnostic

- The existing float16 encoder traces show a small first-convolution difference
  (maximum .0009765625 / RMS .000003527837) that grows through later stages;
  `paint-vae-encoder-diagnostic-v5.json` records the per-stage comparison. This
  observation alone does not establish the cause of the final discrepancy.
- The capture harness can now disable cuDNN explicitly for a **diagnostic**
  Torch-native convolution run. The default production reference remains cuDNN.
  `paint-vae-native-conv-oracle-v1/` records the selected backend and checkpoint
  hashes. It does not improve parity: `paint-vae-native-conv-cuda-v1.log` fails
  the unchanged latent maximum bound with maximum .021972656 / RMS .003787731.
  Its first-convolution RMS is worse (.000155045), and the full stage comparison
  is retained in `paint-vae-native-conv-stage-comparison-v1.json`. No production
  backend or qualification tolerance was changed.

## Paint convolution and shared normalization checkpoint

- `paint_conv.rs` ports Diffusers 0.30 timestep-conditioned residual blocks,
  padding-one stride-two downsampling and nearest-neighbor/convolution upsampling
  including explicit output dimensions. Initial residual and epsilon mismatch
  tests failed (`paint-conv-red-v1.log`, `paint-conv-norm-red-v1.log`). The
  standalone capture's first installed attempt exposed duplicate downsampler
  aliases; it now uses the UNet's actual `name="op"` constructor. Failed capture
  and missing-fixture run remain retained; successful captures are v2.
- `DiffusersGroupNorm` is shared by the paint VAE and UNet and receives epsilon
  explicitly. Both VAE `1e-6` and residual `1e-5` fixtures match CUDA exactly
  (`paint-conv-cuda-norm-v1.log`), as do the existing 32-thread, 512-thread and
  spatial-one VAE fixtures. Spatial Transformer2D normalization uses `1e-6`.
  The native ABI now carries the epsilon as float32 and rounds it to half in
  the kernel. Independent review found no actionable ABI, ordering or bounds
  defect. Six shared SD/VAE tests pass, including bit-identical default SD VAE
  behavior (`paint-conv-norm-green-v2.log`).
- Installed residual, downsample and upsample weights pass CUDA in both dtypes
  (`paint-conv-pretrained-cuda-v2.log`), with odd 9x7 inputs and explicit 17x13
  upsample output. Float32 worst maximum is .000025749207 / RMS .000002208311;
  float16 worst maximum is .0078125 / RMS .000759318. Declared bounds remain
  maximum 1e-4 / RMS 1e-5 for float32 and maximum .02 / RMS .002 for float16.
  Full tensors and checkpoint hashes are retained in
  `paint-conv-pretrained-oracle-v2/` and `paint-conv-pretrained-candle-v2/`.
  The updated Hunyuan suite passes 172 tests with one hardware test ignored
  (`paint-conv-hunyuan-tests-v2.log`); CPU Clippy passes (`paint-conv-clippy-v1.log`).
  CUDA warnings-denied Clippy also passes (`paint-conv-cuda-clippy-v1.log`).
  Component parity does not yet qualify the assembled UNet or resolve the VAE
  float16 maximum-error gate.

## Assembled upstream paint UNet captures

- `scripts/capture-hunyuan3d-paint-unet.py` executes the unchanged pinned Tencent
  wrapper and Diffusers 0.30 UNet. It captures two denoiser timesteps (500, 400),
  all inputs, DINO projection, all 16 reference transformer caches and the four
  position-index resolutions. Position maps are saved before Tencent mutates
  them. The tiny architecture retains all four down/up levels, two residual
  layers per down block, all paint branches and nonzero synthetic parameters.
  Its weights remain outside Git in the evidence directory.
- Float32 tiny and installed captures succeeded at two views and 8x8 latents
  (`paint-unet-tiny-oracle-v1/`, `paint-unet-pretrained-oracle-v2/`). All tensors
  are finite; `paint-unet-initial-capture-check-v1.json` records dimensions,
  cache counts and distinct timestep results. The first installed invocation
  used the config-only component directory and failed before loading weights;
  it remains retained. The harness now validates both required files before
  constructing a model and uses the retained joined checkpoint layout.
- Float16 captures succeeded for the tiny architecture and the full installed
  production dimensions: six views, three CFG batches, two materials, 64x64
  latents (`paint-unet-production-f16-oracle-v1/`). Outputs have shape
  `[36,4,64,64]`; both timestep outputs and all caches are finite. PyTorch peak
  allocated/reserved memory is 6,784,477,184 / 7,742,685,184 bytes. This is an
  upstream forward capture, not a Rust parity or full texture-generation claim.
- Read-only review confirmed 12 newest-first residual skips, `[hidden,skip]`
  concatenation and 16 reference cache sites. It also identified dtype-dependent
  position mutation: `.half()` aliases F16 input but copies F32 input on each
  pyramid scale, so invalid pixels zeroed in the first F16 scale become valid
  zeros on later scales. Any channel equal to one after conversion is invalid;
  half sum/product rounding and ties-to-even voxel rounding must be retained.
- The subsequent two-reference tiny capture
  (`paint-unet-tiny-multiref-oracle-v1/`) adds invocation counters: two denoiser
  calls run the main network twice, but reference inference and DINO projection
  once each. Both the multiple-reference shape and caching behavior are now
  executable fixtures. Assembled Rust parity remains the next open gate.

## Main synchronization before denoiser assembly

Fetched and merged `origin/main` through `4ec048a0` (three upstream fixes).
The sole conflict was generated `website/guide/prompting.md`; rerunning the
canonical prompting generator resolved it and regenerated both outputs.
Focused merged-state checks passed: 52 generation-profile tests, 293 validation
tests, 89 chain tests and 172 Hunyuan tests (one hardware test ignored). Logs are
retained as `main-sync-{prompting,profiles,validation,chain,hunyuan}-v1.log`.

## Complete paint spatial UNet assembly

- `paint_unet.rs` now assembles all four down/up levels, twelve LIFO skips,
  timestep embeddings, the mid block, sixteen spatial transformers, and both
  main/reference network variants from the previously qualified components.
- RED evidence: `paint-unet-red-v1.log`, identity output max error 1.5528353.
- Tiny F32 complete dual-network parity passed (`paint-unet-assembled-v1.log`):
  all sixteen reference caches and the DINO projector match; maximum denoising
  error is 0.0000071525574 across timesteps 500 and 400.
- Installed F32 checkpoint parity passed at 8x8 latent/two-view dimensions
  (`paint-unet-pretrained-cuda-v2.log`). The strict PTH loader consumed the
  entire 1747-tensor checkpoint, including both networks, all learned text
  embeddings and the projector. Denoising max error is 0.0000023841858;
  reference-cache worst max is 0.000013768673. Actual tensors are retained in
  `paint-unet-pretrained-candle-v2/`. The v1 invocation failed on a test-harness
  path expecting tiny weights; that failure is retained too.
- Read-only peer review found no architecture mismatch; its unchecked
  view-count multiplication finding is fixed with `checked_mul` and invalid
  zero/overflow counts are exercised by the full-network test.
- Tiny F16 first attempt failed the declared max 0.02 / RMS 0.002 gate at the
  reference mid cache (max 0.017089844, RMS 0.004933512), retained in
  `paint-unet-tiny-f16-cuda-v1.log`. No tolerance was relaxed. Subsequent runs
  collect all cache/output errors before asserting so a failed cache cannot
  hide downstream diagnostic tensors.
- Rust position preparation is now integrated; installed F32 passes unchanged
  (`paint-unet-integrated-cuda-v1.log`) with Rust positions and all overflow guards.
  Two-reference F32 also passes (`paint-unet-multiref-cuda-v1.log`), worst final
  max 0.000008936971. CPU and CUDA all-target Clippy pass with warnings denied
  (`paint-unet-position-clippy-v1.log`, `paint-unet-position-cuda-clippy-v1.log`).
- Full installed F16 production execution completed in 72.50 seconds
  (`paint-unet-production-f16-cuda-v1.log`, actual tensors in the matching
  `paint-unet-production-f16-candle-v1/` directory). Both denoising outputs pass
  max 0.02 / RMS 0.002: worst max 0.0029296875, RMS 0.00044950502. However,
  `up_1_2_0` and `up_2_0_0` reference-cache max errors are 0.041992188 and
  0.109375; the overall half gate therefore remains FAILED.
- Diagnostic `--attention-backend math` leaves the upstream source unchanged.
  Tiny F16 Torch math-vs-default alone reaches reference-cache RMS 0.0044758
  and final-output max 0.0057373. Rust-vs-math does not eliminate the discrepancy
  (`paint-unet-math-comparison-v1.json`), so this is evidence about backend
  sensitivity, not a parity waiver or tolerance change.

## Paint position-map pyramid

- `paint_positions.rs` implements all four scales, validity from all three
  channels after half conversion, half sum/division/product boundaries,
  F16-only between-scale mutation semantics and IEEE ties-to-even coordinates.
  Caller maps remain unchanged. Reduction runs on the input device; the small
  cached integer coordinate tables round on the host for backend consistency.
- Initial RED: `paint-position-red-cuda-v1.log` (also CPU red retained).
  F32 CPU, F16 CUDA and production three-batch/six-view CUDA captures match all
  indices exactly (`paint-position-green-v1.log`,
  `paint-position-half-cuda-v1.log`, `paint-position-production-cuda-v1.log`).
- Peer review identified mixed half/int64 division rounding that those captures
  missed. The new 2049-valid-pixel regression failed (coordinate 3 vs expected 4)
  in `paint-position-denominator-red-v1.log`, then passed after denominator
  conversion to half before division. Checked-in F32/F16 fixtures are extracted
  from the retained upstream captures with source/capture SHA provenance.
- 180 CPU Hunyuan tests pass, two external tests ignored
  (`paint-unet-position-cpu-v1.log`). Read-only review verified the denominator
  fix and found no further defect for finite maps. End-to-end sampling and the
  unresolved VAE/UNet half gates remain required.

## Paint UniPC trajectories and three-branch guidance

- `paint_sampler.rs` ports the pinned Diffusers VP v-prediction/order-two/bh2
  recipe, scaled-linear betas rescaled to zero terminal SNR, trailing timesteps,
  zero final sigma, warmup and lower-order-final. State commits only after a
  successful step; x0 conversion precedes correction. This does not reuse or
  change the Wan flow sampler.
- CPU and CUDA upstream captures cover 1, 2, 3, 15, 30 and 48 steps, retaining
  model outputs, every resulting sample, x0 history and corrected sample.
  The initial CPU capture failed on shared tensor storage; v2 clones tensors
  only for serialization. All failures and captures remain retained.
- Initial RED is `paint-sampler-red-v1.log`. F32 trajectories pass with worst
  max 0.00000059604645. CPU F16 trajectories are bit-identical after reproducing
  left-scalar operand semantics (`paint-sampler-green-v3.log`). CUDA F16 is
  bit-identical through 30 steps; at 48 steps max is 0.00012207031 and RMS
  0.000011143445 (`paint-sampler-cuda-v2.log`). Original bounds remain F32
  max 5e-5/RMS 1e-5 and F16 max .005/RMS .002.
- Review exposed two precision details. NumPy arange computes its actual
  increment from the first rounded addition, rather than the nominal ratio;
  48-step and 242-step regressions were RED before the fix. The corrected
  formula matches all 1..1000 step counts in the retained NumPy sweep.
  PyTorch CPU `F32_scalar * half_array` rounds the coefficient to half, while
  CUDA's CPU-scalar fastpath preserves F32; reversing the CPU operands changes
  that behavior. CUDA scalar division instead multiplies by an F32 reciprocal.
  Direct probes, ATen source and actual corrector-rho captures are retained in
  `paint-scalar-device-review-v1/`. CPU/CUDA rho solves round to identical F16
  coefficients, ruling out that hypothesis.
- Some upstream arange schedules contain both timestep0 and -1, which
  interpolate to identical sigma. An exact upstream 769-step replay returns
  all NaNs at the final corrector in both dtypes; 15 and 242 remain finite
  (`paint-sampler-degenerate-review-v1/results.json`). Mold deliberately refuses
  nondecreasing sigma schedules in `PaintUniPc::new`, before generation;
  `PaintSchedule` still exposes the exact arrays for inspection. Its regression
  was RED in `paint-sampler-degenerate-red-v1.log` before the refusal was added.
- `paint_guidance.rs` implements the two separate guidance updates, material/view
  ordering and optional azimuth weighting. The normal upstream call does not
  forward azimuths and uses weight1 for every view. Fixtures execute the
  original pinned pipeline statements, extracted without arithmetic changes.
  Initial RED: `paint-guidance-red-v1.log`; CPU and CUDA then pass all F32/F16
  1/2/6-view default/azimuth cases (`paint-guidance-green-v1.log`,
  `paint-guidance-cuda-v1.log`). Peer review found no guidance discrepancy.
- Combined CPU qualification passes 187 Hunyuan tests (two external tests
  ignored), including timestep/shape refusal without state advancement,
  training beta/cumulative-alpha arrays, sigma arrays and degenerate-schedule
  refusal (`paint-sampler-all-cpu-v1.log`). CPU and CUDA all-target Clippy pass
  with warnings denied (`paint-sampler-clippy-v1.log`,
  `paint-sampler-cuda-clippy-v1.log`). Read-only review verified the numerical
  fixes and guidance path.
- These are sampler/guidance component qualifications. A complete paint render,
  its VAE/UNet half gates, decoding, texture bake and client integration remain
  required; component passes do not close those gates.

## Remaining gates

Full-pipeline P0 oracle parity and the remaining P1–P15 implementation/qualification
gates remain open. A successful 2.1 render does not close those gates. This ledger will record measured results as each gate is exercised;
it is not a completion checklist with assumed passes.


## Request-owned paint denoising checkpoint

- `paint_denoiser.rs` joins the complete main/reference networks, projector,
  learned material text, Rust position pyramid, guidance and fifteen-step UniPC.
  The strict loader consumes the entire installed checkpoint. Prepared conditions
  borrow their exact loaded owner, preventing cache reuse with different weights;
  geometry is repeated for both materials and all three guidance branches, while
  the first two DINO inputs alone are zeroed. Cancellation is checked before
  preparing conditions and after every sampling step.
- `paint-denoiser-red-v1.log` records the initial failing wrapper test. The
  standalone tiny F32 network passes (`paint-denoiser-green-v1.log`). Review found
  that repeated timestep outputs needed iteration-qualified filenames; the test
  now requires a new output directory and retains every pass separately.
- `paint-denoiser-installed-cuda-v1.log` passes installed-weight F32 conditioning,
  all sixteen reference caches, forwards at 500/400/500, cancellation at callbacks
  zero and two, and an independent complete fifteen-step guided trajectory after
  cancellation. Final maximum error is .0000057816505, RMS .0000014784437. Original
  bounds remain maximum 1e-4 / RMS 1e-5. Capture and tensors are retained under
  `paint-denoiser-installed-oracle-v1/` and `paint-denoiser-installed-candle-v1/`.
- The tiny random-weight guided trajectory **fails** those same bounds late in
  sampling: final maximum .00035363436 / RMS .00006857673
  (`paint-denoiser-guided-cuda-v1.log`). Its conditioning and individual forwards
  pass. Independent source review found no omitted initial-noise scale, model
  input scaling, timestep, guidance or cache-reset operation. This stress case
  remains an open numerical gate; installed-weight success does not erase it.
- The default CPU Hunyuan suite passes 183 tests with two ignored hardware tests
  (`paint-denoiser-all-cpu-v1.log`). CPU warnings-denied Clippy including
  `mesh-texture` passes (`paint-denoiser-clippy-v1.log`), as does the CUDA/cuDNN
  test build (`paint-denoiser-clippy-cuda-v1.log`). Full six-view F16 guided
  trajectory qualification is running separately and is not claimed here.

## Rust cuDNN paint VAE diagnostic

- The ignored VAE qualification test now uses the existing family-scoped
  convolution policy and records the actual thread-local cuDNN dispatch count.
  A requested cuDNN comparison refuses to pass with zero dispatches. The small
  64-pixel fixture hits that guard (`paint-vae-cudnn-cuda-v1.log`), because the
  fork's convolution size threshold keeps it on im2col.
- The real 512-pixel fixture executes 72 cuDNN convolutions and **fails**, with
  sampled latent maximum .168396 / RMS .0030051953690470204
  (`paint-vae-cudnn-real512-cuda-v1.log`). This is worse than the retained im2col
  result. The fork currently uses an F16 convolution compute descriptor for F16
  input; its relationship to Torch's accumulation policy is under investigation.
  No production convolution policy or tolerance has changed. All tensors and
  failed diagnostics are retained.


## Full six-view denoising trajectory

- The unchanged installed checkpoint runs all fifteen steps at six views and
  64x64 latents (512-pixel output resolution), with three guidance branches and
  both materials, in `paint-denoiser-production-cuda-v1.log`. This is an im2col
  run at Candle `bedc2874`, before the later cuDNN correction. All trajectory
  outputs pass the original half bounds: final maximum .010253906 / RMS
  .0013163141. Cancellation and restart within the loaded owner also pass.
- The complete gate still **fails**: reference caches `up_1_2_0`, `up_2_0_0`, and
  `up_2_1_0` have maxima .041870117, .072265625, and .02722168 respectively, above
  .02. Their RMS values remain below .002. The retained tensors include every
  cache and every trajectory sample. Total test time, including cancellation
  runs and repeated forwards, is 520.09 seconds. Synthetic encoded conditions
  were used; this is not yet a complete image-to-PBR qualification.

## Corrected cuDNN accumulation descriptor

- Source review proves the fork's old F16 compute descriptor differs from Torch
  2.5.1 (`aten/src/ATen/cudnn/Descriptors.h:202-205` and
  `aten/src/ATen/native/cudnn/Conv_v8.cpp:124-129`). Both Torch routes promote
  HALF convolution compute to FLOAT while preserving HALF tensor storage.
  Source and review are retained under `paint-cudnn-compute-review-v1/`.
- The isolated Candle branch `work/hunyuan3d-f16-convolution` corrects Conv1D and
  Conv2D in commit `de478c1c47281161bdd4f60d02c69915b918eefd`. Both deterministic
  regressions fail before correction (`candle-cudnn-accumulation-red-v2.log`);
  Conv2D relative maximum deviation is .010695187. Afterward both pass at the
  unchanged .001 bound: Conv1D .0008271299, Conv2D .00076394196. All eleven cuDNN
  tests pass (`candle-cudnn-accumulation-green-v1.log`), including BF16/F32,
  striding, dilation, grouping, disabled policy and below-threshold behavior.
  The tests assert actual cuDNN dispatch and F16 output. Independent review
  confirmed that tensor storage and alpha/beta ABI remain unchanged.
- All mold Candle declarations and both lockfiles now pin that one revision,
  including H3 payload provenance. The Nix source hash is
  `sha256-CREJfuti4jbOkCce8ywfGjtdquCGjW0QQqQr9WUpvJQ=`; the same archive-prefetch
  method reproduced the previous pinned hash before computing this one.
  `candle-cudnn-pin-contract-v1.log` passes the single-identity contract. The
  independent real512 Torch VAE comparison remains required after this backend
  correction; the numerical regression alone does not close it.

- The corrected fork's actual512 VAE comparison dispatched the same 72 cuDNN
  convolutions and improved sampled latent maximum from .168396 to .029296875,
  RMS .0008123263517362374 (`paint-vae-cudnn-f32accum-real512-cuda-v1.log`). It
  **still fails** the original .01 maximum bound. Its complete encoder and
  decoder tensors remain retained in `paint-vae-cudnn-f32accum-real512-candle-v1/`.
  The pinned build passes 194 Hunyuan tests with fourteen hardware/oracle tests
  ignored (`candle-cudnn-pin-hunyuan-tests-v1.log`).


## Spatial projection bias-rounding correction

- CPU review of the full guided run localizes all over-bound cache errors to
  spatial borders (`paint-guided-cache-review-v1/`). The new reference-trace
  capture retains each failing site's GroupNorm, input projection and norm1
  inputs, outputs and weights, plus original input strides in metadata.
  `paint-reference-spatial-trace-cuda-v1.log` replays identical inputs and fails
  only projections (maximum .015625, .25, .03125); both normalization stages
  pass. Float64 CPU calculations confirm a bias boundary mismatch rather than
  float32 summation error (`paint-projection-review-v1/`).
- Torch 2.5.1 `ATen/native/Linear.cpp:94-120` uses fused addmm for 2D or
  contiguous ND, but matmul followed by bias addition for non-contiguous ND.
  The spatial B,HW,C input is the latter. The retained minimal CPU/CUDA oracle
  `paint-projection-bias-oracle-v1.json` returns 452.25 for strided3D and 452.0
  for contiguous3D or strided2D using identical values. Both Rust regression
  tests fail first (`paint-projection-bias-red-v1.log`), then pass
  (`paint-projection-bias-green-v1.log`).
- `stable_diffusion::linear::forward` now shares that dtype/layout policy between
  paint UNet and opt-in VAE; existing SD numerics remain separate. The spatial
  UNet reshapes before transposing, preserving the non-contiguous layout that
  Candle's previous permute/reshape sequence copied away. Native half matmul
  and a separate half bias addition reproduce **all three captured projections
  bit-for-bit** (`paint-reference-spatial-trace-cuda-v2.log`). Their GroupNorm
  and norm1 replays also pass the unchanged .002 maximum / .0003 RMS bounds.
- Full six-view installed UNet replay still **fails** two cache maxima:
  `up_1_2_0` .056640625 and `up_2_0_0` .048828125. The third previously failing
  cache now passes at .01940918. Individual model outputs continue to pass
  (`paint-projection-unet-cuda-v1.log`). The exact-input projection defect is
  corrected, but inherited network drift remains a separate gate.
- The VAE had the same fused-only assumption and forced its Q/K/V view
  contiguous, contrary to Diffusers 0.30 `AttnProcessor2_0:2188`. Its wrapper
  regression fails before correction (`paint-vae-linear-bias-red-v1.log`). With
  the shared policy and original transpose layout, seven shared SD/VAE tests
  pass, including the old default behavior (`paint-vae-linear-bias-green-v1.log`).
  Real installed VAE parity is rerun separately and is not inferred from these
  focused tests.

- After sharing the projection policy, the Hunyuan CUDA suite passes 196 tests
  with fifteen explicit hardware/oracle tests ignored
  (`paint-projection-all-hunyuan-v1.log`). The installed F32 fifteen-step
  denoiser still passes unchanged bounds, final maximum .000008225441 / RMS
  .0000016663199 (`paint-projection-denoiser-f32-cuda-v1.log`).
- The real512 half VAE maximum remains **over bound**: im2col .034179688 / RMS
  .0008929751412068291 (`paint-vae-linear-bias-real512-cuda-v1.log`); corrected
  cuDNN .029296875 / RMS .0008084002150359077 with 72 dispatches
  (`paint-vae-linear-bias-cudnn-real512-cuda-v1.log`). This correction reproduces
  the missing operation boundary but does not close the full VAE gate. Both
  encoder/decoder traces are retained. Independent review confirms the shared
  helper is confined to paint UNet and the VAE's opt-in Diffusers path.
- CUDA warnings-denied Clippy for both inference and shared model libraries
  passes after removing the import made unused by extraction
  (`paint-projection-shared-clippy-v2.log`); the initial lint failure remains
  retained in v1. Rust formatting checks pass.


## Paint image boundaries and staged runner

- `paint-pixels-oracle-v1` captures unchanged Tencent `encode_images` through
  AST extraction and Diffusers 0.30 `VaeImageProcessor.postprocess`. Checked-in
  fixtures cover input-dtype normalization, half-to-float model boundaries,
  material ordering and ties-to-even RGB conversion. Three CPU tests pass
  (`paint-pixels-green-v1.log`) after the absent-implementation failure.
- Position maps now retain their own F16/F32 precision independently of encoded
  conditioning. Tencent's PIL converter hardcodes half even for a float model;
  the focused guidance regression fails before this correction and passes after
  it (`paint-map-dtype-red-v1.log`, `paint-map-dtype-green-v1.log`).
- `paint_pipeline` connects DINO, VAE reference/normal/position encoding, the
  fifteen-step denoiser and material decoding with explicit posterior/diffusion
  noise. Lexical scopes release each network before loading the next; VAE
  decoding reloads after denoising. Tensor-boundary callbacks support retained
  comparisons and cancellation. Validation covers all image/noise inputs before
  checkpoint loading. This is internal integration, not a qualified paint engine.
- Read-only peer review caught the reference-image cast boundary: reference
  pixels must reach model precision BEFORE normalization, whereas geometry
  pixels retain their input precision. A distinct failing regression captures
  that difference (`paint-reference-cast-red-v1.log`).
- `scripts/capture-hunyuan3d-paint-pipeline.py` runs Tencent's actual multiview
  entrypoint, retaining DINO input/output, three image/posterior/latent groups,
  initial diffusion noise, fifteen samples, decoded pixels and material PNGs.
  `paint-pipeline-oracle-v2` completed six 512-pixel views in 10.455 seconds with
  13,771,083,776 allocated / 20,476,592,128 reserved peak CUDA bytes. These are
  Torch allocator measurements, not a Rust memory claim. The earlier failed
  module-loader invocation remains in v1. Model loading resolves Tencent's
  published legacy class name without replacing any network forward.

- The staged runner's three validation/cancellation/reference-cast regressions
  pass (`paint-pipeline-green-v3.log`); the complete paint component selection
  passes 36 tests with thirteen explicit oracle tests ignored
  (`paint-pipeline-all-paint-v1.log`). Inference Clippy with CUDA/cuDNN and
  mesh-texture features passes with warnings denied (`paint-pipeline-clippy-v1.log`).
- Full installed Rust integration is captured separately in
  `capture-20260906T100514Z-368852622599` using the exact Tencent input/noise
  tensors. Conditioning maxima already exceed the unchanged .01 VAE limit
  (reference .011230469, normal .013671875, position .010009766), so this run
  is not a parity pass regardless of its eventual material output. All stages
  continue for diagnostic comparison; the test reports failure after saving
  its material outputs and comparison table.

- Follow-up read-only review confirms oracle hooks preserve returned values and
  posterior draw order. Its remaining scope caveats are explicit: prepared
  tensors bypass PIL/DINO image preprocessing, and the first integration run
  records but does not gate final PNGs. The follow-up test now compares upstream
  decoded pixels converted by Rust against all twelve actual Tencent PNGs,
  and measures generated PNG errors with bounds propagated from the unchanged
  decoder limits. That follow-up compiles under all-target warnings-denied
  Clippy (`paint-pipeline-clippy-v2.log`); execution remains a separate gate.


## Full prepared paint result and source image preparation

- `capture-20260906T100514Z-368852622599` completed every inference stage and
  retained all twelve material PNGs in 391.898 seconds, with a sampled board
  peak of 18,507 MiB. It **fails parity**: final latent max .17724609375 / RMS
  .0024956353; decoded max .13848877 / RMS .00082838815. The existing maximum
  bounds and final latent RMS gate remain open. This also does not prove a
  16 GiB execution budget.
- Offline comparison of the actual material files is retained in
  `paint-pipeline-image-comparison-v1/comparison.json`. PSNR spans 60.576–65.304
  dB; channel RMS is .138–.239 bytes. Albedo view5 has six channels differing
  by more than8 bytes (max13), and MR view2 has eighteen (max18); all other
  views stay at max2–6. High PSNR does not waive the maximum-error gates.
- `paint_images::PaintImages::prepare` implements the actual two appearance
  resizes, white composition, DINO preprocessing and ordered condition image
  conversion. RGBA premultiply/unpremultiply follows Pillow12.3 Convert.c and
  preserves Image.py's same-size bypass; RGB geometry keeps the existing
  bicubic resizer. Cancellation precedes work and propagates through rows and
  resize callbacks. Two tests pass after the absent-implementation failure,
  including exact down/up/same-size alpha fixtures (`paint-images-green-v3.log`).
- The first real-image comparison exposed a fixture-mode mismatch, retained in
  `paint-images-real512-v1.log`: the saved source is a palette PNG carrying
  transparency, while the v2 neural oracle explicitly converts it to opaque
  RGB. The preparation test now names RGB versus RGBA, and the capture script
  can feed either mode to Tencent before its unchanged resize/composition.
  There is no production-pixel workaround for the mismatched fixture.
- All four prepared tensor boundaries match **bit-for-bit** on six512 views
  for opaque RGB (`paint-images-real512-v2.log`) and full-resolution transparent
  RGBA (`paint-images-rgba-real512-v1.log`). The latter compares against a new
  full Tencent capture, `paint-pipeline-rgba-oracle-v1`, using the original
  1024-pixel source converted to RGBA before Tencent's512 resize. This proves
  the source preprocessing, not neural parity on those RGBA conditions.
- Independent read-only review found no alpha/order/dtype defect and confirmed
  cancellation propagation. It correctly distinguished opaque qualification
  from alpha fixtures before the separate full RGBA capture was added.
- The complete paint component selection now passes38 tests, with fourteen
  explicit oracle tests ignored (`paint-images-all-paint-v1.log`); the two
  real-image oracle tests above were run explicitly. All-target CUDA/cuDNN
  Clippy passes with warnings denied (`paint-images-clippy-v2.log`).


## P7 projection sampling and weighted merge

- `paint_bake::TextureBaker` streams projected views into color/weight sums.
  It mirrors `ViewProcessor.bake_from_multiview` and `MeshRender.fast_bake_texture`:
  camera weight times cosine^4, STRICT >99% positive-weight overlap skipping,
  positive accumulated coverage independently of final trust, and final
  denominator clamp / trust threshold at1e-8. Neither material stream changes
  gamma encoding. `paint-bake-oracle-v1` executes both unchanged upstream
  methods via AST extraction. All four prefixes of overlap and tiny-weight
  scenarios pass with exact trust masks and color error <=1e-7 after the RED
  absent-implementation test (`paint-bake-green-v1.log`).
- `paint_back_sample` mirrors Tencent's actual back_sample branch, including
  resolution-based pixel coordinates, clamped endpoint interpolation, lower
  pixel visibility/cosine/depth decisions, strict depth difference <.003 and
  unique UV texel placement. `back-sample-oracle-v1` executes that unchanged
  branch with image-edge, outside-frustum, occluded, invisible and near-threshold
  cases. Rust colors match within1e-7 and cosine/boundary maps match exactly
  (`back-sample-green-v1.log`), after the absent-function RED run.
- Read-only peer review found no arithmetic/addressing defect but identified
  missing mid-loop cancellation in the merger. Constructors and every long
  validation/merge/finalize loop now poll checkpoints; cancellation during
  accumulation invalidates the session so a partial texture cannot be reused
  or finalized. Validation errors preserve prior state. Back-sampling keeps
  partially computed outputs local and polls validation and sampling loops.
  The new cancellation test fails before the callback API exists, then passes
  (`paint-bake-cancel-green-v1.log`). Follow-up review confirms the lifecycle
  correction and reports no remaining cancellation defect.
- The paint component selection passes43 tests with fourteen explicit oracle
  tests ignored (`paint-bake-all-paint-v1.log`); all-target CUDA/cuDNN Clippy
  passes with warnings denied (`paint-bake-clippy-v1.log`).
- These are qualified P7 primitives, not the complete bake path. Camera/UV
  projection construction, reliability masks, RealESRGAN view integration,
  unseen-texel propagation, Navier–Stokes fill and final PBR GLB qualification
  remain part of the active campaign. Existing neural parity failures remain
  unchanged; no texture capability is newly advertised by this checkpoint.

## P7 UV geometry and camera qualification

- `paint_uv::UvGeometry` rasterizes prepared, V-flipped UVs and retains covered
  paint-frame positions, source face normals and row-major texel indices.
  Degenerate geometry keeps Tencent's zero normal. In-place compaction avoids
  allocating another two full-resolution geometry arrays. Raster traversal,
  extraction and projection poll cancellation callbacks.
- `capture-hunyuan3d-paint-uv.py` executes the pinned Tencent renderer's actual
  `set_mesh` / `extract_textiles` and camera matrices. The checked-in small
  fixture covers asymmetric geometry, UV seams and a degenerate geometric face
  with a valid UV chart (`paint-uv-oracle-v1`, RED `paint-uv-red-v1.log`, GREEN
  `paint-uv-green-v2.log`).
- The first actual 4096 atlas comparison failed on one extra covered texel,
  index 8085715 (`paint-uv-real4096-candle-v1` and `paint-uv-real4096-cpu-v1.log`).
  All shared positions and normals already met the unchanged 2e-6 gate. The
  isolated edge test reproduces that failure (`paint-uv-edge-red-v1.log`).
  Tencent derives alpha from beta/gamma, tests both bounds, and uses distinct
  fused gamma arithmetic for coverage and interpolation. Read-only peer review
  verified the installed CUDA module's SASS, including double-precision alpha
  subtraction (`paint-uv-edge-review-v1/review.md`, retained disassembly).
  The correction is confined to the existing Tencent paint raster branch;
  ordinary gallery raster arithmetic remains unchanged.
- Final `paint-uv-real{1024,2048,4096}-candle-v3/comparison.json` records exact
  coverage at **638,538 / 2,557,480 / 10,239,775 texels** on the retained
  250,396-face mesh. Across all sizes, maximum position error is 1.20e-7,
  normal error 1.56e-6, and six-camera projection error 2.39e-7. Corresponding
  `paint-uv-real*-cpu-v3.log` tests all pass. The broad Hunyuan3D selection
  passes 212 tests with 18 explicit oracle tests ignored
  (`paint-uv-raster-green-v3.log`). These results establish UV construction,
  not final textured GLB or end-to-end paint completion.
- Follow-up review found that the early degeneracy check also needed paint's
  fused determinant: separately rounded products can cancel to zero when
  Tencent's determinant is nonzero. The exact-power-of-two vertex regression
  fails first (`paint-uv-area-red-v1.log`), then passes with the correction;
  it separately asserts that the ordinary raster keeps its prior behavior.
  Peer review confirms the finding is closed. Final functional selection:
  **215 passed, 18 ignored** (`paint-uv-edges-final-tests-v1.log`).
- `paint-uv-final-qualification-v1/run.json` retains the tested binary, binary
  and source digests, oracle metadata/digests, commands, exit codes, artifact
  digests, timings and child-process RSS accounting. All three size checks
  pass again with the final degeneracy correction. The subsequent Clippy
  cleanup only replaces `Option::unwrap_or_else` with `unwrap_or` for a pure
  arithmetic fallback; it does not change the selected depth calculation.
  All-target CUDA/cuDNN Clippy passes with warnings denied
  (`paint-uv-edges-clippy-v2.log`); formatting and diff whitespace checks pass.

## P7 depth-edge primitive

- `paint_edges::depth_edges` ports the exact `cv2.Canny(depth_bytes, 30, 80)`
  recipe used by Tencent's `render_sketch_from_depth`. It uses replicated
  Sobel borders, L1 magnitude, OpenCV's integer direction thresholds and
  asymmetric nonmaximum comparisons, then eight-connected hysteresis.
  Dimensions are bounded at 2048; allocations and long loops poll cancellation.
- The executable oracle uses installed OpenCV 4.10.0, matching the read-only
  source clone at `71d3237a093b60a27601c20e9ee6c3e52154e8b1`.
  `capture-hunyuan3d-paint-edges.py` records eight cases: singleton, one-row,
  one-column, random strong and weak gradients, directional edges, plateaus,
  and weak edges connected to a strong segment versus detached weak edges.
  All final masks match exactly (`paint-edges-oracle-v1`, RED absent-function
  `paint-edges-red-v1.log`, GREEN `paint-edges-green-v1.log`). Cancellation at
  every observed checkpoint also passes. Read-only peer review reports no
  Sobel, suppression, hysteresis or cancellation defect.
- This qualifies the byte-image edge detector. Camera depth normalization,
  cosine rejection, erosion/dilation and their integration into back-projection
  remain active implementation work, alongside the remaining paint pipeline.

## P7 texture reliability and camera geometry

- `paint_reliability::ReliabilityMask` now composes visible-only depth
  normalization, the qualified Canny detector, cosine rejection at75 degrees,
  visibility erosion and edge dilation. Integral images implement the exact
  binary square convolution. Radius0 does not remove edges from visibility;
  outside-image samples contribute zero even for inverted visibility; the
  dilated boundary map remains independent of final visibility. Flat depth
  retains upstream's NaN-to-zero byte behavior, and empty visibility is refused.
- `capture-hunyuan3d-paint-reliability.py` executes the unchanged upstream
  statements and sketch method on six cases spanning holes, depth steps,
  radius0/1/2/8, flat depth and neighboring cosine threshold values. The first
  harness run lacked its namespace's device field and is retained as a failure;
  `paint-reliability-oracle-v2` is the successful fixture. The absent-type RED
  test is `paint-reliability-red-v1.log`; the GREEN fixtures and cancellation
  sweep are `paint-reliability-green-v2.log`. Peer review found no filtering,
  integer-bound, border or cancellation defect.
- The actual CUDA renderer captured all six2048 views of the retained mesh
  (`paint-reliability-mesh-oracle-v1`). Filtering those raw camera fields in
  Rust produces exact visibility and boundary masks, with cosine error no
  greater than1.20e-7 (`paint-reliability-mesh-candle-v1/comparison.json`).
- `paint_camera::CameraGeometry` constructs those fields from the mesh, using
  normals computed AFTER camera transformation and interpolated camera-space Z.
  The existing paint raster now exposes cancellable projection and traversal.
  The six-view small CUDA fixture catches a depth-reduction ordering difference:
  `custom_rasterizer/render.py` multiplies independently, then Torch sums the
  three contributions in0+2+1 order. Applying that order matches every captured
  depth pixel on both the small and actual mesh when fed the oracle barycentrics
  (`paint-camera-depth-reduction-v1/comparison.json`).
- Full camera comparison exposed near-overlap triangle selection differences.
  Installed CUDA SASS establishes fused screen mapping and weighted coverage
  depth accumulation (`paint-uv-edge-review-v1/depth-review.md`). After those
  changes only six pixels across the six real views selected another triangle;
  all other camera depth pixels matched exactly. A further actual `pos_clip`
  capture proves that Torch's preceding Z projection is SEPARATE multiplication
  and addition, followed by CUDA's FUSED depth mapping. The exact-bit regression
  fails first (`paint-camera-depth-rounding-red-v1.log`) and pins that boundary;
  `paint-camera-projection-diagnosis-v1/comparison.json` records the independent
  comparison on all vertices/views. Ordinary gallery raster arithmetic remains
  unchanged throughout these paint-specific corrections.
- Final mesh-to-reliability qualification passes on all six2048 views:
  **camera depth, raw visibility, reliable visibility and boundaries match
  exactly**, maximum normal error is1.78e-6 and cosine error1.17e-6, below the
  unchanged2e-6 gates (`paint-camera-mesh-candle-v4/comparison.json`). The isolated
  competing-face calculation also selects the correct triangle at all six
  disputed pixels (`paint-camera-face-diagnosis-v2/comparison.json`).
- The final Hunyuan3D selection passes **220 tests, 20 explicit oracle tests
  ignored** (`paint-camera-reliability-final-tests-v1.log`). All-target CUDA/cuDNN
  Clippy passes with warnings denied (`paint-camera-reliability-clippy-v1.log`).
  The full4096 UV oracle comparison still passes after the shared paint depth
  correction (`paint-uv-real4096-camera-regression-v1.log`). Both constructors'
  cancellation sweeps and geometry validation tests pass. Read-only peer review
  confirms the numerical findings are closed with no remaining concern in scope.
- `paint-camera-final-qualification-v1/run.json` preserves the test binary,
  source and artifact digests, oracle device identity, command/environment and
  comparison results. All earlier failures and intermediate captures remain.
  This completes camera-to-reliability construction; RealESRGAN integration,
  complete baking/filling/export, neural parity and the remaining campaign
  deliverables continue as active work.

### P7 RealESRGAN scalar arithmetic and oracle capture

- `scripts/capture-hunyuan3d-paint-upscaler.py` calls the unchanged pinned
  Tencent `imageSuperNet`, including its RGB-to-BGR wrapper behavior and
  untiled half-precision x4 inference. It verifies all 702 `params_ema`
  tensors against the installed safetensors, records source/artifact hashes,
  captures pre-activation convolution and RRDB boundaries, and refuses an
  existing output directory. Hooks clone outputs before subsequent in-place
  LeakyReLU. Read-only peer review confirmed capture validity.
- `paint-upscaler-oracle-small-v1` preserves a 16px crop and all 63,488 finite
  FP16 scalar inputs. The CUDA regression first fails with 20,410 residual
  multiplication differences and 10,205 LeakyReLU differences
  (`paint-upscaler-scalars-red-v1.log`). Torch uses F32 scalar opmath; Candle's
  half affine prematurely rounds 0.2 to half. The shared RRDB correction
  widens these operations, rounds their results back to half, and leaves
  residual addition separate. Both exhaustive comparisons now have **zero
  differing values** (`paint-upscaler-scalars-green-v1.log`). Non-F16 paths
  remain unchanged; the release fragment and upscaling guide disclose the
  half-precision output correction.
- `paint-upscaler-oracle-albedo00-v1` captures the retained actual 512px albedo
  view through 2048px output: 1.4723 seconds, 2,350,578,688 bytes peak Torch
  allocation. This is oracle allocation with capture hooks, not a Rust memory
  measurement or board-level qualification. Rust stage/image parity, all
  material views, cancellation and staged-engine integration remain open.
- Exhaustive finite-half clamping and byte conversion finds no rounding
  disagreement: half output multiplied by 255 is exact in F32, and its only
  half-integer tie in the byte interval is 127.5, rounded to 128 by both
  algorithms. Thus existing Rust final rounding needs no paint override for
  the untiled F16 route. This conclusion does not apply to F32 CPU output or
  blended tiles. Peer review independently confirmed the argument.
- Peer review found no scalar correction defect. Actual 512-to-2048 Rust
  memory must still be measured: widening LeakyReLU creates F32 temporaries,
  particularly at the final 64-channel 2048px convolution.
- The upscaler unit selection passes all 28 tests, with the explicit CUDA
  oracle test separately qualified above (`paint-upscaler-unit-green-v1.log`).
  All-target CUDA/cuDNN Clippy passes with warnings denied
  (`paint-upscaler-clippy-v2.log`). The first Clippy log is retained; its only
  findings were exact dyadic decimal literals in the scalar regression,
  now annotated to preserve the oracle values verbatim.

### P7 RealESRGAN network comparisons (qualification remains open)

- The RRDB observer exposes the same nine boundaries captured from Torch,
  including convolution outputs before in-place activation. The ignored
  `pretrained_paint_upscaler_matches_tencent` comparison requires every oracle
  stage and F16 dtype, saves each actual stage plus PNG and comparison JSON,
  rejects nonfinite tensors, and checks raw maximum error 0.01 and final byte
  maximum 8. Read-only review caught the initial missing-stage skip; the final
  test requires all stages and checks the comparison count.
- The 16-to-64 crop passes (`paint-upscaler-small-candle-v1`): final raw max
  0.000732421875, final PNG max 1 byte, largest intermediate max 0.0040283203125.
- Full actual albedo view 00 is retained at
  `capture-20260906T121551Z-05b48acb8c19/upscaler`: final raw max
  0.00927734375, RMS 0.00039832581, PNG max 2 bytes. Full MR view 00 is at
  `capture-20260906T121552Z-1825b9e2a101/upscaler`: final raw max 0.009765625,
  RMS 0.00035947499, PNG max 3 bytes. Both final outputs pass those limits,
  but both tests correctly **FAIL** intermediate limits: error grows after
  body 11, reaching 0.14621 albedo / 0.15576 MR at the pre-activation HR
  convolution. These failures are retained and not waived.
- Both isolated GPU captures sampled 7,175 MiB board use. Runtime including
  observer copies, comparisons and artifact writes is 19.67 / 19.51 seconds;
  this is not uninstrumented inference timing. Each capture records the test
  binary hash, exact checkpoint, oracle stage file and command. The production
  GPU was not used. Rust is fed the captured normalized half input here, so
  these results do not yet qualify the paint adapter's image preprocessing.
- The observer preserves all 28 existing upscaler tests
  (`paint-upscaler-observer-unit-v1.log`); all-target CUDA/cuDNN Clippy passes
  (`paint-upscaler-parity-clippy-v1.log`). All six views in both streams,
  intermediate-error diagnosis, cancellation and engine integration remain
  active requirements.

### P7 cancellation and convolution-layout investigation

- The regression `cancellation_during_last_forward_never_returns_an_image`
  first reproduces a cancelled final tile returning an image
  (`paint-upscaler-cancel-red-v1.log`). Both single-pass and tiled routes now
  check cancellation after the forward, before output can escape. Blend rows
  and normalization chunks also poll, and the engine checks after encoding.
- RRDB inference polls at entry and after each network block/convolution
  boundary. Its cancellation sweep stops at every boundary, verifies the
  cancellation error identity, and reruns the same model to check unchanged
  output and reuse. The callback-free and checkpointed outputs match exactly.
  A running GPU convolution itself remains non-preemptible. Read-only review
  found no issue in these changes.
- The upscaler oracle now records live input/output strides before fixture
  serialization. `paint-upscaler-layout-oracle-v1` reproduces the previous
  albedo PNG and stage-file hashes exactly. Its first input is channels-last
  according to `is_contiguous`, but the first output and every later stage
  are contiguous NCHW. Torch 2.5.1 `ConvUtils.h:328` instead consults
  `suggest_memory_format`; `MemoryFormat.h:139-145` rejects the singleton
  batch stride 3 after accumulating the spatial extent. Thus the first
  convolution also selects NCHW. The earlier hypothesis that NHWC propagation
  explains accumulated drift is ruled out, not used to justify a backend
  change. The independent source and CPU proof remain in
  `paint-upscaler-layout-review-v1`; intermediate numerical gates remain open.
- Final upscaler selection passes 30 tests, with two separately run CUDA oracle
  tests ignored (`paint-upscaler-cancel-green-v2.log`); all-target CUDA/cuDNN
  Clippy passes with warnings denied (`paint-upscaler-cancel-clippy-v1.log`).
  The first build log is retained: its callback error-conversion compile
  errors were corrected without changing cancellation behavior.

### P7 first-convolution dispatch isolation

- Canonical Torch cuDNN logging records 351 successful backend executions;
  read-only review separates executed plans from heuristic candidates and
  duplicate internal logs (`paint-upscaler-cudnn-review-v1`). Every dense block
  uses the same engines, so there is no algorithm transition after body 11.
  A controlled Torch legacy-API run is identical at every captured stage to
  the canonical newer-API run (`paint-upscaler-v7-comparison-v1`).
- Rust records 350 cuDNN calls: its first RGB convolution falls below Candle's
  automatic size threshold and takes im2col. That first stage differs in just
  1,242 of 16,777,216 values. The oracle's explicitly diagnostic
  `--diagnostic-first-features` option substitutes that retained Rust stage
  and runs the remaining Torch network unchanged. All eight downstream
  captured stages then match Rust **byte for byte**, including signed zero
  (`paint-upscaler-common-first-comparison-v2`). This isolates the accumulated
  drift to the first convolution for this fixture. The normal Rust parity
  test rejects diagnostic substitutions as qualification oracles.
- Candle commit `5c74a518ce1c11e31036949b736a35649a1d96ca` lets an explicit
  Conv2D algorithm bypass the size heuristic while preserving the enabled
  policy requirement and unsupported-launch fallback. The GPU regression
  first fails (`candle-explicit-cudnn-red-v1.log`); all 12 convolution tests
  then pass (`candle-explicit-cudnn-green-v1.log`). The RRDB first convolution
  explicitly selects the observed legacy `ImplicitPrecompGemm` algorithm;
  a disabled cuDNN policy still uses im2col. Qualification checks that this
  first convolution actually dispatched, not merely that cuDNN was enabled.
- All dependency declarations, both Candle-containing lockfiles, and H3's
  backend provenance point to that commit. The Nix source archive hash is
  `sha256-9Q+34Aow5+d9xzz6uedU6zTbKYtyOugbUEOyXRCnLbU=`. Single-Candle identity,
  desktop lock sync and desktop Nix source-hash contracts pass. Full upscaler
  comparison against the original unsubstituted oracle remains the exit gate.
- That original-oracle comparison now passes on both full 512-to-2048 view-00
  streams. Albedo (`paint-upscaler-explicit-first-albedo00-v1`) has zero error
  at all nine stages and identical final pixels; the independent comparator
  confirms every stage is byte-identical, including signed zero
  (`paint-upscaler-explicit-first-bits-v1`). MR
  (`capture-20260906T124151Z-2a259017c113/upscaler`) likewise has zero numerical
  stage error and identical pixels; its log asserts the first convolution
  actually dispatched on cuDNN. Peak sampled MR board use is 7,169 MiB, and
  instrumented runtime is 19.13 seconds. No tolerance changed.
- The final upscaler unit selection passes 31 tests, with two explicit GPU
  oracle tests ignored (`paint-upscaler-explicit-first-unit-v1.log`). All 220
  Hunyuan3D regression tests pass, with 20 explicit oracle tests ignored
  (`paint-upscaler-explicit-first-hunyuan-v1.log`). The remaining five views
  per stream, actual image preprocessing/adapter integration and complete
  paint publication remain active requirements.
- All-target CUDA/cuDNN Clippy passes with warnings denied after the fork pin
  and qualification guards (`paint-upscaler-explicit-first-clippy-v1.log`).
