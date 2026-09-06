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
  `uv-sliver-green-v1.log` passes all six UV tests. The full-mesh repeat is pending.
- `capture-20260906T044756Z-08905857bcb7` produced 4096-pixel albedo/MR maps and
  `textured.glb`, but remains FAILED: its final Blender version query inherited
  incompatible library paths and prevented measurement metadata publication.
  Export itself succeeded with a clean environment. Both native invocations now
  share that environment, and the version probe runs before model loading.
  The repeat is `capture-20260906T045720Z-89be259ea074`; no success is assumed.

## Remaining gates

Full-pipeline P0 oracle parity and the remaining P1–P15 implementation/qualification
gates remain open. A successful 2.1 render does not close those gates. This ledger will record measured results as each gate is exercised;
it is not a completion checklist with assumed passes.
