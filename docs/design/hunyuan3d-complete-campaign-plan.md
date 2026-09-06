# Hunyuan3D complete campaign: #1511 and #1496

Status: implementation in progress; see `docs/qualification/hunyuan3d-campaign.md`.
Research date: 2026-09-06 UTC (2026-09-05 America/Phoenix).
Base: `origin/main` at `3fb527e1`.
Branch: `work/issues-1511-1496`.
Worktree: `/home/jamesbrink/Projects/mold-1511-1496`.

## Delivery agreement

Implement the complete paint engine in [#1511](https://github.com/utensils/mold/issues/1511)
and all eight follow-on areas in [#1496](https://github.com/utensils/mold/issues/1496)
on this single long-running branch. Commit and push coherent checkpoints regularly.
The user has now authorized opening the PR only after complete implementation,
feature proof across every affected surface, and subagent peer review. The full
scope has no deferred work, including mesh-to-latent support below.
Feature completion requires executable upstream comparisons, real-weight CUDA runs,
durability and scheduling evidence, and all affected client/documentation surfaces.

Preserve every downloaded model, converted checkpoint, oracle fixture, generated
image, mesh, texture, intermediate, log, and failed-run output. Never clean these
as part of build cleanup or worktree removal. Do not reset or rewrite the user's
existing checkout. Do not modify the running Nix service to test this branch.

## Host and artifact policy

Read from the installed Nix `mold.service` environment, not an old qualification script:

```bash
export MOLD_HOME=/storage/mold
export MOLD_MODELS_DIR=/storage/mold/models
export MOLD_OUTPUT_DIR=/storage/mold/output
```

- Hardware observed: four NVIDIA L40S, each reporting 46,068 MiB, driver 595.71.05.
  One GPU was actively rendering during research. Choose devices from current
  utilization and stable UUIDs for every experiment; never assume all four are free.
- Observed capacity: 279 GiB available on the mold dataset; 145 GiB on the root
  filesystem. Recheck before weight acquisition, conversion, fixture capture and builds.
  Report insufficient space rather than deleting existing data. Store large build
  caches on an appropriate persistent volume if root headroom becomes limiting.
- Existing files include Hunyuan3D 2.0 fp16 and RealESRGAN x4plus fp16. Reuse them.
  This research downloaded configuration files only, not model weights.
- Campaign evidence root:
  `/storage/mold/output/verification/hunyuan3d/campaign-1511-1496/`.
  Raw issue snapshots and HF configs are in `research-20260906/`.
- Each execution gets a new UTC/nonce directory. Record branch commit, upstream
  commit, binary digest, model identities, inputs/digests, exact argv, resolved
  generation profile, frozen runtime policy, seed/noise fixtures, device UUID,
  timing, peak VRAM/RSS, exit status, metrics, and artifact hashes in `run.json`.
  Preserve both success and failure directories. Never overwrite a prior run.
- Oracle environments and caches remain persistent under the Nix mold home;
  reference source clones are in gitignored worktree `tmp/`. Explicitly direct
  Hugging Face caches into persistent storage and map existing weights into oracles
  instead of downloading duplicate repositories.
- Integration servers use loopback, a distinct port, an explicit campaign DB and
  instance identity while retaining the same MOLD_HOME/model store. Validate queue
  ownership isolation first: changing only the port is not sufficient. Never let
  a test server adopt production's retained queue. Recovery test fixtures and all
  test media stay in the campaign root. Tests of deletion operate only on copies.

## Research corrections and implementation consequences

1. **Paint admission is not present.** A repository-wide search finds no
   `ActivationFamily::Hunyuan3dPaint`. `hunyuan3d_admission.rs` prices only 2.0 shape
   with fixed 3,072 latents/giant encoder geometry. Add paint admission and make
   shape estimates recipe-aware; do not simply activate a supposed existing arm.
2. **Video chains are not a generic mesh workflow.** `mold-core/src/chain.rs`
   embeds frame counts, motion tails and video stitching. Reuse durable ownership,
   stage checkpoint and scheduling patterns from `chain_job_runner.rs`; add typed
   mesh workflow stages without fake video frames or changing legacy video semantics.
3. **Paint text is learned, not an empty CLIP prompt.** Upstream sets
   `use_learned_text_clip = True`; material tokens `albedo`, `mr`, and reference
   tokens are learned weights. The active branch bypasses text encoding. Preserve
   ignored-prompt behavior for image/mesh-conditioned Hunyuan generation. A text-to-3D
   workflow separately requires a prompt for its selected text-to-image stage.
4. **12 channels are 4 + 4 + 4.** The UNet wrapper changes `conv_in` from the JSON's
   four channels to twelve, and concatenates noisy sample, VAE-encoded normal map,
   then VAE-encoded position map. Material streams and views are batch axes, not
   twelve RGB channels. Reference UNet, multiview attention, material attention,
   positional RoPE and three guidance branches are also part of the port.
5. **The bundled image encoder is CLIP vision, but paint's active conditioning is
   a separately loaded DINOv2-giant.** HF `image_encoder/config.json` names
   `CLIPVisionModelWithProjection` (hidden 1280), while `multiview_utils.py` explicitly
   loads `facebook/dinov2-giant`. Reuse geometry's implementation only after checking
   tensor identities/key mapping and paint preprocessing; do not assume the bundled
   CLIP weights are DINO or that geometry's finetuned encoder is interchangeable.
6. **Scheduler JSON alone gives the wrong execution recipe.** The bundle declares
   DDIM with v-prediction, scaled-linear betas and zero-terminal-SNR rescaling.
   The wrapper replaces it with UniPC, trailing spacing, 15 steps, guidance 3,
   fixed generator seed 0. Port this diffusion schedule, not Wan's flow-UniPC.
   Preserve operation order in the three-branch guidance calculation even where
   algebra suggests cancellation. Freeze any later user seed extension explicitly.
7. **Views and texture resolution are different controls.** Upstream uses six
   primary cameras plus 24 candidates, 1024 visibility selection, nominal 512
   diffusion views, 2048 render/bake sizing and 4096 textures. Six primary views
   are always selected; extra views add uncovered face area above 0.01. Freeze a
   six-view baseline first, then qualify the bounded adaptive selection policy.
8. **Upscale precedes bake.** Tencent upscales both generated albedo and MR views,
   then bakes and fills holes. Do not silently reverse that order based on the
   issue's abbreviated phase description. Inpainting includes vertex propagation
   plus OpenCV Navier–Stokes filling: a Rust equivalent needs fixture comparisons,
   not an unacknowledged nearest-neighbor substitution.
9. **2.1 uses DINOv2-large.** The published 2.1 shape config explicitly gives
   1024 hidden width, 24 layers, 16 heads, GELU MLP, 518 input size and CLS inclusion.
   Existing `dino2.rs` already supports the non-SwiGLU branch. Header verification
   of the chosen repackaged checkpoint remains a prerequisite before wiring it.
10. **Some mesh plumbing exists, but it is not complete asset ingestion.**
    `glb.rs` writes embedded PBR maps and reads geometry/UVs from the first primitive.
    Imported scene transforms, multiple primitives, material extraction, and OBJ
    parsing need explicit handling. The shared viewer samples base color but does
    not establish a complete metallic/roughness display contract.
11. **Paint weights include PyTorch `.bin` files.** Use the existing constrained
    Rust/Candle pickle-to-safetensors pattern from `encoders/pickle_convert.rs`,
    extended with model-specific tensor inventories and atomic derived artifacts.
    Preserve the downloaded originals. No Python conversion dependency in mold.
12. **Licence matching needs extension.** Current name-prefix matching maps ordinary
    `hunyuan3d*` names to 2.0 and only `hunyuan3d-paint*` to 2.1. Adding 2.1 shape
    without changing that authority would ask for the wrong agreement. Add tests
    for shape, paint, multiview, delight, dependencies and all acquisition surfaces.

The old `~/.claude/plans/research-and-plan-only-expressive-newell.md` is absent
on this host. This plan reconstructs its required context from issues, current
mold code and pinned upstream sources rather than pretending it was available.

## Scope and order

The dependency spine is P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7.
Shape extensions P8/P9 and matting P10 need P0/P1 and can be implemented as later
independent commits on the same branch. P11 depends on P2/P10 and a shape path;
P12 reuses the paint SD components; P13 follows dense 2.1 qualification; P14
finishes asset distribution and client parity; P15 is the campaign completion gate.
These are commit boundaries, not separate PRs. Research was performed without
subagents; the user subsequently requested subagent peer review before the final PR.

### P0 — Reproducible baseline and executable oracles

- Add a host-configurable CUDA qualification runner using the artifact policy
  above, with contract tests for paths, retention, exit propagation, device pins
  and report integrity. Do not reuse the hard-coded old `/mnt/storage20tb` paths.
- Capture the current 2.0 shape render and ComfyUI comparison using identical
  preframed inputs and exported initial noise; a numeric seed alone does not
  align PyTorch RNG with mold's CPU ChaCha noise generator.
- Set up pinned Tencent 2.1 and ComfyUI scratch environments on available GPUs.
  Build Tencent's raster extensions only in the oracle environment. Run full
  paint there and save its intermediates before designing optimization shortcuts.
- Inventory actual checkpoint keys/shapes/dtypes and downloads. Capture learned
  tokens, DINO features, conditioning VAE latents, reference attention cache,
  selected denoise blocks/steps, decoded material views and pre/post-bake maps.
- Exit: current baseline reproducible; oracle runs executable; fixture metadata
  and dependency identities recorded. No visual-parity claim from code reading.

### P1 — Core request, capability and artifact contracts (TDD first)

- Extend the generation profile as the single authority for mesh input, named
  views, texture support/resolution/view limits, matting, delight and workflow modes.
  Bind availability to compiled features and executable dependencies.
- Keep `edit_images` semantics unchanged. Named 2mv slots use typed references
  with a constrained role enum (`front`, `left`, `back`, `right`) rather than
  unvalidated names. Roles must survive missing slots, serialization and recovery.
- Add mesh reference kind, bounded byte/count/coordinate validation, media
  descriptors, upload/retained-media hydration and redacted provenance.
- Define a versioned mesh workflow specification with typed stage inputs/outputs,
  per-stage recipes/seeds, predecessor artifact digests and execution identities.
  An explicit text-to-3D workflow composes a selected existing image model and
  selected shape model; it is not a new native text-conditioned Hunyuan family.
- Reject duplicate/unknown roles, all-empty multiview sets, mixed source/reference
  ambiguity, unsupported mesh controls, invalid sizes and incompatible workflow
  graphs at admission. Do not silently ignore parameters.
- Exit: Rust/Studio shared fixtures cover positive and negative admission at
  CLI, HTTP and durable replay boundaries; old image and video contracts still pass.

### P2 — Durable mesh workflow and resource handoffs

- Extract narrowly reusable stage persistence/ownership helpers from durable
  chains, then implement a mesh runner. Prefer a separate typed mesh route/schema
  to overloading video `ChainStage`. Reuse queue tickets and media publication.
- Stages: optional text-to-image → optional matting → optional delight → shape
  (or mesh ingest) → CPU mesh preparation → paint conditioning/denoise/decode →
  view upscale → CPU bake/inpaint → final asset publication.
- Each GPU phase obtains its own admitted placement; release the lease and drop
  GPU model residency before entering CPU work or acquiring the next stage.
  Persist CPU artifacts, not live tensors/device handles. Upscaling is separately
  charged; the presence of a helper engine cannot bypass scheduler ownership.
- Add shape 2.1/multiview and paint activation families/estimates, host-memory
  bounds for UV/bake grids and chunked attention, frozen execution equivalence,
  dependency licence checks, H3 non-interference and runtime-env audit coverage.
- Journal stage intent, durable artifact publication and completion atomically
  enough to resume after every interruption point. Completed stages are reused
  only if their inputs/recipe/version hashes still match. Preserve the source mesh
  and generated image after a later failure. Expose progress, cancellation, pause,
  explicit resume and retained partial results across clients.
- Exit: an image job executes on the *same GPU* while mesh work is in a CPU
  phase; shape and paint residency never overlap; crash/restart and publication
  replay produce one final gallery print with complete provenance.

### P3 — Mesh ingestion, preprocessing and UV unwrap

- Extend GLB parsing to validate supported scene/node transforms, triangle
  primitives, accessors and indices; flatten supported scenes predictably. Add
  OBJ positions/UVs/normals, negative indices and polygon triangulation. Reject
  unsupported compression, external resources or malformed structures clearly.
  Never fetch arbitrary texture URLs or follow OBJ/MTL filesystem traversal.
- Normalize once with reversible coordinate metadata; preserve supplied valid
  UVs when requested, otherwise unwrap. Handle seam vertex duplication, normals,
  winding, degeneracy and face-budget limits. Make remeshing optional and explicit.
- Add pinned xatlas FFI behind `mesh-texture`, forwarded across Cargo roots and
  Nix build variants. This is the issue's narrow native-library exception; model,
  sampler, bake and inference implementations remain Rust/Candle. Include native
  source licence and reproducible build inputs.
- Tests: cube/sphere/seamed/disconnected/open/degenerate meshes, deterministic
  chart output, UV range, texel-aware non-overlap tolerance and attribute remapping;
  parser fuzz/property tests and truncated/oversized inputs.
- Exit: imported and generated meshes can produce validated UV artifacts without
  holding a GPU lease; feature-off builds refuse texture requests before downloads.

### P4 — CPU G-buffers, camera conventions and view selection

- Extend `hunyuan3d/raster.rs` with face IDs, barycentrics, normal, position,
  depth and visibility buffers. Separate paint cameras from the existing poster
  camera authority so poster/turntable framing remains stable.
- Port pinned camera axes/projection, normal encoding, positions, visibility,
  boundary reliability and the six-plus-24 selection algorithm. Preserve candidate
  order/tie behavior. Bound RAM by tiling/chunking, not allocating every full-size
  candidate buffer simultaneously.
- Tests compare analytic primitives and Tencent buffers at silhouette, seam,
  occlusion and mirrored-axis cases. Compare selections and face-area coverage,
  with explicit edge-pixel tolerances for different raster implementations.
- Exit: matching conditioning maps and view selection before any paint UNet work.

### P5 — Paint model loading and component parity

- Validate complete checkpoint tensor inventory, convert pinned `.bin` data in
  Rust as necessary, and load every required learned/reference/material tensor.
  Fail on missing or unexpected architecture-critical tensors.
- Reuse SD VAE and DINO implementations through clean interfaces, not copied
  forks; prove normalization, patch interpolation, VAE latent scaling and mode vs
  sampled latent behavior against fixtures. Resolve DINO weights explicitly.
- Implement the dual UNet structure and SD2.1 blocks, multiview/reference/material
  attention, DINO projection, position RoPE and learned tokens in application-owned
  Candle code. Preserve tensor layouts and branch/cache lifetime.
- Begin with tiny synthetic exported-contract tests, then real-weight component
  fixtures. Check finite values after each boundary and report the first divergence.
- Exit: the reference pass and one conditioned UNet forward match frozen upstream
  fixtures within predeclared dtype tolerances on CUDA.

### P6 — Paint sampling, decode and memory qualification

- Implement the actual UniPC v-prediction trajectory, beta rescaling, trailing
  timesteps, three guidance branches and material/view ordering. Validate against
  the exact diffusers version exercised by the oracle, not an unrelated sampler.
- Decode albedo and MR independently; compare per-step trajectories and output
  maps before changing batching, attention kernels or dtype policies.
- Split or tile attention/CFG/VAE work where necessary, keeping all cross-view
  dependencies intact. Calibrate admission from measured peaks including the
  reference UNet/cache, not only the main UNet or `views × 512²`.
- Exit: six-view 512 baseline renders with quantitative parity and no non-finites.
  Attempt a measured 16 GiB execution budget after the 48 GB baseline; upstream
  reports 21 GB for paint, so stage separation alone is not proof it fits 16 GB.
  A software budget test is not physical 16 GB GPU qualification.

### P7 — View upscale, bake, inpaint and textured GLB (#1511)

- Use existing `UpscaleEngine`/RealESRGAN for both material streams before bake.
  Port back-sample projection and cosine-to-the-fourth weighting with the pinned
  trust mask. Validate albedo sRGB versus linear MR interpretation and orientation.
- Port unseen-texel vertex propagation and Navier–Stokes fill in Rust; validate
  seams and holes against the oracle. Record any justified boundary divergence.
- Write self-contained GLB with UVs, baseColorTexture and metallicRoughnessTexture
  (G roughness, B metallic), correct factors and sampler/material links. Validate
  dimensions at 1024/2048/4096. Validate with a glTF validator and an independent
  renderer; do not declare PBR successful from a colored thumbnail alone.
- Wire `--texture` to this durable path, remove early refusals only where the
  feature is executable, expose `MeshData.textured` truthfully, and publish once.
- Exit: the issue's image → textured chair GLB example works locally and remotely;
  outputs and material maps match the oracle; untextured baseline still passes.

### P8 — Hunyuan3D 2.1 shape (#1496.1)

- Add manifest/catalog/profile/loader detection for 2.1 with correct licence and
  verified checkpoint identity. Validate bundled versus split component mappings.
- Implement `HunYuanDiTPlain`: 21 blocks, width 2048, 64 latent channels, 4096
  tokens, skip connections, six MoE layers/eight experts/top-2 routing, qk RMS
  normalization and fp16 epsilon. Use published scale 1.0039506158752403 and
  DINO-large configuration; verify the VAE's actual weights/config before reuse.
- Test MoE selection/normalization, empty experts, tied routing, gather/scatter,
  skip placement, timestep dtype, occupancy axes and threshold contracts.
- Add 30-step CFG-5 Euler/normal reference recipe after verifying ComfyUI's
  effective schedule. No invented Turbo variant or silent 2.0 architecture fallback.
- Exit: component and full-shape CUDA comparisons pass; memory admission uses
  actual 2.1 tokens, heads and expert storage; 2.1 also feeds the paint workflow.

### P9 — Named multiview 2mv and 2mv-turbo (#1496.2)

- Add both checkpoint tiers with verified component mappings and tier-specific
  sampling. Encode only present views, add their fixed slot-index sine/cosine
  embeddings, concatenate in front/left/back/right order, and use zero negative
  context as upstream does. Missing slots must not renumber surviving views.
- Test all 15 nonempty subsets, reordered serialized requests, duplicate slots,
  per-view preprocessing, durable media restore and reference conditioning memory.
- Exit: real-weight normal and Turbo multiview shapes match upstream; named input
  wells, CLI/MCP syntax, TUI/Discord and phone request construction agree.

### P10 — Background removal and matting (#1496.4)

- Prefer a pure Rust U²-Net port to follow upstream rembg's model family, with
  a pinned checkpoint and preprocessing contract. Read/run the exact upstream
  U²-Net/rembg implementation before porting; model choice is not yet weight-qualified.
- Expose explicit auto/on/off policy: auto preserves useful existing alpha and
  processes opaque input; on reprocesses; off keeps existing input behavior.
  Center/framing is a separate recorded operation. Bound image dimensions/RAM.
- Test alpha preservation, thin structures, hair/fur, translucent boundaries,
  empty masks and white/black backgrounds against exported upstream masks.
- Exit: durable matting pre-stage works for single/multiview and text-generated
  inputs, records both original and processed images, and releases GPU resources.

### P11 — Text-to-3D and supplied-mesh texturing (#1496.3/.5)

- Finish user-facing typed workflow creation: explicit existing t2i recipe and
  shape recipe, prompt routed only to the image model, then optional matting,
  delight and paint. Resolve dependencies/capacity before starting; persist the
  generated image so a resumed paint failure never reruns successful text-to-image.
- Add mesh-input texturing through P1/P3: GLB/OBJ plus a reference appearance image,
  bypassing shape inference entirely. Define existing-UV versus regenerate policy,
  material replacement and normalized coordinate handling in the profile.
- Test HTTP uploads, inline/retained media, CLI files, resumed jobs, cross-host reuse
  and explicit refusal of unsupported mesh forms. Show source mesh and appearance
  image as distinct controls on clients.
- The shape-VAE encoder/mesh-to-latent round-trip is included in this campaign
  under the user's no-deferred-work instruction. After mesh-input paint, port
  point/sharp-edge sampling, farthest-point sampling and PointCrossAttention with
  separate oracle fixtures and loss/geometry validation.
- Exit: prompt → final GLB and supplied mesh → textured GLB both resume durably;
  a supplied mesh never loads a shape model.

### P12 — Delight pre-stage (#1496.7)

- Port `hunyuan3d-delight-v2-0` as the pinned InstructPix2Pix pipeline, reusing SD
  components where equivalent. Upstream uses empty prompt, Euler ancestral,
  50 steps, seed 42, image CFG 1.5/text CFG 1.0, alpha erosion and RGB correction.
  Mirror its actual effective branch behavior rather than inferring from names.
- Make delight opt-in and preserve its input/output. Do not conflate it with
  matting or apply it automatically to existing 2.0 jobs.
- Validate lit/specular/flat-color sources, constant-color correction and empty
  alpha; require finite statistics and comparison to upstream results.
- Exit: delight composes before shape/paint, survives restart and pays its own lease.

### P13 — Shape DiT quantization (#1496.6)

- Build a local deterministic Rust quantization/conversion route from verified
  dense weights, with source digest, quantizer version and per-tensor policy in
  artifact metadata. Preserve all original weights. Start with Q8 linear/MoE
  weights; keep norms, routers, encoders and VAE dense until separately qualified.
- Reuse Candle quantized dispatch only after testing tensor rank/dtype/layout,
  unsupported-kernel fallback and non-finites. Exercise warm, cold and repeated
  model switching, offload, and execution-equivalence invalidation.
- Compare dense and quantized layer outputs, denoise trajectories and geometry
  over the campaign fixture set; measure actual VRAM and runtime benefits. Establish
  tolerances before inspecting candidate quant results. Do not hide failure with
  a silent full-model dense fallback.
- Exit: at least the 2.1 Q8 path is measured and usable through a documented local
  conversion workflow. Qualify 2.0 separately; advertise only passing tiers. Qualify
  FP8 and lower-bit policies against the dense baseline before advertising them.

### P14 — Generation assets, exports and complete surfaces (#1496.8)

- Include the issue's conditionally proposed `generation_assets` work because
  this campaign includes per-map downloads and a complete material export bundle.
  Keep one self-contained GLB as the canonical gallery print; derived albedo/MR
  maps and OBJ/MTL/texture bundles are child assets, not new generation targets.
- Add the next migration with FK → `generations(id)`/ON DELETE CASCADE, stable
  asset IDs/roles, media type, digest, size and internal location authority. Avoid
  serializing local paths. Use current migration numbering, not the issue's lines.
- Extend archive authority, replay/rebuild projection, `to_gallery_image`, list
  overlay and all GalleryImage mirrors together. Coordinate publication and pins:
  archive commits before projection/settlement; trash retains assets; permanent
  deletion removes authority then releases only owned data; sibling prints survive.
- Add authenticated opaque asset downloads and bounded ZIP export with sanitized
  names. Test old gallery rows, missing/corrupt assets, publication failure,
  migration/backup/rebuild, authorization and retained-source reuse.
- Finish shared viewer/poster/turntable material support and color space behavior;
  test base color and varying metallic/roughness on an independent reference mesh.
  Preserve existing camera/framing, CSP and WebGL1 fallback contracts.
- Update Create, Library, Reuse settings, workflow progress, assets and exports in
  web/desktop/iPhone; native bridge and TUI/CLI/Discord/MCP must build equivalent
  requests or give explicit capability refusal where an attachment cannot be supplied.
- Exit: maps and material bundles are downloadable, correctly retained, and visible
  across clients; no client loses named inputs or workflow provenance on reuse.

### P15 — Full qualification, docs and final review

- Sync README, affected CLAUDE/rules, canonical prompting corpus, skill renderer,
  MCP descriptions, website model/API/config pages, desktop docs and mobile README
  at each feature milestone. Track related [#1528](https://github.com/utensils/mold/issues/1528)
  requirements as part of this work; no divergent hand-written agent guides.
- Regenerate generation profiles and prompting guides with their canonical binaries.
  Maintain one campaign changelog fragment; never hand-edit CHANGELOG or versions.
- Remove "no engine yet" paint caveats only after P7 passes. Document qualified
  hardware, memory limits, optional xatlas feature, supported import subsets,
  text-to-image provenance and quantization limits accurately.
- Run final requirements-to-evidence audit against both issue bodies. Completion
  means every mandatory gate below passes; elapsed effort is not a substitute.
- Push final clean checkpoint with the qualification report. Continue to withhold
  PR creation until requested by the user.

## Verification matrix and acceptance rules

| Area | Required evidence |
| --- | --- |
| Existing geometry | 2.0 baseline, mini/Turbo regression where affected; fixed framing, axes, occupancy and fp16 behavior |
| Paint components | DINO/VAE/reference cache/UNet block/three CFG branches/UniPC step fixtures; finite tensors and layout checks |
| Texture output | 1024, 2048, 4096 maps; UV coverage/seams; correct GLB PBR channels/factors; independent renderer and glTF validator |
| Shape extensions | 2.1 MoE + DINO-large; 2mv and Turbo; all nonempty view subsets in contract tests |
| Image preprocessing | alpha/no-alpha, fine detail, empty foreground, delight bright/specular/flat-color cases |
| Mesh inputs | GLB/OBJ; valid/invalid UVs, multi-primitive/transforms, malformed/oversized/degenerate inputs |
| Quantization | dense vs Q8 metrics, finite outputs, peak memory/runtime, cold/warm/offload transitions |
| Scheduling | same-device image progress during mesh CPU work; no simultaneous shape/paint residency; multi-GPU placement/cancel |
| Durability | restart at every stage and publication window; disconnect; cancellation; interrupted explicit resume; idempotent gallery |
| Assets/security | authorized downloads, retained sources, no paths/secrets, trash/restore, isolated permanent-delete copies, sibling independence |
| Surfaces | CLI local/remote, server, TUI, MCP, Discord, web, desktop, iPhone request/reuse/export/progress; browser and native evidence |
| Build coverage | CPU/feature-off, CUDA, CUDA+cuDNN, flash-attn policy, Nix sm89; other release targets compile where toolchains available |

Use at least eight owned/licensed input fixtures: upholstered chair, metallic
object, thin-legged object, concave object, asymmetric color markings, furry
subject, flat-color toy and difficult background. Use three fixed seeds for shape
comparisons. Paint's upstream fixed-seed baseline is separate from any future
configurable seed tests. Store sample noise to isolate RNG differences.

Before implementing each numerical component, commit explicit tolerances derived
from same-oracle repeats and fp32/fp16 comparisons: tensor abs/relative error,
finite fraction, mask IoU/coverage, normalized Chamfer/normal consistency, material
map PSNR/SSIM and seam/unseen-texel error. Never adjust a threshold merely to pass
the Rust output. The existing 2.0 framing-matched Chamfer gate of 0.02 is baseline
evidence, not automatically the right threshold for 2.1, quantization or paint.
Attach images/turntables and human visual inspection notes alongside metrics.

Use four GPUs for independent oracle/mold/cold-memory/scheduler experiments when
available; prove constrained-memory behavior separately. An L40S run does not
qualify Metal, 16 GB hardware, mobile native packaging or every NVIDIA architecture.
Keep unavailable platform gates explicitly pending until tested on real targets.

Targeted failing tests precede each feature/fix. At milestone boundaries run the
relevant `scripts/ci-local.sh` routes. Final required checks include:

```bash
./scripts/ci-local.sh rust web docs contracts gpu nix -k
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p mold-ai-core --bin generate_prompting_guides -- --check
cargo run -p mold-ai-core --bin generate_generation_profiles -- --check
bash scripts/tests/candle-single-identity.sh
bun run check:frontend
bun run check:dead-code
bun run fmt:check
nix flake check
```

The runner's `--help` was checked and supports multiple route arguments.
Also explicitly compile/test new feature
combinations and excluded desktop/mobile Cargo roots: workspace tests do not
compile every feature-gated module. Run the repository's MSRV gate. Use
`ensure-web-dist.sh` before local CLI/server builds that embed the SPA.

## Branch management and risk checkpoints

- Commit this plan first. Thereafter commit each coherent test/implementation/docs
  milestone and push after its targeted checks, and before ending a work session.
  Record what passed and what remains in a checked-in qualification ledger.
- Fetch `origin/main` before major milestones; merge it into the shared long-running
  branch with focused regression tests. Avoid rebases/force-pushes that erase reviewed
  checkpoints. Never include weight files, credentials or large evidence in Git.
- Keep one eventual PR narrative organized by final behavior and evidence, with
  links to the phase commits. No PR creation or issue-closing comments in this plan pass.
- Highest risks: paint attention/scheduler parity; Rust unseen-texel fill fidelity;
  durable heterogeneous-stage ownership; CPU bake memory/time at 4096; 2.1 MoE
  quantization; cross-platform xatlas packaging and client material rendering.
- Resolve P0 oracle feasibility before promising optimization results; resolve P2
  ownership before adding long-lived GPU stages; freeze dense P8 before Q8 P13.
  Keep unsupported modes refused until they pass their gate.
- The issue estimates paint alone at roughly 10–12 engineering weeks; the full
  campaign adds substantial model, workflow, asset and client work. Treat this as
  a multi-month engineering scope, not a quick wiring PR. Re-estimate after P0/P2/P5
  using measured parity and performance; no delivery date is asserted here.

## Pinned source index

References were cloned into ignored `tmp/` and `git pull --ff-only` was run before
inspection. Upstream is read-only; Python is only an oracle/fixture capture tool.

- Tencent 2.1 commit `82920d643c0dc2f7bfd7255f45f62d386edfe60c`:
  [paint orchestration](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dpaint/textureGenPipeline.py#L39),
  [scheduler/DINO/defaults](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dpaint/utils/multiview_utils.py#L49),
  [UNet flags and loading](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dpaint/hunyuanpaintpbr/unet/modules.py#L785),
  [channel packing](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dpaint/hunyuanpaintpbr/unet/modules.py#L966),
  [learned prompt branch](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dpaint/hunyuanpaintpbr/pipeline.py#L267),
  [guidance](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dpaint/hunyuanpaintpbr/pipeline.py#L662),
  [selection/bake](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dpaint/utils/pipeline_utils.py#L40),
  [inpaint](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dpaint/DifferentiableRenderer/MeshRender.py#L1381).
- ComfyUI commit `15eb748b3ec5f8a0a2d470b7fb280e2d7579f916`:
  [2.1 DiT and MoE](https://github.com/comfyanonymous/ComfyUI/blob/15eb748b3ec5f8a0a2d470b7fb280e2d7579f916/comfy/ldm/hunyuan3dv2_1/hunyuandit.py),
  [named view embedding](https://github.com/comfyanonymous/ComfyUI/blob/15eb748b3ec5f8a0a2d470b7fb280e2d7579f916/comfy_extras/nodes_hunyuan3d.py#L57).
- Tencent 2.0 commit `f8db63096c8282cb27354314d896feba5ba6ff8a`:
  [delight](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/f8db63096c8282cb27354314d896feba5ba6ff8a/hy3dgen/texgen/utils/dehighlight_utils.py#L24),
  [rembg entry](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/f8db63096c8282cb27354314d896feba5ba6ff8a/hy3dgen/rembg.py).
- HF configuration snapshots fetched during research (mutable main URLs; copies
  retained in the evidence directory; pin repository revision during P0):
  [paint model index](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/hunyuan3d-paintpbr-v2-1/model_index.json),
  [CLIP vision config](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/hunyuan3d-paintpbr-v2-1/image_encoder/config.json),
  [shape 2.1 config](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/hunyuan3d-dit-v2-1/config.yaml).

## Planning checkpoint validation

Completed: clean fresh worktree from fetched main; both issues and comments read;
Nix service paths and GPU/storage inventory inspected; current implementation
and pinned upstream paths inspected; configuration/issue snapshots retained.
No model inference, golden capture, build, unit test or CUDA qualification was
performed in this planning pass. All implementation phases remain pending.
