# Render Chain v1 — Implementation Plan

> Server-side chained video generation for LTX-2: generate videos of arbitrary length by stringing together multiple per-clip renders and stitching the results. v1 exposes a single-prompt/arbitrary-length UX; the request shape is **stages-based from day one** so the eventual movie-maker (multi-prompt, multi-keyframe) extends without a breaking change.

## Confirmed design decisions (signed off 2026-04-20)

1. **Trim over-production from the tail** of the final clip, not the head. The head carries the user's starting image anchor and is perceptually load-bearing; tail frames are the freshest continuation but cheapest to lose.
2. **Per-stage seed derivation: `stage_seed = base_seed ^ ((stage_idx as u64) << 32)`.** Deterministic, reproducible, avoids identical-noise artefacts when prompts match across stages. `ChainStage::seed_offset` stays reserved as the v2 movie-maker override hook.
3. **Fail closed on mid-chain failure.** If any stage errors, return 502 and discard all prior stages. No partial stitch is ever written to the gallery. Partial-resume is a v2 movie-maker feature.
4. **1 GB RAM ceiling for the accumulation buffer.** Hold decoded `RgbImage`s in memory through the stitch — acceptable for the 400-frame 1216×704 target. Revisit with streaming encode when someone pushes 1000+ frames.
5. **Single-GPU per chain.** The orchestrator runs every stage on the GPU the engine was loaded onto. Multi-GPU stage fan-out is a v2 perf win; docs mention it, code doesn't build it.

**Goal:** `mold run ltx-2-19b-distilled:fp8 "a cat walking" --image cat.png --frames 400` produces a single 400-frame MP4, stitched from ~4 coherent sub-clips, each seeded by a motion tail of latents from the prior clip.

**Scope (v1):**

- LTX-2 only (other video engines intentionally out of scope).
- Single prompt replicated across all stages. Optional starting image on stage 0.
- Motion-tail carryover **using cached latents in-process** (no VAE re-encode between clips).
- Single stitched output to the gallery. No per-clip gallery rows, no `chain_id` grouping.
- Sequential execution (clip N+1 waits for N). Multi-GPU fan-out is v2.
- Server-side orchestration under a new `/api/generate/chain[/stream]` route. CLI auto-routes when `--frames > max_per_clip`.

**Explicitly NOT in v1:**

- Movie maker UI (that's v2, built on the same server API).
- Per-stage prompts/keyframes (the request shape supports them; the CLI doesn't expose them yet).
- Crossfade / colour-matching at clip boundaries.
- Pause/resume/retry of a partial chain.

**Base branch:** `main` · **Feature branch:** `feat/render-chain-v1` · **PR target:** `main`

---

## The compatibility contract

The key architectural decision: **the wire format is already multi-stage.** v1 auto-synthesises the stages list from a single prompt + total length, but the server only ever sees the stages form. That means v2 (movie maker) is additive — the SPA just lets the user author the stages list by hand, no server breaking changes.

```json
POST /api/generate/chain
{
  "model": "ltx-2-19b-distilled:fp8",
  "stages": [
    { "prompt": "a cat walking", "frames": 97, "source_image": "<base64 PNG>" },
    { "prompt": "a cat walking", "frames": 97 },
    { "prompt": "a cat walking", "frames": 97 },
    { "prompt": "a cat walking", "frames": 97 }
  ],
  "motion_tail_frames": 4,
  "width": 1216, "height": 704, "fps": 24,
  "seed": 42, "steps": 8, "guidance": 3.0, "strength": 1.0,
  "output_format": "mp4"
}
```

Or the auto-expand form (what v1 CLI sends):

```json
POST /api/generate/chain
{
  "model": "ltx-2-19b-distilled:fp8",
  "prompt": "a cat walking",
  "total_frames": 400,
  "clip_frames": 97,
  "source_image": "<base64 PNG>",
  "motion_tail_frames": 4,
  "width": 1216, "height": 704, "fps": 24,
  "seed": 42, "steps": 8, "guidance": 3.0, "strength": 1.0,
  "output_format": "mp4"
}
```

Server-side, a canonicalising function collapses the auto-expand form into stages. From the engine's POV there's only ever a `Vec<ChainStage>`.

---

## File map

### New

```
crates/mold-core/src/chain.rs                      -- ChainStage, ChainRequest, ChainResponse types
crates/mold-inference/src/ltx2/chain.rs            -- LTX-2 chain orchestrator + latent-tail carry
crates/mold-server/src/routes_chain.rs             -- POST /api/generate/chain[/stream]
```

### Modified

```
crates/mold-core/src/lib.rs                        -- re-export chain types
crates/mold-core/src/client.rs                     -- MoldClient::generate_chain[_stream]()
crates/mold-inference/src/ltx2/mod.rs              -- pub use chain::{Ltx2ChainOrchestrator, ChainTail}
crates/mold-inference/src/ltx2/pipeline.rs         -- expose internal render path that returns (VideoData, ChainTail)
crates/mold-inference/src/ltx2/runtime.rs          -- thread ChainTail through run_real_distilled_stage
crates/mold-server/src/lib.rs                      -- route registration
crates/mold-server/src/queue.rs                    -- chain handler uses ModelCache but does NOT enqueue via the existing video job queue (reason in §3)
crates/mold-cli/src/main.rs                        -- auto-route --frames > clip_max to /api/generate/chain
crates/mold-cli/src/commands/generate.rs           -- chain client + progress rendering
CHANGELOG.md
website/guide/video.md                             -- document --frames N and the chain endpoint
```

---

## Conventions

- All new Rust code gets unit tests where the logic is pure (stage expansion, tail shape math, concat-drop math). The orchestrator's end-to-end path is covered by an integration test that swaps in a fake engine.
- `mold-inference` crate has `test = false` on the `lib` target — new tests in `ltx2/chain.rs` must either run under `#[cfg(test)] mod tests` with logic that doesn't touch candle weights, or use the fake-engine pattern. Keep tests weight-free.
- CLI manual UAT runs against BEAST (`MOLD_HOST=http://beast:7680`) with `ltx-2-19b-distilled:fp8`.
- Commit scopes: `feat(chain): …`, `fix(chain): …`, `test(chain): …`, `docs(chain): …`.
- Every task ends with a commit. No mid-plan push.

---

## Phases

### Phase 0 — core types (no-op at runtime)

**0.1. Add `mold-core::chain` module with wire types.**

```rust
// crates/mold-core/src/chain.rs
pub struct ChainStage {
    pub prompt: String,
    pub frames: u32,
    pub source_image: Option<Vec<u8>>,   // PNG bytes
    pub negative_prompt: Option<String>, // future-proof; v1 ignores if Some
    pub seed_offset: Option<u64>,        // v2 hook; v1 derives from base seed
}

pub struct ChainRequest {
    pub model: String,
    pub stages: Vec<ChainStage>,                // canonical form
    #[serde(default)]
    pub motion_tail_frames: u32,                // 0 = single-frame handoff; >0 = multi-frame tail
    pub width: u32, pub height: u32, pub fps: u32,
    pub seed: Option<u64>, pub steps: u32, pub guidance: f64,
    pub strength: f64,                          // applied to stage[0].source_image only
    pub output_format: OutputFormat,
    pub placement: Option<DevicePlacement>,
    // auto-expand form (server normalises):
    pub prompt: Option<String>,
    pub total_frames: Option<u32>,
    pub clip_frames: Option<u32>,
    pub source_image: Option<Vec<u8>>,
}

pub struct ChainResponse { pub video: VideoData, pub stage_count: u32, pub gpu: Option<u32> }
```

- Add a `normalise(self) -> Result<ChainRequest>` that collapses the auto-expand fields into stages when `stages.is_empty()`.
- Validation: at least one stage, each stage has `frames` satisfying 8k+1 and > 0, total stages × clip_frames ≤ 16 (early guardrail — users aren't generating feature films with this yet).
- Tests: `normalise_splits_single_prompt_into_stages`, `normalise_preserves_first_stage_image`, `normalise_rejects_empty`, `normalise_rejects_non_8k1_frames`.

Commit: `feat(chain): add core wire types and request normalisation`.

**0.2. Re-export from `mold_core`, add `MoldClient::generate_chain`/`generate_chain_stream`.**

Mirror the existing `generate` / `generate_stream` shape. No server changes yet — client just has the surface area.

Commit: `feat(core): MoldClient chain methods`.

---

### Phase 1 — LTX-2 chain orchestrator (single GPU, in-process)

**1.1. Define `ChainTail` as the latent-carryover payload.**

```rust
// crates/mold-inference/src/ltx2/chain.rs
pub struct ChainTail {
    pub frames: u32,                      // number of pixel frames this tail represents
    pub latents: Tensor,                  // [1, C, tail_latent_frames, H/32, W/32] on the engine device
    pub last_rgb_frame: RgbImage,         // for fallback + debugging
}
```

The VAE temporal ratio is 8 with causal first frame, so `tail_latent_frames = ((tail_pixel_frames - 1) / 8 + 1).max(1)`. For `motion_tail_frames=4` this is 1 latent frame. For `motion_tail_frames=9` it's 2 latent frames. Tests cover the arithmetic.

**1.2. Extend `Ltx2Engine` with a chain-aware generate path.**

Add a method that `generate` proper delegates to:

```rust
impl Ltx2Engine {
    pub fn generate_with_carryover(
        &mut self,
        req: &GenerateRequest,
        carry: Option<&ChainTail>,
    ) -> Result<(GenerateResponse, ChainTail)>;
}
```

When `carry = None`, behaviour is identical to `self.generate(req)` (use the source_image path as today). When `carry = Some(tail)`, the engine:

1. Skips VAE encode on `stage_conditioning` for the keyframe at frame 0.
2. Instead, threads `tail.latents` straight into `maybe_load_stage_video_conditioning` via a new optional parameter. The patchified tail tokens go into `StageVideoConditioning::replacements` with `strength = 1.0` and `start_token = 0..tail_token_count`.
3. Extracts the last `K = motion_tail_frames` pixel frames' worth of latents from the completed denoise (before VAE decode) and returns them as the new `ChainTail`.

The new latent extraction hook needs to run **after the last denoise step, before `vae.decode`** in the distilled and two-stage paths. Surface it as a single helper `extract_tail_latents(&final_latents, motion_tail_frames) -> Tensor` that narrows along the time axis.

- Tests for the helper: `extract_tail_computes_correct_latent_slice`, `extract_tail_preserves_device_and_dtype`, `extract_tail_handles_single_frame_edge_case`.

**1.3. Stage conditioning: accept pre-encoded latents instead of a staged image.**

Currently `maybe_load_stage_video_conditioning` (`runtime.rs:1215`) reads an image path, decodes, VAE-encodes. Add a sibling path that accepts `Option<&Tensor>` as pre-patchified tokens (or raw latents to be patchified in place). Route through it when the orchestrator passes carryover.

Concretely: a new variant on `StagedImage` or a parallel `StagedLatent` struct carried through `StagedConditioning`. Prefer the latter — keeps the existing image path pristine.

```rust
pub struct StagedLatent {
    pub latents: Tensor,   // [1, C, T, H/32, W/32]
    pub frame: u32,        // start frame (0 for chain carryover)
    pub strength: f32,     // 1.0 for chain
}

pub struct StagedConditioning {
    pub images: Vec<StagedImage>,
    pub latents: Vec<StagedLatent>,     // NEW, empty for today's callers
    pub audio_path: Option<String>,
    pub video_path: Option<String>,
}
```

`maybe_load_stage_video_conditioning` iterates `images` then `latents`, patchifying the latter directly without calling `vae.encode`. All existing call sites pass an empty `latents` Vec.

- Test: `staged_latent_produces_same_replacement_token_shape_as_image_for_single_latent_frame`.

**1.4. Build `Ltx2ChainOrchestrator`.**

```rust
// crates/mold-inference/src/ltx2/chain.rs
pub struct Ltx2ChainOrchestrator<'a> {
    engine: &'a mut Ltx2Engine,
}

impl<'a> Ltx2ChainOrchestrator<'a> {
    pub fn run(
        &mut self,
        req: &ChainRequest,
        progress: Option<ProgressCallback>,
    ) -> Result<ChainResponse>;
}
```

Internal loop:

```
let mut tail: Option<ChainTail> = None;
let mut accumulated_frames: Vec<RgbImage> = Vec::new();
let tail_drop = req.motion_tail_frames as usize;

for (idx, stage) in req.stages.iter().enumerate() {
    let per_clip = build_clip_request(stage, &req, tail.is_some())?;
    let (resp, new_tail) = self.engine.generate_with_carryover(&per_clip, tail.as_ref())?;
    let frames = decode_video_frames_from_response(&resp)?;
    if idx == 0 {
        accumulated_frames.extend(frames);
    } else {
        // drop the leading `tail_drop` pixel frames; they duplicate the prior clip's tail
        accumulated_frames.extend(frames.into_iter().skip(tail_drop));
    }
    tail = Some(new_tail);
    emit_progress(progress.as_ref(), ChainStageDone { idx, total: req.stages.len() });
}

let stitched = encode_mp4(&accumulated_frames, req.fps)?;
Ok(ChainResponse { video: stitched, ... })
```

- Stage-1 request has `source_image = stage.source_image`, `keyframes = None`.
- Stage-N request (N ≥ 2) has `source_image = None`, `keyframes = None`; the carryover is passed via the `tail` parameter to `generate_with_carryover`, not through the request DTO.
- Progress events: forward engine events with an added `stage_idx`, plus emit `ChainStageStart` / `ChainStageDone` / `ChainStitching` / `ChainComplete`.

- Tests (fake engine): `chain_runs_all_stages_and_drops_tail_prefix_from_continuations`, `chain_with_zero_tail_concats_full_clips_without_drop`, `chain_progress_forwards_engine_events_with_stage_idx`, `chain_empty_stages_errors`.

Commit: `feat(ltx2): chain orchestrator with latent-tail carryover`.

---

### Phase 2 — server route

**2.1. `POST /api/generate/chain` (non-streaming).**

Handler flow:

1. Parse & normalise the `ChainRequest`.
2. Validate model is an LTX-2 family (`anyhow::bail!` with a clear error otherwise).
3. Grab the model's engine from `ModelCache` (load if needed, same as the existing video path).
4. Construct `Ltx2ChainOrchestrator` against it and call `run()`.
5. Save the stitched MP4 via the same save path as single-clip videos (`save_video_to_dir`), populating `OutputMetadata` with a synthetic prompt (`stages[0].prompt` for v1) and a note in a new optional metadata field `chain_stage_count: Option<u32>`.
6. Return `ChainResponse` as JSON.

Do **not** go through the existing single-job queue — a chain is a long-running compound job and would block the queue for 10+ minutes. Instead, the handler holds the `ModelCache` mutex the same way the multi-GPU worker does, for the full chain duration. This is OK because the multi-GPU pool already has per-GPU thread isolation.

**2.2. `POST /api/generate/chain/stream` (SSE).**

Same flow but progress events stream as `data:` frames. Event types:

- `chain_start { stage_count, estimated_total_frames }`
- `stage_start { stage_idx }`
- `denoise_step { stage_idx, step, total }` (forwarded from engine with `stage_idx` wrapped in)
- `stage_done { stage_idx, frames_emitted }`
- `stitching { total_frames }`
- `complete { video_frames, video_fps, video_base64, filename, seed, ... }` (same shape as `/api/generate/stream` complete event)
- `error { message }`

The existing SSE completion-event helper (`build_sse_complete_event` in `queue.rs`) is not reusable as-is because it takes a single `GenerateResponse`; write a sibling `build_chain_sse_complete_event(&ChainResponse)` that produces the same JSON structure plus `chain_stage_count`.

- Tests: route-level tests with a fake engine that exercise both non-streaming and SSE shapes; verify SSE emits events in the expected order.

Commit: `feat(server): chain render endpoint and SSE stream`.

---

### Phase 3 — CLI

**3.1. Auto-route `mold run` to `/api/generate/chain` when `--frames > max_per_clip`.**

Add a constant in `mold-cli` for LTX-2 clip caps (97 for 19B distilled, 97 for 22B — same as today's single-clip validation). When `frames > cap`:

- Build a `ChainRequest` with `prompt=…`, `total_frames=…`, `clip_frames=cap`, `source_image=…`, `motion_tail_frames=4` (default).
- Call `MoldClient::generate_chain_stream`.
- Render a progress bar per stage stacked with a parent "chain" bar.

When `frames ≤ cap`, path is unchanged (`/api/generate/stream`, single clip, today's behaviour).

- New flag: `--clip-frames N` to let advanced users override the per-clip length (default = model cap).
- New flag: `--motion-tail N` to override the tail (default 4, 0 to disable).
- Help text for `--frames` updates to mention chained output when > cap.

- Tests: `run_frames_above_cap_selects_chain_endpoint` (argparse-level; doesn't invoke the network).

**3.2. `--local` chain mode.**

For parity with `mold run --local`, the CLI should run the orchestrator in-process when `--local` is passed. Factor the orchestrator invocation into a helper so both the server handler and the CLI local path share it.

Commit: `feat(cli): chain rendering for --frames above clip cap`.

---

### Phase 4 — docs & changelog

**4.1. Website.** Add a new section in `website/guide/video.md` explaining chained video output, how motion tail works, and the CLI flags. Link it from the LTX-2 model page.

**4.2. CHANGELOG.** Unreleased / Added entry describing the `/api/generate/chain` route, the CLI auto-routing behaviour, and the motion-tail carryover.

**4.3. Skill file.** Update `.claude/skills/mold/SKILL.md` with the new CLI flags and endpoint.

Commit: `docs(chain): guide, changelog, and skill updates`.

---

## Integration test: a realistic end-to-end

One integration test lives in `crates/mold-server/tests/chain_integration.rs` (or inline in `tests/` if an integration dir exists). It:

1. Stands up an in-process server with a **fake LTX-2 engine** (not real weights) whose `generate_with_carryover` returns a deterministic gradient pattern + a synthetic `ChainTail` whose latents are zeros but whose RGB tail frame is the last frame of the emitted clip.
2. POSTs an auto-expand chain request with `total_frames=200`, `clip_frames=97`, `motion_tail_frames=4`.
3. Asserts:
   - Three stages fired.
   - The stitched MP4 has `ceil((200 - 97) / 93) * 93 + 97 = 97 + 93*2 = 283 ≥ 200` frames before trim; after trim it's 200 frames.
   - SSE stream emitted events in the expected order.
   - The gallery DB got one row with `chain_stage_count = 3`.

The fake-engine pattern keeps this test out of the GPU path and makes it safe to run in CI.

---

## Open design decisions I'm flagging for your sign-off

1. **Trim policy.** If `total_frames = 400` and chain math produces 469 frames, should we trim from the tail (final clip's final frames get cut — but those are the freshest continuation) or from the head (stage-0 frames get cut — but those are the user-anchored ones)? I recommend **trim from tail** because the head is where the user's starting image landed and matters more perceptually.

2. **Seed handling across stages.** Should each stage get the same seed (reproducible but with artifacts from identical noise when prompts match), or derive per-stage seeds (`base_seed ^ (stage_idx << 32)`)? I recommend **derive per-stage**. `seed_offset` on `ChainStage` lets the movie maker override.

3. **Failure mode mid-chain.** If stage 3 of 4 fails, do we return a 502 and discard everything, or return the partial stitch of stages 1–3? I recommend **fail closed for v1** — no partial output. Partial resume is a v2 movie-maker feature where individual stage regen is first-class.

4. **Memory.** 400 frames × 1216×704×3 ≈ 1 GB of RgbImages held in RAM before MP4 encode. Acceptable for v1. If users push to 1000+ frames we revisit with streaming encode.

5. **Placement.** Chain always runs on a single GPU for v1 (the one the engine was loaded onto). Multi-GPU fan-out (stage N and N+1 on different cards) is a v2 perf win; mention in docs but don't build.

---

## What `mold run` looks like after this ships

```console
$ mold run ltx-2-19b-distilled:fp8 "a cat walking through autumn leaves" \
    --image cat.png --frames 400

⏳ Chain render: 4 stages × 97 frames (motion tail: 4) → 388 stitched frames
▸ Stage 1/4 · denoise step 8/8 · 47s
▸ Stage 2/4 · denoise step 8/8 · 44s   (tail carried from stage 1)
▸ Stage 3/4 · denoise step 8/8 · 44s
▸ Stage 4/4 · denoise step 8/8 · 44s
▸ Stitching 388 frames @ 24fps …
✔ Saved mold-ltx-2-19b-distilled-{ts}.mp4 (400 frames, 16.7s, 16MB)
```

---

## Out-of-scope for v1 but in-scope for v2 (movie maker)

- SPA route `/movie` with a timeline authoring UI.
- Per-stage prompts and keyframes exposed in the request body (the server already supports this — only the UI needs to change).
- Per-clip gallery rows with `chain_id` grouping so users can iterate on individual stages.
- Selective stage regeneration (replace stage 2 without redoing 1/3/4).
- Crossfade blending at clip boundaries.
- Multi-GPU stage fan-out.

The whole point of v1 is to ship a stable foundation these land on top of without breaking changes.
