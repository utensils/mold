# render-chain-v1 — context handoff

> Paste the prompt at the bottom of this file into a fresh Claude Code session
> to resume work on render-chain v1. Everything above it is reference material
> that the prompt points at.

## Status

Branch: `main` (local). **6 commits stacked ahead of `origin/main`, not pushed**
per plan convention (no mid-plan push):

| # | Commit    | Scope    | Phase |
|---|-----------|----------|-------|
| 1 | `d13a554` | `fix(ltx2): use pure source latents as i2v denoise-mask target` | Fix A (prereq) |
| 2 | `b4ed487` | `feat(chain): add core wire types and request normalisation`    | 0.1 |
| 3 | `0328e76` | `feat(core): MoldClient chain methods`                           | 0.2 |
| 4 | `e89826f` | `feat(ltx2): ChainTail type and latent-tail extraction helper`   | 1a |
| 5 | `e917210` | `feat(ltx2): staged latent conditioning bypasses VAE encode`    | 1b |
| 6 | `14801c7` | `feat(ltx2): chain orchestrator with motion-tail carryover loop` | 1c |

Test status on commit 6: `mold-core` 617 pass, `mold-inference` 586 pass,
`cargo fmt --check` clean, no candle weights loaded by any test.

Pre-existing clippy warnings on main (NOT introduced by this branch):
- `crates/mold-core/src/download.rs:1451` — `manual_repeat_n`
- `crates/mold-core/src/placement_test.rs:167` — `field_reassign_with_default`

These only fire on newer clippy versions than CI pins and are unrelated to
the chain work. Don't "fix" them as part of render-chain.

## Signed-off design decisions (do NOT re-litigate)

User confirmed these 2026-04-20 and they're recorded at the top of
`tasks/render-chain-v1-plan.md`:

1. **Trim over-production from the tail** of the final clip, not the head.
2. **Per-stage seed derivation: `stage_seed = base_seed ^ ((stage_idx as u64) << 32)`.**
   `ChainStage::seed_offset` overrides this; reserved for the v2 movie-maker.
3. **Fail closed on mid-chain failure.** 502 + discard all prior stages. No
   partial stitch.
4. **Accept ~1 GB RAM ceiling** for accumulated `RgbImage` buffer. Streaming
   encode revisited at 1000+ frames.
5. **Single-GPU per chain.** Multi-GPU stage fan-out is v2.

The orchestrator already encodes 1, 2, 3 and Phase 2 server route handles 3.

## What's done

- **`mold_core::chain`** — wire types (`ChainRequest`, `ChainResponse`,
  `ChainStage`, `ChainProgressEvent`, `SseChainCompleteEvent`) and
  `ChainRequest::normalise()`. Re-exports from `mold_core`.
- **`MoldClient::generate_chain{,_stream}`** with 422 → Validation, 404-with-
  body → ModelNotFound, empty-404 → hard error (non-streaming) / `Ok(None)`
  (streaming). Wiremock integration tests pin all four paths.
- **`ltx2::chain::ChainTail` + `extract_tail_latents`** — pure tensor math,
  VAE formula `((pixel - 1) / 8) + 1`. Errors (not panics) on rank
  mismatch / oversize tail.
- **`StagedLatent` + `StagedConditioning.latents`** — threaded through
  `maybe_load_stage_video_conditioning` in `runtime.rs`. When the latents
  vec is non-empty, the function builds `VideoTokenReplacement`s straight
  from pre-encoded tokens and **skips VAE load entirely** (conditional
  `Option<VAE>` — confirmed only loaded when images or reference video are
  present).
- **`Ltx2ChainOrchestrator<R: ChainStageRenderer>`** — fully tested against
  a fake renderer. Handles seed derivation, motion-tail trim on
  continuations (stage 0 keeps all frames, continuations drop leading K),
  progress forwarding with `stage_idx` wrapping, fail-closed error handling.
  Orchestrator does NOT trim to a target total or encode MP4 — those are
  caller responsibilities.

## What's remaining

### Phase 1d — `impl ChainStageRenderer for Ltx2Engine` (engine integration)

The one-sentence contract: given `stage_req`, optional `carry: &ChainTail`,
and an optional stage-progress callback, return
`StageOutcome { frames, tail, generation_time_ms }`.

Three sub-tasks:

1. **Tail capture slot.** Add a mechanism for `render_real_distilled_av`
   (`crates/mold-inference/src/ltx2/runtime.rs:1722`) to clone the
   pre-VAE-decode `latents` tensor into a caller-provided slot. The exact
   capture point is immediately before `vae.decode(&latents...)` at
   `runtime.rs:2010` — shape is `[1, 128, T_latent, H/32, W/32]` F32.
   Preferred mechanism: a field on `Ltx2RuntimeSession` (or a method
   argument threaded down) holding `Option<Arc<Mutex<Option<Tensor>>>>`.
   Production non-chain callers leave it `None` and pay no overhead.

2. **`Ltx2Engine::generate_with_carryover(&mut self, req, carry)`**:
   - Validate the request is a supported family (v1 scope: distilled LTX-2
     only — see `select_pipeline` at `crates/mold-inference/src/ltx2/pipeline.rs:108`).
   - Build a `Ltx2GeneratePlan` via the existing `materialize_request` flow.
     When `carry.is_some()`, wipe `source_image` and append a
     `StagedLatent { latents: carry.latents.clone(), frame: 0, strength: 1.0 }`
     to `plan.conditioning.latents`. The runtime already handles the rest
     (`maybe_load_stage_video_conditioning` skips VAE, builds a frame-0
     replacement from patchified tokens).
   - Enable the tail-capture slot.
   - Run the existing render → decode → encode pipeline.
   - Pull the captured latents out of the slot.
   - Call `ltx2::chain::extract_tail_latents(&captured, motion_tail_frames)`
     to get the tail slice.
   - Decode the stitched MP4 once to extract `last_rgb_frame` (or capture
     it alongside the `frames` Vec from `decoded_video_to_frames`).
   - Return `(GenerateResponse, ChainTail)`.

3. **`impl ChainStageRenderer for Ltx2Engine`** that delegates to
   `generate_with_carryover`. The orchestrator's fake-renderer tests
   define the exact contract; no new test harness needed for the impl —
   real-engine coverage is Phase 2's integration test.

**Gotchas:**
- `CLAUDE.md` claims `[lib] test = false` on `mold-inference` and
  `mold-server` — **this is stale.** Both have normal test configs. Verified
  in Phase 1a/b/c by running 586 tests.
- `run_real_distilled_stage` already takes
  `video_clean_latents: Option<&Tensor>` and `video_denoise_mask: Option<&Tensor>` —
  don't add new parameters unnecessarily. The tail carryover rides on
  `conditioning.replacements` via `StagedLatent`, not on `video_clean_latents`.
- VAE temporal ratio is **8× with causal first frame** (`model/shapes.rs:20`).
  `extract_tail_latents` already encodes this; just call it.
- `motion_tail_frames` defaults to 4 per plan; orchestrator validates
  `motion_tail < stage.frames` up front, but the engine should still
  tolerate `motion_tail = 0` (simple concat, no latent carryover — `carry`
  will be `None` for every stage in that configuration).

### Phase 2 — `POST /api/generate/chain[/stream]` server route

Plan §2. Handler flow:

1. Parse + `ChainRequest::normalise()`.
2. Reject non-LTX-2 models with a clear error.
3. Grab the engine from `ModelCache` (`crates/mold-server/src/lib.rs` —
   holds `AppState.model_cache: Arc<tokio::sync::Mutex<ModelCache>>`).
4. Construct `Ltx2ChainOrchestrator` against it, call `run()`.
5. Trim accumulated frames to target total (the ChainRequest no longer
   carries `total_frames` after normalise — if you want tail-trim support,
   add a `target_total_frames: Option<u32>` field that normalise
   populates). Per the sign-off: trim from the tail.
6. Encode stitched MP4. Reuse `ltx2::media::encode_frames_to_mp4` or the
   existing `encode_native_video` path — scout during Phase 2.
7. Save via `save_video_to_dir` with an `OutputMetadata` synthesised from
   `stages[0].prompt`; optionally add `chain_stage_count: Option<u32>` to
   `OutputMetadata`.
8. Return `ChainResponse` JSON.

**Do NOT go through the existing single-job queue.** A 10+ minute chain
would block the queue. Instead hold the `ModelCache` mutex directly for
the chain duration, same pattern as the multi-GPU pool. Reason in plan §2.1.

SSE variant: same flow, stream `ChainProgressEvent` as `event: progress`
JSON frames and a final `SseChainCompleteEvent` as `event: complete`.

Tests: route-level with a fake engine (same trait seam as Phase 1c). No
real weights.

### Phase 3 — CLI auto-routing + flags

When `--frames > clip_cap` (97 for LTX-2 19B/22B distilled), build a
`ChainRequest` from the CLI args and route to
`MoldClient::generate_chain_stream`. New flags: `--clip-frames N`,
`--motion-tail N` (default 4).

Stacked progress bar: one parent bar per chain (estimated total frames),
one per-stage bar wiping between stages.

`--local` parity: factor the orchestrator invocation so both server
handler and CLI local path use the same code.

### Phase 4 — docs

- `website/guide/video.md`: new "Chained video output" section explaining
  `--frames N`, motion tail, and the server endpoint.
- `CHANGELOG.md`: Unreleased/Added entry.
- `.claude/skills/mold/SKILL.md`: new CLI flags + endpoint.

## Verification commands

Run these in order after any Phase 1d change to verify nothing regressed:

```bash
cargo fmt -p mold-ai-inference -- --check
cargo check -p mold-ai-inference
cargo test -p mold-ai-inference --lib ltx2::chain::     # orchestrator + tail helpers
cargo test -p mold-ai-inference --lib                   # full 586-test sweep (~35 s)
cargo test -p mold-ai-core                              # sanity
```

Phase 1d's own tests should live alongside existing `pipeline.rs::tests`
patterns (using `with_runtime_session` injection at
`crates/mold-inference/src/ltx2/pipeline.rs:1062` — the existing test
exercises the runtime without real weights).

## File map — where everything lives now

```
NEW (this branch):
  crates/mold-core/src/chain.rs                            # wire types + normalise
  crates/mold-core/tests/chain_client.rs                   # wiremock integration
  crates/mold-inference/src/ltx2/chain.rs                  # ChainTail + orchestrator

MODIFIED (this branch):
  crates/mold-core/src/lib.rs                              # re-exports
  crates/mold-core/src/types.rs                            # pub(crate) base64_opt
  crates/mold-core/src/client.rs                           # generate_chain{,_stream}
  crates/mold-inference/src/ltx2/mod.rs                    # pub use chain::*
  crates/mold-inference/src/ltx2/conditioning.rs           # StagedLatent
  crates/mold-inference/src/ltx2/runtime.rs                # latents loop + Fix A

TARGETS (Phase 1d):
  crates/mold-inference/src/ltx2/pipeline.rs               # Ltx2Engine::generate_with_carryover
  crates/mold-inference/src/ltx2/runtime.rs                # tail-capture slot on session

TARGETS (Phase 2+):
  crates/mold-server/src/routes_chain.rs                   # NEW
  crates/mold-server/src/lib.rs                            # route registration
  crates/mold-cli/src/main.rs                              # auto-route
  crates/mold-cli/src/commands/generate.rs                 # chain path + local parity
  website/guide/video.md                                   # docs
  CHANGELOG.md
  .claude/skills/mold/SKILL.md
```

## Convention reminders

- Feature branch: `feat/render-chain-v1` (currently committing directly to
  local `main` since pre-push). PR target: `main`.
- Commit scopes: `feat(chain)`, `fix(chain)`, `test(chain)`, `docs(chain)`
  (core), or `feat(ltx2)`, `feat(server)`, `feat(cli)` depending on crate.
- **No mid-plan push.** All work accumulates locally until Phase 4 ends.
- Every phase step ends with a commit; verification (`fmt`, `test`)
  between every step.
- Tests must be weight-free. Use the trait-seam pattern (Phase 1c) or the
  `with_runtime_session` injection pattern (`pipeline.rs:1062`).

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I'm continuing work on **render-chain v1** — server-side chained LTX-2 video
generation for the mold repo.

## Read first, in this order

1. `CLAUDE.md` (both global at `~/.claude-personal/CLAUDE.md` and
   `/Users/jeffreydilley/github/mold/CLAUDE.md`).
2. `tasks/render-chain-v1-plan.md` — full design, signed-off decisions.
3. `tasks/render-chain-v1-handoff.md` — status, remaining work, gotchas.
   **This is your primary briefing.** Read it end-to-end before writing code.

## Status on entry

- 6 commits stacked locally on `main`, not pushed (per plan convention).
  Last commit: `14801c7 feat(ltx2): chain orchestrator with motion-tail carryover loop`.
- Phase 0 (core wire types + client) and Phase 1a/b/c (ltx2 chain types,
  StagedLatent plumbing, orchestrator + fake-renderer tests) are done.
- `mold-inference` has 586 tests passing, `mold-core` 617. Nothing loads
  candle weights. Fmt clean.
- `CLAUDE.md`'s claim that `mold-inference` has `[lib] test = false` is
  **stale** — the previous session verified tests run normally.

## What you're doing

**Phase 1d** — the engine integration that makes the orchestrator actually
render. Spec in `render-chain-v1-handoff.md` under "Phase 1d". In one
sentence: implement `impl ChainStageRenderer for Ltx2Engine` by adding a
tail-capture slot to `Ltx2RuntimeSession` and a
`Ltx2Engine::generate_with_carryover` method that populates
`plan.conditioning.latents` from the `ChainTail` input and returns the
captured tail alongside the response.

Key surgery points already scouted:
- Tail capture immediately before `vae.decode` at
  `crates/mold-inference/src/ltx2/runtime.rs:2010`
- Plan's staged-latents plumbing already works —
  `maybe_load_stage_video_conditioning` accepts pre-encoded latents when
  you populate `plan.conditioning.latents` (Phase 1b).

After Phase 1d, Phases 2 (server route), 3 (CLI), and 4 (docs) per the plan.

## How to work

- Use `superpowers:subagent-driven-development` — the plan is sized for it.
- Use `superpowers:verification-before-completion` before claiming any
  phase done. The handoff doc has the exact verification commands.
- Every step ends with a commit. Commit scope `feat(ltx2)` for Phase 1d.
- Do NOT push anything — plan convention is no mid-plan push.
- Do NOT re-litigate the signed-off design decisions in the handoff doc.
- Tests must be weight-free (use the `with_runtime_session` injection
  pattern from `pipeline.rs:1062` or the trait seam shipped in Phase 1c).

## Start here

1. Run `git status && git log --oneline -7` to confirm the 6 commits are
   on the tree.
2. Read `tasks/render-chain-v1-handoff.md` end-to-end.
3. Delegate an Explore subagent to map `Ltx2RuntimeSession` and the full
   `Ltx2Engine::generate` → `generate_inner` → `render_native_video` call
   chain end-to-end before writing code. Cite file:line throughout. Keep
   the report under 2000 words.
4. Then plan the tail-capture mechanism (decide: field on
   `Ltx2RuntimeSession` vs. threaded parameter, ergonomics tradeoffs).
5. Implement. Commit. Then Phase 2.

If you hit a surprise that invalidates an assumption in the plan or
handoff doc, stop and re-plan rather than papering over it.
