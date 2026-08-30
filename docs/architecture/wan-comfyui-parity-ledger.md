# Wan ComfyUI parity ledger

The decision record for every Wan variant ComfyUI supports: what its
conditioning contract is, whether mold runs it, and — when mold does not —
whether that is an implementation issue, an intentional defer, or an
intentional drop, with the reason.

It exists so upstream capability growth cannot become invisible. A Wan node
that appears upstream and is never dispositioned here is the failure mode
this document prevents: a user finds it in ComfyUI, assumes mold's silence
means "coming soon", and mold's catalog eventually offers a checkpoint whose
conditioning no engine implements.

**Rule:** a newly discovered variant does not enter catalog discovery until
its row here says what mold does with it. See
[`crates/mold-catalog/src/civitai_map.rs`](../../crates/mold-catalog/src/civitai_map.rs)
for the drop list this feeds.

## Pinned revisions

The node and architecture tables below describe one exact upstream revision.
Refreshing them is a mechanical procedure (see the end of this document);
the phases and risks are stable prose that a refresh does not rewrite.

| Reference | Revision                                   |
| --------- | ------------------------------------------ |
| ComfyUI   | `2eb609766a749e3104485979615e062e401bab97` |
| diffusers | `e504b0496da440442b2b2352a97ac8530bab9d6a` |
| Wan2.1    | `9737cba9c1c3c4d04b33fcad41c111989865d315` |
| Wan2.2    | `42bf4cfaa384bc21833865abc2f9e6c0e67233dc` |

Surveyed surface at those revisions: **31 Wan node classes across 8 files**,
and **15 Wan architecture entries** in `comfy/supported_models.py`.

Five of the 31 nodes do **not** carry `wan` in their registered id —
`TrimVideoLatent`, `GenerateTracks`, `SCAIL2ColoredMask`,
`WanContextWindowsManual` (assigned dynamically), and
`WanUni3CControlnetApply` (old-style registration). A name grep finds 26; the
refresh procedure below accounts for the other five explicitly.

## Dispositions

`supported` — mold runs it today. `impl issue` — earns an engine; needs its
own issue. `defer` — a real capability, not now, reason recorded. `drop` —
will not implement, reason recorded. `n/a` — graph plumbing with no mold
analogue.

### Text-to-video and image-to-video (the shipped core)

| Variant                                | Contract                                                                              | Disposition                                                                                                          |
| -------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Plain text-to-video (`WAN21_T2V`, `WAN22_T2V`) | No conditioning channels; the DiT consumes latents directly                     | **supported** — the family's shipped core: 1.3B, A14B T2V pairs, and TI2V-5B in its text-only mode                     |
| `WanImageToVideo` (2.2 A14B)           | 36-ch channel-concat + 4-group mask                                                    | **supported** — `wan/conditioning.rs` `build_a14b_conditioning` / `build_a14b_mask`                                    |
| `WanImageToVideo` (2.1 I2V)            | same, plus a CLIP-vision `k_img`/`v_img` cross-attention branch                        | **drop** — a separate encoder and attention branch for a checkpoint 2.2 supersedes; refused by name today              |
| `Wan22ImageToVideoLatent` (TI2V-5B)    | 48-ch latent inpaint, leading frames pinned, `noise_mask` zeroed                       | **supported** — `WanTi2vInpaint` with per-token timesteps and post-step re-imposition                                  |
| `WanFirstLastFrameToVideo`             | both endpoints on the canvas; trailing flag in mask channel 3 only                     | **supported** (#779) — the two-entry `keyframes` layout                                                                |
| `WanFunInpaintToVideo`                 | delegates verbatim to the first/last-frame path                                        | **supported** as a contract — it *is* FLF under a Fun brand name; the 2.1 Fun-InP weights stay dropped (CLIP branch)   |

`WanFunInpaintToVideo` deserves a line in the user docs: people search for
"Fun Inpaint" and will not guess it is spelled `keyframes` in mold.

### Earns an engine

| Variant                    | Contract                                                                                                                              | Why it ranks here                                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **VACE** (`WanVaceToVideo`) | 96-ch context: 32-ch VAE (inactive/reactive) + 64-ch pixel-shuffle mask, plus an optional reference latent prepended on the time axis | **Phase 1.** Highest community value; 1.3B fits any consumer card. Real work is the parallel `vace_blocks` tower |
| `WanFunControlToVideo`      | 48-ch DiT; `concat_latent` is control video + start image, no mask                                                                     | **impl issue** — pure channel-concat arithmetic, no new tower; GGUFs exist                                       |
| `Wan22FunControlToVideo`    | same, VAE-generation aware, plus a mask group and `reference_latents`                                                                  | **impl issue** — bundle the reference-image wire decision with VACE so the surface is designed once              |

### Deferred, with the blocker named

| Variant                                        | Blocker                                                                                                     |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Camera (`WanCameraImageToVideo`)               | Plücker camera-embedding adapter tower **and** a trajectory authoring surface; see risk R1                    |
| Phantom (`WanPhantomSubjectToVideo`)           | No new tower — but needs 3-way CFG (positive / negative-text / negative-image) and an extra forward per step  |
| Track (`WanTrackToVideo`, `nodes_wanmove.py` ×5) | UI-bound, not model-bound: needs a track wire type and an authoring surface. `WanMove*` is the modern variant |
| S2V (`WanSoundImageToVideo` + Extend)          | A wav2vec2 encoder tower in candle plus audio ingest. Weights exist; the tower is the cost                    |
| HuMo (`WanHuMoImageToVideo`)                   | Same wav2vec2 dependency; upstream marks it experimental. Must not precede S2V                                |
| InfiniteTalk (`WanInfiniteTalkToVideo`)        | wav2vec2 **plus** a loadable model-patch concept and a sampler wrapper. Deepest audio integration; last       |
| WanDancer (`nodes_wandancer.py` ×4)            | A music encoder tower **and** ~700 lines of beat-tracking/CQT DSP that would have to be written in Rust       |
| Animate (`WanAnimateToVideo`)                  | Pose extraction and face cropping are upstream *preprocessing* mold does not host. Decide hosted vs required  |
| Animate's cache path                           | Chunked continuation via frame offset — the same carry-tail shape as mold's chain/extend, so it would reuse it |
| SCAIL / SCAIL-2 (`nodes_scail.py`)             | SAM3 segmentation as a preprocessing dependency — strictly heavier than Animate                               |
| Uni3C (`WanUni3CControlnetApply`)              | A second checkpoint type, a ModelPatch abstraction, and point-cloud-warp preprocessing                        |
| `WanContextWindowsManual`                      | **Covered by a different mechanism** — mold's long-video answer is chain/extend (#783), not windowed sampling |
| `WAN21_CausalAR_T2V`, `WAN21_FlowRVS`          | No dedicated node; both load through the plain T2V nodes. See risk R1 — they would render *wrong*, not error   |

### Dropped

| Variant        | Reason                                                                                                                             |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `wanBlockSwap` | Upstream NOPs it and calls native block-swap "placebo at best" that breaks its memory management. mold's equivalent is `MOLD_OFFLOAD` |

`TrimVideoLatent` is **n/a**: graph plumbing whose semantic (trim the
reference frames before decode) is internal to a VACE engine.

## Risks this ledger carries

**R1 — `in_dim=36` is not a unique fingerprint.** mold classifies
conditioning from the channel ratio alone, and three `supported_models.py`
classes declare 36 channels against the 16-channel 2.1 VAE — `WAN21_I2V`
(covering both 2.1 and 2.2 I2V checkpoint variants), `WAN22_Camera`, and
`WAN22_WanDancer`. Wan 2.1 I2V is caught positively by a `cross_attn.k_img`
probe. Camera and WanDancer are caught only *incidentally*: they ship as
single checkpoints, and `reject_unwired_channel_concat_checkpoint` refuses
any 36-channel checkpoint without a low-noise expert. So the gate holds for
every published file today, and the exposure is narrower than "any future
checkpoint" — it is specifically a **hand-configured expert pair** (an
explicit `--transformer` / `--low-noise-transformer` pointing at camera or
WanDancer weights), which bypasses the pair check and would then render with
its camera or music conditioning silently ignored. Positive fingerprints
(`cam_adapter.*`, `music_encoder.*`) would close that hole and, more
importantly, keep the gate honest if a future 36-channel variant ever ships
as a pair. `WAN21_CausalAR_T2V` and `WAN21_FlowRVS` are the sharper version
of this hazard on the T2V side: they match the plain-T2V shape exactly, have
no pair check to fall back on, and would render wrong rather than error.

**R2 — the `hf:` path fails closed only after the bytes land.** Civitai
discovery is name-gated, but Civitai publishes no base-model label for
VACE/Fun-Control/Animate/S2V, and a raw `hf:` VACE repository is not gated at
all: it downloads, then dies at the channel-count check. That is fail-closed
but wasteful, and it does not meet "variants do not enter discovery until
their contract is explicit". This ledger's disposition column should drive a
pre-download header gate.

**R3 — the `source_video` refusal is a family-wide blanket.** Wan refuses
`source_video` and `mask_image` for every checkpoint with one message
(`reject_unsupported_conditioning`). `extend_video` already left that blanket
in #783 — it routes to `extend_inner` before the guard and is decided per
checkpoint from the `source_image` contract — so the remaining blanket is those
two. Phase 0 below makes it wrong for `Plain` checkpoints; it must become a
per-shape decision for `source_video` too.

## Phase 0 — plain video-to-video restyle (SDEdit)

The load-bearing simplification: **V2V changes only the initial latents and
the schedule start.** It does not touch the DiT input, so the existing
no-conditioning arm is already correct — which is why this is small, and why
it works on every `Plain` checkpoint (1.3B, A14B-T2V, TI2V-5B) rather than
only image-capable ones. Reference: diffusers `pipeline_wan_video2video.py`.

Six seams, four of which are reuse of tested helpers:

1. **Request/admission** — `source_video`, `source_video_path`, and
   `strength` already exist; extend the strength validation (today gated on
   `source_image`) and replace the blanket refusal with the per-shape one
   from R3. Advertise capability so a non-capable checkpoint gets an
   admission-time error, not a load-time one.
2. **Ingest** — reuse LTX-2's frame decoder; conform the clip length **down**
   to `4k+1` and report the substitution; **reject** resolution and fps
   mismatches rather than rescaling (mold's standing no-bucketing rule).
3. **Encode** — the Wan VAE's existing normalization is already identical to
   diffusers'. No new code.
4. **Noise mix** — reuse `flow_match_interpolate`; noise must come from the
   CPU-seeded generator or cross-backend seed determinism breaks.
5. **Start-sigma offset** — the index arithmetic exists, **but the loop
   cannot simply skip iterations**: the UniPC solver asserts a monotonic step
   index and indexes its schedule internally. Build a *truncated* schedule
   and construct the solver over it, so step indices stay zero-based and the
   multistep warm-up restarts cleanly.
6. **Parity** — golden fixture against diffusers at a fixed seed, plus cheap
   units: a zero-truncation schedule is bit-identical to the untruncated one,
   `strength = 1.0` reproduces the plain T2V schedule, and the `4k+1` conform
   round-trips.

## Phase 1 — VACE 1.3B

Weights are consumer-friendly and no second modality is involved, which is
why it outranks everything deferred above. Context assembly is tensor work
the conditioning layer is already shaped for. The genuine addition is the
**parallel block tower**: `vace_blocks` on every second layer with zero-init
projections and a `context_scale`. The reference-image prepend needs an
additive wire field shared with `Wan22FunControlToVideo`, and VACE's own
defaults differ from the family's, so its manifests carry their own:
**shift 16**, 50 steps, guidance 5.0. Take the shift from upstream's
entrypoint (`Wan2.1/generate.py:76-81`, which selects 16 for any `vace` or
`flf2v` task), not from the lower default in the `WanVace.generate`
signature — the entrypoint is the recipe users actually run, and issue #799
records 16 as well. **Phase 0 lands first** — VACE's control video and masks
reuse the ingest and per-shape refusal it builds.

## Refresh procedure

1. `git -C tmp/ComfyUI pull --ff-only`; record the new SHA above.
2. `grep -rn 'node_id="[^"]*[Ww]an' tmp/ComfyUI/comfy_extras/` and diff the
   node list against the tables. **This finds 26 of the 31** — it matches on
   the registered id, and five Wan-relevant nodes are not named for the
   family. Also re-check, by hand, the files those five live in:
   `nodes_wan.py` (`TrimVideoLatent`), `nodes_wanmove.py` (`GenerateTracks`),
   `nodes_scail.py` (`SCAIL2ColoredMask`), `nodes_context_windows.py`
   (`WanContextWindowsManual`, assigned dynamically), and
   `nodes_model_patch.py` (`WanUni3CControlnetApply`, old-style
   registration). A whole-file sweep of those five plus the grep is the
   complete enumeration.
3. `grep -n "^class WAN" tmp/ComfyUI/comfy/supported_models.py` and diff
   against the architecture set (15 at the pinned revision). Architectures
   without a dedicated node are invisible to a node-only sweep — that is how
   `WAN21_CausalAR_T2V` and `WAN21_FlowRVS` were found.
4. Any new row gets a disposition **before** its base-model string may enter
   catalog discovery (R2).
5. Re-check R1 whenever a new architecture declares `in_dim = 36`.
