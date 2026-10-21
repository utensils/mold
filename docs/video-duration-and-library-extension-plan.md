# Video duration and Library extension

Implementation plan for the follow-up to the authored clip interfaces. This
document records acceptance criteria; unchecked work is not shipped behavior.

## Default interaction

One prompt and **Duration** remain the default on web, Tauri desktop/mobile,
native macOS and native iOS. Show the selected server's actual safe
single-generation duration. Beyond it, show **N clips joined into one video**.
**Define scenes** optionally exposes the existing ordered prompt rows. Avoid
another output mode or timeline. Library **Extend video** asks for a prompt
and added duration, then publishes a new full movie and retains the original.

The delivered duration accounts for every overlap and final frame-grid trim.
The server supplies clip count and join semantics. Positive motion/frame carry
uses the existing model contract; independent shots use deterministic distinct
seeds and the default crossfade, disclosed as blended shots.

## Shared execution

- [x] Constant-space, lazy exact-frame planner with actual H3 and other grids.
- [x] Exact published-source descriptor and streamed, checksummed private copy;
      accepted input survives original Trash/purge and is never overwritten.
- [x] Versioned video-job validate/create/status/events/cancel/resume contracts,
      affirmative capability, installation/revision fences and operation UUID.
- [x] Small durable parent executes bounded chain parts through the existing
      renderer. Checkpoint part ownership and reconciliation before admission.
- [x] Bounded published-pixel boundary extraction, including final trim.
- [x] File-backed video assembly with compatible packet passthrough, preserved
      decoder/presentation metadata and sequential packet access.
- [x] One continuous AAC encode from a bounded PCM spool, exact priming and
      padding edits, model-rate/channel conversion and audio crossfade.
- [x] Bounded decoded fallback for incompatible video parts; no full-prefix
      decoded-frame or payload allocation. State unsupported inputs explicitly.
- [x] Publish one final full movie, retain exact provenance/owned inputs and
      clean only authenticated owned intermediates.

There is no composer duration or lifetime part-count cap. Each part retains
its existing admission budgets. Checked container representations, parser
allocation budgets and disk availability remain necessary resource constraints.
GPU work retains one clip's inference envelope, independent of movie length.

## Settings reuse

- [x] Published Duration authoring restores the exact requested duration, literal
      scenes, full-width seed and authenticated original media after recovery
      data is discarded. Extend restores added time and discloses that the
      original prefix is not attached to the fresh draft.
- [x] Legacy authored sequences retain ordered clip prompts, effective seeds,
      frames, joins, adapters and typed inputs. Verified automatic sequences
      restore a single-prompt Duration with an explicit saved-length notice.
      Missing historical provenance refuses before changing the current draft.
- [ ] Complete actual browser and native settings-reuse UAT, including extended
      output, media-bearing legacy clips and iOS keyboard dismissal.

## Recovery and delivery

- [x] Failed ordinary video retains its encrypted input in a non-retryable
      hold; cancellation and image settlement retain their existing meanings.
- [x] Failed/interrupted chain parents remain discoverable after reload.
- [x] Resume preserves frozen execution; Restart creates a new current-recipe
      attempt; Send captures the destination installation and credentials.
- [x] Web/Tauri and native actions share these authorities and visibly explain
      unavailable historical inputs or nonportable conditioning.
- [x] API, CLI, canonical skill, Discord and user documentation agree.
- [ ] Independent code review, actual CLI and Playwright generations, native
      Simulator UAT, exact-head CI and a fresh minimal-reasoning skill blind test.
- [ ] All generated UAT movies publish in Plato's active `/storage/mold/output`
      and **Video chaining UI UAT** collection. Receipts remain under
      `/storage/mold/uat/video-chaining-ui-20261020`.
- [ ] Merge through PRs, deploy main to Plato and HAL9000, update the local
      nightly CLI and installed skills, and verify native nightly delivery.

Physical-device evidence must be distinguished from Simulator evidence. Parser,
audio waveform and long-prefix memory/performance tests complement real model
generations; none is a substitute for a published, playable final video.
