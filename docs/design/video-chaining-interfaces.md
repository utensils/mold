# Multi-clip video across interfaces

## Product flow

Keep the existing Short clip destination, model picker and ordinary one-prompt generation. Add clip expands ordered prompt and length rows in place. Add, move and remove are the only structural operations; no timeline or second output mode is needed. The browser editor has one primary Make video action after Check length reports delivered duration. Native apps use Add another clip and Generate video, validating automatically before submission. The compact editor uses Smooth joins. Advanced Cut/Fade scripts remain available through CLI/API; the apps avoid adding per-clip source controls just to expose transitions.

Model, canvas, frame rate and audio apply to the whole part. Server generation profiles and chain limits remain the capability authority. A model that concatenates independent clips must be described honestly. Unsupported ordinary input combinations stop with an actionable reason instead of being dropped.

Library Continue video is a separate action from Reuse settings. It verifies the original installation and latest completed published take, freezes the predecessor model/canvas/fps/audio/tail contract, and makes the next separate MP4. Parts have bounded clips and media, with no lifetime part-count limit. Ordinary automatic long clips have no authored predecessor identity and cannot offer exact linked continuation.

## Surface ownership

| Surface           | Composition                      | Shared authority                                      |
| ----------------- | -------------------------------- | ----------------------------------------------------- |
| Web               | Existing Create and Library      | Studio VideoClips, draft/controller and chain API     |
| Tauri desktop     | Existing Generate and Library    | Same Studio components and controller                 |
| Tauri mobile      | Existing Create and viewer       | Same Studio components and controller                 |
| Native macOS      | Generate, Library menu and Queue | MoldClient wire/drafts; NativeSequence session/editor |
| Native iOS/iPadOS | Generate, Library menu and Queue | Same MoldClient and NativeSequence sources            |
| Discord           | Compact video command group      | HTTP SDK validation and durable jobs                  |
| CLI/API/skill     | Existing scripts and jobs        | Existing canonical chain contracts                    |

## Correctness and recovery

Capture host, credentials, persistent installation identity and draft revision before asynchronous preparation. Recheck them after every await and before mutation. Persist a known operation UUID before POST. A dropped or uncertain reply is reconciled against that UUID; inference is never blindly replayed. Admission cancellation must cover the period before a successful response reaches the client.

Retain failed or paused jobs for recovery. Show completed publication only after a latest gallery filename is available. Display parent clip progress instead of multiplying child jobs. Persist scalar unsent drafts and continuation locators, without media bytes or credentials. When retained references are missing after relaunch, require explicit restoration or removal.

H3's durable stage renders are intermediate artifacts. Authenticate an existing child against its canonical admission request before reattaching or cancelling it. Retain its exact video privately with a durable streaming copy, then move its Library print to recoverable Trash. Publish the joined part as the ordinary Library result; recovery can use the private copy even after public Trash is purged.

Authored references must be inline retained media, not ordinary short-lived upload handles. Repeat typed references across clips; map a first endpoint to clip zero and a last endpoint to the final clip's own last frame. Check aggregate bytes and member counts before copying. Carry source-fit provenance through request, script and output metadata only when the server advertises support.

## Review and acceptance

Independent reviews cover the product plan, shared wire contracts, browser lifecycle, native lifecycle, Discord admission and final integrated diff. Tests first exercise exported contracts, actual wire shapes, stale owner/revision refusals, uncertain admission, cancellation, recovery, bounded inputs and latest-take checks.

Real browser UAT uses Playwright and submits through visible controls. Native UAT uses the built apps; Simulator evidence is identified separately from physical-device evidence. All real generations target Plato's active `/storage/mold` installation. Store receipts under `/storage/mold/uat/video-chaining-ui-20261020` and organize published videos into Video chaining UI UAT so they can be monitored from the ordinary Library. Verify output frames, frame rate, audio and publication on the actual host. Include a fresh create, Library continuation, reconnect during rendering and typed-reference case.

Before completion, obtain independent final reviews, run affected frontend/Rust/native checks, verify CI on the final PR head, merge through a PR, deploy main to Plato and HAL9000, and verify both running revisions. Refresh local nightly CLI and installed skills after the corresponding nightly is available.
