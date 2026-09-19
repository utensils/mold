# Native generation and Library parity verification

## Reproduction

The unmodified app at `b44e75639` was built with Xcode and launched as a separate
`io.utensils.mold.parity-uat` application, using disposable preferences and a
loopback fixture server. The fixture server supplied the real H3 generation
profile fetched read-only from Plato on 2026-09-18. No generation was submitted
to a production queue.

Using the native source well's Library picker, attaching a PNG and pressing
Generate captured `strength: 0.75`, with audio absent, on
`minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p`. The freshly adopted recipe's
Generate audio checkbox was off. The app displayed the reported strength
failure. Sending the captured body to Plato's read-only placement-preview
endpoint independently returned `MINIMAX_H3_FIXED_STRENGTH`.

The Library menu lacked image source/reference actions, and the full-size viewer
had no shared context menu. Inspection also found generated-result attachment
bypassed the normal original-pixel and source-size handling.

## Acceptance checks

- MoldClient: 940 tests across 51 suites passed.
- Native Xcode application tests: 769 passed, zero failures or skips, on arm64
  macOS with the isolated UAT instance closed. An earlier rerun ended with
  two test-runner early exits (no assertion failures); a full repeat passed.
- All 11 native release/linkage/workflow shell checks passed.
- Native size/architecture lint and `git diff --check` passed (the existing
  HTTPBackend size advisory is unchanged).
- Library-first launch, before Generate mounted: right-click Use as Source Image
  retained a saved LTX-2.5 model, prompt and explicit audio-off preference, then
  attached the selected PNG and navigated to Generate.
- Fresh Library-first launch adopts H3's recipe defaults before attachment:
  five sampler points, guidance zero, and 124 frames. The native submitted
  request retains those values and strength 1.
- Library tiles and the full-size viewer expose the shared right-click menu.
  Use as Source Image preserves the prompt/model. A Flux.2 recipe also offers
  Add as Reference; the resulting reference appears in its conditioning strip.
- A fresh LTX draft enables audio. Explicit off survives switching through a
  non-audio model and back. GIF selection disables optional audio; enabling
  audio again selects MP4. H3 displays Audio is always included.
- A native H3 submission captured by the local backend carries strength 1,
  the source name/bytes and dimensions, and omits unsupported enable_audio and
  video_only flags. No production generation was submitted.
- Behavioral tests cover stale attachment completion after navigation,
  selection/viewer, draft/host/model/reuse, media eligibility and reachability
  changes; cancelled completions do not navigate or attach media.

## Review follow-up

The independent plan review preceded implementation. Claude Code's first review
identified an audio/container conflict and the Library-first saved-draft hazard.
Both were fixed and regression-tested, along with clearing stale capabilities
when selecting a model without a recipe. Legacy persisted audio false remains
explicitly off: old descriptors cannot distinguish an intentional choice from
an old default, so restoring them must not silently change that choice.

A second review found that a Library-first fallback must apply the newly chosen
model's defaults and that an unavailable audio checkpoint must preserve its
parked preference. Both received regression coverage. The suggestion to loosen
result attachment fencing was not adopted: cancelling an in-flight attachment
when the authored draft changes is intentional, matching Library attachment.
The user can repeat the action against the new draft; delayed bytes must not
alter a draft that changed after the action began.

The final reuse audit also covered prints whose model was removed or omitted.
Their reused settings now regain the currently selected recipe's audio and
strength contract, with regression tests for LTX audio-off and H3 fixed strength.
Result-conformance error reporting uses the same stale-draft guard. Native UAT
reused a print naming a removed model while LTX was selected: the audio control
remained visible/off and the captured request explicitly sent enable_audio false.

Final Claude Code (Sonnet, medium) review: **No actionable findings.** All valid
findings from earlier passes were fixed before opening the pull request.

## CI follow-up

The first GitHub native run exposed a pre-existing queue-test ordering assumption:
two synchronous clicks can enter the asynchronous fake backend in either order.
The test incorrectly assumed batch-1 always belonged to the first click and left
the other stream unconfigured. The failure reproduced locally before the fix.
The test now keeps both streams open and verifies first-click/second-click order
using each admission's client ID. All four queue tests passed ten repetitions
(40 runs), retaining the synchronous double-submit coverage.

## Limits

The original failing request was independently reproduced against Plato's
read-only placement preview. The corrected placement preview timed out, so this
verification does not claim production admission or completed GPU inference.
Native interaction and request capture used an isolated application bundle,
disposable preferences and a loopback backend carrying the real model profiles.
The user's running application and production queue were not modified.
