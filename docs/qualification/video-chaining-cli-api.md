# Video chaining CLI/API qualification

PR [#1910](https://github.com/utensils/mold/pull/1910) restores durable scripted
video chaining without adding GUI authoring. The implementation and review plan
is in [video chaining through CLI and API](../plans/video-chaining-cli-api.md).

## Real CUDA runs

UAT used Plato's L40S GPUs 2 and 3 through isolated loopback servers, the
shipping H3 CUDA feature graph, and existing installed model checkpoints. GPU
0/1 services were left running. Scripts, videos, job snapshots, FFprobe JSON,
logs and blind-test receipts are retained under
`/storage/mold/uat/video-chaining-20261019`. Final verified outputs are also
transferred into Plato's main Mold library for viewing using the authenticated
gallery import protocol; retained originals are verified at the destination.

| CLI output | Exact model | Frames / fps | Job identity |
| --- | --- | --- | --- |
| `legacy-ltx.mp4` | `ltx-video-0.9.8-2b-distilled:bf16` | 50 / 30 | `485726b6-d24c-45d4-aa20-db7068ec6428` |
| `ltx2-19b.mp4` | `ltx-2-19b-distilled:fp8` | 49 / 24 | `e3a0cfd3-355d-433b-b68e-2e33f35ced2b` |
| `ltx25-distilled.mp4` | `ltx-2.5-22b-distilled:q8` | 49 / 24 | `a426365c-3366-45e0-a39d-f327e9a35720` |
| `ltx23-dev.mp4` | `ltx-2.3-22b-dev:fp8` | 49 / 24 | `56b8282f-3a45-4842-9b50-ae2652de4b1a` |
| `wan-t2v.mp4` | `wan21-t2v-1.3b:turbo` | 34 / 16 | `37538b86-8cff-492a-a581-33ad78ef4db3` |
| `wan-ti2v.mp4` | `wan22-ti2v-5b:turbo` | 33 / 24 | `9d812a38-1b1d-4e35-ad63-4735be0b8eef` |
| `wan-i2v.mp4` | `wan22-i2v-a14b:q4` | 33 / 16 | `33860118-026a-4c74-895e-e9de99559364` |
| `h3-fl2va-sugar.mp4` | `minimax-h3-fl2va:comfy-pruned-int8-pdmd-2step` | 213 / 24 | `869fedaa-270a-4dac-b5bf-08faf1a56ded` |
| `luna-blind-two.mp4` | `minimax-h3-ref2va:comfy-pruned-int8-turbo-4step` | 248 / 24 | `355f8a7c-3c49-462c-bbb9-774a77b40d75` |

The H3 FL2VA command uses repeated `--prompt` arguments and
`--frames-per-clip 107`; the CLI resolves its one-frame seam automatically.
Contact frames around the seam show the robot and garden continuing into the
second clip. The Ref2VA test uses an authored image reference and independent
cuts. These are different continuity contracts, as advertised by the skill.

These runs qualify representative installed checkpoints across every supported
video family and both H3 task partitions. They do not claim that every weight
precision/tier or every GPU architecture was rendered. Capability/admission
regressions cover all reviewed runnable H3 identities; download-only identities
remain unavailable for inference.

## Gaps found and corrected by UAT/review

- H3 frozen chain aliases were incorrectly re-admitted as user-selected model
  IDs, producing HTTP 451 before accepting a job. The internal alias now must
  match the recomputed frozen fingerprint; original model gates remain active.
- LTX-2.3 dev failed after successful denoising because debug inspection
  demanded a raw velocity from multimodal guidance, which returns denoised
  output. Diagnostics now inspect the available prediction; sampler arithmetic
  is unchanged. The same accepted job was resumed after the fix.
- Native LTX audio was about 79 ms short in a 49-frame output. Stage PCM and
  final AAC now follow the video clock, including fractional H3 sample seams;
  cached stages can be refinalized without another generation pass.
- Independent review found that final metadata hashed scrubbed endpoint bytes
  and described reference media using the maximum clip duration. Finalization
  now hydrates authenticated inputs before provenance creation and records each
  stage's actual prepared duration. Regressions include a prompt/seed retake.
- Exporting the real random H3 seed exceeded TOML's signed integer range.
  Private TOML wire types now preserve full `u64` seeds and offsets using
  decimal strings when needed; JSON/API types stay unchanged. Fourteen TOML
  tests cover boundary values and the exact observed seed.
- Copying conditioned clips exposed a legacy encrypted-member reader refusal.
  Gallery reads now authenticate the complete legacy bundle while retaining
  only the selected bounded payload. All 59 queue-media tests pass, including
  tampering, owner/pin authority and legacy scheduling-projection refusal.

## Blind skill testing

Fresh Luna agents received only the rendered canonical skill, CLI help, host
and output requirements. The lowest supported reasoning setting was used
(`low`; the delegation interface does not offer `none`). The first test authored
and validated the script but correctly stopped at the backend's pre-acceptance
451 refusal. The second fresh test exercises actual Ref2VA generation and
downloads the published `finalizes[-1].gallery_filename`. The private
job-relative `output` field is never used as a library filename. Receipts remain
beside the videos.

The final authoring-only blind test used the clarified complete-block guidance.
Both fresh Luna scripts (LTX-2 and H3 Ref2VA) passed their first CLI validation;
no repair or additional generation was needed. Its remote script is
`luna-final-authoring.toml` under the same Mold-home UAT directory.

## Linked bounded parts

Each part remains limited to sixteen stages; there is no lifetime part count or
chain-follow wall-clock deadline. Successors emit separate MP4s and own their
incoming tail before acceptance, preserving frozen model artifacts and the
canvas/fps/audio contract. A 20-successor execution regression deletes every
predecessor before dispatch, checks that only the new stage renders, and checks
that final gallery tail bytes remain readable after successor GC.

The final server chain suite passes 267 tests. Independent review also found
and corrected missing-source lock ordering, malformed retry-media panics and
pre-index creation recovery. Interruption regressions cover an empty creation
directory, a creation marker, a completed tail snapshot, a sealed-media manifest,
and partial snapshot cleanup. Changed request markers, nonempty markers and
symlinked ancestors fail closed. Ownership markers are removed last so another
interruption remains recoverable. Continued manifests use v2 to prevent older
servers from ignoring their incoming-tail authority during rollback.

A part limits decoded inline authored inputs to 32 MiB and retained media to 64
members, including incoming carry. Projected auto-expanded copies are checked
before allocation, and scoped HTTP body limits account for base64 expansion.
The SDK and CLI require an affirmative linked-part capability before submission;
older servers cannot silently accept a context-free replacement.
