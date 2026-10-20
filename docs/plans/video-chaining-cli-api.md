# Video chaining through CLI and API

Scope: restore and qualify scripted durable video chains across runnable LTX-2,
Wan, legacy LTX-Video, and MiniMax H3 models. No GUI authoring changes.

## Findings

The durable chain-job API and CLI remain implemented. H3 is explicitly refused
because its reference media and per-request owned execution are not integrated.
The skill points to a script without providing one, and documentation overstates
some models' continuity. Downloaded remote chain audio facts are discarded.

## Implementation

1. Keep `mold.chain.v1`, `mold run --script`, repeated `--prompt`, and
   `/api/chain-jobs` as the public entry points. Extend stage media additively
   for typed generation references and first/last keyframes. Resolve authored
   file paths before submission; never trust a client path on the server.
2. Seal every input under the existing authenticated media store. Hydrate
   inputs for validation, execution, resume, retake, and amendment; media changes
   invalidate the completed prefix. Preserve the seed, frozen model identity,
   cancellation, restart, and gallery ownership contracts.
3. Run H3 stages through the ordinary per-request preparation/admission and
   owned-attempt protocol. Never reuse the generic engine cache or an earlier
   stage's executable authority. Preserve each tier's sampler and runtime gates.
4. Advertise capability truthfully: LTX-2 uses motion-tail continuation;
   image-conditioned Wan and H3 FL2VA can use the last frame; text-only Wan,
   legacy LTX-Video, and H3 Ref2VA compose independent clips. Cuts and fades
   work without claiming continuous motion. Keep unsupported runtime identities
   and specialized non-video pipelines explicitly refused.
5. Update the canonical generated skill with a complete minimal script and
   discovery, validation, submission, job recovery, and exact-output retrieval.
   Sync affected README, website and owning rules. Preserve remote audio facts.
6. Support open-ended duration through linked bounded parts. A new part pins
   a completed predecessor revision, inherits its frozen model authority and
   owns an authenticated copy of its incoming tail. It renders only new clips;
   prior manifests, media and MP4s stay finite. The 16-stage budget applies to
   one part, with no total part-count limit. Persist linked parts with a new
   manifest schema so older runtimes refuse them rather than ignore the carry.
   Require positive server capability discovery before CLI submission.

## Verification and delivery

- Add failing regression tests before implementation: wire/TOML round trips,
  family capabilities, sealing/recovery, media invalidation, admission, stage
  execution, and decoded/stitch frame and audio facts.
- Independently review this plan and final changes. Resolve findings before merge.
- Build and run real CLI chains on Plato, using an isolated server when needed
  to preserve unrelated queue work. Save videos, scripts, job identities,
  ffprobe facts and UAT receipts under Plato's `/storage/mold`.
- Cover each video family and both H3 task partitions; test tier-specific
  contracts exhaustively without pretending every checkpoint was rendered.
- Give a fresh Luna agent with minimal reasoning only the rendered skill and
  a task; record failures, improve the skill, and repeat the blind test.
- Open a conventional-commit PR, verify required checks on its exact final head,
  merge to main, then deploy that main revision to HAL9000 and Plato. Verify
  live revision, service health, GPU inventory and retained queue/media state.
- After the merged main nightly artifacts are published, install the nightly
  CLI on James's system through the documented curl installer. Reinstall all
  supported agent skill bundles and verify the installed chain guide and CLI
  revision match the delivered fixes.
- Finish with a real 60-second FastH3 V2 chain on deployed Plato and a real
  chained run on HAL9000's verified 24 GB RTX 4090. Use live admission for
  opening and conditioned-stage shapes, record peak GPU memory, and qualify
  the actual tested recipe rather than claiming every resolution fits 24 GB.
  Keep all remote outputs beneath that host's `MOLD_HOME`; the production
  server publishes viewable prints in its active `output` directory.

## H3 continuation reference and durable children

Reviewed ComfyUI checkout `tmp/chain-comfy-reference` after `git pull --ff-only`,
revision `b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c`:
`comfy_extras/nodes_minimax_h3.py:139-163` accepts optional first and last
images, places them at frame zero and `frame_count - 1`, and VAE-encodes each
condition. FL2VA chaining reuses that existing first-frame conditioning by
feeding the preceding delivered clip's last RGB frame; the shared stitcher
removes the one repeated boundary frame. This anchors appearance; it does not
carry a multi-frame motion history. Ref2VA uses authored independent clips;
there is no inferred temporal continuation contract.

H3 stages publish ordinary child gallery prints, titled with the chain and
stage and tagged `chain:<id>` / `chain-stage:<index>`, plus the final stitched
print. The child MP4 is retained under the parent's `h3-children/` directory
before decoding, so later deletion of that gallery print cannot destroy an
already-retained stage. Child identities derive from persisted authored
retake/amend history rather than the scheduler attempt counter: recovering a
crashed scheduler attempt reattaches a completed child instead of rerendering.
Cancelling a parent cancels and settles its active child before releasing the
parent's ownership.

## Native chain audio alignment qualification

The first real LTX-2 19B CLI UAT exposed a delivery gap. Re-probing Plato's
`/storage/mold/uat/video-chaining-20261019/ltx2-19b.mp4` with FFprobe showed
49 video frames at 24 fps (`49000/24000` seconds), while the stereo 24 kHz AAC
track ended at `47104/24000` seconds: 1,896 samples, or 79 ms, short. Padding
only the final mux would restore duration while still allowing a short first
stage to start the second stage's audio early.

The durable writer now preserves the native PCM prefix and trims or zero-pads
only its tail to the raw stage's video duration. Cached sidecars receive the
same conformance in memory before smooth/fade boundary operations, allowing a
saved chain to be refinalized without another inference pass. Each emitted
stage is then fitted between rounded cumulative sample boundaries on the final
video clock. For H3's fractional 32 kHz / 24 fps ratio, this keeps each seam
within half a sample of its rational position instead of accumulating a
per-stage rounding error. Every chain uses the existing `MatchVideo` AAC mux
policy, including exact final-packet duration. No resampling or sampler change
is involved. An explicitly audio-enabled stage missing its PCM sidecar fails
before publication; legacy silent chains remain valid.

Focused regressions (`tmp/chain-all-audio-green.log`) pass: unchanged native
sample prefixes with tail-only trim/padding; sixteen H3 stage boundaries with
no cumulative drift; missing expected audio versus legacy silent delivery;
and real encoded 49-frame / 24 kHz and 213-frame / 32 kHz final MP4s whose AAC
track durations equal the complete video timeline. The encoded-output test
also replaces the saved sidecars with short pre-fix data before refinalizing,
proving recovery repairs existing cached stages without inference.

## Bounded parts with no lifetime duration cap

A part keeps at most sixteen stages; `continue_from` references the newest
completed finalize take and snapshots the required tail at admission. Successors
inherit the exact frozen model artifact identity, canvas, fps, carry and audio
contract. Each part emits a separate MP4. There is no maximum successor count
and no hard chain-follow deadline; memory, manifest size and finalization work
remain bounded by the current part. No historical inference is repeated.

Part media is limited to 32 MiB decoded inline payload and 64 retained members,
including incoming tail frames. Admission checks projected auto-expanded input
copies before allocation. Linked manifests use `mold.chainjob.v2`, preventing
older binaries from silently ignoring the carry link during rollback. Ordinary
manifests stay v1. Creation retries must survive interruption before indexing,
reject changed operation payloads, and avoid holding a nonexistent source lock.
