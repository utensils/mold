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
