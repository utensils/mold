# MiniMax H3 authorization gate

Status: **blocked**

Mold does not currently activate, advertise, download, or execute MiniMax H3.
The reviewed H3 Community License revision limits the territory in which its
rights apply, and this repository has no written authorization record covering
the current development and distribution environment.

This is a product compliance boundary, not a location-detection feature. The
gate is enforced from model identity and resolved family at the shared catalog,
request-validation, nested generation artifact, prompt-expansion, download,
cloud-provisioning, durable-sequence, and frozen-engine boundaries. Weight
presence, an environment variable, an HTTP header, a client flag, or a
geographic guess must never bypass it.

Artifact checks are relative to the configured trusted models root. Storage
ancestors are placement, not model identity: no path is safe or authorized for
H3 merely because of its name or location. An H3-named model, sidecar
directory, ControlNet, LoRA, upscaler, or nested component below any root
remains gated, and every proposed `MOLD_HOME` must independently pass the
storage qualification gate.

As of 2026-08-07, `/Volumes/ExternalStorage` is operationally excluded. Mount
metadata remained readable and reported a mounted APFS volume with about 966
GiB available, but time-bounded directory enumeration did not complete; an
earlier bounded create/fsync/read/checksum/delete probe and later filesystem
commands also did not complete. Nominal mount, capacity, and SMART metadata do
not override that failure. Do not enumerate, write, remount, or repair it as
part of H3 work. No H3 bytes were read from or written to the volume. A separate
owner must authorize storage recovery, and a fresh clean probe must pass before
the volume can be considered for any Mold UAT. Storage recovery would not
change the H3 authorization gate.

## Work allowed while blocked

The current decision permits static review of public implementation source,
small textual repository metadata, and license materials. It also permits
weight-free compilation and deterministic tests that construct only synthetic
tensors and fixtures.

Until issue #831 contains an accepted authorization record, do not:

- download, clone through Git LFS, cache, open, range-read, or hash an H3
  checkpoint shard or other binary model payload;
- fetch or inspect a production safetensors header, execute a real H3
  checkpoint, retain generated H3 media, or call any of those activities UAT;
- seed any `MOLD_HOME`, external volume, fixture root, model cache, catalog,
  manifest, installer, cloud image, or release artifact with H3 model bytes; or
- treat source compilation, synthetic CUDA probes, UI authoring, or existing
  local files as evidence that H3 is licensed, qualified, or available.

## Reviewed sources

- [MiniMax H3 Community License, pinned revision](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/bfc8ed0353f5a9733be73e6b2c98ec0948195b86/LICENSE)
- [MiniMax license Q&A and authorization process, pinned revision](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/bfc8ed0353f5a9733be73e6b2c98ec0948195b86/docs/QA-about-License.md)
- [Official implementation, pinned revision](https://github.com/MiniMax-AI/MiniMax-H3/tree/8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea)
- [Authorization tracking issue](https://github.com/utensils/mold/issues/831)

## Activation record

No authorization evidence has been accepted as of 2026-08-07.

| Field                  | Current record                                                                  |
| ---------------------- | ------------------------------------------------------------------------------- |
| Decision               | Unavailable; fail closed                                                        |
| Policy owner           | `utensils/mold` maintainers through issue #831                                  |
| Last review            | 2026-08-07                                                                      |
| Authorization evidence | None accepted                                                                   |
| Next mandatory review  | Any upstream license/Q&A revision, proposed H3 artifact, or release touching H3 |

Activation requires all of the following in a reviewed change:

1. Written authorization whose scope explicitly covers implementation, local
   inference, automated tests and fixtures, distribution, and any hosted use.
2. A durable repository record of the allowed territories, products, users,
   attribution, downstream terms, generated-content duties, and expiry or
   revocation conditions.
3. Tests proving every disallowed path remains fail-closed and every newly
   allowed path follows that exact record.
4. Named ownership for recurring license review and immediate revocation.

If authorization expires, is narrowed, or is revoked, the compile-time gate is
restored before any release or hosted deployment proceeds.

## Release checklist

- [ ] Compare the pinned H3 license and Q&A revisions with their current
      upstream versions. Any change, or any proposed H3-specific artifact, blocks
      the release until the authorization record and policy tests are reviewed by
      the named compliance owner in issue #831.
- [ ] While the decision remains blocked, prove that the exact release contains
      no H3 catalog entry, public manifest or download URL, runtime activation,
      model payload, or generated fixture.
- [ ] Prove that every published binary omits the local H3 attention release
      candidate and every other development-only H3 execution feature.
- [ ] If written authority is accepted, replace this blocked decision in review
      before fetching a checkpoint payload or starting real-checkpoint UAT; a
      storage repair or a green synthetic test is not a substitute.
