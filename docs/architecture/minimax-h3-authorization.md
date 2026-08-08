# MiniMax H3 authorization gate

Status: **private qualification authorized; public product activation blocked**

Ordinary Mold builds do not currently activate, advertise, download, or execute
MiniMax H3. On 2026-08-08, the project maintainer accepted a direct attestation
that MiniMax granted permission to integrate H3 with Mold. That evidence is
accepted only for a private, access-controlled implementation and qualification
campaign on project-controlled private storage and compute. It is not accepted as
authority for public product activation, distribution, redistribution, hosted
access, or third-party use.

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

## Authorized private qualification scope

The current decision permits:

- direct download of revision-pinned official or Comfy H3 artifacts into the
  private qualification root on the authorized host;
- private local inference and benign UAT on that host by the project maintainer;
- private conformance outputs and fixtures needed to compare Mold with the
  pinned official and Comfy implementations; and
- static source review, small textual repository metadata, weight-free
  compilation, and deterministic synthetic tests.

The qualification host must not expose an H3 endpoint to a third party. Model
payloads, headers, generated media, and real-checkpoint evidence stay private
until a later reviewed decision explicitly permits publication. No copy may be
moved across a different host, operator, organization, or territory under this
record.

Until issue #831 contains a broader accepted authorization record, do not:

- expose H3 through a public or shared CLI, server, Discord bot, desktop, web,
  iPhone, gallery, remote client, cloud image, or hosted service;
- enable ordinary catalog/search/install/download planning, public manifests or
  artifact URLs, release capabilities, or a shipping production factory;
- redistribute official weights, transformed/quantized weights, safetensors
  headers, generated media, or real-checkpoint fixtures; or
- treat source compilation, synthetic CUDA probes, UI authoring, or existing
  local files as evidence that H3 is publicly licensed, qualified, or available.

## Reviewed sources

- [MiniMax H3 Community License, pinned revision](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/bfc8ed0353f5a9733be73e6b2c98ec0948195b86/LICENSE)
- [MiniMax license Q&A and authorization process, pinned revision](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/bfc8ed0353f5a9733be73e6b2c98ec0948195b86/docs/QA-about-License.md)
- [Official implementation, pinned revision](https://github.com/MiniMax-AI/MiniMax-H3/tree/8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea)
- [Authorization tracking issue](https://github.com/utensils/mold/issues/831)

## Decision record

The direct authorization correspondence is retained privately because it may
contain personal or contact information. The repository records its accepted
scope and a content identity for the maintainer-supplied corroborating image,
not the private correspondence itself.

| Field                  | Current record                                                                                                                                                                                           |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Decision               | Private implementation and qualification on the authorized private host are permitted; every public product path remains fail-closed                                                                   |
| Decision owner         | James Brink, `utensils/mold` maintainer                                                                                                                                                                  |
| Revocation owner       | James Brink, `utensils/mold` maintainer                                                                                                                                                                  |
| Last review            | 2026-08-08                                                                                                                                                                                               |
| License revision       | `bfc8ed0353f5a9733be73e6b2c98ec0948195b86`; LICENSE SHA-256 `59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44`                                                                         |
| Authorization evidence | Maintainer attestation that MiniMax authorized H3 integration with Mold; corroborating image SHA-256 `8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d`; direct grant retained privately |
| Permitted artifacts    | Revision-pinned official and Comfy H3 artifacts downloaded directly to the private qualification root; private benign outputs and conformance evidence                                                    |
| Permitted users        | Project maintainer operating authorized private host only                                                                                                                                               |
| Prohibited scope       | Third-party access; public/hosted product activation; distribution or redistribution; public weights, headers, outputs, fixtures, manifests, URLs, or release capabilities                               |
| Expiry/revocation      | Immediate on MiniMax revocation, narrowed authority, license/Q&A change, loss of access control, or maintainer decision                                                                                   |
| Next mandatory review  | Any upstream license/Q&A revision, scope expansion, proposed public artifact or service, new operator/host/territory, or release touching H3                                                             |

Broader public activation requires all of the following in a reviewed change:

1. Written authorization whose scope explicitly covers implementation, local
   inference, automated tests and fixtures, distribution, and any hosted use.
2. A durable repository record of the allowed territories, products, users,
   attribution, downstream terms, generated-content duties, and expiry or
   revocation conditions.
3. Tests proving every disallowed path remains fail-closed and every newly
   allowed path follows that exact record.
4. Named ownership for recurring license review and immediate revocation.

The private qualification path must remain separate from shipping features and
must fail release-exclusion verification. If authorization expires, is narrowed,
or is revoked, stop private execution, remove qualification credentials and
access, and preserve the ordinary compile-time/product gate before any further
run, release, or hosted deployment.

## Release checklist

- [ ] Compare the pinned H3 license and Q&A revisions with their current
      upstream versions. Any change, or any proposed H3-specific artifact, blocks
      the release until the authorization record and policy tests are reviewed by
      the named compliance owner in issue #831.
- [ ] While public activation remains blocked, prove that the exact release contains
      no H3 catalog entry, public manifest or download URL, runtime activation,
      model payload, or generated fixture.
- [ ] Prove that every published binary omits the local H3 attention release
      candidate and every other development-only H3 execution feature.
- [ ] If broader written authority is accepted, replace this private-only
      decision in review before any public activation, distribution, hosted use,
      or release claim; a private green UAT result is not a substitute.
