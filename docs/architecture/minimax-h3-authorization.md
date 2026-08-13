# MiniMax H3 license and integration decision

Status: **public Mold integration and local execution authorized**

On 2026-08-08, the project maintainer accepted a direct attestation that MiniMax
granted permission to integrate H3 with Mold. On 2026-08-12, the maintainer
explicitly authorized public Mold support: listing and downloading the pinned
upstream H3 artifacts, shipping Mold's H3 loader and runtime code, and local H3
generation by Mold users. H3 is therefore treated like other supported model
families. It does not require a per-user authorization file or a private runtime
record. Ordinary builds may advertise the two compact Comfy manifests, download
their exact revision-pinned files directly from Hugging Face, verify their
recorded SHA-256 identities, retain them in the user's model store, and expose
supported generation capabilities. Mold does not bundle or mirror model
payloads in its own releases.

Execution remains capability-gated for technical reasons. The first supported
runtime is the compact FL2VA graph on CUDA and its documented request profile.
Unsupported Metal/CPU routes, Ref2VA execution, altered weights, and broader
request shapes remain fail-closed because they are not implemented or tested,
not because the user lacks authorization. Hosted third-party access and
Mold-hosted weight redistribution remain outside the current decision.

The upstream license remains a product compliance boundary, while runtime
admission is an engineering capability boundary. Model identity and resolved
family are still enforced at request validation, nested generation artifact,
prompt expansion, cloud provisioning, durable sequence, and frozen-engine
boundaries. Weight presence alone must never bypass artifact integrity,
hardware, memory, task, or request-profile validation.

Artifact checks are relative to the configured trusted models root. Storage
ancestors are placement, not model identity: no path is safe or authorized for
H3 merely because of its name or location. An H3-named model, sidecar
directory, ControlNet, LoRA, upscaler, or nested component below any root
remains gated, and every proposed `MOLD_HOME` must independently pass the
storage qualification gate.

The earlier operational exclusion of `/Volumes/ExternalStorage` was superseded
on 2026-08-08 after the maintainer explicitly selected it for private H3 UAT and
a fresh qualification campaign completed successfully. The isolated root is
`/Volumes/ExternalStorage/mold/uat-h3`; every directory is owner-only mode 0700,
and model and evidence files are mode 0600. It currently retains about 760 GiB
free.

That root completed revision-pinned downloads and repeated verification for 50
official payloads totaling 144,028,152,581 bytes at `bfc8ed0` and 17 practical
Comfy payloads totaling 42,482,090,318 bytes at `eb8a161`. Size and SHA-256
checks matched every expected payload, no partial files remained, and repeat
dry runs reported zero missing files. The canonical private `MOLD_HOME` is
`/Volumes/ExternalStorage/mold/uat-h3/mold-home`; authorization evidence and its
validated external record live below the sibling owner-only `compliance`
directory. This storage qualification remains the authority for private
qualification evidence. Ordinary upstream-direct downloads use each user's
configured models root and do not expose or copy that private evidence.

## Authorized product and qualification scope

The current decision permits:

- public listing of the two compact Comfy FL2VA and Ref2VA manifests in Mold's
  Models surfaces;
- user-initiated, upstream-direct download of their revision-pinned files from
  `Comfy-Org/MiniMax-H3` and required support files from
  `MiniMaxAI/MiniMax-H3`, with existing SHA-256 verification and repair flows;
- no raw repository, arbitrary live-catalog recipe, configured alias, or
  caller-authored manifest may substitute for those two registered graphs;
- local storage, inventory, deletion, and repair of those downloaded artifacts;
- public distribution of Mold's H3 integration, manifests, documentation, and
  loader/runtime code;
- local FL2VA execution for Mold users when a server advertises the supported
  CUDA capability and request profile;

- direct download of revision-pinned official or Comfy H3 artifacts into the
  private qualification root on the authorized host;
- private local inference and benign UAT on that host by the project maintainer;
- private conformance outputs and fixtures needed to compare Mold with the
  pinned official and Comfy implementations; and
- static source review, small textual repository metadata, weight-free
  compilation, and deterministic synthetic tests.

Private qualification artifacts, authorization correspondence, checkpoint
headers, and real-checkpoint evidence remain private. A normal user download is
made from the upstream repositories to that user's own model store; Mold does
not copy payloads out of the private qualification root.

The current decision does not permit Mold to:

- bundle or mirror official/transformed weights, safetensors headers, private
  evidence, or real-checkpoint fixtures in Mold releases;
- offer a Mold-hosted model download mirror or third-party hosted H3 inference
  service under this record;
- claim support for unimplemented tasks, devices, artifacts, or request
  envelopes merely because acquisition succeeded; or
- treat source compilation, synthetic CUDA probes, UI authoring, or existing
  local files as evidence that a runtime is qualified.

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
| Decision               | Public Mold integration, upstream-direct discovery/download, local storage, and supported local execution are permitted; Mold-hosted weight redistribution is not authorized                              |
| Decision owner         | James Brink, `utensils/mold` maintainer                                                                                                                                                                  |
| Revocation owner       | James Brink, `utensils/mold` maintainer                                                                                                                                                                  |
| Last review            | 2026-08-12                                                                                                                                                                                               |
| License revision       | `bfc8ed0353f5a9733be73e6b2c98ec0948195b86`; LICENSE SHA-256 `59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44`                                                                           |
| Authorization evidence | Maintainer attestation that MiniMax authorized H3 integration with Mold; corroborating image SHA-256 `8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d`; direct grant retained privately |
| Qualification root     | Owner-only `/Volumes/ExternalStorage/mold/uat-h3`; validated external authorization record under its `compliance` directory; no evidence or model payload committed                                      |
| Permitted artifacts    | Public compact manifest metadata and upstream-direct revision-pinned H3 downloads; private official/qualification artifacts and conformance evidence                                                     |
| Permitted users        | Mold users downloading directly from the reviewed upstream repositories and running supported local H3 generation                                                                                       |
| Prohibited scope       | Mold-bundled or mirrored weights; Mold-hosted third-party inference; unsupported runtime/task/device/envelope activation; publication of private evidence                                                |
| Expiry/revocation      | Immediate on MiniMax revocation, narrowed authority, license/Q&A change, loss of access control, or maintainer decision                                                                                  |
| Next mandatory review  | Any upstream license/Q&A revision, hosted-service or weight-redistribution proposal, new operator/host/territory, or material scope expansion                                                             |

Broader hosted service or redistribution authority requires all of the
following in a reviewed change:

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
- [ ] Prove that the exact release contains no bundled or mirrored H3 model
      payload, private checkpoint header, authorization correspondence, or
      generated qualification fixture. Public compact manifests and upstream
      Hugging Face URLs are expected.
- [ ] Prove that public SM89 H3 binaries retain consistent H3-scoped attention
      provenance while omitting global FlashAttention, qualification/capture
      executables, private evidence producers, and every private marker.
- [ ] If hosted inference or redistribution authority is accepted, update this
      decision in review before that use; a green UAT result is not a substitute.
