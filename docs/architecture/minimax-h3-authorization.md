# MiniMax H3 license and integration decision

Status: **all Mold integration, distribution, and use authorized**

On 2026-08-08, the project maintainer accepted a direct attestation that MiniMax
granted permission to integrate H3 with Mold. On 2026-08-12, the maintainer
explicitly authorized public Mold support: listing and downloading the pinned
upstream H3 artifacts, shipping Mold's H3 loader and runtime code, and local H3
generation by Mold users. On 2026-08-14, the maintainer completed the broader
governance review and approved H3 for every territory, operator, product
surface, local or remote client, shared or hosted server, generated-output
flow, and distribution or redistribution scenario. H3 is therefore treated
like other supported model families. It does not require a per-user
authorization file, private runtime record, location check, license-acceptance
dialog, or H3-specific downstream control. Ordinary builds may advertise every
registered H3 manifest identity — the two compact Comfy FL2VA and Ref2VA
graphs, the eight reviewed Turbo LoRA tags built on them, and the
download-only `official-bf16` and `comfy-pruned-nvfp4` tags — download their
exact revision-pinned files directly from Hugging Face, verify their recorded
SHA-256 identities, retain them in the user's model store, and expose supported
generation capabilities.
Mold's current releases do not bundle or mirror model payloads.

Execution remains capability-gated for technical reasons. The supported
runtimes are the compact FL2VA and Ref2VA graphs on CUDA and their documented
request profiles; Ref2VA joined them on 2026-08-24 (#825), and the pinned
license was re-reviewed against that change with no term affected — the
governance decision below already covers every task partition, and Ref2VA
adds no new artifact, repository, or distribution path (it shares the compact
stack and differs only by its task transformer, which was already listed,
downloadable, and covered).
The Apple Silicon Metal route is admitted and shipped but not yet
hardware-qualified. The unsupported CPU route, altered
weights, and broader request shapes remain fail-closed because they are not
implemented or tested,
not because the user, location, deployment topology, or distribution path lacks
authorization. Authorization does not claim that an unimplemented route works.

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

Ordinary model storage is operator-managed. Mold does not treat Unix ownership
or group/other write-mode bits as an execution eligibility signal: shared model
roots and files created with collaborative modes such as `0664` remain valid.
H3 instead authenticates pinned content, refuses symlinks and non-regular model
files, validates canonical containment, and fences opened descriptor identity
through admission and use. Staged VAE construction reads retained process
descriptors, so replacing a path beneath a shared staging parent cannot replace
the authenticated bytes. The stricter owner-only rules below describe the
private authorization/evidence campaign, not a requirement for runnable model
weights or public upstream-direct downloads.

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

- public listing of every registered H3 manifest identity in Mold's Models
  surfaces: the two compact Comfy FL2VA and Ref2VA graphs; the eight reviewed
  Turbo tags built on them (`minimax-h3-fl2va:comfy-pruned-int8-turbo-8step`,
  `minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p`,
  `minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-v1.1`,
  `minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-768p`,
  `minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-r21`,
  `minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-r21`,
  `minimax-h3-ref2va:comfy-pruned-int8-turbo-4step`, and
  `minimax-h3-ref2va:comfy-pruned-int8-turbo-4step-r21`, each the compact
  stack of its own task plus one pinned adapter, stored once under
  `shared/minimax-h3/loras/` and shared, from the adapter's own source: three
  from `Comfy-Org/MiniMax-H3` `loras/` at `COMFY_TURBO_LORA_REVISION`, the
  v1.1 4-step 768p and 8-step 768p adapters from `lightx2v/Minimax-h3-Turbo`
  at the repository root, and three lossy SVD-resized rank-21 adapters — a
  per-module dynamic-rank derivative of the full-rank adapter each was
  resized from; the A/B against each full-rank tier ran 2026-09-02 (the
  8-step r21 tier cleared the acceptance thresholds at 768x768 and sits in
  the 20-24 dB maintainer band at 1344x768; the 4-step 768p r21 tier sits in
  the band at both canvases; the Ref2VA r21 tier measured 16-17 dB with
  visual parity), and the per-tier registration decision is pending the
  maintainer's call on this branch's pull request (see
  `docs/qualification/minimax-h3.md`, "The rank-21 Turbo tiers campaign") —
  from `drbaph/MiniMax-H3-Turbo-Lora-ComfyUI` at the repository root); and the
  pinned download-only tiers
  (`minimax-h3-{fl2va,ref2va}:official-bf16` and `:comfy-pruned-nvfp4`), which
  acquire, verify, inventory, and remove normally but are refused at
  generation as `MINIMAX_H3_RUNTIME_UNAVAILABLE`;
- user-initiated, upstream-direct download of their revision-pinned files from
  `Comfy-Org/MiniMax-H3` and required support files from
  `MiniMaxAI/MiniMax-H3`, the pruned NVFP4 transformers from
  `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot`, the two lightx2v Turbo LoRA
  adapters from `lightx2v/Minimax-h3-Turbo`, and the three SVD-resized
  rank-21 Turbo LoRA adapters from `drbaph/MiniMax-H3-Turbo-Lora-ComfyUI`,
  with existing SHA-256 verification and repair flows;
- no raw repository, arbitrary live-catalog recipe, configured alias, or
  caller-authored manifest may substitute for those registered graphs;
- local storage, inventory, deletion, and repair of those downloaded artifacts;
- public distribution of Mold's H3 integration, manifests, documentation, and
  loader/runtime code;
- local, remote-client, shared-server, and hosted execution for Mold users when
  a server advertises the supported CUDA capability and request profile;
- use by any person or organization in every territory and publication or
  distribution of generated outputs;
- distribution or redistribution of official or transformed H3 artifacts,
  provided the H3 license link and notice described below accompany the user
  documentation; and

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

- publish private authorization correspondence, owner-only qualification
  evidence, or private real-checkpoint fixtures;
- claim support for unimplemented tasks, devices, artifacts, or request
  envelopes merely because acquisition succeeded; or
- treat source compilation, synthetic CUDA probes, UI authoring, or existing
  local files as evidence that a runtime is qualified.

## Reviewed sources

- [MiniMax H3 Community License, pinned revision](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/bfc8ed0353f5a9733be73e6b2c98ec0948195b86/LICENSE)
- [MiniMax license Q&A and authorization process, pinned revision](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/bfc8ed0353f5a9733be73e6b2c98ec0948195b86/docs/QA-about-License.md)
- [Official implementation, pinned revision](https://github.com/MiniMax-AI/MiniMax-H3/tree/8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea)
- [Authorization tracking issue](https://github.com/utensils/mold/issues/831)
- [Third-party pruned NVFP4 transformers, pinned revision](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/tree/908eccad7e68751190d04c171956f163bfeed741)
- [Third-party Turbo LoRA adapters, pinned revision](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/05ef678438e84933c406131b59abbf86919b3aac)
- [Third-party SVD-resized Turbo LoRA adapters, pinned revision](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/tree/be8eb3ea3466cbb7def202ffec0d2fdc054256ac)

### The pruned NVFP4 third-party source

This is the first pinned H3 source that is neither MiniMaxAI nor Comfy-Org,
and three facts about it are load-bearing.

**Only the transformer comes from it.** The conditioner, both VAEs, the task
config, and every runtime support file still resolve to `Comfy-Org/MiniMax-H3`
and `MiniMaxAI/MiniMax-H3` at their existing pinned revisions, byte for byte.
Comfy-Org publishes no NVFP4 diffusion model at all, so there is no
first-party artifact to prefer.

**The repository declares the reviewed license but does not ship its text.**
Its card records `license: other`, `license_name:
minimax-h3-community-license-agreement`, `license_link: LICENSE`, and
`base_model: MiniMaxAI/MiniMax-H3`, and its README describes itself as "a
community-compiled collection of quantized and pruned weights". At the pinned
revision, `GET /resolve/908eccad…/LICENSE` returns **404** — the declared
license file is absent. Mold's own artifact contract stamps the canonical
MiniMaxAI license URL and `LICENSE_SHA256` on every H3 artifact, so the
authoritative text is unchanged and unaffected; this is recorded as a gap in
the re-uploader's packaging, not a different licence.

**Every object in the repository carries an appended marker**, of the form
`\nL2P_bypass_<filename>_<unix_ts>\n`, past the end of the safetensors
payload. It is content-dedup defeat rather than tampering, and that is
checkable: the same repository's INT8 copy has its payload end at exactly
`20,970,379,616` — the byte count mold already pins for the Comfy-Org
object — and its header hashes to exactly the pinned
`H3_COMFY_PUBLISHED_INT8_HEADER_SHA256`. The marker is **inside** the pinned
size and digest, so it is part of the reviewed content identity: a future
re-upload without it is a different artifact and must be re-pinned, never
silently accepted.

Because a personal namespace can be deleted or relicensed without notice —
a risk that does not exist for MiniMaxAI or Comfy-Org — the release checklist
below requires re-confirming this source before every release that ships it.

### The SVD-resized adapter source

`drbaph/MiniMax-H3-Turbo-Lora-ComfyUI` @
`be8eb3ea3466cbb7def202ffec0d2fdc054256ac` is pinned in
`crates/mold-candle/src/minimax_h3/turbo_lora.rs`
(`H3_TURBO_LORA_DRBAPH_REPOSITORY` / `_SOURCE_REVISION`) as the source of
three SVD-resized rank-21 Turbo adapters: lossy, per-module dynamic-rank
derivatives of three of the full-rank adapters above.

These three tags carry PINNED-IDENTITY evidence — size, SHA-256, header
identity, and the welded source tier — plus MEASURED A/B evidence against the
exact full-rank tier each was resized from: the gate ran 2026-09-02 (the
8-step r21 tier cleared the acceptance thresholds at 768x768 and sits in the
20-24 dB maintainer band at 1344x768; the 4-step 768p r21 tier sits in the
band at both canvases; the Ref2VA r21 tier measured 16-17 dB with visual
parity). The per-tier registration decision is pending the maintainer's call
on this branch's pull request; the gate, its per-tier acceptance rule, and
the measured evidence are in `docs/qualification/minimax-h3.md` ("The
rank-21 Turbo tiers campaign (2026-09-02)").

The repository declares `license: apache-2.0` for these SVD-resized
derivatives specifically, and `base_model: Comfy-Org/MiniMax-H3` with
`base_model_relation: adapter` — a different declared base than the two rows
above record, because these files were resized from the Comfy-Org re-hosts.
It ships no LICENSE file (404 at the pinned revision), the same packaging gap
already recorded for the pruned NVFP4 source and shared by
`lightx2v/Minimax-h3-Turbo`. The `apache-2.0` declaration governs the resize
transform the publisher applied, not the underlying weights: the MiniMax H3
Community License still governs the base compact checkpoint every tag
executes on and the full-rank Turbo adapter each `-r21` file was resized
from, exactly as it does for the Comfy-Org and lightx2v adapters above.
Reviewed 2026-09-02, against the same standard already applied to
`Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` and `lightx2v/Minimax-h3-Turbo`:
acceptable as a third-party derivative source, recorded in the reviewed-source
links above and the decision table's Third-party sources row below.

## Decision record

The direct authorization correspondence is retained privately because it may
contain personal or contact information. The repository records its accepted
scope and a content identity for the maintainer-supplied corroborating image,
not the private correspondence itself.

| Field                  | Current record                                                                                                                                                                                           |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Decision               | All territories, users, Mold surfaces, local/remote/hosted execution, outputs, distribution, and redistribution are authorized; technical support remains capability-gated                               |
| Decision owner         | James Brink, `utensils/mold` maintainer                                                                                                                                                                  |
| Revocation owner       | James Brink, `utensils/mold` maintainer                                                                                                                                                                  |
| Last review            | 2026-09-02                                                                                                                                                                                               |
| License revision       | `bfc8ed0353f5a9733be73e6b2c98ec0948195b86`; LICENSE SHA-256 `59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44`                                                                           |
| Authorization evidence | Maintainer attestation that MiniMax authorized H3 integration with Mold; corroborating image SHA-256 `8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d`; direct grant retained privately |
| Scope approval         | Maintainer legal/compliance determination on 2026-08-14 that use is permitted everywhere and that the README plus H3 user guide satisfy the remaining obligations                                        |
| Upstream review        | At upstream HEAD `42ed227ee7df40d41602854ae760620d6eb651fe`, LICENSE and Q&A hashes still match the pinned review; Q&A SHA-256 `c39dcfc5dc3e546918509b57709db826a9b1945311bffaa01e80501101b8abe4`        |
| Qualification root     | Owner-only `/Volumes/ExternalStorage/mold/uat-h3`; validated external authorization record under its `compliance` directory; no evidence or model payload committed                                      |
| Permitted artifacts    | Mold code/docs/manifests, upstream or transformed H3 artifacts, and generated outputs; private correspondence and owner-only qualification evidence remain confidential                                  |
| Third-party sources    | `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` @ `908eccad7e68751190d04c171956f163bfeed741`, pruned NVFP4 transformers only. Declares the reviewed MiniMax H3 Community License and `base_model: MiniMaxAI/MiniMax-H3`; ships no LICENSE file (404 at the pinned revision). Reviewed 2026-08-22; downloadable, no runtime arm. `lightx2v/Minimax-h3-Turbo` @ `05ef678438e84933c406131b59abbf86919b3aac`, Turbo LoRA adapters only (v1.1 4-step 768p, v1.0 8-step 768p). Declares `apache-2.0` for the adapters and `base_model: MiniMaxAI/MiniMax-H3`; its v1.0 files are byte-identical to Comfy-Org's re-host; ships no LICENSE file (404 at the pinned revision), the same packaging gap as the NVFP4 source. Reviewed 2026-09-01; runnable on the FL2VA compact stack. `drbaph/MiniMax-H3-Turbo-Lora-ComfyUI` @ `be8eb3ea3466cbb7def202ffec0d2fdc054256ac`, SVD-resized rank-21 Turbo LoRA adapters only (three, lossy per-module dynamic-rank derivatives of the full-rank adapters above). Declares `apache-2.0` for these derivatives, `base_model: Comfy-Org/MiniMax-H3` and `base_model_relation: adapter`; ships no LICENSE file (404 at the pinned revision), the same packaging gap as the two sources above; the base MiniMax H3 Community License still governs the checkpoint and full-rank adapter each was resized from. Reviewed 2026-09-02; runnable on the compact stack of its own task, carrying pinned-identity evidence plus the measured A/B against each full-rank source, which ran 2026-09-02 with the per-tier registration decision pending the maintainer's call on this branch's pull request (see `docs/qualification/minimax-h3.md`, "The rank-21 Turbo tiers campaign (2026-09-02)") |
| Permitted users        | Any person or organization in every territory, using local, remote-client, shared-server, hosted, or redistributed Mold/H3 paths                                                                         |
| Prohibited scope       | Claiming technical support for an unimplemented runtime/task/device/envelope; publication of private correspondence or owner-only qualification evidence                                                 |
| Expiry/revocation      | Immediate on MiniMax revocation, narrowed authority, license/Q&A change, loss of access control, or maintainer decision                                                                                  |
| Next mandatory review  | Any upstream license/Q&A revision or maintainer revocation                                                                                                                                               |

## User-facing license and acceptable-use delivery

The maintainer's completed review requires documentation, not an additional
product gate. The root README and H3 user guide identify the MiniMax H3
Community License, link its pinned text, distinguish the weights from Mold's
MIT-licensed code, and tell users to review the upstream terms for their use.
That documentation is the required license, notice, attribution, disclosure,
downstream-term, and acceptable-use delivery for Mold.

No H3-specific clickthrough, acceptance record, modified-file notice, commercial
UI attribution, AI-generation label, provenance field, downstream contract,
reporting path, safeguard, geolocation rule, or periodic in-product review is
required. Existing Mold authentication, request validation, capability
admission, safety settings, and ordinary abuse/operations controls continue to
apply uniformly to CLI, server/API, Discord, desktop, web, iPhone, TUI, gallery,
remote-client, shared-server, and hosted use. No surface requires a separate H3
acceptable-use control; the README and H3 user guide are sufficient.

The private qualification path must remain separate from shipping features and
must fail release-exclusion verification. If authorization expires, is narrowed,
or is revoked, stop private execution, remove qualification credentials and
access, and preserve the ordinary compile-time/product gate before any further
run, release, or hosted deployment.

## Release checklist

- [ ] Compare the pinned H3 license and Q&A revisions with their current
      upstream versions. Any change, or any proposed H3-specific artifact, blocks
      the release until this decision and the user-facing license links are
      reviewed by the named compliance owner.
- [ ] Prove that the exact release contains no bundled or mirrored H3 model
      payload, private checkpoint header, authorization correspondence, or
      generated qualification fixture. Public compact manifests and upstream
      Hugging Face URLs are expected.
- [ ] Confirm every pinned third-party H3 source still exists at its pinned
      revision and still declares the license recorded in the decision table.
      A personal-namespace re-upload can be deleted or relicensed without
      notice, and a reviewed source that has vanished blocks the release. This
      does not apply to `MiniMaxAI/MiniMax-H3` or `Comfy-Org/MiniMax-H3`.
- [ ] Prove that public SM89 CUDA and Apple Silicon Metal H3 binaries retain
      consistent H3-scoped attention provenance while omitting global
      FlashAttention, qualification/capture executables, private evidence
      producers, and every private marker.
- [ ] Verify the README and H3 user guide still carry the pinned license link and
      clearly distinguish H3 assets from Mold's MIT-licensed code.
