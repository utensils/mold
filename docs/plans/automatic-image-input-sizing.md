# Automatic image input sizing across all surfaces

Status: implementation and local verification in progress, 2026-10-07. Base: a07fb0c11.

## Required behavior

A supported, decodable picture that is larger than a model's processing size
must queue successfully. Mold automatically prepares it at the required size.
No manual resize, confirmation, or new toggle is required. Preserve originals,
aspect/composition, reference order, orientation, transparency and input roles.
Do not silently change explicitly selected output dimensions or invent support
for a role that a recipe does not implement.

Keep a separate, bounded ingestion envelope for corrupt files, unreasonable
decode allocations and transport limits. These are not model processing sizes.
Ordinary large photographs should be normalized before transport/role checks;
truly unsafe or undecodable inputs should still give a precise refusal.

## Confirmed root cause

`StillReferenceValidation.swift:12-18` interprets `max_pixels_single/multi` as
limits on original images. Qwen Image 2.1 advertises 1024² there, so a perfectly
ordinary 1280×853 JPEG disables Generate on both native apps. FLUX.2 and
Qwen-Image-Edit have the same native error path.

In contrast, `generation_profile.rs:1278` validates reference roles, counts,
formats and broad decode bounds, not those model pixel caps. The engine already
resizes. This is primarily a client/contract semantics bug, not missing Qwen
inference support. The previous success banner is independent of the current
draft's refusal and can misleadingly remain visible.

## Surface audit

| Surface | Current path/evidence | Gap and implementation requirement |
| --- | --- | --- |
| Native macOS and iOS/iPadOS | Shared `PictureImport.swift:99` bounds newly imported pictures to 4096 axis / 2 MiB; `StillReferenceValidation.swift` then rejects processing-size overflow | Remove the false processing-cap refusal; keep genuine decode/role checks. Cover file, drag/drop, paste, Photos, Share, Library, restored drafts, replacement/reorder and model changes. Shared fix must reach both apps. |
| Web SPA | `studio/lib/inputImage.ts` provides proportional 4096-axis / 2-MiB normalization; CreatePage, SourceMediaPanel, IdentityPanel and ImagePicker use it | Verify every authoring route uses this authority, including Library and prepared submissions; never confuse output fitting with reference resizing. |
| Tauri desktop | `desktop/src/lib/image.ts` delegates normal imports to shared inputImage; some gallery/lightbox paths use raw base64 | Trace raw imports to final submission and close actual authoring bypasses. Raw export/copy stays lossless. |
| Tauri mobile / Android | `desktop/src/mobile` uses shared inputImage in pickers, but MobileApp/viewer also contain raw base64 paths | Audit camera/picker/Library/Share/restore/use-as-source consumers; normalize fresh authoring inputs consistently. |
| CLI remote and `--local` | `commands/generate.rs` has explicit source fitting and canvas derivation; references generally pass through to engine; `commands/identity.rs` validates originals before submission | Share Rust input preparation before role limits/transport; preserve exact source-fit semantics. Local fallback and remote must agree. Cover every still flag and repeated/grouped forms. |
| CLI MCP | `commands/mcp.rs` decodes reference base64 and derives canvas | Use the same Rust preparation as CLI/server; cover ordinary source, references, identity and typed routes exposed by its actual schema. |
| Discord | `commands/generate.rs:925` rejects source attachments above 10 MiB and checks aggregate original sizes; downloads generally preserve originals; H3 uses `h3_references.rs` | Distinguish bounded download limits from normalized request size. Normalize supported stills before request-size checks; preserve typed metadata and names. Do not add unsupported slash-command image roles. |
| HTTP/API, batch, queue, chains, upload leases, retained reuse | `queue_media_admission.rs` resolves media then validates; `reference_uploads.rs` owns immutable media facts; engine performs model preparation | All admission doors must accept model-size overflow within bounded ingestion. If canonical bytes change, do so before hashes, descriptors, placement, persistence and one-use lease consumption; preserve original provenance and repeat/recovery behavior. Do not mutate existing upload authority in place. |

## Model and input-role audit

| Family/input | Existing processing | Required outcome |
| --- | --- | --- |
| SD1.5, SDXL, SD3/3.5, FLUX.1, FLUX.2 source, Z-Image, ordinary Qwen-Image, Wuerstchen img2img | Call `img_utils::decode_source_image`, which resizes to requested canvas | Keep existing fitting/composition; consistently bound transport/decode without a new model-size refusal. Only supported roles count. |
| Inpaint masks and ControlNet hints | `decode_mask_image` resizes to latent size; source decoder also serves control hints | Transform paired masks with their source geometry; do not independently crop/stretch them. Preserve grayscale/soft masks. Audit all input paths including CLI. |
| FLUX.2 Klein/base/dev ordered references | `img_utils.rs:38` already downsizes to 2024² single or 1024² multiple, then aligns to 16 | Native must accept oversized originals; adding a second reference must correctly switch effective sizing without destroying the original. Keep upstream minimum-side/aspect rules. |
| Qwen-Image-Edit ordered target/references | `qwen_image/pipeline.rs` uses source-pixel budget and separate encoder paths | Remove native false refusal; preserve target-driven canvas and per-reference preprocessing. |
| Qwen Image 2.1 all tiers | `qwen_image21/reference.rs:110` and `conditioning.rs:40` prepare each reference at its own aspect around 1024² | Accept >1 MP inputs; preserve RGBA, last-reference canvas intent, EXIF and ICC handling. Do not introduce a second blanket 1-MP resize that changes qualified engine pixels. |
| SD1.5/SDXL IP-Adapter | Profile deliberately has no pixel ceiling because vision tower resizes to 224 | Keep supported img2img + adapter combinations and batching; normalize only when ingestion/transport requires it. |
| PuLID / identity photos, singular and plural | `identity.rs:897` rejects >16 MiB, >8192 axis, >32 MP; inference performs face detection/alignment and vision resizing | Safely normalize ordinary large originals before identity validation; never fit identity photos to output canvas. Preserve faces and ordering; maintain bounded decoding and multi-photo limits. |
| LTX-Video image conditioning | `ltx_video/pipeline.rs` resizes image conditioning | Same transparent size handling, preserving pipeline-specific shape. |
| LTX-2 / 2.5 still sources and keyframes | `ltx2/preprocess.rs` owns oriented sRGB and float resize/center crop | Preserve qualified crop/interpolation; cover keyframe lists, chain entry and retained sources. Do not indiscriminately re-encode generated continuation/retake frames. |
| Wan 2.1/2.2 first/last/source images | `wan/pipeline.rs` and `wan/conditioning.rs` have model-specific preparation; some paths require finalized geometry | Normalize at the correct source-fit boundary before strict tensor checks; preserve paired geometry and advertised canvas. Do not remove meaningful pipeline assertions. |
| MiniMax H3 Ref2VA still references | `minimax_h3/reference_media.rs` resizes; core owns down-only 2048-short-edge / 32-grid reference geometry, broad 100-MP ingestion bounds | Keep geometry/row-budget/crop calculations tied to effective image facts; synchronize descriptor and upload metadata after any transport normalization. |
| MiniMax H3 FL2VA endpoints | `minimax_h3/pipeline.rs:1210` owns anchor/follower resize | Preserve anchor canvas and follower crop; size alone must not force manual editing. |
| Hunyuan3D single source, named multiview and paint appearance | Tier-specific image preprocessing; `hunyuan3d/paint_images.rs` resizes/composites appearance | Cover every supported named role, preserve alpha/matting semantics, and avoid generic flattening of source silhouettes. |
| Families/recipes with no image role | Capability authority already hides/refuses unsupported roles | Retain that refusal; automatic sizing is not a new model capability. |

This is a source audit, not a claim of hardware qualification for every model.
Implementation must trace final call sites and record coverage rather than
treat this table as proof that every raw-byte caller is broken.

## Implementation sequence

1. **Regression first.** Add failing native tests reproducing the 1280×853
   Qwen reference and equivalent FLUX.2/Qwen Edit cases. Test native validation
   against server acceptance. Add fixtures for transport overflow, alpha,
   orientation, multiple references and source/mask alignment where changed.
2. **Clarify the authority.** Document processing caps versus ingest bounds in
   the profile contract and generated docs. Prefer an additive explicit sizing
   policy only if necessary; avoid a schema migration just to remove a wrong
   client check. Older-host absence remains compatible, not unsupported.
3. **Fix native refusal and feedback.** Allow engine-resizable references;
   show genuine current-draft refusals clearly and avoid stale success messages
   implying acceptance. Use shared MoldClient logic for both native apps.
4. **Close ingestion gaps.** Add/reuse one bounded, CPU-only Rust image
   normalization utility in mold-core for CLI/Discord/MCP and necessary HTTP
   paths. Normalize only for real transport/role-ingest requirements, not a
   blanket 1-MP cap. Apply oriented proportional downscaling, preserve alpha,
   sniff resulting format, update filenames/metadata, and avoid re-encoding
   already-valid inputs. Check total request budgets after normalization.
   Preserve explicit download/body ceilings as bounded ingestion guards.
5. **Finish every client route.** Reuse PictureImport and studio inputImage,
   close verified bypasses at shared authoring boundaries, and prepare a stable
   submission snapshot. Re-evaluate on count/model/role changes using originals.
   Keep exports, original assets and retained-media evidence intact.
6. **Admission and recovery parity.** Test direct, batch, local, preview,
   uploads, reuse and queue restart against effective media facts. Preserve
   idempotency, reference scope, source digests and immutable upload leases.
   Avoid GPU allocations before admission and excessive decode work on async
   executors. Do not add duplicate resizing where qualified engines already
   perform it correctly.
7. **Documentation and delivery.** Update this checklist with exact coverage;
   add changelog fragment and affected README, canonical CLI skill renderer,
   website and owning rule docs. Regenerate generated files with their owners.
   Run appropriate local CI, independent parent review, PR exact-head checks,
   then merge and synchronize. Conventional branch/commits required.

## Acceptance matrix

- Every supported image role queues from every surface that exposes it with an
  ordinary oversized JPEG/PNG; transparent and EXIF-oriented fixtures retain
  semantics. Explicit output size remains unchanged.
- Include the reported 1280×853 reference, 12/48-MP phone photos within safe
  decoding envelopes, >transport-budget compressible images, odd sizes, and
  mixed-size multi-reference lists. Test both smaller and larger second images.
- Below-limit inputs remain byte-identical unless a required format conversion
  or user-selected source fit applies. Never upscale merely for ingestion.
- Check drag/drop, file, clipboard, Photos/camera, Library, Share, restored
  draft, retained reuse, CLI local/remote, MCP and Discord request construction.
- Verify actual engine preprocessing sizes without requiring full expensive
  generations for every variant; retain existing upstream pixel goldens.
- Corrupt images, impossible dimensions, unsupported formats/roles, excessive
  counts and genuine hard resource limits still fail precisely and promptly.
- Native UAT reproduces the disabled-button case using a benign fixture with
  the same dimensions. Do not queue the user's private reference or launch a
  real generation without a deliberate test need.

## Progress

- [x] Source audit and cross-surface plan.
- [x] User unblocked with resized local reference copy; original preserved.
- [x] Failing regression tests captured (native processing refusal and Library bypass).
- [x] Shared policy and native validation corrected.
- [x] Authoring surfaces and exposed roles audited; verified fresh-input gaps corrected.
- [x] Admission/transport/recovery parity verified at source/contract level; hardware generation and live restart qualification remain outside this task.
- [x] Documentation and generated contracts synchronized.
- [x] Parent peer review and fixes complete.
- [ ] Exact-head PR checks green, merged, local state synchronized.

## Implemented coverage and boundaries

- Shared native validation accepts the reported 1280×853 fixture, references
  above single/multiple processing budgets, retained attachment replacement,
  and older-host profiles with no pixel budget. It retains required/count,
  unreadable-header and the server's 16384-axis/100-MP/200:1 ingestion guards.
  File/drop/paste/Photos/Share/Library imports already use `PictureImport`;
  restored Mac snapshots and parked/reference mutations reach this shared
  validation. Persistent prior acceptance feedback remains an event record.
- Web and Tauri normal file/camera/identity/typed/keyframe imports already use
  `inputImage`, `readFileBase64` or normalized desktop IPC. The shared desktop
  Library/Lightbox/History **Use as source** path now applies that authority
  before H3 dimensions/MIME/names are constructed. Raw export/save paths and
  retained restore readers intentionally keep exact original authority bytes.
- CLI single/plural identity originals are securely read within a 64-MiB
  original envelope, then normalized as one set before role validation.
  PuLID-enabled HTTP single and batch admission prepare fresh inline identity photos on a
  blocking worker before validation, placement and media sealing. Placement preview applies the
  same fresh preparation before its identity gate; replay
  fingerprints remain about the original submitted operation. Existing upload
  scopes are frozen before host defaults; identity preparation does not change
  any typed reference, upload handle, digest or retained-resolution bytes.
- CPU preparation preserves byte identity below real role/transport limits,
  proportionally downsizes, applies EXIF once, retains the ICC profile without
  converting pixel colour space, and uses premultiplied-alpha resampling.
  Transformed names/MIME describe the PNG bytes. The identity set respects
  both individual (8192 axis/32 MP/16 MiB) and aggregate (64 MP/32 MiB) bounds.
- CLI/MCP ordered-reference groups normalize only above 32 MiB aggregate;
  engine processing caps never trigger this transport preparation. MCP mesh
  and CLI named Hunyuan views prepare fresh inline images before shape/MIME/
  provenance facts, keeping four views within a 32-MiB image budget. CLI H3
  boundary originals can be prepared within the safe ingestion envelope.
- CLI H3 Ref2VA already streams authenticated file uploads with descriptors
  and hashes, independently of the inline body budget; those originals remain
  untouched. Discord still downloads have a separate bounded original limit;
  fresh source/reference/keyframe images are normalized before aggregate
  request checks, and identity uses the shared role policy. Discord H3 image
  preparation occurs before descriptor SHA/dimensions and upload construction;
  audio/video bytes remain unchanged.
- Existing SD/SDXL/SD3/FLUX/Z-Image/Wuerstchen source, mask, ControlNet, IP-Adapter,
  PuLID inference, LTX still/keyframe, Wan endpoints, H3 and Hunyuan pipeline
  preprocessing remain the qualified engine authorities. No second model-size
  resize, output-canvas change, crop policy or mask geometry was introduced.
- Hard HTTP body and bounded file/download/decode ceilings remain deliberate
  ingestion guards. This work does not promise arbitrary inputs larger than
  those envelopes or change unsupported roles, count bounds, video/audio
  limits, upload ownership or retained-media recovery semantics.

## Local evidence

- Native regression before fix: five expected failures, including 1280×853
  against Qwen 2.1's 1048576 processing budget; after fix shared tests pass.
- Full MoldClient package: 1219 tests passed. Parent native suite: 969 tests
  across 140 suites passed (25.549 seconds), with strict `make lint` green.
- Library authoring regression: one expected failure; seven tests passed after
  the fix. Frontend architecture check passed.
- Core preparation tests cover original byte identity, proportional alpha-safe
  resize, EXIF, ICC retention/no implicit colour conversion, transparent edges,
  ordered-reference transport normalization, native/server processing-budget
  agreement and immutable non-identity media authorities.
- A real benign 8000×6000 grayscale PNG fixture was accepted after identity
  preparation (162.64 seconds in debug, no GPU generation). That expensive
  fixture is explicit/ignored in routine tests and can be run with `--ignored`.
- Default CLI/Discord/server type check passed. Final all-target Clippy with
  PuLID enabled passed for core/CLI/Discord/server with `-D warnings`. Core
  preparation tests: 10 passed and the explicit expensive fixture ignored in
  the routine run. Studio input tests: 5 passed; desktop restore/source tests:
  22 passed. Formatting and generated-profile checks passed.
  The first all-target Clippy run was stopped
  when combined task-created Cargo and native build caches exhausted disk.
  Only this task's generated incremental artifacts were cleaned up.
- PuLID HTTP placement-preview regression passed with a compact original above
  identity's axis envelope. Fresh single/plural identity helper tests exercise
  role preparation and immutable upload facts; direct and batch admission call
  that shared helper before sealing. No live durable restart was performed.
- Full Discord library suite: 190 tests passed.
- Documentation local CI passed: install, formatting, reference verification
  and VitePress build. Parent native UAT verified portrait image/video Fit and
  Actual Size, scrolling/return to Fit, replay and accessible slider seeking.
- This is source/contract and CPU preprocessing evidence. It is not new GPU
  qualification of every family and no private user media or real generation
  was used.
