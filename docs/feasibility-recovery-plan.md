# Plan: Fix "No selected machine has the model and required components" + missing-component auto-recovery

**Status:** phases 0–1 implemented; phase 3 messaging implemented; repair/resume
blocked on download authority · **Owner:** Codex · **Date:** 2026-07-28

## Implementation status

Implemented in this change:

- Shared strict classification for `planned`, `unsupported`, `infeasible`, and
  `temporarily_unavailable`, including validated additive recovery metadata.
- Web, desktop, and iPhone retain the server reason, transport/HTTP error, malformed
  response, and routing-authority race instead of collapsing them into a generic
  unavailable route.
- Web and desktop multi-host routing retry one status-only authority race, preserve
  exact URL/key/instance fences, and keep the origin eligible for strict legacy
  `404`/`405` fallback while its first status poll is still connecting.
- Known registry-backed encoder dependencies produce a read-only planned preview with
  `pending_downloads`. Their preview fingerprints use a separate, domain-tagged
  repo/filename/size/format/quantization identity; no download is registered or
  started, and admission recomputes the plan from the landed bytes.
- Hard preparation failures include absent manifest components and `repair_model`
  metadata where the existing component-status authority can prove them. Stale device
  pins remain hard-infeasible.

Not implemented in this change:

- Automatic component repair and held-request resume. The web download singleton and
  API helpers are origin-only, but repair must POST, list, stream, and cancel on one
  immutable remote target. A catalog repair can also enqueue a primary job plus
  companion jobs, so one returned job id is not sufficient ownership for a visible,
  cancellable recovery.
- Native automatic repair/resume. Existing missing-model pull recovery starts after a
  failed submission; placement recovery instead needs a finalized held-request
  authority spanning source preprocessing, form/prepared staleness, route identity,
  unmount, and a grouped pull lease.

Phase 2 therefore requires a shared target-aware, all-host download authority with a
server-issued repair group id (or equivalent complete job-id set) and group-level
list/stream/cancel semantics. Only after that exists should any surface auto-start a
repair and promise generation will resume.

## Symptom

Web Create submissions fail with the toast *"No selected machine has the model and
required components for this print."* regardless of which model is chosen, even though
every model in the picker comes from the machine's own `/api/models` inventory (the web
picker already filters to `downloaded && standalone`).

## Diagnosis (verified against the live server on :7680)

The toast is emitted by `web/src/pages/CreatePage.vue` (`resolveFeasibleSubmitRoute` and
the inline copy in `onSubmit`) whenever `useHostRouting.resolveFeasible()` returns
`null`. That null is a **collapse of at least six unrelated failure classes**, and the
server's precise `reason` string is discarded on every one of them:

| # | Failure class | Where it happens | What the user should have seen |
|---|---------------|------------------|--------------------------------|
| 1 | Authoritative `infeasible` from `POST /api/generate/placement-preview` | `routes.rs::placement_preview_for_request` → `variant_dependencies::prepare_execution_inputs_existing_only` | The server's `reason`, e.g. `required local dependency 't5-….gguf' from '…' is not installed`, `model 'X' has no concrete local artifacts`, `component placement references unavailable device 'cuda:…'` |
| 2 | `temporarily_unavailable` outcome (scheduler preview transiently errored) | `classifyPlacementPreview` has no case for it → **classified "invalid"** → host silently dropped | "Machine X couldn't compute a plan right now, try again" |
| 3 | Probe HTTP error other than 404/405 (401 key rotated, 5xx, network/timeout/CORS) | `resolveFeasibleWithPreview` catch-block → host silently dropped | "Machine X didn't answer the feasibility probe (401)" |
| 4 | Routing-authority race (server `instance_id` changed mid-probe, localStorage registry/target mutated, first-poll instanceId write) | `routingAuthorityGeneration` mismatch → unconditional null | Silent retry, or "routing state changed, resubmit" |
| 5 | Legacy fallback (`unsupported` outcome — fires **whenever post-upscale or local prompt-expand is enabled**) requires origin to be `ready` in `resolveRoute`'s Auto path, though the candidates filter deliberately exempts origin from that check | `lib/hostRouting.ts::pickAutoHost` filters `status === "ready"`; origin is `connecting` for up to the first poll and `error` on any status blip | Origin should be routable in the legacy path exactly as it is in the probe-candidate path |
| 6 | Genuinely missing components (partial install: VAE, encoder, tokenizer file gone) | `ModelPaths::resolve` fails or `manifest_model_needs_download` → infeasible | Name the components + offer download/repair, then resume |

Live verification performed:

- `placement-preview` returns `outcome: "planned"` for every downloaded model
  (`flux-dev:bf16`, `flux2-klein:q8`, `qwen-image:q8`, `sd15:fp16`, `cv:2442439`) with a
  minimal request → the server inventory itself is currently healthy.
- A non-downloaded model returns authoritative `infeasible` with reason
  `model 'flux-schnell:q8' has no concrete local artifacts`.
- A placement pin to a nonexistent device returns authoritative `infeasible` with reason
  `component placement references unavailable device 'cuda:deadbeef'`. Device ids are
  content hashes and this machine's topology changed recently (multi-GPU restore, #585),
  so stale pins in restored requests ("Reuse settings" copies `request.placement`
  verbatim into the form) reproduce the symptom for every model until cleared.
- `upscale_model` or `expand` in the request → non-authoritative `unsupported` → the
  client goes down the class-5 legacy path on **every** submission.
- No stale server state on this box: no saved `[models.*].placement` in config, no
  `model_prefs.lora_path` rows.

**The key architectural finding:** the actual admission path already self-heals almost
everything the preview refuses. `prepare_execution_inputs` (policy `Admission`)
auto-downloads missing encoder variants with SSE `DependencyWait`/`DownloadProgress`
progress, and `/api/models/pull` re-fetches only the missing files of a partial install.
The preview (`ExistingOnly` policy) fails closed on exactly the conditions admission
would have healed, and the client turns that into a dead end with a misleading message.

## Design

Three principles:

1. **Never discard the server's reason.** Every routing failure names its cause and the
   machine it happened on.
2. **The preview stays read-only** (authority-boundary invariant), but it must
   distinguish *"infeasible forever"* from *"feasible after downloads the admission path
   already knows how to do"*.
3. **Recovery is automatic where a mechanism already exists**, explicit where it
   doesn't:
   - Missing **encoder variant** (T5/Qwen3/Qwen2-VL GGUF): admission auto-downloads →
     preview should return `planned` (with pending-download metadata), the client
     submits normally, and the existing dependency-download SSE stream carries progress.
     Generation "resumes" by never having stopped — it's one job.
   - Missing **model component** (VAE, transformer shard, tokenizer — a partial
     install): preview stays `infeasible` but carries structured components; the client
     runs `POST /api/models/pull` on the same host (existing endpoint, per-file skip of
     present files = repair), tracks it in the downloads center, and auto-resubmits the
     held request when the pull completes.

### Wire contract (additive, version stays 1)

`GenerationPlacementPreview` gains optional fields; absent fields keep today's exact
bytes so old desktop/iPhone clients are unaffected:

```jsonc
// planned, but admission will download encoder deps first:
{
  "version": 1, "authoritative": true, "outcome": "planned",
  "candidate": { /* unchanged */ },
  "pending_downloads": [            // NEW, additive
    { "kind": "text_encoder", "name": "t5-v1_1-xxl-q8.gguf",
      "repo": "…", "bytes": 5100000000 }
  ]
}
// infeasible with a repair route:
{
  "version": 1, "authoritative": true, "outcome": "infeasible",
  "reason": "model 'flux-dev:bf16' is missing 2 components",
  "missing_components": [           // NEW, additive
    { "kind": "vae", "name": "ae.safetensors", "present": false,
      "repair_model": "flux-dev:bf16" }
  ]
}
```

`missing_components` reuses the `ModelComponentStatus` shape from
`GET /api/models/:model/components` (which already computes `repair_model`).
The durable-chain preview remains non-authoritative `unsupported`; it carries empty
recovery fields until frozen per-device chain-stage plans exist.

Compatibility: old clients classify `infeasible` exactly as today (fail closed, queue
nothing), and `planned` + extra fields routes normally — strictly better, since
admission then downloads the deps it always could.

### Server changes

`crates/mold-server/src/variant_dependencies.rs`:

- `ensure_downloaded` under `ExistingOnly` returns a structured
  `MissingDependency { kind, name, repo, filename, bytes }` instead of a prose-only
  error. Materializers thread it up; `prepare_inputs_for_devices` aggregates per-device
  outcomes into `{ prepared | pending(downloads) | failed(reason) }`.
- New rule: if every failure on an otherwise-eligible device is an
  **admission-materializable** dependency (a known variant-registry entry), preparation
  succeeds with the deterministic would-be cache paths and the pending list. Estimate
  confidence drops to `"low"`; the execution-equivalence fingerprint for a pending
  artifact hashes its identity (repo + filename + size) instead of file contents —
  called out below as its own risk item.
- Hard failures stay hard: broken BF16/FP16 encoder expected from the model's own
  manifest, malformed Gemma shard sets, unknown variant pins, stale device pins.

`crates/mold-server/src/routes.rs::placement_preview_for_request`:

- Map preparation `pending` → `planned` + `pending_downloads`.
- Map preparation failure → `infeasible` + `reason` + `missing_components` (computed via
  a strictly local existing-only component-status path when the model resolves, so VAE
  and friends are named individually without catalog/network mutation).
- A manifest model with no concrete local artifacts emits every absent manifest
  component. Opaque models that cannot be resolved locally keep the authoritative
  reason but omit guessed component metadata.

### Client changes (studio-first, all three surfaces)

`studio/api/generationPlacement.ts`:

- `classifyPlacementPreview` gains `"infeasible"` (strict: version 1,
  `authoritative: true`, `outcome: "infeasible"`, no candidate) and
  `"temporarily_unavailable"` classes. Unknown outcomes remain `"invalid"` (fail
  closed). Parse `pending_downloads` / `missing_components` when present.

`web/src/composables/useHostRouting.ts`:

- `resolveFeasible*` returns a **discriminated result** instead of `HostRoute | null`:
  `{ kind: "route", route }` | `{ kind: "infeasible", perHost: [{hostId, label, reason,
  missingComponents}] }` | `{ kind: "unreachable", perHost: [{hostId, error}] }` |
  `{ kind: "transient" }` (authority race / temporarily_unavailable). No information is
  thrown away.
- Class-5 fix: the legacy fallback resolves through the same origin-exempt candidate
  rule as the probe filter (origin is routable while `connecting`; a genuinely dead
  origin fails at dispatch with an honest network error).
- Class-4 fix: one silent re-resolve on authority-generation mismatch before reporting
  `transient`.

`web/src/pages/CreatePage.vue`:

- Replace all four generic toasts with per-cause messages built from the result:
  - infeasible + missing components → inline **recovery card** (not a toast): "Mold
    Studio on *host* is missing the VAE and text encoder for *flux-dev* (7.2 GB). —
    **Download and generate** / **Cancel**". Given the product decision that recovery
    should be automatic, the card starts the pull immediately and shows it; the buttons
    are Cancel/Details.
  - unreachable/probe-error → name the host and the HTTP/network error.
  - transient → "try again" phrasing (and the automatic retry usually absorbs it).
- **Held-request resume:** on starting a repair pull, freeze `{request, decision, route,
  model, host instanceId}` exactly like the prepared-work pattern. When the download
  completes: re-run `resolveFeasible` (must still plan on the same frozen host — the
  prepared-route staleness rules apply: any prompt/model/host/form change or instance-id
  mismatch cancels auto-resume but leaves the download running), then submit. Surface
  the pull in the existing downloads center; failure/cancel of the pull cancels the
  resume with a visible row, never silently.
- `pending_downloads` on a planned route needs no new flow — submit as normal; the
  activity strip already renders the job, and the SSE `DependencyWait` /
  `DownloadProgress` events should be surfaced on the in-flight row ("downloading text
  encoder — 42%") instead of appearing as a stalled job.

Desktop (`GenerateView.vue`) and iPhone adopt the same classification results from
`studio/`; desktop reuses `useDownloadsStore`, iPhone reuses `useMobileDownloadsStore`
and its existing pull-lease flow (per the native prepared-expansion invariant: same
host, retained prompt list, inline Connecting → … → Pulling states). Their generic
"authoritative route" toasts get the same per-cause split.

## Phases

**Phase 0 — stop lying to the user (small, ships first).**
Web-only: thread `reason` / probe errors / classification through `resolveFeasible*`
and render per-cause toasts. Add the `"infeasible"` + `"temporarily_unavailable"`
classes to `classifyPlacementPreview` (server already sends these outcomes today). Fix
class-5 origin-ready inconsistency and class-4 single retry.
*This alone converts the reported bug from a mystery into an actionable message and
likely resolves the "regardless of model" experience.*
Tests first: `classifyPlacementPreview` new classes; `useHostRouting` result-shape
tests (per-host reasons retained, origin routable while connecting, retry-on-race);
CreatePage submit-flow message selection.

**Phase 1 — structured wire contract.**
Server: structured `MissingDependency`, preparation outcome tri-state,
`pending_downloads` / `missing_components` on both preview endpoints.
Tests first: existing-only preparation with a missing known-variant encoder returns
pending (not error) and never registers a download (extend
`existing_only_dependency_check_never_starts_a_download`); missing-VAE manifest model
returns infeasible with the VAE named; stale device pin stays hard-infeasible; wire
snapshots for old-client byte-compatibility when the new fields are empty.

**Phase 2 — auto-download + resume.**
Web recovery card, held-request freeze/resume, downloads-center integration,
`DependencyWait`/`DownloadProgress` rendering on activity rows.
Tests first: resume submits exactly the frozen request on the frozen host; form edit /
host change / instance-id change cancels resume but not the download; pull failure
surfaces and cancels resume; multi-host — if any other candidate plans cleanly it wins
and no download is offered (never trigger downloads when another selected machine can
print now); recovery card only when *no* candidate plans.

**Phase 3 — desktop + iPhone parity.** Same split messages and recovery flow through
the shared studio results; respect each surface's existing pull authorities and the
prepared-work staleness rules.

**Phase 4 — docs + invariants sync (required by repo workflow).**
`CHANGELOG.md` `[Unreleased]`; CLAUDE.md placement-preview invariant paragraph gains:
"authoritative `infeasible` carries additive `missing_components`; `planned` may carry
additive `pending_downloads`; previews remain read-only and never start downloads;
clients must keep queueing nothing on infeasible but may offer the structured repair
flow"; `.claude/skills/mold/SKILL.md`, `website/` guide pages, `apps/mobile/README.md`
if the iPhone flow changes.

## Risks / open questions

- **Execution-equivalence fingerprints for pending artifacts** hash identity rather
  than content until the file lands. **Resolved:** pending previews use an explicit
  preview-only fingerprint/storage domain keyed by registry repo, filename, declared
  bytes, container, and quantization. Admission never carries it and recomputes from
  landed bytes; focused tests reject any leaked opaque missing-path identity.
- **Estimate honesty:** a planned candidate whose deps still need a 5 GB download has a
  wrong `predicted_completion_after_ms`. Add a conservative download-time term from
  `bytes` (unknown link speed → `estimate_confidence: "low"`), or accept the skew and
  document it. **Decision:** this change accounts for declared dependency bytes plus
  encoder headroom in feasibility and forces `estimate_confidence: "low"`, but does not
  invent a transfer-time estimate without a measured link rate.
- **Auto-start vs confirm for repair pulls:** this plan auto-starts per the product
  direction ("they should auto download and then the generation should resume"), with a
  visible, cancellable card. If disk-space concerns surface (multi-GB VAE/encoder
  pulls), demote to one-click.
- **Stale device pins** (class 1, placement flavor) are fixed as *messaging* here
  ("references unavailable device X — clear the pin"). A follow-up could auto-clear
  pins referencing devices absent from `/api/devices`, but silent clearing conflicts
  with the never-silently-reroute rule — keep it explicit.
- Per repo policy this touches a scheduling/authority boundary: Phase 1 needs
  independent exact-SHA Sol and Opus 5 high-effort reviews.

## Acceptance evidence

- Core recovery-wire tests, server pending/no-download tests, missing-manifest-component
  tests, and stale-pin tests pass.
- Studio classification tests, focused web routing/Create tests, focused native
  routing/Generate/mobile tests, and desktop/iPhone production builds pass.
- Native slice Opus review: Claude Code 2.1.220, `claude-opus-5`, high effort,
  session `fa302b5c-771a-4524-b902-c82bd6555af6`; all accepted findings were fixed.
- Final integrated exact-SHA Sol and Opus acceptance provenance is recorded in the task
  handoff after those reviews complete.
