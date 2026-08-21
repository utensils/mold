# Identity photos on iPhone — staging notes for doc consolidation

Written for the orchestrator consolidating #1231's documentation. Everything
here is the invariant text that would otherwise have gone into `CLAUDE.md`
(iPhone app section) and `crates/mold-cli/src/skill/SKILL.md`. Nothing in this
file is user-facing; the user-facing copy already landed in
`website/guide/iphone.md` and `apps/mobile/README.md`.

## Proposed CLAUDE.md addition (iPhone app section)

> **Identity photos are a phone surface too.** Create mounts the shared
> `@studio/components/IdentityPhotoWell.vue` (through
> `desktop/src/mobile/MobileIdentityWell.vue`) in the PRIMARY form beside the
> source wells — never inside the source-media plan, and never in the Advanced
> sheet — gated on positive `supportsIdentity` knowledge only and hidden for
> Sequence output. A photo staged before a capability-losing model switch is
> PARKED: retained in the form, kept off the wire by `buildRequest`, Develop
> still enabled, and rendered again the moment a qualified checkpoint returns.
> The photo is never fitted, cropped, or resized against the canvas and carries
> no `source_fit` provenance. `Identity strength` and `Identity start step`
> live in the full-screen Advanced sheet, feed its active-count badge, clear on
> its Reset while the attached face survives, and stay absent from the request
> until touched. Every refusal is inline beside the control and repeated in the
> Develop blocker — never a toast. Prepared Batch N siblings inherit the whole
> partition. The Library viewer's Info sheet lists identity provenance (name ·
> short digest · strength · from step) and opens for it even on a host with no
> organization capability; **Use as prompt** restores both knobs and
> re-attaches the photo from the shared content-addressed stash through
> `restoreIdentityPhoto`, disclosing a miss in the persistent inline status
> line rather than rendering a different face. `desktop/src/mobile/identity.ts`
> holds the pure phone-shaped helpers (45 MiB request-media budget, native
> ingest, Info rows, reuse outcome) so `MobileApp.vue` stays an orchestrator.

## Proposed shared-policy addition (applies to every Studio surface)

> **A changed identity photo is a conditioning-media change.**
> `studio/lib/promptTransform.ts`'s `conditioningFingerprint` reads `id_image`
> alongside `source_image`, the video/audio sources, keyframes, and H3
> references, so reviewed prompt work (remix on iPhone; remix, prepared, and
> quick expansion on desktop and web) stales on a face swap exactly as it does
> on a source-image swap. There is deliberately no second, identity-only
> staleness rule. `ExpandTask` is untouched: a face reference does not make a
> text-to-image print an img2img one, and `/api/expand` still never receives
> media bytes.

## Notes for the consolidator

- Mobile's plain (non-remix) prepared batches carry no conditioning fingerprint
  at all today — they freeze `ExpandTask` only, which is why a source-image
  change does not stale them either. Identity was deliberately given exactly
  the same treatment rather than a special case. If that gap is ever closed,
  close it for all conditioning media at once.
- `crates/mold-cli/src/skill/SKILL.md` already documents the four request
  fields from #1253; the only phone-specific fact worth adding there is that
  the iPhone surface now ships them too.
