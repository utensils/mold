# Durable source media across the library

## Scope and decisions

Follow-up scope: native iOS and macOS library connectivity notices must not shift the grid when machines become reachable/unreachable. Keep the notice accessible while reserving stable layout geometry or placing status outside the grid's layout flow; visually compare grid positions across a connectivity transition.

Expose retained inputs in every output's details and in a distinct Source Media browser on web, Tauri desktop, Tauri mobile, native iOS and native macOS. Users can select a retained input and attach it to a new creation independently of the original recipe. This is a projection of existing library ownership, not a second media store: no independent upload, pin or delete. Trash keeps ownership; permanent deletion releases only that output's pins. Existing whole-recipe reuse remains separate.

Ship one PR after independent plan review, implementation review, focused contract tests and visual UAT. Leave the PR unmerged after final-head CI for the user to inspect all surfaces. The latest user instruction supersedes prior merge/deploy authorization: do not merge, deploy either host, or publish client releases. Include polished screenshots for review.

## Architecture

1. Preserve archive authority and encrypted retention. Extend retained-member presentation additively with media kind, role/position label and preview eligibility. Keep raw member download and original-role reuse unchanged. Add authenticated member content access for normal attachments, including decoding a retained keyframe's JSON envelope into its original image. Use that same extraction for keyframe thumbnails. Never resolve provenance paths or expose pin/store identities.
2. A bounded batch inventory API accepts exact output filenames selected by the client's existing library projection. It returns each output's inventory or explicit unavailable state, with limits and private/no-store responses. No decrypting media bodies for enumeration. Clients page visible owners with bounded concurrency and cancellation, never silently stop at a fixed library size. Older servers fall back to bounded per-output inventories; unsupported content actions explain the upgrade requirement.
3. Source Media uses the existing logical visibility and physical machine projection before reading sources. Hidden collection sources remain hidden from the general view. List each retained member with owner, machine, role and preview (or a media-kind tile for audio/video/unknown). Preserve output-scoped identities; do not deduplicate different owners by basename or guess content equality. Provide owner navigation, search, loading, empty, partial-error and retry states. Trash sources remain accessible from their owner's details.
4. Shared Studio components and helpers own Vue member cards, authenticated bounded thumbnails/content, lazy inventory loading and labels. All three Vue shells bind their exact routes and existing attachment flows. Shared MoldClient owns equivalent transport/presentation policy; native apps own their SwiftUI views. Fence asynchronous completion by owner, host route, instance and view lifetime; revoke object URLs and cancel obsolete work.
5. Details always ask the server, regardless of metadata markers. Display every retained binary role, including source/start/end/keyframes, ordered references, identity, mask/control, continuation video, audio and processed/stage sources. Missing/legacy/auth states are honest; source-free outputs avoid misleading damage messages. Individual reuse uses original bytes through normal attachment APIs, never a thumbnail or an original-role reuse session. Offer only destinations supported by the current authoring flow; preserve edits made while a read is pending.
6. Tauri desktop gains expandable library navigation with Prints, Favorites, Collections (expandable named rows), Source Media and Trash. Collection rows retain physical counts/availability, hidden state, creation/rename/visibility/delete actions and drag filing where supported by native macOS. Keep existing collection shelf, filters, history, selection, search, viewer and keyboard actions. Route state remains authoritative and works with a collapsed sidebar.

## Implementation and verification

- First add failing tests at the changed boundaries: member classification/extraction, batch bounds and authorization; source presentation/reuse and stale request rejection; privacy/host projection; collection navigation.
- Implement server/Studio contracts, native clients and shell integration in coordinated file ownership groups. Update changelog, READMEs, owning rules, canonical skill and website documentation for the delivered user/API behavior.
- Run targeted Rust, Studio/Vue and Swift tests, frontend architecture and builds, native builds and repository CI routing checks. Avoid implementation-mirroring tests or unrelated suite expansion.
- Use an isolated fixture server/library with authentic retained media to exercise detail inventory, thumbnails and individual reuse, next-owner navigation, unavailable hosts, hidden collections, Trash/restore and sibling retention. Do not consume production generations for UAT.
- Visually inspect web desktop/mobile layouts, the actual Tauri desktop and mobile surfaces, native macOS and iOS Simulator. Verify cards render, role labels distinguish ordered/end frames, actions stage original media, filters do not leak hidden owners, and collection expansion/actions work. Record commands and evidence; distinguish simulator proof from physical-device proof.
- Have a sub-agent review the completed changes. Address valid findings, push a conventional commit, open one PR, resolve CI/review findings and leave the green PR open for user review. Do not deploy.

## Review log

Independent native/client reviewer identified three blockers, accepted before implementation:

- Make the wire contract explicit: `POST /api/gallery/source-media/inventory` accepts at most 100 exact filenames and returns `{instance_id, inventories: {filename: Inventory}}`. Member additions are optional `label`, `media_kind` (`image`, `video`, `audio`, `mesh`, `unknown`), `position`, `frame_index` and `image_role`. The authenticated `/content` route preserves original payload bytes except for extracting the image from the existing keyframe JSON envelope. Raw download and full-recipe hydration remain unchanged. Unknown/binary members remain visible without guessed image actions.
- Batch fallback is permitted only for an unsupported route (404/405); auth failures, unavailable hosts, malformed responses and corruption must not become empty success. Display partial failure and retry explicitly. Reload when owner visibility, route credentials, instance, media revision or retained repair changes. No fixed scan cap may silently masquerade as the entire library.
- Individual attachment must be atomic and fenced against owner/host/credential/instance changes, destination model/recipe and attachment revisions, and component lifetime. Download first, verify the destination is still current, then apply through its existing capability-aware attachment path. Cancellation or stale completion leaves the user's newer edits intact and permits deliberate retry.

Server research independently confirmed keyframe preview failure and the need for the additive content route. Inventory must read one committed authority snapshot for a bounded batch and never decrypt payloads during listing. Implement these revisions before integration.

## Additional parity and regression scope

- LTX 2.5 duration: verify capability-qualified prediction versus explicit length on every surface. Native iOS and macOS must decode the additive model capability, expose prediction alongside explicit duration, omit frames only when prediction is selected and supported, and restore saved intent. Web must hide its exact-frame control while prediction is selected. Existing server prediction semantics remain authoritative.
- Native iOS unread badge: individual displayed media marks all merged physical copies viewed. Fix the delayed thumbnail/preload race using current selection and displayed identity; merely opening Library, preloading, or visiting Trash must never clear unread state. Test delayed selection and verify the navigation badge in Simulator.
- Visual polish: refine Tauri collection navigation hierarchy, spacing and actions, hide irrelevant output controls in Source Media, and provide review screenshots across all requested surfaces.

- Native Queue badges: use one shared waiting-work count for iOS and macOS, excluding running and terminal jobs. Confirm existing paused/held semantics and add a Mac badge if absent. Verify the count with mixed queue states and an isolated UI fixture.
