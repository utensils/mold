# Native authoring and media parity — local acceptance record

Local session: 2026-10-06, macOS 26.6.2, iOS 26.5 Simulator. Changes were planned, reviewed adversarially, implemented, reviewed independently, and corrected before publication. No CI workflow or UAT job was added.

## Scope and decisions

- Keep iOS Generate exclusively for authoring, with one prompt expansion action, stable attachment hierarchy and pinned submission controls. Queue owns live render previews; Library owns results.
- Open Mac queue rows into job details. Default Discover to manifest models with an explicit community-catalog route and visible pagination.
- Normalize new GUI still inputs proportionally with orientation/alpha preserved, accurate MIME/name/dimensions/digest, and bounded transport. Preserve raw exports and retained conditioning authority; replace stale masks when their source changes.
- Make Library source selection available from the iOS grid/viewer and searchable across hosts; route into the chosen recipe's source, reference, boundary-frame or named-camera contract.
- Keep completion notifications, disable generic persistent Live Activities, and show foreground queue counts.
- Default fresh generation to Random while retaining deliberately fixed zero. Display unpinned queue placeholder seeds as Random.
- Prefer proven LAN routes over Tailscale/relay, coalesce refreshes and fence stale responses. Avoid speculative infrastructure changes or widening the embedded engine's loopback listener.
- Add native autoplay/repeat preferences and Mac GIF controls. Reduce iOS 3-D memory pressure with file downloads, selected-page loading and retryable errors.

## Visual checks performed

Used real native builds against two loopback HTTP fixture machines, UAT Coast and UAT Ridge. Fixtures returned real image/video/GLB bytes and generation profiles, while generation submission was captured and refused deliberately so no GPU work ran.

| Surface               | Observed result                                                                                                                                                                                                                                                                                                             |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mac Queue             | Clicking a running row opened Job Details with source, prompt, Random seed and a changing live step/preview.                                                                                                                                                                                                                |
| Mac Discover          | Featured manifest models displayed on entry; Browse Community Catalog showed 20/40 results and a prominent Load More Models action; clicking it added the second page.                                                                                                                                                      |
| iPhone Generate       | Stable prompt/model/source form and queue link; no running/result canvas. Generate remained available for another request. Captured fresh request omitted seed.                                                                                                                                                             |
| iPhone source picker  | All Machines showed Coast and Ridge prints; Ridge filter and search narrowed to one picture. While the picker remained open, a fixture job changed from running to queued and the underlying queue count updated from 2 rendering/4 waiting to 0 rendering/6 waiting. Filter/search survived; selected picture attached.    |
| iPhone Library/viewer | More offered Use as Source and returned to Generate with the picture attached.                                                                                                                                                                                                                                              |
| Native settings       | Both apps exposed Play videos automatically and Repeat videos; initial values were on/off. Repeat was enabled in disposable profiles. iOS completion/failure notification preferences remained present.                                                                                                                     |
| Native video          | Both apps displayed the MP4 fixture. Mac used inline controls without a full-frame hover scrim; audio/mute controls remained available.                                                                                                                                                                                     |
| Mac GIF               | Video Export opened Loop/Forever/pause/size/FPS controls. Saved `/tmp/mold-uat-loop.gif` was a valid GIF. Captured request contained `playback: loop`, `repeat: forever`, `pause_ms: 500`, `fps: 12`, `max_dimension: 720`. The fixture supplied encoded output; this verifies the client workflow, not the server encoder. |
| Native 3-D            | Both apps opened a GLB in their interactive renderer; iPhone also displayed an embedded-texture GLB using the file-backed loading path.                                                                                                                                                                                     |

## Focused validation

- Native Mac and iOS builds and architecture/accessibility lints passed.
- Mac: 39 tests across discovery, queue-detail identity and GIF/mesh export routing passed. Initial test-host launch failed while the disposable app was open; closing it and rerunning succeeded.
- iOS: 21 GenerateController tests passed on iPad Simulator, including typed image/named-camera source attachment. The existing rendered Generate test was updated to inject its new QueueStore dependency.
- Shared Swift: 18 focused image sizing, source/mask, random/fixed seed, route preference, refresh-coalescing and file-size tests passed; 16 relay transport tests passed, including file-backed staged-object download and no credential forwarding to object storage.
- Studio/web: 55 focused shared tests and 12 web drop tests passed; web, desktop and mobile frontend builds/typechecks passed.
- Desktop Rust image helper: exact production helper/import logic compiled in a narrow harness; unchanged-small-input and oversized-transparent-input cases passed. Full Tauri engine suite was not run.
- Chain seed contracts: a narrow Rust harness using the exact production manifest constructor and seed-materialization method reproduced the missing persisted seed before the fix, then passed random-resolution/stage-offset and explicit-zero cases. Core/server/orchestrator regression tests were added; full package/engine suites were not run.
- Android production image helper compiled with the existing toolchain. Orientation/alpha instrumentation cases were added to the existing suite but not executed because no Android emulator was connected.

## Limits and production follow-up

This is local simulator/native-app evidence, not physical-device or production network qualification. The reported failing production 3-D filename/host was not identified during this session. No claim is made that the specific incident is reproduced or that cellular relay throughput is measured. Relay protocol tests preserve status/range headers and object-origin/credential boundaries; they do not measure cloud cost or sustained throughput. Read-only cloud inspection found the relay frontend active and no matching media-staging failure log in the inspected interval. No cloud deployment or Terraform change was justified or performed.

Native routing requires advertised, reachable endpoints and valid pairing proofs. The Mac embedded engine remains loopback-only; this work changes paired clients' route preference, not server network exposure.

Desktop native imports retain 256 MiB encoded / 128 MiB decoded resource limits; automatic fitting is not unlimited image acceptance. Browser decode memory limits and supported source formats remain platform-dependent. Retained source/mask restoration and original exports bypass new-input fitting intentionally.
