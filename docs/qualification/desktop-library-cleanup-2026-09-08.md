# 3-D Studio cleanup UAT — 2026-09-08

Implementation: `b5462fa1` (admission) and `49cb683e` (shared UI and queue restoration), on `fix/desktop-library-cleanup`. No PR or production deployment.

## Environment

Real inference ran on Plato GPU 3 (L40S), using an isolated server at loopback port 7689, a separate durable home/output directory, and the installed model files. The test binary was built from `9ec27747` plus the admission fix later committed as `b5462fa1`, with CUDA, cuDNN, preview, mesh-texture, mesh-matting, and mesh-delight enabled. Existing accepted licenses were copied into the isolated home. Production service and existing paused jobs were untouched.

Browser UAT exercised the actual desktop MeshWorkflowView, web MeshWorkflowPage, and mobile MobileApp through local Vite servers. The initial desktop browser harness replaced native APIs. Follow-up UAT used the actual macOS development binary in a separate test bundle with WKWebView and native IPC. Physical iPhone/Android hardware and release installer delivery are not part of this qualification.

## Real workflows

| Surface / operation | Durable workflow ID | Result |
| --- | --- | --- |
| Desktop From words | `81afaf06-7afd-4fa5-9d51-2b93ce366242` | Z-Image Turbo image → Hunyuan3D mini turbo geometry; completed, 7,166,532-byte GLB |
| Desktop Add texture | `bc1259bc-32ed-4a18-b27f-ccdec1875967` | Generated mesh and PNG → matting → delight → paint → finalize; completed, 12,197,164-byte GLB with two embedded textures |
| Web Rebuild / Most capable | `424aa4e7-e747-485a-9005-45bec784464d` | Hunyuan3D 2.1 Shape VAE round trip; completed, 7,076,676-byte GLB |

SHA-256 of downloaded outputs:

- Geometry: `bb19628968388e9e93d74d1ffef11bc1a15ce10bc44514289fa3092f5c184c50`
- Texture: `c241414968dfc043c45b8b3b8a62d38c3202d513fc37cefb3f9d555cacf2d1a8`
- Rebuild: `9d2906ed556ebd8a47ceaa7f81478ffcdc1c4f5bd486e7b100d4515051090454`

Desktop rendered the textured fox with 304,062 triangles. Web and mobile rendered the rebuilt fox with 299,638 triangles. No JavaScript page errors occurred. Completed geometry history survived an isolated-server restart.

## Interaction and regression checks

- Desktop style pickers are the same components as Generate, filtered by workflow mode and image/mesh capability.
- Mouse resize, keyboard resize, saved preference updates, and double-click reset passed in the desktop browser harness. Follow-up native keyboard resizing changed the saved preference from 480 to 460 pixels and persisted it to the native settings file.
- Telemetry refresh preserves draft text and focus; unchanged results are not fetched repeatedly.
- Web uses its existing Create picker, shared segmented mode control and switches, and Auto/Most capable routing constrained to a host supporting every stage.
- Web 390×844 layout has no horizontal overflow. Mobile connects to Plato, lists generated outputs, and opens the shared mesh viewer.
- Production inspection confirmed the two reported paused jobs were recovered ordinary generation jobs after the 11:30 service shutdown, not newly admitted workflows. Their single-job endpoints retained metadata that payload-free listings omit. Shared queue restoration now fetches that detail on desktop, web, and mobile.
- The PNG/JPEG rejection came from a one-byte future-image admission placeholder. The fix uses valid image bytes only in the validation clone and does not persist a fake source.

## Checks

`NODE_OPTIONS=--no-experimental-webstorage bun run check:frontend` passed: 1,725 Studio, 1,844 web, and 6,470 desktop/mobile tests, architecture checks, and desktop/web production builds. The mobile production build passed separately. The final history attachment-clear adjustment passed its five component tests. The Rust admission regression passed with Rust 1.93 and the system Clang linker.

Local screenshots, API receipts, downloaded GLBs, and harness scripts are retained under `/tmp/mold-library-cleanup/`. The isolated remote output is under `/home/jamesbrink/.local/share/mold-cleanup-uat/output`.

## Native follow-up and progress verification

The actual NYC LTX video `mold-ltx-2.5-22b-dev-int8-conv-1788888568154.mp4` was opened in the native Library. Its playback context menu offered Reuse settings; that action restored the original `mold-z-image-turbo-q8-1788888016740.png` source without the missing-media warning. The persistent Sound off control synchronized with the native player and remained muted in Generate. Inspector actions fit inside the panel. The native mesh viewer rendered the first textured fox successfully.

A fourth workflow was submitted through the native file pickers and Generate button to the isolated Plato endpoint: `d53b958c-6a41-4220-8b2c-6a03143f8346`. It completed with a 12,482,632-byte textured GLB, SHA-256 `c6f7d7ba2e365e1c0e7590f98bf82ecfb9e809c9e3f8382ff6b9f3b86adc5f3f`.

The first run had an apparent stale “Unwrapping mesh” observation while a later debugger stack showed denoising. To resolve the uncertainty, the fourth run used temporary event logging in the isolated build only. Unwrapping completed in 88 seconds; the API and native queue then reported “Generating PBR views” with steps 1–15, followed by decode, upscale, bake, fill, and GLB publication. Denoising took 371 seconds. The earlier discrepancy was not reproduced; no paint-progress source change was needed. Test logging is not included in this branch. Production paused jobs were not resumed by this UAT.

Native development reload exposed a separate inventory timing issue: 3-D Studio mounted before host readiness did not refresh its model inventory later. The ready-authority watcher now refreshes only on connection/authority changes, not telemetry; cached styles remain available while reconnecting. Regression tests cover both behaviors.

## Independent review

Claude Sonnet reviewed the full branch and found no critical/high issues. Its valid redundant localStorage-write finding was fixed in `b0e0f508`, with regression coverage. Queue-detail error handling and workflow-owner cancel/resume tests were also added. The appearance picker deliberately accepts PNG/JPEG, matching server admission. Claude's follow-up review found no actionable issues in that fix.
