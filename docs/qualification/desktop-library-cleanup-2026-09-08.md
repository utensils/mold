# 3-D Studio cleanup UAT — 2026-09-08

Implementation: `b5462fa1` (admission) and `49cb683e` (shared UI and queue restoration), on `fix/desktop-library-cleanup`. No PR or production deployment.

## Environment

Real inference ran on Plato GPU 3 (L40S), using an isolated server at loopback port 7689, a separate durable home/output directory, and the installed model files. The test binary was built from `9ec27747` plus the admission fix later committed as `b5462fa1`, with CUDA, cuDNN, preview, mesh-texture, mesh-matting, and mesh-delight enabled. Existing accepted licenses were copied into the isolated home. Production service and existing paused jobs were untouched.

Browser UAT exercised the actual desktop MeshWorkflowView, web MeshWorkflowPage, and mobile MobileApp through local Vite servers. Desktop native APIs were replaced by the browser harness; this is not a packaged Tauri or physical iPhone/Android qualification.

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
- Mouse resize, keyboard resize, saved preference updates, and double-click reset passed in the desktop browser harness. Actual OS preference persistence uses the existing app preference store and was not exercised through a packaged app.
- Telemetry refresh preserves draft text and focus; unchanged results are not fetched repeatedly.
- Web uses its existing Create picker, shared segmented mode control and switches, and Auto/Most capable routing constrained to a host supporting every stage.
- Web 390×844 layout has no horizontal overflow. Mobile connects to Plato, lists generated outputs, and opens the shared mesh viewer.
- Production inspection confirmed the two reported paused jobs were recovered ordinary generation jobs after the 11:30 service shutdown, not newly admitted workflows. Their single-job endpoints retained metadata that payload-free listings omit. Shared queue restoration now fetches that detail on desktop, web, and mobile.
- The PNG/JPEG rejection came from a one-byte future-image admission placeholder. The fix uses valid image bytes only in the validation clone and does not persist a fake source.

## Checks

`NODE_OPTIONS=--no-experimental-webstorage bun run check:frontend` passed: 1,725 Studio, 1,844 web, and 6,470 desktop/mobile tests, architecture checks, and desktop/web production builds. The mobile production build passed separately. The final history attachment-clear adjustment passed its five component tests. The Rust admission regression passed with Rust 1.93 and the system Clang linker.

Local screenshots, API receipts, downloaded GLBs, and harness scripts are retained under `/tmp/mold-library-cleanup/`. The isolated remote output is under `/home/jamesbrink/.local/share/mold-cleanup-uat/output`.

## Remaining observation

During the texture run, the queue preview continued reporting “Unwrapping mesh” while a debugger stack showed paint denoising on the GPU. It later advanced to upscaling and completed in about nine minutes. This qualification proves successful output, not accurate intermediate paint progress; the stale progress observation remains unresolved. The production paused jobs were not resumed as part of this test.
