# Qwen Image 2.1 — mobile UAT in the iOS Simulator (2026-09-27)

Issue [#1769](https://github.com/utensils/mold/issues/1769) asks for the Qwen
Image 2.1 reference and transparency features from PR #1767 to be exercised
on real iPhone and Android devices. This record covers everything that can be
proven without them: the shared mobile Vue layer running inside the real iOS
app's WKWebView, and the native Tauri bridge writing to a real Photos library.
The device-only items are listed at the end and stay open on the issue.

## Setup

- App: `scripts/ios.sh simulator` debug build of `feat/qwen21-macos-mobile`
  (the Tauri crate in `apps/mobile/src-tauri`), installed on an iPhone 16 Pro
  simulator (iOS 26.5) on an Apple M4 Max.
- Host: the `workstation` CUDA server running `main` at `5b61d1757`, keyless,
  with every Qwen Image 2.1 tier installed. All renders ran there; nothing ran
  on the Mac's GPU.
- Driving: the WebView was reached through the simulator's Web Inspector
  socket (`ios-webkit-debug-proxy`), which evaluates JavaScript in the running
  app. Controls were pressed through their `data-test` elements, and native
  commands through `window.__TAURI_INTERNALS__.invoke`. Screenshots came from
  `simctl`; Photos assets were read from the simulator's
  `data/Media/DCIM/100APPLE`.

## Results

| Checklist item                                                | Result                  | Evidence                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Alpha survives upload with no flattening, order kept          | Pass                    | An RGBA PNG, an RGBA WebP and an opaque JPEG were fed to the reference input (it advertises `image/png,image/jpeg,image/webp` for the recipe). A `turbo:q8` render with the toggle **off** and the RGBA WebP as reference 2 came back with `has_alpha: true` and no `transparent_background`. The alpha could only have come from the reference's original bytes. |
| Reorder and remove; canvas follows the last reference         | Pass                    | Tiles numbered 1–3, **SETS CANVAS** on the last. Canvas 1248 × 832 with the 768 × 512 JPEG last (upstream's 1024² area, ties-to-even on the 32 px grid). Moving the square WebP last gave 1024 × 1024, and removing tile 1 kept 1024 × 1024 with 2 tiles.                                                                                                         |
| Transparent background on/off; JPEG → PNG; field only when on | Pass                    | With JPEG selected, turning the toggle on moved the format to PNG. The render's metadata recorded `transparent_background: true`. The render with the toggle off carried no field.                                                                                                                                                                                |
| Gallery checkerboard (list and `MobileGalleryViewer`)         | Pass                    | The Library grid draws the bed behind the transparent PNG, the transparent WebP and an older transparent print, and the plain bed behind opaque prints. The viewer sizes the bed to the picture (512² and 1024² prints checked).                                                                                                                                  |
| Photos auto-save keeps alpha                                  | Pass (fixed in this PR) | A phone render with the toggle on auto-saved `IMG_0015.PNG`: RGBA, byte-identical to the host's file.                                                                                                                                                                                                                                                             |
| Reuse restores the toggle and references                      | Pass (toggle)           | The toggle was switched off, then **Reuse settings** on the transparent print turned it back on. The prompt, style and PNG format were restored too. Restoring references is covered by the shared `applyMetadataToForm` tests.                                                                                                                                   |
| Licence prompt before a Qwen Image 2.1 / turbo pull           | Not re-run              | The workstation had already accepted `qwen-research` and has every tier installed, so no pull could trigger it. Covered by the shared `runWithLicenseConsent` flow the Catalog and Create pulls use.                                                                                                                                                              |
| WebP still opens as an image, not a video                     | Pass                    | The WebP still opens in the viewer with the **IMAGE** badge and the checkerboard.                                                                                                                                                                                                                                                                                 |

### Bug found and fixed: WebP stills never reached Photos

Photos auto-save and Library multi-select save filtered filenames to PNG/JPEG,
so a WebP still was silently skipped. Both native bridges refused WebP, and the
viewer's **Save photo** failed with "not a PNG or JPEG image". iOS also saved
through `UIImageWriteToSavedPhotosAlbum(UIImage)`, which re-encodes the
picture, so a transparent print's alpha was not guaranteed.

After the fix, the iOS bridge hands the original bytes to PhotoKit as the
asset's photo resource (`PHAssetCreationRequest` +
`addResourceWithType:data:options:`). Both bridges accept a still WebP and
refuse an animated one (VP8X animation flag). In the simulator:

- `save_image_to_photos` with the transparent WebP print stored
  `IMG_0012.WEBP`, byte-identical to the host file with alpha.
- The transparent PNG stored `IMG_0013.PNG`, also byte-identical.
- The viewer's **Save photo** on the WebP reported "Sent to Photos" and stored
  `IMG_0014.WEBP`, byte-identical.

`xcrun simctl addmedia` refuses WebP ("File type unsupported"), but that is a
limit of the tool only. PhotoKit itself stores a WebP resource.

## Still open on the issue (needs physical devices)

- Picking references through the native Photos picker on iPhone and Android.
- Android end to end: MediaStore save (covered by the new instrumented test
  `savesAStillWebpVerbatimSoItsAlphaSurvives`, which CI runs on an emulator),
  the Android picker, and the gallery checkerboard.
- A real iPhone's Photos app displaying the saved WebP and PNG with transparency.
