# Library deletion navigation plan

## Surface inventory

| Surface | Owner | Removal behavior to repair |
| --- | --- | --- |
| Native macOS | LibraryPane, LibraryStore.runBulk | Missing viewed media shows grid; successful trash leaves a completion banner |
| Native iOS / iPadOS | PrintViewer, LibraryGridProjection | Changed projection dismisses missing current media |
| Web | LibraryPage | Permanent delete closes; reactive URL hydration can clear optimistic next selection |
| Tauri desktop (macOS / Windows / Linux) | LibraryView | Single trash advances, but pruneSelection closes after permanent/bulk removal |
| Tauri mobile (iOS / Android) | MobileApp, MobileGalleryViewer | Permanent delete explicitly dismisses its viewer |

## Intended behavior

1. Capture the previous visible filtered order and logical copy identities.
2. Keep the current logical media if any displayed copy survives, including partial failures.
3. Otherwise choose its next surviving neighbor; at the end choose the previous survivor. Close only when none remain.
4. Preserve existing scope/filter navigation, confirmation, undo, host targeting, unread display boundaries and manual close behavior.
5. Update web print deep links when the selected media changes.
6. Hide only the native Mac successful trash completion notice. Retain ongoing progress, Stop, and interrupted/failed/uncertain result messages.

## Delivery and verification

- Add regression tests for neighbor order, end/empty lists, surviving renamed mirrors and native Mac feedback.
- Exercise web and desktop component deletion paths, build all frontend bundles, and run native unit/package checks.
- Update the existing native iOS Simulator viewer-deletion UI test and inspect its screenshot evidence.
- Have an independent subagent review the implementation; resolve findings and obtain a final review.
- Open a conventional fix PR, verify exact-head CI and merge when checks pass.
