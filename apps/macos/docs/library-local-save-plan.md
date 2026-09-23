# Native Library: local saving and host scoped actions

## Current behavior

The macOS app (`apps/macos`) already lists one row per host copy, supports Command-A and range selection, and accepts `on:machine` search tokens. “Save a Copy…” writes to a Finder destination; it does not add a print to This Mac’s Library. The embedded engine exposes the same authenticated gallery import API as other hosts. Tauri has “Save to this device’s Library” for one or many remote prints.

## Plan

1. Give the native Library an obvious Machine picker with All Machines, This Mac, and each connected remote host. It drives the existing machine token query, works in every shelf, and clears a grid selection when changed. Retain typed `on:` search support. The picker makes the host scope visible before a bulk action.
2. Add “Save to This Mac’s Library” to the shared Library action plan, tile menu, viewer, and menu bar. For a multi-selection, save only remote entries; leave existing local rows untouched. Show progress and a final count. Require the embedded local engine to be reachable, and clearly explain when it is not.
3. Download each supported remote picture through its authenticated backend and import into the embedded host’s gallery through `importPrint`. Add a mirrored-print import descriptor carrying original generation metadata, `metadata_synthetic`, timestamp, and bytes; keep Finder imports synthetic. Avoid writing a Finder copy and avoid replacing a different local print on filename collision. Refresh This Mac’s gallery after import. The current buffered media path has a size ceiling, so report a too-large picture as a failed save rather than silently truncating it.
4. Keep each host copy as a distinct row. With a remote machine filter active, Move to Trash and Delete Immediately act on selected remote rows only; a local copy remains in This Mac. Make the confirmation name the affected machine(s) and clarify the local copy remains. Never turn a remote deletion into a fleet wide delete.
5. Cover query and menu gating, duplicate/collision and import failure, bulk selection, and remote deletion preservation with focused tests. Run native lint/test/build and rendered UAT with isolated media and a test remote host; verify local bytes/metadata after import and again after deleting the remote row. Then independent final review, PR, exact head CI, merge, and local sync.

## Acceptance

- Selecting a remote machine visibly narrows the grid; Command-A selects its visible prints.
- Save to This Mac places the selected remote prints in the local Library with the same original media and recipe, separately from Save a Copy.
- Saving the same remote print again does not create duplicate local entries; same name with different content does not overwrite a local entry.
- Deleting remote rows after saving leaves local rows and bytes present.
- Offline local engine and partial batch failures are explained to the user.
