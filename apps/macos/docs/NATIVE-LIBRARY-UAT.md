# Native library acceptance checks

2026-09-30, macOS native Debug build, disposable preferences and home, and a
loopback fixture on port 18769. No live generation or production gallery data
was used.

- Generate: the controls sit at the bottom of a tall maximized window. The
  rendered layout regression also checks 300-point and 900-point panels.
- New collection: select eight prints, open Move to Collection → New Collection,
  create UAT Selection, and verify eight members. The fixture recorded one bulk
  mutation containing all eight filenames and the new collection name.
- Hidden collection: hiding Drafts changes All Prints from eight to six. Opening
  hidden Drafts directly shows its two members. Show in All Prints restores eight.
- Menu lifetime: a fixture delays each media copy by ten seconds. Open the
  eight-print context menu during Copying 2 of 8, leave it open through 6 of 8
  and 8 of 8, then open Move to Collection. The native menu and submenu keep
  their accessibility identities through completion. Escape closes the menu
  and clears the completed footer. A subsequent right-click with no selection
  offers Quick Look for the clicked print, rather than the former eight prints.

Focused native tests cover bottom alignment, captured collection targets,
failed collection creation, nested menu tracking, lazy pure menu providers,
pending owner inputs, footer removal, and deferred local-save reports. Shared
query tests cover merged remote/local hidden membership and Recently Deleted.
The UI stress test uses copy progress; local-save completion report deferral is
covered by the native contract tests rather than a running local engine.
