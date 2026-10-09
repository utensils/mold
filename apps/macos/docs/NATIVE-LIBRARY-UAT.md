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

## Library scrolling and media filters — 2026-10-01

Native Debug build with ad-hoc signing, disposable preferences/home, and a
read-only loopback fixture containing 120 mixed prints (45 favourites).

- All Media shows 120 prints. Videos and 3D each show 40; Videos within
  Favourites shows 15. Photos shows 40. Changing media type after scrolling
  deeply resets the new result list to the top, retaining the shelf.
- Pointer selections, including a partly clipped row, leave the viewport in
  place. Arrow navigation reveals later rows when the keyboard cursor moves
  outside the visible area.
- Opened Fixture 42 in the still viewer and returned with Library; its selected
  tile was revealed. Scrolled away, visited Queue, and reopened Library: the
  old viewer anchor was not replayed.
- Native build and architecture lint pass. The independent reviewer approved
  the final one-shot viewer restore and scope/query reset. Shared MoldClient
  tests pass (1,095 tests). No generation or production data mutation occurred.

## Host-scoped Trash — 2026-10-08

Native Debug build, disposable preferences/home and two loopback fixture hosts.
No generation or production media was used for mutation testing.

- A shared hidden-collection print merges into one tile. Filtering to Origin
  Fixture and moving it to Trash sends one request to that fixture; Other Copy
  remains live. The fixture action log records no request to Other Copy.
- The Trash sidebar selection originally failed on click and native table-row
  selection. Applying the list tag outside its context-menu owner fixes it;
  the rendered native regression passes and clicking Trash opens the grid.
- Trash retains the Origin Fixture filter and shows the deleted hidden member.
  Its toolbar offers Put Back, Delete Immediately and Empty Trash. Selecting
  Delete Immediately names only Origin Fixture; Cancel keeps the print.
- Put Back sends one restore request to Origin Fixture. Both fixtures end with
  one live print and no trash entries. Source/query and lifecycle regressions
  pass in 32 native tests across five affected suites; native lint passes.
- Shared HTTP tests cover trash video URLs, keyed tickets and relay fallback
  to authenticated temporary trash media, including cleanup at the caller.

The full native suite also exposes pre-existing error-message and reuse-name
expectation mismatches and a queue-layout harness missing its QueueStore. Those
unrelated failures are separate from the passing affected suites.
