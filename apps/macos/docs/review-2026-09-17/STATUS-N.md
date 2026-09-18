# Lane N — visual symmetry and the error voice

Worktree `.claude/worktrees/lane-n`, branch `lane-n`, cut from `feat/macos-native-app`
(`be5a4907c`). Photographs in `uat/lane-n-*.png`; every one of them was read.

| id  | status | commit      | test |
| --- | ------ | ----------- | ---- |
| N#1 Library toolbar / column symmetry | fixed | `f0500e225` | photographed (see below) |
| N#1b Failure banner ✕ offset | fixed | `f0500e225` | photographed (`uat/lane-n-03-failure-banner-after.png`) |
| N#2 Inspector section alignment | fixed | `efcf07812` | `InspectorSectionLayoutTests` (3) |
| N#3 Error voice | fixed | `8b57a1508` | `FailureVoiceTests` (8+8+8+3), `HostFailureTests.aBannerEndsWithTheWayForwardWhenThereIsOne` |

## N#1 — which layout won, and why

Prototype (a), SwiftUI's own `.inspector`, with one correction the brief could
not have known. `.inspector` DOES carry the divider up through the window
toolbar — photographed at 2x in `uat/lane-n-01-library-divider-after.png` — but
**`.searchable(placement: .toolbar)` on the content does NOT keep the field over
the content region.** macOS pins that field to the window toolbar's trailing end
at a fixed width and no placement moves it, so on the first prototype it still
hung 14 points over the divider (the exact difference between the field's width
and the old 320-point column).

So the column follows the field: `TrailingColumn.width` is the search field's
width plus the toolbar's own edge inset, and the field then sits flush against
the column's leading edge. A pane with no search field (Generate) stretches the
inspector switch to reserve the same width, which is what stops its model and
recipe capsules at the divider instead of over the column.

- Library: `uat/lane-n-01-library-before.png` → `-after.png` (+ `-divider-after.png`)
- Generate: `uat/lane-n-02-generate-inspector-before.png` → `uat/lane-n-02-generate-toolbar-after.png`

Option (b) — a hand-built search bar in the content's own top bar — was NOT
taken: it means reimplementing `.searchable`'s token chips, suggestion list,
`.onSubmit(of: .search)` and `.searchFocused` (Edit ▸ Find) by hand, which is a
bigger change than this pass and would degrade the token search Lane-era work
just shipped. Recorded as a follow-up if the owner wants search over the
content rather than over the column.

The banner turned out to float for a different reason than it looked: its stack
shrank to its content whenever the content did not fill the pane, so the whole
banner — not just its ✕ — drifted to the middle. It is pinned under the toolbar
now, on the toolbar's trailing inset, with the ✕ on the first line's baseline.

## N#2 — the alignment rule

**A section's content starts where its TITLE does, and a line narrower than the
column leads like every other row.** `InspectorSection` owns both halves and
every group of the Generate inspector is one, Recent included (its Refresh is a
title accessory now, not a thing inside the content's leading edge).

Measured, not eyeballed: `InspectorSectionLayoutTests` renders each section
twice and takes the columns that gain ink, so nothing is asserted about a pixel
the test did not choose the colour of. One honest limitation — **the AppKit
disclosure row is not drawn by `ImageRenderer` at all**, so the title's own
inset cannot be measured in a test; it is read off
`uat/lane-n-02-generate-inspector-top-before.png` (14 points, chevron leading
edge to first glyph) and named as such in `InspectorSection.titleInset`. What
the test does pin is the half that regressed: where the CONTENT lands, and that
a narrow line does not float. Both fail against a plain leading `VStack`.

Lane I's `InspectorLayoutTests`, which the brief said to read first, does not
exist on this branch (`command grep -rln InspectorLayout Sources Tests` finds
nothing) — so the measuring approach above is new rather than copied.

## N#3 — the voice

`Error.advice` is the one place each route is worded, and `failureSentence`
joins reason + route for a surface that would otherwise be a dead end.
"Try again" is asked of `MoldClientError.isTransient` rather than decided a
second time.

Sentences changed: **13** — five routes newly worded (`unreachable`,
`unauthorized` moved out of the reason clause, transient `http`,
`malformedResponse`, `licenseRequired`), two canvas headlines ("That didn't
arrive/finish" → "This render didn't arrive/finish", the first now carrying
Show in Library), four call sites moved from `reasonSentence` to
`failureSentence` (result fetch, clip open, clip playback, mesh transport), and
two `MeshViewFailure` lines that stopped at the reason (`.upload`,
`.unreadable`) given the same "so here's the poster" / "it is still in the
Library" route the others already had.

Deliberately NOT changed: `HostStore+Reachability.check`'s `.down(...)`, which
is a compact STATUS line in the sidebar. Telling a machine row to "check the
machine under Machines" while you are looking at it is noise; that is stated in
`Error+Sentence`.

## Cross-lane edits

Smallest possible, all listed for sequencing:

- `Generate/GeneratePane.swift` (Lane L) — `.trailingColumn` moved ahead of
  `.toolbar { toolbar }` and takes a `Binding`. Required: a later `.toolbar`
  renders BEFORE an earlier one, so the column's switch has to be declared
  after the pane's own items to land last in the row.
- `Generate/GeneratePane+Toolbar.swift` — its inspector `ToolbarItem` removed
  (it is inside `trailingColumn` now, which is what knows where the column is).
- `Generate/GenerateController+Run.swift` (Lane L) — one line,
  `.failed(error.sentence)` → `.failed(error.failureSentence)`. `sentence` is
  the full `LocalizedError` description, which repeats the canvas's own
  headline and carries no route.
- `Generate/RunCanvas.swift`, `RunCanvas+Result.swift`, `RunCanvas+Clip.swift`,
  `Mesh/MeshViewFailure.swift` — failure wording only, no behaviour.
- `Packages/MoldStyle/.../Chrome.swift` — one new token,
  `Chrome.toolbarEdgeInset`.

## Not swept, and why

These `reasonSentence` producers are other lanes' files and were left alone;
each would be a one-word change to `failureSentence` once those lanes land:
`SourceImageWell+Import`, `ReferenceStrip+Import`, `ControlPictureWell`,
`IdentityGroup+Import`, `MediaWell`, `KeyframeTable` (Lane M's wells);
`GenerateController+Expand` (Lane L); `UpscaleStore+Jobs`, `ConfigStore`.

## Nothing judged wrong

The three items were all reproduced on screen before being fixed. The only
correction to the brief is the `.searchable` finding under N#1 above.

## Gates

`make lint` green (the size advisories are the branch's existing ones; no file
I touched grew past what it was). `Packages/MoldClient` `swift test`: 920 tests
in 50 suites passed. Full app bundle suite under the shared lock: **740 tests in
107 suites passed**.

## UAT housekeeping

Driven from a throwaway prefs domain (`io.utensils.mold.native.fresh`) and
`MOLD_HOME=/Volumes/ExternalStorage/mold-uat-home-lane-n`, seeded with
`MOLD_NATIVE_HOSTS=hal9000=…,offline=192.0.2.1`. Build went to
`/Volumes/ExternalStorage/xcb-lane-n`, never the main checkout's `build/`. No
renders were started, so nothing was written to hal9000's library and nothing
on it needs undoing.
