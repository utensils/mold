# Lane I — the inspector follows the open print

The owner's observation, from a screenshot: with a print OPEN in the Library
viewer, the right-hand inspector still read **"Nothing selected"**.

Round two, rebuilt on the Reuse lane's tip. Round one widened `OutputMetadata`
independently; **that commit is dropped and the Reuse lane's widening is the
one definition** — this lane reads it and adds exactly two things to it, below.

| id | status | commit | test |
| --- | --- | --- | --- |
| I1 the inspector follows the open print | fixed | `a1d4934` | `MoldClientTests/LibraryInspectionTests` (6), `MoldTests/LibraryInspectedTests` |
| I2 the Library menu acts on the open print | fixed | `a1d4934` | `MoldTests/LibraryInspectedTests` (source scan) |
| I3 `mesh_workflow` + the identity photograph's label | fixed | `d19f7b1` | `MoldClientTests/ProvenanceTests+Workflow` (3) |
| I4 the inspector shows every detail | fixed | `0be6cbf` | `MoldClientTests/PrintDetailsTests` (12) |
| I5 every row copyable, reuse borrowed not re-declared | fixed | `0be6cbf` | `PrintDetailsTests.everyRowOffersCopyingAndBorrowsUseTheseSettings` |
| I6 ⌥⌘I, Escape and ← → unchanged | verified, no change | — | (`MoldCommands` and `LibraryViewer` untouched) |
| I7 the label column fits its labels | fixed | `c42cd3a` | `LibraryInspectedTests.everyDetailLabelFitsTheColumnItIsDrawnIn` |

## I1/I2 — why it said "Nothing selected"

The inspector was handed `showing.selected`, the GRID's selection, and opening
a print never selects it:

- a tile's `.onTapGesture(count: 2)` opens; the single-tap selector does not
  fire for a double-click,
- a tile's right-click **Open** acts on the clicked print, which a right-click
  deliberately does not select,
- `step(_:)` moves the viewer and never touches the selection, so ← → left the
  inspector behind even when the first print had been selected.

`LibraryShowing.inspected(viewing:)` is the ONE rule, pure and in MoldClient:
**the open print while the viewer is showing one, the grid's selection
otherwise, and the selection again for a `viewing` the list no longer holds** —
the same moment `LibraryPane.content` falls back to the grid, so nothing goes
on describing a print that is not on screen.

**Decided against writing the selection from the viewer.** The selection is the
cursor a ⇧-click extends and ⌘A replaces; opening one print out of a careful
selection and then stepping would destroy it. The two are read together at one
seam instead. `menuSelection` and `menuFile` read the same derivation, which
was the second half of the same bug: Favourite, Move to Collection, File ▸
Export… and Move to Trash acted on whatever the grid still held.
`LibraryPane+Empty.subtitle` still reads `showing.selected` on purpose —
"3 selected" is a statement about the grid — and is the one name the scan
excludes.

## I3 — what this lane added to the landed widening, and why

The Reuse lane's `OutputMetadata` covers everything `PrintDetails` reads except
two things, both added THERE with the fixture-driven tests the others have:

1. **`meshWorkflow`** (`MeshWorkflowProvenance`, its own file so their
   `OutputMetadata.swift` stays small). Not decoded at all, so a stage of a
   text-to-3-D run was indistinguishable from a hand-authored picture — the
   confusion that block exists to end. Pinned against the capture's own stage
   row (`mold-qwen-image-q8-1789529980561.png`: `text_to_mesh` /
   `generated_image` / stage 0), and against an ordinary print carrying none.
2. **`identityPhotoNames`**, the names twin of their `identityDigests`, same
   singular/plural rule. The digests had a reader and the LABEL did not.
   Fixture-pinned on the two face prints; the plural precedence is synthetic
   for exactly the reason theirs is — nothing on hal9000 has ever produced
   `id_image_names`.

**Three rows from round one were dropped rather than added**, because no
captured print carries the field and a non-nil test would have to be invented:
`true_cfg`, `cfg_start_step` and `mesh.texture_resolution`. If a capture ever
carries them they are two lines each. In their place the inspector now shows
what the landed widening does carry and round one did not: CFG+, both distill
strengths, the LTX-2 pipeline, the extend overlap, and the keyframe count.

`editImageSha256S` needed no finding in round two — the Reuse lane hit the same
`convertFromSnakeCase` trap and documented it at the field. This lane reads
`editImageDigests` and `identityDigests` and never the stored names.

## I4/I5 — what it shows

`PrintDetails` (pure, MoldClient) returns nine groups in the order a person
reads them: **Prompt, Model, Settings, Clip, 3-D, Made from, Sequence /
Rendered in clips, 3-D workflow, File**. A group exists only when the print has
something to put in it, so a picture shows four headings and no em-dashes.
Drawn by `InspectorDetails` under the existing disclosure, now titled
**Details** with its `@AppStorage` key unchanged. `ProvenanceGrid` is deleted —
it was the eight hard-coded rows this replaces.

The tests read the same fourteen captured prints `ProvenanceTests` does, so the
mesh case is a real GLB, the sequence a real three-clip chain, the adapters a
real stack beside its legacy singular twin, and the face a real one.

Wording decisions worth a reviewer's eye:

- a **seed is unformatted digits** — one with separators in it cannot be pasted
  back, and the test asserts that against the print's own seed.
- **a sequence and an auto-chain read differently.** Both carry `chain`; only
  `output_mode` says which. "Sequence · Made of 3 clips, one prompt each"
  against "Rendered in clips · 3 clips". Crediting somebody with a split they
  never authored is the thing to avoid.
- **a sequence's prompt is shown whole**, newlines and all. This is provenance;
  `firstStagePrompt` is reuse's reduction and belongs to authoring.
- an **adapter is its file's name and its strength**, never the server-side
  path — the rest of that path is a stranger's directory layout.
- **"Rendered at" appears only when the file is not the canvas.**
- a **workflow stage is named in words** ("The picture it started from", "The
  finished mesh"); a role this build has never heard of is opened out, not
  dropped.

Every row's menu is `Copy <label>` plus, where the Library would let you reuse
the print, its own **Use These Settings** filtered out of
`LibraryMenuPlan.items` rather than declared a second time. Attached through
`.rowActionMenu`, the one door.

## Cross-lane edits

1. `Packages/MoldClient/Sources/MoldClient/OutputMetadata.swift` — one stored
   property, `meshWorkflow`, with its doc comment (I3).
2. `Sources/Mold/Library/LibraryPane.swift` — the trailing column calls
   `inspector(showing.inspected(viewing: viewing))`; the body of that call is
   in this lane's own `LibraryInspector+Pane.swift`, so the file is a line
   SHORTER (157 → 156).
3. `Sources/Mold/Library/LibraryPane+Menu.swift` — `menuSelection` and
   `menuFile` read `showing.inspected(viewing: viewing)` (2 lines + a comment).
4. `Sources/Mold/Library/LibraryInspector.swift` — the disclosure's title and
   its one child view (4 lines).
5. `apps/macos/README.md` — one sentence appended inside the **Library** row,
   before "Refreshes by ETag".

Everything else is new files, or this lane's own. `ProvenanceGrid.swift` is
deleted.

## Findings judged wrong

None. The one assumption that was not so is the coordinator's own and is
already acknowledged: `OutputMetadata` had not been widened when round one
started. Round one's widening commit is not carried here.

## Gates

Run last, on the settled tree. No helper agents.

- `make lint` — layers ok, colour ok, a11y ok. Every new file under 150 lines;
  `LibraryPane.swift` a line smaller than it landed.
- `cd Packages/MoldClient && swift test` — **913 tests, 49 suites, green.**
- app bundle `xcodebuild … test` under the shared lock — **694 tests, 100
  suites, green.**
