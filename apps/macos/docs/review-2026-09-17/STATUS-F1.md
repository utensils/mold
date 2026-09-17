# STATUS — Lane F1 · native mesh viewer + mesh exports

Branch `worktree-agent-ac7de2eea3506f1ea`, cut from `feat/macos-native-app` at `a5172d65`.

| item | status | commit | test |
| --- | --- | --- | --- |
| GLB reader, port of `studio/lib/glb.ts` incl. every bounds check | done | 9cf3e4e2 | `GLBSuite` (22) |
| `GLBFixture`, port of `glbFixture.ts` | done | 9cf3e4e2 | used by `GLBSuite`, `MeshViewerCameraSuite` |
| `MeshViewerCamera` + `MeshMatrix`, ports of `meshViewerCamera.ts` | done | 9cf3e4e2 | `MeshViewerCameraSuite` (29) |
| `MeshViewerMath`, port of `meshViewerMath.ts` | done | 9cf3e4e2 | `MeshViewerMathSuite` (12) |
| `capabilities.mesh` + `export_geometry` wire types | done | 9cf3e4e2 | `MeshExportSuite` |
| `splitMeshExportFormats` / `meshGeometryDefaults` / `takesGeometryOptions` / `meshExportRequest` | done | de3d36c8 | `MeshExportSuite` (28) |
| Rust contract: a Swift arm on `the_viewer_mirrors_the_poster_camera` | done | ea28591f | `the_viewer_mirrors_the_poster_camera`, `the_swift_parser_reads_what_it_claims_to` |
| `MeshInteraction` (drag/zoom/keys, no pan) | done | fdb6c7fe | `MeshInteractionSuite` (12) |
| `MeshViewMenuPlan` — one `RowAction` list for controls, tile menu, menu bar | done | fdb6c7fe | `MeshViewMenuSuite` (7) |
| `MeshView` / `MeshRenderer` / `MeshShaders.metal` — the interactive view | done | 5c37928d | `MeshViewerSuite` (8) |
| Plugged into `LibraryViewer` (mesh arm before the image arm) | done | 5c37928d | `MeshViewerSuite`, manual UAT below |
| Plugged into the Generate result canvas | done | 5c37928d | builds; UAT below |
| Arrow keys stand down for a focused mesh view | done | 5c37928d | `aFocusedMeshViewClaimsTheArrows` |
| Exports from the advertised list; real request body | done | 2bdc1cde | `MeshExportSuite`, `ExportOptions` tests |
| Quick Look on a mesh shows its poster | done | 2bdc1cde | UAT below (needs a real host) |
| Menus: `RowAction` on the view, the tile and the menu bar | done | 2bdc1cde + follow-up | `MeshViewMenuSuite`, `LibraryMenuPlanTests` |

## Gates

- `make lint` — green. `lint-size` (advisory) still names `LibraryViewer.swift` (159)
  and `PrintMaterializer.swift` (151); both were already 147/150 before this lane and
  each grew by the smallest change that could be made (an arrow-claim watcher; a
  `named:` parameter).
- `cd Packages/MoldClient && swift test` — 750 tests, 34 suites, green.
- app-bundle `xcodebuild test` under the shared lock — 566 tests, 85 suites, green.
- `cargo test -p mold-ai-inference --lib the_viewer_mirrors` — green (the one cargo run
  the brief allowed), plus the two parser tests in the same warm binary.

## Cross-lane edits

Each is the smallest line that could be written; none reorganises the file.

- `Sources/Mold/Generate/ArrowKeyClaim.swift` — `MeshMetalView` added to `claims`.
  This IS the existing responder-chain gate the brief named; no new mechanism.
- `Sources/Mold/Library/LibraryActions.swift` — one optional `meshExport` closure,
  set by whoever owns the export sheet.
- `Sources/Mold/Library/LibraryActions+Menu.swift` — the `.exportTurntable` arm (the
  switch is exhaustive, so it had to be edited) and `.export` routed through
  `requestExport`.
- `Sources/Mold/Library/LibraryActions+Files.swift` — `quickLook` routes a mesh to
  its poster.
- `Sources/Mold/Library/LibraryPane.swift` / `LibraryPane+Menu.swift` /
  `LibraryMenu.swift` — mount the one export sheet; feed `meshExports` to the two
  plan sites and to the File menu.
- `Sources/Mold/Library/PrintMaterializer.swift` — additive `named:` parameter with a
  default, so no existing call site changed.
- `Sources/Mold/Shell/LibrarySelection.swift`, `MoldCommands.swift`,
  `MoldCommands+FocusedValues.swift` — carry the mesh split to the menu bar so it and
  the tile menu offer the same things.
- `Packages/MoldClient/.../MoldBackend+Gallery.swift` — two additive requirements
  (`export(_:request:)`, `thumbnail(_:size:trashed:)`).
- `Tests/MoldTests/FakeBackend.swift` — those two witnesses, plus `exportRequests`
  so a test can assert what a mesh export ASKED for.
- `Tests/MoldTests/LibrarySelectionTests.swift` — one new argument.

## Judgement calls a reviewer should check

- **"Unknown names skipped."** The brief says the export list skips names this build
  does not know. The reference (`studio/lib/meshExport.ts:59-64`) is deliberately
  PERMISSIVE — an unknown container becomes a one-click transcode, because the app
  saves bytes to a file and needs no per-format knowledge. I kept the reference's
  behaviour: the SERVER already drops every name it does not know
  (`known_mesh_export_formats`, `crates/mold-core/src/types.rs:11999-12009`), so the
  skipping happens before this client ever sees one. Pinned by
  `keepsAContainerThisClientHasNeverHeardOfAsADirectTranscode`.
- **`ExportOptions.forMesh` is deleted, not fixed.** Its `{obj,stl,ply,zip}` set was
  the client constant the brief said to replace, and `capabilities.mesh` is the
  authority. A host with no `mesh` block now offers no mesh exports — which is
  correct: absence there means no mesh family on that host, not an older server.
- **The Generate result canvas offers NO Export.** Its menu would otherwise promise a
  sheet that pane does not mount. Save and Show in Library are there, and the Library
  is one click away. Say if you would rather it mounted the sheet too.
- **Metal's clip range.** `MeshMatrix.orthographic` is the reference's GL projection
  (near → -1), pinned through the TypeScript by the Rust test, so the renderer remaps
  z to Metal's `[0, 1]` at the uniform boundary (`MeshRenderer+Draw.metalDepth`)
  rather than changing the shared matrix.
- **Polygon offset.** GL's `polygonOffset(1, 1)` has no exact Metal twin; the fill
  pass uses `setDepthBias(1e-4, slopeScale: 1, clamp: 0)`. The slope term is the
  same; the constant term is in depth units rather than GL's smallest resolvable
  difference. Worth a look on a real mesh (below).

## UAT — what only a human can check

I cannot drive the UI. On a host with a mesh print (plato holds several):

1. **Home view == the tile.** Open a mesh in the Library. Its first frame must be
   pixel-for-pixel the gallery tile's poster — same angle, same size in frame. This
   is the whole parity claim; if it is off, `sweep_fit_for` and the viewer disagree.
2. **Orbit.** Drag right: the object must turn the way a turntable GIF of it spins.
   Drag up and down: it must stop just short of looking straight down/up, never flip.
3. **Zoom.** Scroll and pinch both ways; it must stop at about a quarter size and
   about six times, and scrolling down then back up must land exactly where it began.
4. **Keys.** Arrows turn it (Shift turns further), `+`/`-` zoom, `0` and a
   double-click go home. While the mesh has focus, ← and → must NOT step to the next
   print — click the grid first, then they should again.
5. **Auto-rotate.** It should start turning by itself and stop for good the moment you
   drag, press a key or scroll. Turn on System Settings ▸ Accessibility ▸ Display ▸
   Reduce Motion and reopen: it must not turn at all.
6. **Wireframe.** The control must outline the mesh without the lines shimmering
   through the surface (that is the depth-bias note above). On a mesh with no edges
   the control should be ABSENT, not greyed.
7. **A textured mesh.** One with a baked texture must show it, right way up — a
   flipped V would show the texture upside down.
8. **Quick Look.** Space on a mesh tile must show its poster, not a generic icon.
9. **An STL export at 100 mm, Z-up, floor.** Export ▸ STL, keep the defaults, then
   open it in a slicer: the longest side must measure 100 mm, it must stand upright
   rather than lie on its side, and its base must sit on the plate.
10. **A turntable.** Export ▸ Turntable…, 36 frames at 10 fps: a 3.6 s spin whose
    first frame is the same picture as the tile.
11. **Failure lands on the poster.** Disconnect the host mid-open: the poster must
    stay with one sentence under it, never a black rectangle or a spinner.

## Deferred

- Fullscreen. The reference gates it behind a prop and only the Create result areas
  pass it; on this Mac the viewer already fills the pane and the window has its own
  full-screen button, so I did not add a second one. Say if you want it.
- A mesh's bounding box does not reach the export sheet's size sentence from the TILE
  menu (it does not know one yet), so it reads "longest side 100 mm" there and names
  all three extents only where a viewer has reported a box. The wire request is
  identical either way.
