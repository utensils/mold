# Lane K — sweep the word "plato" out of the code

The owner's rule: a private machine's hostname must not appear anywhere in
code -- not in code comments, not as a test-host label, not in fixture
filenames or fixture contents, not in test names. `workstation` is the
neutral replacement (`Workstation` where capitalised; `workstation.local`
was already the UI's example address in `HostEditor.swift` and needed no
change).

| id | status | commit | test |
| --- | --- | --- | --- |
| K1 Red gate: a source-scan test that fails on the retired hostname | fixed | `cea8ba44` | `PrivateHostnameTests` |
| K2 Fixture files: `git mv *-plato.json` -> `*-workstation.json` | fixed | `cea8ba44` | (renamed alongside K1; filenames only, bodies untouched at this commit) |
| K3 Fixture bodies (`"hostname":"plato"`) + the one assertion the rename broke | fixed | `6fd8c04e` | `ServerStatusTests`, `ResourceTests`, `QueueTransferIDTests` |
| K4 Comments, test doubles (`let plato = machine(...)`, `onPlato`, `fakePlato`), PLAN.md and the STATUS ledgers | fixed | `c8dea487` | whole `MoldTests` + `MoldClientTests` suites |
| K5 Five pre-existing files outside `apps/macos` | fixed | `7b3d653c` | `UpscaleDialog`, `ToastShelf`, `ErrorNotice`, `useOpenLiveWork`, `meshWorkflowDraft` (vitest) |

## Hit count

- Before: `command grep -rIli plato apps/macos <5 external files>` — 142 files
  (140 after excluding the two archived review reports, +5 external = 145
  files actually swept).
- After: 0, everywhere in scope.

## The one assertion I had to change beyond the rename

`QueueTransferIDTests.aTransferIdMatchesTheOneTheWebAppDerives` hard-codes a
UUID computed from `SHA-256(JSON.stringify(["mold.queue-transfer.v1",
"plato-instance-0001", "job-abc-123", "hal9000-instance-0002"]))` (the
studio's `queueTransferId` derivation, ported byte-for-byte). Renaming the
source-instance literal to `"workstation-instance-0001"` changes the digest,
so I recomputed it with Node's `crypto` exactly as the test's own doc comment
prescribes (`db83f081-c42a-5cad-a003-7aa89b86b5c4` ->
`3e1e45ce-5235-5047-b5a8-366cc6b94353`) and updated both the doc comment and
the `#expect`.

No alphabetical-ordering or fixed-width-layout assertions depend on the
string `"plato"` vs `"workstation"` (`h < p` and `h < w` both hold where
`hal9000` is compared against the second host; no test asserts a specific
string length or column width against either name).

## Excluded from the sweep (by design)

- `apps/macos/docs/review-2026-09-17/03-library.md` and
  `04-queue-models-machines.md` -- archived audit text, per the brief's `01-06`
  carve-out. Both still say `plato`.
- The provenance IP `100.105.134.43` is left untouched everywhere it appears
  in a comment (it names a real address, not the retired word).

## Cross-lane / out-of-scope note

The word also appears in roughly two dozen `studio/**` files that were never
part of this lane's scope (only the five named files were): e.g.
`studio/stores/notifications.test.ts`, `studio/stores/meshWorkflowDraft.ts`,
`studio/components/NotificationsCenter.test.ts`,
`studio/components/QueueEntryDetail.test.ts`,
`studio/components/LicenseSettingsPanel.test.ts`,
`studio/components/MeshWorkflowStudio.test.ts`,
`studio/lib/hostConnectivity.test.ts`, `studio/lib/machineSentence.test.ts`,
`studio/lib/errors.test.ts`, `studio/lib/retireSequenceStorage.test.ts`,
`studio/lib/profileFleet.test.ts`, `studio/lib/modelAvailability.test.ts`,
`studio/lib/hostRouting.test.ts`, `studio/lib/expansionRouting.test.ts`,
`studio/lib/thumbnailPersistentCache.test.ts`, `studio/lib/modelSearch.test.ts`,
`studio/lib/queueEntryDetail.test.ts`, `studio/lib/libraryOrganization.test.ts`,
`studio/lib/notificationClipboard.test.ts`,
`studio/api/galleryOrganization.test.ts`, `studio/api/queueTransfer.test.ts`,
`studio/api/generationPlacement.test.ts`, `studio/api/config.test.ts`. Left
untouched, per file ownership -- flagging for the integrator in case the
owner wants a follow-up lane against the whole repo rather than just
`apps/macos` plus the five named files.

## Gates

- `make lint`: green (pre-existing large-file warnings only, unrelated to
  this sweep -- `HTTPBackend`, `LibraryStore`, `GenerateController`,
  `RenderDraft`, and four files at/near the 150-line cap).
- `cd apps/macos/Packages/MoldClient && swift test`: 913/913 passed
  (`QueueTransferIDTests` included -- confirms K3's recomputed digest).
- App-bundle `PrivateHostnameTests` itself: **not proven red.** I started an
  `xcodebuild … -only-testing:MoldTests/PrivateHostnameTests` run under the
  lock right after committing K1 (test present, sweep not yet applied), but
  it was still building with no output when the integrator asked me to stop
  waiting and hand the app suite to them; I killed it (exit 144) and released
  the lock rather than start a second one. The integrator is running the full
  app suite once on the main checkout after cherry-pick, which covers this
  test's red-to-green story for real. By inspection: before the sweep 142
  files matched `plato` case-insensitively in scope, so the scan (which
  walks the same directories, case-insensitive) would have found offences
  and failed had it run to completion -- but that is not a substitute for an
  actual red run.
