# Hidden collection acceptance checks

2026-09-30. Native iOS/iPadOS 26.5 Simulator checks against a loopback fixture;
no physical-device validation or live inference was performed. Collection PATCH
requests change only `FixtureMachine`'s in-memory collection.
Each fixture registers teardown before pairing: restore All Prints, remove the
exact loopback machine, verify its card disappears, terminate the app, and stop
the listener even when an interaction fails. This keeps the full shell audit
free of stale pairings from earlier tests.

## Interaction coverage

`HiddenCollectionTests.testHideShowAndOpenHiddenCollection` drives the native UI:

1. Pair the loopback machine with two prints, one filed into UAT Drafts.
2. Open Library → View Options → Manage Collections.
3. Enable Hide from All Prints and verify its filed print disappears from All Prints.
4. Open the hidden collection and verify its print remains accessible.
5. Disable hiding and verify All Prints shows both prints again.

Passing runs:

| Device | Result | Test result |
| --- | --- | --- |
| iPhone 17 Pro Simulator | 1 test, 0 failures | `/tmp/mold-hidden-collection-test/DerivedData/Logs/Test/Test-MoldCompanion-2026.09.30_09-12-11--0700.xcresult` |
| iPad Pro 11-inch (M5) Simulator | 1 test, 0 failures | `/tmp/mold-hidden-collection-test/DerivedData/Logs/Test/Test-MoldCompanion-2026.09.30_09-21-02--0700.xcresult` |

The four screenshot attachments per device show the enabled hidden toggle,
All Prints without the hidden member, explicit hidden-shelf access, and restored
All Prints. Exports are under `/tmp/mold-hidden-uat-iphone-evidence/` and
`/tmp/mold-hidden-uat-ipad-evidence/`, with filenames in each `manifest.json`.

## Accessibility matrix

Focused fixture-backed tests are `testCollectionsExtraSmallAccessibility`,
`testCollectionsLargeAccessibility`, and `testCollectionsAX5Accessibility` in
`HiddenCollectionTests`. Each audits Collections at xSmall, Large, and AX5,
on iPhone and iPad, in light and dark appearance. Contrast is checked before
Dynamic Type, clipping, hit region, and sufficient description.

Contrast excludes only the dimmed background outside actual Collections list
and navigation-bar descendants. Native navigation-bar Dynamic Type caps use the
same Large Content Viewer exception as the existing shell audits; their contrast,
clipping, and hit regions remain checked. Custom collection content has no
Dynamic Type, clipping, or contrast exemption.

The initial matrix caught an undersized collection-open target, small blue text
contrast, and label sizing. The collection-open button now has a full-width
44-point minimum target, primary body text, and a vertically flexible title.

`LibraryCollectionPickerTests` separately checks the populated phone Library's
inline shelf picker contrast at all three text sizes. The picker uses primary
body text. These focused checks retain its root and actual descendants in the
audit and leave unknown targets as failures.
Run `apps/ios/scripts/audit-library-collection-picker.sh` for a disposable phone
Simulator sweep in both appearances; its result bundles are retained and its
Simulator is removed on exit.

Light appearance results on the final sheet implementation:

| Device | xSmall | Large | AX5 |
| --- | --- | --- | --- |
| iPhone 17 Pro Simulator | Pass | Pass | Pass |
| iPad Pro 11-inch (M5) Simulator | Pass after clean simulator reset | Pass | Pass |

The light matrix result is
`/tmp/mold-hidden-collection-test/DerivedData/Logs/Test/Test-MoldCompanion-2026.09.30_09-41-22--0700.xcresult`.
Its iPad xSmall run timed out in XCTest's auditor without a content finding;
the clean repeat passed in
`/tmp/mold-hidden-collection-test/DerivedData/Logs/Test/Test-MoldCompanion-2026.09.30_09-53-18--0700.xcresult`.
The focused harness allows one retry for the same auditor timeout domains the
existing shell audits handle; it never ignores accessibility findings.
