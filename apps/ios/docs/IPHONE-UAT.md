# iPhone usability follow-up

Scope: native app, iPhone only; exploratory UAT against existing media and
installed models, with no generation submissions. Simulator cannot establish
physical speaker output, camera pairing, or device-only extension delivery.

## Findings and implementation plan

| Finding | Reproduction | Improvement | Verification |
| --- | --- | --- | --- |
| First-run Machines clips its explanation and action at AX5 | Open Machines with no saved hosts, largest text and nearby discovery | Keep discovery count in the scrolling explanation; wrap Add; reserve opaque space above tabs for the action | First-run iPhone UAT |
| More Options becomes unreadable at AX5 | Select an installed picture model, set largest accessibility text, open Options | Give Shape, Steps, Batch and Length separate form rows; wrap values; remove the inert More Options button inside its own sheet | Populated iPhone UI regression and exploratory UAT |
| Seed choice ignores large text and loses its label | In the same sheet, scroll to Look | Use a labeled system menu for Repeat this look | AX5 UAT |
| Clip and 3-D drafts reopen as Still picture | Choose either kind, background, terminate, relaunch before model profiles arrive | Restore kind from the saved recipe once profiles arrive, preserving valid authored options and reconciling audio capabilities | Failing-then-passing cold-start controller tests and relaunch UAT |
| Empty model search gives no feedback | Search installed models for an unmatched word | Explain that no models match and allow clearing the search | Populated UI regression and UAT |
| Models pane chooser ignores large text | Open Installed/Discover at AX5 | Use a labeled system menu that scales with the text | iPhone UAT |
| Offline inventory falsely claims zero installed | Add an unreachable local test machine, open its detail and Models | Distinguish an unread inventory from a confirmed empty one; omit unknown counts | Model-store regression and UAT |
| Removing a machine leaves a dead detail page | Remove the temporary machine from its detail | Return to Machines after removal | iPhone UAT |
| Info has no explicit dismissal action | Open Info at its large detent | Add a standard Done button | Viewer UAT |
| Shared video has an extensionless temporary filename | Share an existing clip | Preserve each original filename in an owned export directory; clean it after sharing/copying/saving | Failing-then-passing sharing test and system share-sheet UAT |
| Slow machine responses replace the saved model | Restore a clip while another machine answers first | Keep the saved choice pending; explicit kind/model choice cancels restoration | Two-host failing-then-passing controller regressions |
| Offline Generate and Queue claim missing models or an empty queue | Open both tabs with only unreachable machines | Explain unavailable data and direct the user to Machines; invalidate failed queue reads | Controller/queue tests and SE UAT |
| A removed machine silently drops a selected export | Remove a source machine before an export begins | Abort the entire export and clean staged files, preserving entry/file alignment | Export regression |
| Viewer actions are covered by the main tabs | Open an existing print, try Info at the bottom | Hide the main tab bar while the viewer is open, preserving its own controls and back navigation | Viewer UAT on iPhone |

## Verification record

- Baseline exploratory screenshots reproduced the Options and viewer failures.
- Cold-start regression failed for both clip and mesh with `.picture` and no
  recipe. A second assertion exposed sound being disabled on restored clip
  drafts until recipe capabilities were reconciled.
- Populated UI tests use a loopback fixture with generated model profiles.
  The fixture cannot generate, download, or modify remote data.
- Baseline Pro exploration covered Generate long text and model/kind choices,
  Library browsing/search/selection and existing video playback, Models
  installed/discover/search, Machines add/edit/cancel/offline/remove, invalid
  address/pairing and scanner fallback, and Settings in light/default and
  dark/AX5. The Library selection toolbar was correctly positioned.
- After the final draft, offline and export fixes, all 107 native unit tests
  passed. Native architecture lints and documentation verification passed.
- Populated SE AX5 Options/search regression passed, including test-machine
  removal. The updated viewer's actions are visible and Info opens in ordinary
  pushed navigation; the agent restored the original favourite state after
  testing a neutral existing print.
- Post-fix Pro/SE exploration verified all visible fixes, including Info Done,
  video-aware Share, clip/audio cold restoration, first-run AX5 scrolling,
  offline inventory/Generate guidance and machine-removal navigation.
- The SE populated fixture regression also verified landscape prompt typing,
  keyboard dismissal, reachable Generate (never pressed), and Options, followed
  by portrait AX5 Options and no-match search.
- The SE offline Queue regression reaches the complete explanation above the
  pinned action at AX5. A generic full-screen swipe started on the action;
  dragging the visible explanation proves that this was a test-gesture issue.
- Independent peer review found and resolved staggered-host restoration,
  explicit-choice cancellation, and partial-export alignment defects.
- No generation submissions or observed crashes during exploratory UAT.
  Existing-media browsing was used; physical audio, camera pairing and
  device-only extension delivery remain outside Simulator verification.
- Full light/dark accessibility results and exact-head CI are recorded on the
  pull request. The audit includes xSmall, Large and AX5.

## Phone layout, library and cache follow-up (2026-09-29)

The TestFlight screenshot exposed an idle Generate canvas that left too little
room for the composer. At AX5 on iPhone SE, Options was initially covered by
the pinned Generate action, and the model search sheet had no visible exit
while the keyboard was open. The final phone form has Kind and Machine at the
top, full-width Generate above the tabs in portrait, extra bottom scroll travel,
and a Close Model Search action above the keyboard. Compact-height landscape
keeps Generate inside the scrollable form.

The phone Library now shows its shelf menu above the grid. Settings is directly
reachable from the main destinations; it reports image and listing storage
separately, and clearing it drops saved listings, images and stale ETags. Gallery
work avoids repeated tag extraction, per-tile host scans and tile fade
animations. The viewer keeps back navigation visible.

- Native unit suite: 110 tests in 21 suites passed; native lint passed.
- iPhone SE at AX5: first-run offline Queue explanation and Machines return
  passed; populated fixture regression passed with landscape prompt/Options,
  portrait model search, visible Close action, Options controls and cleanup.
- No generation submission was made. The fixture is loopback only.
- Peer review found and fixed races in clearing a replaced thumbnail save and
  clearing during a detached listing restore. Settings now separates listing
  size from the image cache limit.

## Library viewer follow-up (2026-09-29)

The full-screen clip page previously loaded in a paused state on first entry.
Playback now begins when that page is selected and pauses on page change or
viewer dismissal. The Library retains the last visible print while a viewer is
open and scrolls the opened tile back into view on return.

- A local audio/video fixture advanced under the selected-page playback action
  and paused when deselected.
- An iPhone SE Simulator UI run opened print 50 in a 60-print loopback gallery,
  returned with Back, found print 50 still visible, then changed to a populated
  Favourites shelf and found its first print at the top. The temporary machine
  was removed. No generation or remote mutation was used.
- Physical-device audio output and playback on a remote machine still require
  acceptance in TestFlight; Simulator confirms the page and player behavior.

## Render notifications and Generate options — 2026-09-29

- Reproduced the reported CancellationError banner with failing gallery/trash/collection refresh tests. Cancellation now retains loaded prints and produces no failure banner; genuine server errors remain visible.
- iPhone 17 Pro Simulator, iOS 26.5: opened the completed-print deep link against a loopback fixture and confirmed the viewer appeared. Notification copy/link/deduplication and the bundled AppIcon are verified by unit tests; physical-device notification delivery/presentation was not exercised.
- Visually confirmed aspect-menu icons distinguish 1:1, 4:3, 3:4, 16:9 and 9:16 in their actual proportions.
- Attached an existing picture without submitting a render; Options showed Seed = Random, Fit = Crop to fill and both alignment controls = Center. Changed to Fit with borders, then Reset restored centered Crop to fill while retaining Random.
- All 123 companion unit tests and native architecture lints passed. All 1,077 shared MoldClient tests passed, including source-fit pixels before batch admission, painted source-space mask alignment, retained Mac canvas-space masks and rotated JPEG geometry.
- The existing LibraryViewerTests scroll-position UAT failed to reach its viewer controls locally after a Simulator runner launch retry. This unrelated test is not counted as passing validation for this change.
- Independent sub-agent review identified source-mask coordinates and JPEG orientation; both were repaired and the final re-review reported no actionable findings.
- Hosted accessibility CI exposed an existing offline-queue fixture cleanup failure: a card long press could select its address instead of opening Remove, contaminating the next appearance pass. The UI test now removes via machine details, clears stale fixtures, always cleans up, and verifies Add-sheet dismissal. Its unchanged largest-text scrolling assertion passed locally in consecutive light/dark runs.
- A subsequent hosted light-mode prompt test read its accessibility value immediately after typing; the full dark-mode suite passed. The text assertion now waits up to five seconds for the same phrase and reports the actual value on failure, without retyping or relaxing the keyboard assertion. Both normal and largest-text prompt tests passed three local repetitions each; independent review approved the synchronization change.

## Notification activation crash — 2026-09-30

- Reproduced on iPhone 17 Pro Simulator, iOS 26.5, by tapping a real local notification after backgrounding. UIKit threw `NSInternalInconsistencyException: Call must be made on main thread` in `_updateSnapshotAndStateRestorationWithAction`, called by the synthesized Objective-C completion bridge for `Notifier.userNotificationCenter(_:didReceive:)`.
- Regression test first failed because the delegate's completion ran off-main. Explicit completion-handler delegates now route and complete on MainActor. All 126 iOS unit tests pass, including default/View/Favourite/dismissed/unknown responses, queue/malformed links and foreground suppression.
- `NotificationTapTests`: both warm and cold launches through real local system banners pass. The unknown-print fixture proves graceful missing-media handling. A separate manual system-notification tap (injected with `simctl push` into the same delegate) opened a populated still viewer against a loopback-only, read-only HTTP fixture from both background and terminated states; the image and Done/Share/Favourite/Info/Delete controls appeared. No render was submitted. The Simulator machine list was restored afterward.
- Notification Center's grouped rows exposed unreliable XCTest row-tap/swipe behavior; the automated regression uses the real system banner, and manual UAT uses an ungrouped Lock Screen notification.
- Architecture lint and Release Simulator build pass. The Debug-only notification fixture is absent from the Release binary. The native macOS notification delegate already explicitly routes and completes on MainActor and was not changed. Physical-device confirmation remains unverified: the paired iPhone was locked when diagnostics were requested.
- Follow-up: both notification UI tests also pass with an empty saved-machine list. Dismissing the viewer verifies the Generate destination instead of assuming a populated model chooser.

## Live Activity card — 2026-09-30

- iPhone 17 Pro Simulator, iOS 26.5: visually checked the real ActivityKit Lock Screen card in light/dark appearances. The prompt is subordinate to the status, progress spans the card, and machine/queue labels share one footer instead of a large monospaced stack. The fallback preview uses a blue gradient tile.
- At accessibility-extra-extra-extra-large, the compact variant retains the status, progress and 44 pt Stop target without clipping. The finished variant uses a checkmark when no preview exists and removes the repeated machine footer. Tapping the finished activity opens the missing-print viewer and keeps the app alive.
- Presentation contracts cover legacy step/machine separation, unknown progress, stale messaging and terminal controls. The ActivityKit payload and deep-link contract are unchanged.
- UAT uses the Debug-only `--live-activity-fixture` launch argument; no render was submitted. Physical-device presentation remains to be confirmed in TestFlight.
- All 129 iOS unit tests and architecture lints pass. The Release Simulator build passes, with both Debug UAT fixture types absent from the binary. Also visually checked a real preview image and the no-step "Working on it" state; the latter shows no fabricated progress bar.
- The final suite has 130 passing tests, including a height invariant that measures all three normal rows with UIKit text metrics and accounts for the card padding. The final Release build also passes.

## Library selection, title and media filters — 2026-10-01

- Reproduced selection moving a scrolled tile upward by 39 points with a
  failing iPhone Simulator regression. The fixed grid observes visibility
  separately from explicit viewer-return and pinch scroll requests.
- iPhone 17 Pro Simulator, iOS 26.5: tapping multiple prints retains tile
  position within two points. Horizontal finger sweeps select and deselect a
  range; holding near the bottom edge scrolls the grid upward to expose later
  rows. A subsequent vertical swipe still scrolls normally in Select mode.
- Videos filters pictures out, remains active when switching to Favourites,
  and All Media restores the unfiltered shelf. The loopback fixture supplies
  picture, clip and mesh rows; no generation or remote mutation occurs.
- The retained-position screenshot shows one inline All Prints title/menu
  above the grid, with no duplicate blue shelf link. Focused title contrast
  checks pass at extra-small, normal and AX5 text. Viewer-return UI coverage
  and all 134 native unit tests pass.
- Independent review approved the selection lifecycle and shared filter
  contract. Physical-device gesture behavior remains unverified; the evidence
  above is Simulator UAT.
- iPad Pro 13-inch Simulator, iPadOS 26.5: the same selection/edge-scroll/filter
  regression passes against 300 mixed rows. The populated model/Options,
  landscape prompt and largest-text model-search regression also passes. The
  two explicitly phone-specific layout/Settings-route tests skip on iPad;
  the full shell audit retains iPad composer, Machines, sidebar and Settings
  coverage at all three text sizes in both appearances.
- The full iPad run exposed a pre-existing Collections modal audit timeout.
  The same AX5 test on main (`158e7690e`) reproduced it. A runtime sample
  caught the private Dynamic Type auditor spinning in UIKit floating-tab
  pagination. Hiding the presenting tab chrome while Collections is open
  makes the actual Large/AX5 modal pass every original audit type; no audit
  type or issue was exempted.
- Follow-up iPad regression passes after both Done and a full downward
  dismissal drag: the Library tab becomes hittable again and the same print's
  vertical position is restored within two points. Viewer-return also passes,
  explicitly checking that the main Library tab stays hidden in the viewer.
- Populated Library/Search AX5 audits exposed status notices covering the
  date heading and first row. The focused date-heading contrast regression
  failed at all three sizes before replacing the overlay with a top safe-area
  inset; notices now reserve their own space above the grid.
- The saved-gallery notice itself failed small/normal-text contrast on its
  translucent material. Primary text on an opaque semantic background passes
  focused XS/Large/AX5 contrast, with the fixture deliberately taken offline.

## Library long press and notification appearance (2026-10-02)

A populated iPhone Simulator reproduced an EXC_BREAKPOINT on long press:
`No Observable object of type ThumbnailLoader found` in the UIKit-hosted
context-menu preview. Both context-menu and drag preview roots now receive
the existing loader explicitly. The Library regression opens the menu, uses
its settings action, and opens/dismisses the viewer without crashing.

Source-library UAT exposed healthy reconnect checks temporarily removing the
source controls. A routine refresh now retains the last answering state until
its response; a unit regression covers success and actual authorization failure.
The source chooser remains usable after a long press and attaches a real
loopback PNG. Notification Center warm/cold taps pass, and the app icon catalog
uses the same authored image for Any/Dark appearances. Card backgrounds remain
system controlled. Tests retain notification screenshots before activation.

The native unit suite passes (136 tests); lint and CI routing/branding/runner
contracts pass. Independent peer review corrected the PNG fixture checksum and
isolated controller tests from saved machine preferences. No generation was
submitted; physical-device acceptance remains separate from Simulator evidence.

The iPad sweep also exposed source selection fetching an offline merged lead.
The picker now prefers a reachable copy; the source regression deliberately
pairs a saved offline copy and a live copy before selection. iPad drag and
context-menu interactions pass, including opening the viewer afterward.

## Live Activity material correction (2026-10-02)

A physical-device report showed the stale Live Activity with a pale custom
background and white text. Removing its explicit UIKit background tint restores
ActivityKit's system material, which resolves background and semantic text
appearance together. The change applies outside the content-state branches.

Independent UAT on a fresh iOS 26.5 iPhone 17 Pro Simulator confirmed readable
running cards on the Lock Screen in light and dark appearances. The original
five-second stale fixture retained its launch arguments, but the Simulator kept
showing the running state after the deadline and sleep/wake cycles. Stale-state
visual acceptance remains unverified; existing content-model tests cover its
refresh title, progress suppression, and retained Stop control. No fixture or
production stale-date changes were made for this check.

## Visible server model unloading (2026-10-02)

Machines → Models → Installed now exposes per-model Unload and Unload All Models
under Server Memory. The regression uses an opt-in loopback fixture with two
resident, downloaded models: unload one, unload the remaining models, and verify
the installed model ID remains reachable. It never contacts a live inference
server or submits a generation. The fixture restores residency for the audit.

Local iOS 26.5 iPhone 17 Pro and iPad Pro 11-inch (M5) Simulator checks cover
XS, Large and AX5 in light and dark. Every case exercises the visible controls,
44-point targets, retained inventory, and contrast, Dynamic Type, clipping,
descriptions, traits and element detection. Contrast exclusions are restricted
to List content obscured by bottom system glass; bar controls remain audited.

UAT required a full-width wrapping Unload All Models title. On iPad, per-machine
Models now hides inherited floating tab chrome and restores it on Back. A named
installed-family heading failed Dynamic Type prediction even with an explicit
text style; ordinary heading rows within the family cards pass without adding
a Dynamic Type or clipping exception. Family headings retain VoiceOver header
traits, and model rows retain their existing context menus and swipe actions.

The native unit suite passes 140 tests, including selected-host nil/nil unload,
named unload, inventory refresh, server refusal, older residency metadata,
same-host serialization and independent-host concurrency. Native lint, routing
contracts and independent source/visual peer review pass. Simulator evidence
remains separate from physical-device or live-GPU-server acceptance.

### First-frame aspect selection (2026-10-04)

On an isolated iPhone 17 Pro Simulator running iOS 26.5, the loopback
`testStartFrameSelectsClosestAspectAndMenuMarksIt` imported a 160×90 first frame
and a 90×160 closing frame through Choose from Library for the H3 FL2VA 768p
recipe. Shape became 16:9 (1024×576), stayed there after the closing frame, and
the top-level aspect menu displayed a checkmark and “16:9 · Selected”. Other
aspects remained available without the selected label. The captured menu was
visually inspected. This test performs no inference and uses disposable media.
Shared attachment regressions cover both Wan and H3, replacement and manual
intent, plus the centered crop-fill default. Physical-device UAT was not run.

## 2026-10-04 — permission recovery and Photos export

The original Photos change callback reproduced an `EXC_BREAKPOINT` /
`_dispatch_assert_queue_fail` on iOS 26.5 Simulator after authorizing a real
H.264 MP4 save. The stack entered Swift's actor check in
`PrintActions.saveToPhotos` from PhotoKit's serial queue. PhotosWriter now owns a
nonisolated callback with immutable file descriptors; the same MP4 reached
“Saved to Photos” without that crash. Cold Simulator Photos imports can take
longer than 15 seconds, so the regression allows 60 seconds for completion.

`LibraryViewerTests` also exercises first denial, the native recovery alert,
Not Now, repeated denied saves, and Open Settings reaching the Settings app.
PermissionRecoveryTests cover granted/limited/undetermined/denied/restricted
policy and local-network error classification. PrintActionsTests distinguish
MP4/MOV/M4V video resources from GIF/WebP/APNG/still photo resources, including
older-host listings without `format`. The permission inventory and Apple
references are in [PERMISSIONS.md](PERMISSIONS.md).

These are Simulator observations. Physical camera capture, VisionKit QR
scanning, Screen Time/MDM restrictions and Local Network policy changes remain
physical-device checks. No model generation was submitted for this work.

## 2026-10-05 — media export parity

Real-host conversion UAT used an isolated loopback Mold server, a disposable
six-second MP4 and a small GLB. No inference or generation was submitted.
The production Tauri iOS app on iPhone 17 Pro / iOS 26.5 saved three GIFs into
its visible Mold folder. ImageIO decoded every delivered frame: Loop/Forever/0
produced 72 frames at 80 ms; Bounce/Forever/250 produced 142 frames, with 140 at
80 ms and the two turn frames at 330 ms; resetting with No pause (0 ms) produced
142 frames at 80 ms. Existing exports were preserved with collision names
`(2)` and `(3)`. Scrolling reached every option and the Export action. APNG
selection removed GIF-specific controls and successfully opened the native
share sheet with a PNG file.

The shared browser export UI against the same real server also delivered GIF,
APNG and animated WebP, transparent mesh GIF, STL, PLY, OBJ and ZIP. Decoded
outputs confirmed GIF timing, APNG animation, WebP animation with duplicate-frame
coalescing, transparent turntable alpha and geometry/archive contents. An
older-host capability fixture omitted `gif_pause`; the visible control and
request field disappeared while real GIF conversion still worked. Browser
console checks reported zero errors and warnings.

Native SwiftUI delivery tests use valid media from an isolated HTTP fixture,
including real PhotoKit writes and native Share/Files presentation. Those tests
verify client requests and delivery ownership; the fixture does not encode the
requested GIF timing. The separate real-server output checks above establish
encoder behavior. Simulator evidence does not establish physical-device UAT.

ImageIO inspection of the actual PhotoKit-persisted GIF resource confirmed all
three fixture frames and their 100 ms delays were retained. A separate native
unit regression suspends Photos authorization, cancels the export, then returns
denial: no recovery alert, error/status, delivery or staged directory survives.
It failed before the cancellation fence and passes with all 186 native unit
tests after the fix.
