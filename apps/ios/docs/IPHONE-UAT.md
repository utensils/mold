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
