# Native gallery badge validation

The iOS and macOS galleries preserve their session-only visit baseline and
whole-gallery seen behavior. Opening selected picture/video/3-D media removes
its New label immediately. Neighboring prepared pages and previews remain unread
within that visit. Home Screen/Dock counts persist locally and clear when Library
opens; they use merged copy identities and exclude hidden collections and Trash.

## Automated acceptance

`LibraryLongPressTests.testAppIconCountsNewMediaAndViewingClearsCurrentVisitImmediately`
pairs a read-only loopback fixture and checks the actual SpringBoard icon:

1. Establish a first-visit baseline, then add a picture and video while Generate
   is showing. The icon must show two new media.
2. Open Library and its picture. Returning to the grid immediately removes only
   that picture's New label; the neighboring video retains New until opened.
3. Confirm the icon has no badge, and the next visit preserves the old clearing
   behavior.
4. Add another picture, refresh after a settled background transition, and verify
   the icon count before and after terminating/relaunching the app. The first
   gallery visit after relaunch still has the old session-only baseline.
5. Assert that the fixture received no generation requests.

The existing clip/machine-placement regression independently checks immediate
viewer-return clearing. Shared `LibraryNewMediaTests` preserves next-visit
semantics; `LibraryUnreadLedgerTests` covers distinct same-name outputs, renamed
copies, persistence, visibility, unavailable inventories and host removal.
`AppIconBadgeTests` checks suspended permission requests/latest-count ordering
and no prompts for background or zero-count writes.

## macOS acceptance

Use a remote-only Debug build, `MOLD_NATIVE_FRESH=1`, a disposable `MOLD_HOME`,
and a read-only loopback gallery. Establish the baseline in Library, leave for
Generate, add media via the fixture, then verify the actual Dock count. Return to
Library, open each new media, and verify immediate label clearing. Use the
existing gallery navigation and viewer controls; never submit a render.

## Recorded validation

On 2026-10-09, the shared suite passed 1,265 tests. The final iOS unit suite
passed 220 tests, and the two initial badge/clip interaction checks passed on an
iPhone 17 Pro Simulator running iOS 26.5 (222 passing tests in that result).
Native architecture lints and the remote-only macOS Debug build passed.
Additional relaunch and macOS interaction results are recorded in PR #1840.
No physical-device UAT is claimed.

An earlier Home Screen failure exposed alerts-only notification authorization:
the Debug fixture had not requested badges, and the production authorizer only
requested access for `notDetermined`. The authorizer now also registers badge
options for previously authorized notifications, while respecting denied access
and iOS Settings choices. The actual two-media Home Screen check passes with
this correction.
