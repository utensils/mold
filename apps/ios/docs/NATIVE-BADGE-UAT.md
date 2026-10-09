# Native gallery badge follow-up — incomplete UAT

Recorded 2026-10-09. The owner requested an immediate wrap-up after a machine
reset; no further builds, tests or app launches were performed afterward.

## Intended behavior

- Preserve the existing session-only gallery-visit baseline and whole-gallery
  seen behavior on iOS and macOS.
- Opening the selected picture/video/3-D page removes its New label immediately;
  neighboring preloaded pages and previews do not count as viewing.
- Home Screen/Dock count new visible merged media before Library opens. Opening
  Library clears the icon count. Persist icon read state locally across launches.

## Completed verification

- Failing shared visit regression was recorded before adding `Visit.markViewed`.
- Shared MoldClient suite: **1,265 tests passed**, including merged-copy identity,
  unequal-size filename collisions, renamed read copies, visibility, unavailable
  inventories, host removal and persisted read state.
- iOS native unit suite: **220 tests passed** before the final ledger identity
  revision. AppIconBadge tests cover permission suspension/latest-count ordering
  and no prompts for background or zero-count updates.
- Native iOS and macOS architecture lints passed.
- macOS Debug build passed after correcting startup Dock access to use
  `NSApplication.shared`; direct `NSApp` access before initialization had crashed
  the first isolated UAT launch.
- Independent peer review identified merge identity and macOS visibility-refresh
  defects; both were fixed. Re-review reported no remaining source findings
  before the final startup-access correction.

## Outstanding verification / known failure

The iPhone 17 Pro Simulator (iOS 26.5) Home Screen test did **not pass**.
Persisted gallery state contained the two newly added media IDs, but the actual
SpringBoard icon did not show the expected count. Notification permission/badge
settings and delivery still require diagnosis. Do not consider the icon feature
accepted based on the internal count alone.

The final native unit suite, same-visit picture/video transitions, icon persistence
and zero-count checks, macOS gallery/Dock interactions, and final peer review with
runtime results remain incomplete. No physical-device UAT was performed.

The preceding landscape playback fix is separately merged as PR #1839. This
follow-up must remain a draft until its remaining acceptance checks pass.
