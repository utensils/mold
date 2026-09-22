# Native macOS notification activation

## Evidence

- The native app uses one SwiftUI `Window(id: "main")`. This limits windows within one process, not the number of processes Launch Services may start.
- Neither the source Info.plist nor the installed application declares `LSMultipleInstancesProhibited`. Four indexed bundles share the native identifier on the development machine. This makes duplicate launch targets plausible, but does not establish which bundle handled the original notification.
- An isolated copy of the current Debug build accepts two explicit Launch Services launches (`open -n`): two distinct running application PIDs were observed. No user preferences or generation jobs were used.
- The notification response callback never launches an executable. It changes navigation through an optional callback installed later by the main scene's task. A response arriving before that task is silently discarded; a response after closing the window does not explicitly reopen it.

## Plan

1. Declare the native application single-instance to Launch Services with `LSMultipleInstancesProhibited`. Verify both the built bundle and actual repeat-launch behavior. Avoid an early process-exit workaround that discards a cold-launch notification before the system delivers its payload.
2. Register the notification delegate in `applicationWillFinishLaunching`. Retain accepted routes until the composition root supplies navigation; preserve arrival order and consume them once. Ignore dismissal/unknown actions.
3. Apply the selected print or Queue route, reopen/reuse `Window(id: "main")`, and activate the application. Keep window presentation in SwiftUI's existing scene, without launching an executable or creating another window family.
4. Cover early and warm delivery, dismissal, handler replacement, notification destination/reveal semantics, and the built bundle launch policy. UAT checks duplicate launch refusal, same-PID ordinary reopen, close/reopen, and a real notification click where the local notification service permits it.
5. Run native lint and tests, obtain a separate implementation review, refresh the scoped knowledge graph and release note, then open and merge a PR after exact-head checks pass.

## Boundaries

The OS policy applies to Launch Services launches, including notification activation. Direct execution of the Mach-O by development/test scripts is not a supported end-user launch mechanism. The app must not kill another running process, migrate user settings, or modify the installed application during verification.

## Sources

- [Apple Launch Services Keys](https://developer.apple.com/library/archive/documentation/General/Reference/InfoPlistKeyReference/Articles/LaunchServicesKeys.html): `LSMultipleInstancesProhibited` rejects a separate instance in the current session and simultaneous use across sessions.
- [Apple notification delegate](https://developer.apple.com/documentation/usernotifications/unusernotificationcenterdelegate): install the delegate before launch finishes.
- [Apple OpenWindowAction](https://developer.apple.com/documentation/swiftui/openwindowaction): target the existing `Window` scene by identifier.

## Review and verification

The separate plan reviewer approved the narrow Launch Services policy and notification lifecycle fix. A process-level exit guard/IPC was rejected because it could discard the launch response before Notification Center delivers it.

On macOS 26, the baseline disposable app admitted two processes (PIDs 38856 and 39026). With the launch policy, the first probe retained PID 40920 for both ordinary open and `open -n`. The implemented, developer-signed disposable build retained PID 61930 for ordinary open, `open -n`, and opening a second bundle copy with the same identifier. All three opens returned success and reused the running process.

An injected default response changed the live window from Generate to Queue in PID 61930 after minimizing it. The sole SwiftUI Window currently exits the app when closed; that existing policy is unchanged, so a click after that is a cold-launch case rather than an in-process reopen.

18 focused notification tests passed, including the built bundle's single-instance policy, early buffered delivery, exactly-once draining, handler replacement, completion ordering, ignored actions and print identity/navigation.

A real system banner click remains unverified: temporary ad-hoc bundles were denied notification authorization, and the developer-signed test bundle reached the OS authorization step, whose `com.apple.UserNotificationCenter` dialog the computer-use tool explicitly refuses to access. No permission database or system policy was modified to work around that restriction. Injected responses and Launch Services process probes are recorded separately from real notification delivery.

The startup UAT injected a response immediately during launch while the launch destination was explicitly Generate; it displayed Queue in a single process (PID 64445). A separate implementation reviewer found no actionable issues and confirmed callback ownership, MainActor acceptance/completion, ordered buffering, default-action filtering, and SwiftUI window reuse.

The full native application suite passed: 784 tests in 110 suites. An initial full run timed out in the existing `HeartbeatTests.oneSlowMachineDoesNotHoldUpTheRest` during concurrent desktop UAT; the unchanged full suite passed after UAT ended. Native architecture/accessibility lint and `git diff --check` passed.
