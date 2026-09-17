import AppKit
import UserNotifications

/// What has to happen before Mold goes away.
///
/// `Info.plist` sets `NSSupportsSuddenTermination` to false and says why: the
/// in-process engine drains on quit and a cut publish loses a render. Nothing
/// was actually doing that draining -- the plist asked macOS to wait, and the
/// app used the wait for nothing.
@MainActor
final class MoldAppDelegate: NSObject, NSApplicationDelegate {
    /// Set by the composition root, which owns all three.
    var engine: MoldEngine?
    var materializer: PrintMaterializer?
    var landedPrints: LandedPrints?
    /// What a notification click should do, applied by the composition root
    /// -- this delegate only decodes the payload (`MoldNotifications.swift`).
    var onNotificationRoute: ((NotificationRoute) -> Void)?

    /// Sets the notification-centre delegate, behind the same bundle guard
    /// `MoldNotifications` posts behind -- installing a delegate touches
    /// `UNUserNotificationCenter` too, and outside a real `.app` that aborts
    /// the process just as posting would.
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Before the bundle guard below: the appearance is a plain AppKit
        // call and must land outside a real `.app` too (the test host).
        Appearance.stored(in: AppStorageSuite.defaults).apply()
        guard MoldNotifications.isInsideBundle() else { return }
        UNUserNotificationCenter.current().delegate = self
    }

    /// Mirrors `NSApp.isActive` onto `LandedPrints`, which is what decides
    /// whether a `gallery_added` frame counts and clears the badge on the
    /// transition back to `true` (decision 21).
    func applicationDidBecomeActive(_ notification: Notification) {
        landedPrints?.isActive = true
    }

    func applicationDidResignActive(_ notification: Notification) {
        landedPrints?.isActive = false
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        // The cache is disposable and local, so it goes first and without
        // ceremony: whatever it holds can be fetched again.
        materializer?.purge()

        guard let engine, case .running = engine.state else { return .terminateNow }
        // `stop()` is a POST to the engine's own shutdown route and a join --
        // seconds, not instants. Answering "later" is what keeps macOS from
        // killing the process mid-publish.
        Task {
            await engine.stop()
            NSApplication.shared.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}

extension MoldAppDelegate: UNUserNotificationCenterDelegate {
    /// A click on a delivered notification. Decoding the payload is the only
    /// AppKit/`UserNotifications`-touching half of routing -- the decision
    /// itself is `NotificationRoute.route(userInfo:)`, pure and tested apart
    /// from this.
    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        defer { completionHandler() }
        // AppKit does not promise the main thread here, so the hop is
        // explicit rather than `MainActor.assumeIsolated`, which would trap
        // if that promise is ever broken.
        guard let userInfo = response.notification.request.content.userInfo as? [String: String],
              let route = NotificationRoute.route(userInfo: userInfo)
        else { return }
        Task { @MainActor [weak self] in self?.onNotificationRoute?(route) }
    }
}
