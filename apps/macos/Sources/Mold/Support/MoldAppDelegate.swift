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
    /// Set by the composition root, which owns all of them.
    var engine: MoldEngine?
    var materializer: PrintMaterializer?
    var thumbnails: ThumbnailCache?
    var landedPrints: LandedPrints?
    /// The fleet's own 10 s tick. It lives here rather than on a view because
    /// the app being active is an APPLICATION fact, and a window closing must
    /// not take the fleet's only unprompted reconciliation with it.
    var heartbeat: HostHeartbeat?
    /// The Dock is an APPLICATION surface, so its observer lives here and
    /// not on a view -- see `DockBadge`.
    let dockBadge = DockBadge()
    /// The "Finishing…" panel and the one-time reply to macOS.
    let quit = EngineQuit()
    /// Retains cold-launch clicks until the SwiftUI scene can apply them.
    let notificationResponses = NotificationResponses()

    /// Sets the notification-centre delegate, behind the same bundle guard
    /// `MoldNotifications` posts behind -- installing a delegate touches
    /// `UNUserNotificationCenter` too, and outside a real `.app` that aborts
    /// the process just as posting would.
    func applicationWillFinishLaunching(_ notification: Notification) {
        guard MoldNotifications.isInsideBundle() else { return }
        UNUserNotificationCenter.current().delegate = self
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Appearance is a plain AppKit call and applies outside a real
        // `.app` too; only the notification-center setup needs a bundle.
        Appearance.stored(in: AppStorageSuite.defaults).apply()
        #if DEBUG
        UATScript.runIfRequested(responses: notificationResponses)
        #endif
        // Touching `shared` is what builds the updater; in a build with no
        // updater this is `nil` and nothing is constructed, scheduled or
        // fetched. It is NOT behind the notification bundle guard: `.commands` is
        // evaluated during scene construction and `UpdateCommands` reads
        // `shared` there, so a guard here would already be too late. The
        // bundle question is asked where it can hold instead, as the fourth
        // condition in `UpdaterActivation` (review F5#7).
        _ = SoftwareUpdates.shared
    }

    /// Mirrors `NSApp.isActive` onto `LandedPrints`, which is what decides
    /// whether a `gallery_added` frame counts and clears the badge on the
    /// transition back to `true` (decision 21).
    func applicationDidBecomeActive(_ notification: Notification) {
        landedPrints?.isActive = true
        // The same signal, one meaning: while Mold is frontmost its panes are
        // being read, so the machines that cannot stream are asked on a tick.
        heartbeat?.start()
    }

    func applicationDidResignActive(_ notification: Notification) {
        landedPrints?.isActive = false
        heartbeat?.stop()
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        // The caches are disposable and local, so they go first and without
        // ceremony: whatever they hold can be fetched again. BOTH of them --
        // the thumbnails are a second on-disk copy of somebody's library, and
        // the README's "emptied when Mold quits" was only ever true of one.
        materializer?.purge()
        thumbnails?.purge()

        // EVERY state with an engine thread in it, not just `.running`:
        // `.starting` is inside `recover_storage` or the one-time v2 -> v3
        // authority upgrade, and `.stopping` is mid-drain after Stop Engine.
        // Both used to be answered `.terminateNow`, which is the hard kill the
        // whole drain exists to avoid (review F2).
        guard let engine, engine.isDraining else { return .terminateNow }
        // Answering "later" is what keeps macOS from killing the process
        // mid-publish; the panel keeps that wait from reading as a hang, and
        // it carries the one way out (review 05-M7).
        quit.present(seconds: EngineShutdownBudget.totalSeconds)
        Task { [quit] in
            await engine.finishForQuit()
            quit.reply()
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
        withCompletionHandler completionHandler: @escaping @Sendable () -> Void
    ) {
        // AppKit does not promise the main thread here, so the hop is
        // explicit rather than `MainActor.assumeIsolated`, which would trap
        // if that promise is ever broken.
        let userInfo = response.notification.request.content.userInfo as? [String: String] ?? [:]
        let action = response.actionIdentifier
        Task { @MainActor [weak self] in
            guard let self else { completionHandler(); return }
            notificationResponses.receive(action: action, userInfo: userInfo, completion: completionHandler)
        }
    }
}
