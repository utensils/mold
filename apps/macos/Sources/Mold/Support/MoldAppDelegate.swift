import AppKit

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
