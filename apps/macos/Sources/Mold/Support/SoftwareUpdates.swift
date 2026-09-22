import Combine
import Foundation
import Sparkle

/// Mold's updater, or nothing at all.
///
/// ABSENT, not disabled: in a Debug build, under the `MOLD_NATIVE_FRESH` UAT
/// suite and in the unit-test host, `shared` is `nil`, no
/// `SPUStandardUpdaterController` is ever constructed, and the app-menu
/// command is simply not there. Settings still explains why this build cannot
/// update, without constructing Sparkle or reading a feed.
///
/// Sparkle's own settings are the single authority for the two toggles
/// (`automaticallyChecksForUpdates`, `automaticallyDownloadsUpdates`): they
/// are backed by the host bundle's user defaults and the SCHEDULER reads
/// them, so a shadow `@AppStorage` copy beside them would be a second answer
/// that nothing acts on.
@MainActor
@Observable
final class SoftwareUpdates {
    /// Built once, on first use. `UpdaterActivation` is the whole gate.
    static let shared: SoftwareUpdates? =
        UpdaterActivation.isEnabled() ? SoftwareUpdates() : nil

    let controller: SPUStandardUpdaterController

    /// Mirrors Sparkle's `canCheckForUpdates`, which is false while a check
    /// is already running. A menu item bound straight to it would not
    /// re-evaluate -- the property is KVO, not `@Observable` -- so it is
    /// observed here and republished, which is the shape Sparkle's own
    /// SwiftUI recipe uses (https://sparkle-project.org/documentation/programmatic-setup).
    private(set) var canCheckForUpdates = false

    /// Sparkle holds its delegate WEAKLY, so this reference is what keeps the
    /// feed choice alive for the life of the app.
    private let feed: UpdateFeedDelegate
    private var watch: AnyCancellable?

    private init() {
        let feed = UpdateFeedDelegate()
        self.feed = feed
        controller = SPUStandardUpdaterController(
            startingUpdater: true, updaterDelegate: feed, userDriverDelegate: nil)
        canCheckForUpdates = controller.updater.canCheckForUpdates
        watch = controller.updater.publisher(for: \.canCheckForUpdates)
            // Sparkle posts this from wherever its check is; `RunLoop.main`
            // is what makes the `assumeIsolated` below sound rather than a
            // hope.
            .receive(on: RunLoop.main)
            .sink { [weak self] value in
                MainActor.assumeIsolated { self?.canCheckForUpdates = value }
            }
    }

    func checkForUpdates() {
        controller.updater.checkForUpdates()
    }

    /// The stored channel, and the feed it resolves to. The picker writes
    /// through here so the one unknown-value rule in `UpdateChannel` is the
    /// only one anywhere.
    var channel: UpdateChannel {
        get { UpdateChannel(stored: AppStorageSuite.defaults.string(forKey: UpdateChannel.storageKey)) }
        set {
            AppStorageSuite.defaults.set(newValue.rawValue, forKey: UpdateChannel.storageKey)
            // The delegate is asked afresh at every check, but nothing was
            // asking: `SUScheduledCheckInterval` is a day, so moving to
            // Nightly could sit idle until tomorrow. `resetUpdateCycle` is
            // Sparkle's own hook for exactly this -- it re-reads the feed URL
            // and, when it differs from the last one checked, schedules the
            // next check immediately (review F5#8).
            controller.updater.resetUpdateCycle()
        }
    }

    var automaticallyChecksForUpdates: Bool {
        get { controller.updater.automaticallyChecksForUpdates }
        set { controller.updater.automaticallyChecksForUpdates = newValue }
    }

    var automaticallyDownloadsUpdates: Bool {
        get { controller.updater.automaticallyDownloadsUpdates }
        set { controller.updater.automaticallyDownloadsUpdates = newValue }
    }

    var lastUpdateCheckDate: Date? {
        controller.updater.lastUpdateCheckDate
    }
}
