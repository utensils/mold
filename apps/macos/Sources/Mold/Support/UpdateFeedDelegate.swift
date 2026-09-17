import Foundation
import Sparkle

/// Which feed Sparkle asks, decided at every check from the stored channel.
///
/// `SUFeedURL` in `Info.plist` names the stable feed and is the default; the
/// updater's own `setFeedURL` is DEPRECATED in Sparkle 2 precisely because a
/// feed written into user defaults outlives the app that wrote it. The
/// supported way to move an app between streams is this delegate method
/// (https://sparkle-project.org/documentation/api-reference/Protocols/SPUUpdaterDelegate.html,
/// `feedURLStringForUpdater:`), which is asked afresh each time, so switching
/// the picker in Settings takes effect on the very next check with nothing
/// persisted on Sparkle's side.
///
/// `nonisolated`: Sparkle makes no promise about which thread it asks on, and
/// the project defaults to `@MainActor`. Nothing here needs the main actor --
/// it is one defaults read and a pure mapping.
nonisolated final class UpdateFeedDelegate: NSObject, SPUUpdaterDelegate {
    /// The preferences suite to read the channel from. `nil` means
    /// `UserDefaults.standard`, which is what a shipped build always uses:
    /// `AppStorageSuite` only swaps in the scratch suite under
    /// `MOLD_NATIVE_FRESH`, and `UpdaterActivation` refuses to build an
    /// updater at all in that case. The parameter exists so a test can point
    /// this at a scratch suite without a launched app.
    private let suiteName: String?

    init(suiteName: String? = nil) {
        self.suiteName = suiteName
        super.init()
    }

    private var defaults: UserDefaults {
        guard let suiteName, let suite = UserDefaults(suiteName: suiteName) else { return .standard }
        return suite
    }

    /// The decision itself, without Sparkle's argument -- so a test can ask
    /// it the same question a check asks, against a scratch suite, with no
    /// `SPUUpdater` and no network.
    ///
    /// Nothing is cached: the stored channel is read on every call, because
    /// Sparkle asks on every check and the picker must take effect on the
    /// next one.
    func currentFeedURLString() -> String {
        UpdateFeed.url(stored: defaults.string(forKey: UpdateChannel.storageKey))
    }

    func feedURLString(for updater: SPUUpdater) -> String? {
        currentFeedURLString()
    }
}
